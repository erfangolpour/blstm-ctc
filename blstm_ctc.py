import pickle
import random
import pathlib
import librosa
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
from dataclasses import dataclass
from collections import defaultdict
from typing import List, Tuple, Dict, Optional
from Levenshtein import distance as levenshtein_distance


@dataclass
class Config:
    """configuration for BLSTM-CTC training and inference."""

    # training hyperparameters
    EPOCHS: int = 10
    HIDDEN_DIM: int = 20
    LEARNING_RATE: float = 0.001
    MOMENTUM: float = 0.9
    GRADIENT_CLIP: float = 5.0

    # data parameters
    MAX_SAMPLES_TO_LOAD: int = 320
    TRAIN_SPLIT: float = 0.8

    # audio processing
    N_MFCC: int = 13
    SAMPLE_RATE: int = 16000

    # CTC parameters
    BEAM_WIDTH: int = 5
    BLANK_LABEL: str = "_"

    def __post_init__(self):
        """calculate derived parameters after initialization."""
        self.HOP_LENGTH = int(self.SAMPLE_RATE * 0.010)  # 10ms hop
        self.WIN_LENGTH = int(self.SAMPLE_RATE * 0.025)  # 25ms window
        self.EPSILON = np.finfo(float).eps  # numerical stability constant


# global config instance
config = Config()


def prepare_target_sequence(
    target_labels_list: List[str], label_map: Dict[str, int]
) -> np.ndarray:
    """
    Convert a list of target label strings to their corresponding integer indices using label_map.
    """
    return np.array(
        [label_map[label] for label in target_labels_list if label in label_map],
        dtype=np.int32,
    )


def ctc_loss_and_gradient(
    probs: np.ndarray, target_labels: np.ndarray, blank_symbol: int = 0
) -> Tuple[float, np.ndarray]:
    """
    Compute the CTC loss and its gradient for a single input sequence.

    Args:
        probs: Softmax probabilities (num_classes, T)
        target_labels: Target label indices (U,)
        blank_symbol: Index of the blank symbol

    Returns:
        loss: Negative log-likelihood
        grad: Gradient of loss w.r.t. probs
    """
    T = probs.shape[1]
    U = len(target_labels)

    # handle empty target sequence
    if U == 0:
        # for empty target, only blank symbols are valid
        loss = -np.sum(np.log(np.maximum(probs[blank_symbol, :], config.EPSILON)))
        grad = probs.copy()
        grad[blank_symbol, :] -= 1.0
        return loss, grad

    # build l' (target with blanks inserted)
    l_p = np.empty(2 * U + 1, dtype=np.int32)
    l_p[::2] = blank_symbol  # fill even positions with blank
    l_p[1::2] = target_labels  # fill odd positions with target labels

    # initialize alpha
    alpha = np.zeros((len(l_p), T))
    alpha[0, 0] = probs[blank_symbol, 0]
    if len(l_p) > 1:
        alpha[1, 0] = probs[l_p[1], 0]

    # rescale
    c_t = np.maximum(np.sum(alpha[:, 0]), config.EPSILON)
    alpha[:, 0] /= c_t
    log_likelihood = np.log(c_t)

    # forward recursion
    for t in range(1, T):
        s_min = max(0, len(l_p) - 2 * (T - t))
        s_max = min(2 * t + 2, len(l_p))

        for s in range(s_min, s_max):
            current_char_idx = l_p[s]

            prev_alpha_sum = alpha[s, t - 1]
            if s > 0:
                prev_alpha_sum += alpha[s - 1, t - 1]

            if s % 2 != 0 and s > 1 and l_p[s] != l_p[s - 2]:
                # current character is different from the previous non-blank
                prev_alpha_sum += alpha[s - 2, t - 1]
            alpha[s, t] = prev_alpha_sum * probs[current_char_idx, t]

        c_t = np.maximum(np.sum(alpha[s_min:s_max, t]), config.EPSILON)
        alpha[s_min:s_max, t] /= c_t
        log_likelihood += np.log(c_t)

    final_prob_sum = alpha[len(l_p) - 1, T - 1] + alpha[len(l_p) - 2, T - 1]
    log_likelihood += np.log(np.maximum(final_prob_sum, config.EPSILON))
    loss = -log_likelihood

    # initialize beta
    beta = np.zeros((len(l_p), T))
    beta[len(l_p) - 1, T - 1] = 1.0
    if len(l_p) > 1:
        beta[len(l_p) - 2, T - 1] = 1.0

    # rescale
    d_t = np.maximum(np.sum(beta[:, T - 1]), config.EPSILON)
    beta[:, T - 1] /= d_t

    # backward recursion
    for t in range(T - 2, -1, -1):
        s_min = max(0, len(l_p) - 2 * (T - t))
        s_max = min(2 * t + 2, len(l_p))

        for s in range(s_max - 1, s_min - 1, -1):
            current_char_idx = l_p[s]

            prev_beta_sum = beta[s, t + 1]
            if s < len(l_p) - 1:
                prev_beta_sum += beta[s + 1, t + 1]

            if s % 2 != 0 and s < len(l_p) - 2 and l_p[s] != l_p[s + 2]:
                # current character is different from the next non-blank
                prev_beta_sum += beta[s + 2, t + 1]
            beta[s, t] = prev_beta_sum * probs[current_char_idx, t]

        d_t = np.maximum(np.sum(beta[s_min:s_max, t]), config.EPSILON)
        beta[s_min:s_max, t] /= d_t

    # compute gradient
    grad_term_numerator = np.zeros_like(probs)
    ab = alpha * beta

    for s_idx in range(len(l_p)):
        char_label_in_probs = l_p[s_idx]
        grad_term_numerator[char_label_in_probs, :] += ab[s_idx, :]

    Z_t_values = np.zeros(T)
    safe_probs_for_Zt = np.maximum(probs, config.EPSILON)
    for t_idx in range(T):
        current_ab_sum_over_y = 0
        for s_prime_idx in range(len(l_p)):
            char_idx_for_y = l_p[s_prime_idx]
            current_ab_sum_over_y += (
                ab[s_prime_idx, t_idx] / safe_probs_for_Zt[char_idx_for_y, t_idx]
            )
        Z_t_values[t_idx] = current_ab_sum_over_y
        if Z_t_values[t_idx] <= 0:
            Z_t_values[t_idx] = config.EPSILON

    grad = probs.copy()
    safe_probs_for_grad = np.maximum(probs, config.EPSILON)
    Z_t_broadcastable = Z_t_values.reshape(1, T)
    term_to_subtract = (grad_term_numerator / safe_probs_for_grad) / Z_t_broadcastable
    grad -= term_to_subtract

    return loss, grad


# --- LSTM Model ---
def sigmoid(x):
    """Numerically stable sigmoid."""
    return 1 / (1 + np.exp(-np.clip(x, -500, 500)))


def dsigmoid(y):
    """Derivative of sigmoid given output y."""
    return y * (1 - y)


def dtanh(y):
    """Derivative of tanh given output y."""
    return 1 - y**2


class LSTM:
    """
    Single LSTM layer (can be used for forward or backward pass).
    Handles its own weights, forward, and backward passes.
    """

    def __init__(self, input_dim: int, hidden_dim: int, direction: str = "forward"):
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.direction = direction

        xavier = np.sqrt(6.0 / (input_dim + hidden_dim))
        # input gate
        self.W_ix = np.random.uniform(-xavier, xavier, (hidden_dim, input_dim))
        self.W_ih = np.random.uniform(-xavier, xavier, (hidden_dim, hidden_dim))
        self.b_i = np.zeros((hidden_dim, 1))
        # forget gate
        self.W_fx = np.random.uniform(-xavier, xavier, (hidden_dim, input_dim))
        self.W_fh = np.random.uniform(-xavier, xavier, (hidden_dim, hidden_dim))
        self.b_f = np.zeros((hidden_dim, 1))
        # output gate
        self.W_ox = np.random.uniform(-xavier, xavier, (hidden_dim, input_dim))
        self.W_oh = np.random.uniform(-xavier, xavier, (hidden_dim, hidden_dim))
        self.b_o = np.zeros((hidden_dim, 1))
        # cell candidate
        self.W_gx = np.random.uniform(-xavier, xavier, (hidden_dim, input_dim))
        self.W_gh = np.random.uniform(-xavier, xavier, (hidden_dim, hidden_dim))
        self.b_g = np.zeros((hidden_dim, 1))

        # momentum buffers
        self._init_momentum()

    def _init_momentum(self):
        # input gate
        self.mom_W_ix = np.zeros_like(self.W_ix)
        self.mom_W_ih = np.zeros_like(self.W_ih)
        self.mom_b_i = np.zeros_like(self.b_i)
        # forget gate
        self.mom_W_fx = np.zeros_like(self.W_fx)
        self.mom_W_fh = np.zeros_like(self.W_fh)
        self.mom_b_f = np.zeros_like(self.b_f)
        # output gate
        self.mom_W_ox = np.zeros_like(self.W_ox)
        self.mom_W_oh = np.zeros_like(self.W_oh)
        self.mom_b_o = np.zeros_like(self.b_o)
        # cell candidate
        self.mom_W_gx = np.zeros_like(self.W_gx)
        self.mom_W_gh = np.zeros_like(self.W_gh)
        self.mom_b_g = np.zeros_like(self.b_g)

    def forward(self, X: np.ndarray) -> Tuple[np.ndarray, np.ndarray, dict]:
        """
        Symmetric forward pass for both directions. If direction is 'backward', input is reversed and output is reversed back.

        Args:
            X: Input features (input_dim, T)

        Returns:
            h: Hidden states (hidden_dim, T+1)
            c: Cell states (hidden_dim, T+1)
            cache: Dictionary of gate activations for backward
        """
        T = X.shape[1]
        X_proc = np.flip(X, axis=1) if self.direction == "backward" else X

        # pre-allocate all arrays
        h = np.zeros((self.hidden_dim, T + 1))
        c = np.zeros((self.hidden_dim, T + 1))
        i_t = np.zeros((self.hidden_dim, T))
        f_t = np.zeros((self.hidden_dim, T))
        o_t = np.zeros((self.hidden_dim, T))
        g_t = np.zeros((self.hidden_dim, T))
        c_tanh = np.zeros((self.hidden_dim, T))

        # pre-allocate temporary arrays
        temp_i = np.zeros((self.hidden_dim, 1))
        temp_f = np.zeros((self.hidden_dim, 1))
        temp_g = np.zeros((self.hidden_dim, 1))
        temp_o = np.zeros((self.hidden_dim, 1))

        for t in range(T):
            x_t = X_proc[:, t : t + 1]
            h_prev = h[:, t : t + 1]
            c_prev = c[:, t : t + 1]

            # vectorized gate computations with pre-allocated temps
            np.add(self.W_ix @ x_t + self.W_ih @ h_prev, self.b_i, out=temp_i)
            i_t[:, t] = sigmoid(temp_i).flatten()

            np.add(self.W_fx @ x_t + self.W_fh @ h_prev, self.b_f, out=temp_f)
            f_t[:, t] = sigmoid(temp_f).flatten()

            np.add(self.W_gx @ x_t + self.W_gh @ h_prev, self.b_g, out=temp_g)
            g_t[:, t] = np.tanh(temp_g).flatten()

            # cell state update
            c[:, t + 1] = f_t[:, t] * c_prev.flatten() + i_t[:, t] * g_t[:, t]
            c_tanh[:, t] = np.tanh(c[:, t + 1])

            np.add(self.W_ox @ x_t + self.W_oh @ h_prev, self.b_o, out=temp_o)
            o_t[:, t] = sigmoid(temp_o).flatten()
            h[:, t + 1] = o_t[:, t] * c_tanh[:, t]

        # handle backward direction output reversal
        if self.direction == "backward":
            h_out = h.copy()
            c_out = c.copy()
            h_out[:, 1:] = np.flip(h[:, 1:], axis=1)
            c_out[:, 1:] = np.flip(c[:, 1:], axis=1)
            # flip gate arrays in-place
            i_t = np.flip(i_t, axis=1)
            f_t = np.flip(f_t, axis=1)
            o_t = np.flip(o_t, axis=1)
            g_t = np.flip(g_t, axis=1)
            c_tanh = np.flip(c_tanh, axis=1)
        else:
            h_out = h
            c_out = c

        cache = {
            "i_t": i_t,
            "f_t": f_t,
            "o_t": o_t,
            "g_t": g_t,
            "c_tanh": c_tanh,
            "X": X,
            "h": h_out,
            "c": c_out,
        }
        return h_out, c_out, cache

    def backward(
        self,
        dL_dh: np.ndarray,
        cache: dict,
        learning_rate: float,
        momentum_coeff: float = 0.9,
    ):
        """
        Backward pass and parameter update for LSTM.
        Args:
            dL_dh: Gradient of loss w.r.t. hidden states (hidden_dim, T)
            cache: Dictionary from forward pass
            learning_rate: Learning rate
            momentum_coeff: Momentum coefficient
        """
        X = cache["X"]
        h = cache["h"]
        c = cache["c"]
        i_t = cache["i_t"]
        f_t = cache["f_t"]
        o_t = cache["o_t"]
        g_t = cache["g_t"]
        c_tanh = cache["c_tanh"]
        T = X.shape[1]

        # pre-allocate gradient arrays
        dW_ix = np.zeros_like(self.W_ix)
        dW_ih = np.zeros_like(self.W_ih)
        db_i = np.zeros_like(self.b_i)
        dW_fx = np.zeros_like(self.W_fx)
        dW_fh = np.zeros_like(self.W_fh)
        db_f = np.zeros_like(self.b_f)
        dW_ox = np.zeros_like(self.W_ox)
        dW_oh = np.zeros_like(self.W_oh)
        db_o = np.zeros_like(self.b_o)
        dW_gx = np.zeros_like(self.W_gx)
        dW_gh = np.zeros_like(self.W_gh)
        db_g = np.zeros_like(self.b_g)
        dh_next = np.zeros((self.hidden_dim, 1))
        dc_next = np.zeros((self.hidden_dim, 1))

        # time range computation
        time_range = reversed(range(T)) if self.direction == "forward" else range(T)

        # pre-allocate temporary arrays for gate computations
        temp_dL_do = np.zeros((self.hidden_dim, 1))
        temp_dL_dc = np.zeros((self.hidden_dim, 1))
        temp_dL_df = np.zeros((self.hidden_dim, 1))
        temp_dL_di = np.zeros((self.hidden_dim, 1))
        temp_dL_dg = np.zeros((self.hidden_dim, 1))

        for t in time_range:
            x_t = X[:, t : t + 1]
            if self.direction == "forward":
                h_prev = h[:, t : t + 1]
                c_prev = c[:, t : t + 1]
                idx = t
            else:
                h_prev = h[:, t + 1 : t + 2]
                c_prev = c[:, t + 1 : t + 2]
                idx = t

            dL_dh_t = dL_dh[:, idx : idx + 1] + dh_next

            # gate gradient computations using pre-allocated temporaries
            np.multiply(dL_dh_t, c_tanh[:, idx : idx + 1], out=temp_dL_do)
            np.multiply(temp_dL_do, dsigmoid(o_t[:, idx : idx + 1]), out=temp_dL_do)

            np.multiply(dL_dh_t, o_t[:, idx : idx + 1], out=temp_dL_dc)
            np.multiply(temp_dL_dc, dtanh(c_tanh[:, idx : idx + 1]), out=temp_dL_dc)
            temp_dL_dc += dc_next

            np.multiply(temp_dL_dc, c_prev, out=temp_dL_df)
            np.multiply(temp_dL_df, dsigmoid(f_t[:, idx : idx + 1]), out=temp_dL_df)

            np.multiply(temp_dL_dc, g_t[:, idx : idx + 1], out=temp_dL_di)
            np.multiply(temp_dL_di, dsigmoid(i_t[:, idx : idx + 1]), out=temp_dL_di)

            np.multiply(temp_dL_dc, i_t[:, idx : idx + 1], out=temp_dL_dg)
            np.multiply(temp_dL_dg, dtanh(g_t[:, idx : idx + 1]), out=temp_dL_dg)

            # gradient accumulation
            dW_ix += temp_dL_di @ x_t.T
            dW_ih += temp_dL_di @ h_prev.T
            db_i += temp_dL_di
            dW_fx += temp_dL_df @ x_t.T
            dW_fh += temp_dL_df @ h_prev.T
            db_f += temp_dL_df
            dW_ox += temp_dL_do @ x_t.T
            dW_oh += temp_dL_do @ h_prev.T
            db_o += temp_dL_do
            dW_gx += temp_dL_dg @ x_t.T
            dW_gh += temp_dL_dg @ h_prev.T
            db_g += temp_dL_dg

            # Compute gradients for next timestep
            dh_next = (
                self.W_ih.T @ temp_dL_di
                + self.W_fh.T @ temp_dL_df
                + self.W_oh.T @ temp_dL_do
                + self.W_gh.T @ temp_dL_dg
            )
            dc_next = temp_dL_dc * f_t[:, idx : idx + 1]

        # gradient clipping
        gradients = (
            dW_ix,
            dW_ih,
            db_i,
            dW_fx,
            dW_fh,
            db_f,
            dW_ox,
            dW_oh,
            db_o,
            dW_gx,
            dW_gh,
            db_g,
        )
        for grad in gradients:
            np.clip(grad, -config.GRADIENT_CLIP, config.GRADIENT_CLIP, out=grad)

        # momentum updates
        self.mom_W_ix = momentum_coeff * self.mom_W_ix - learning_rate * dW_ix
        self.W_ix += self.mom_W_ix
        self.mom_W_ih = momentum_coeff * self.mom_W_ih - learning_rate * dW_ih
        self.W_ih += self.mom_W_ih
        self.mom_b_i = momentum_coeff * self.mom_b_i - learning_rate * db_i
        self.b_i += self.mom_b_i
        self.mom_W_fx = momentum_coeff * self.mom_W_fx - learning_rate * dW_fx
        self.W_fx += self.mom_W_fx
        self.mom_W_fh = momentum_coeff * self.mom_W_fh - learning_rate * dW_fh
        self.W_fh += self.mom_W_fh
        self.mom_b_f = momentum_coeff * self.mom_b_f - learning_rate * db_f
        self.b_f += self.mom_b_f
        self.mom_W_ox = momentum_coeff * self.mom_W_ox - learning_rate * dW_ox
        self.W_ox += self.mom_W_ox
        self.mom_W_oh = momentum_coeff * self.mom_W_oh - learning_rate * dW_oh
        self.W_oh += self.mom_W_oh
        self.mom_b_o = momentum_coeff * self.mom_b_o - learning_rate * db_o
        self.b_o += self.mom_b_o
        self.mom_W_gx = momentum_coeff * self.mom_W_gx - learning_rate * dW_gx
        self.W_gx += self.mom_W_gx
        self.mom_W_gh = momentum_coeff * self.mom_W_gh - learning_rate * dW_gh
        self.W_gh += self.mom_W_gh
        self.mom_b_g = momentum_coeff * self.mom_b_g - learning_rate * db_g
        self.b_g += self.mom_b_g


class BLSTM:
    """
    Bidirectional LSTM for sequence labeling.
    Uses two LSTM instances (forward and backward).
    """

    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int):
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim

        # forward and backward LSTMs
        self.lstm_f = LSTM(input_dim, hidden_dim, direction="forward")
        self.lstm_b = LSTM(input_dim, hidden_dim, direction="backward")

        # output layer weights
        xavier = np.sqrt(6.0 / (2 * hidden_dim + output_dim))
        self.W = np.random.uniform(-xavier, xavier, (output_dim, 2 * hidden_dim))
        self.b = np.zeros((output_dim, 1))

        # momentum buffers
        self._init_momentum()

    def _init_momentum(self):
        self.mom_W = np.zeros_like(self.W)
        self.mom_b = np.zeros_like(self.b)

    def _softmax(self, x: np.ndarray) -> np.ndarray:
        e_x = np.exp(x - np.max(x, axis=0, keepdims=True))
        return e_x / np.sum(e_x, axis=0, keepdims=True)

    def forward(self, X: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        T = X.shape[1]
        # forward LSTM
        h_f, _, self.cache_f = self.lstm_f.forward(X)
        # backward LSTM
        h_b, _, self.cache_b = self.lstm_b.forward(X)
        # combine forward and backward hidden states
        self.combined_h = np.vstack((h_f[:, 1:], h_b[:, 1:]))
        # output layer
        self.output = self.W @ self.combined_h + self.b
        self.label_probs = self._softmax(self.output)
        return self.label_probs, self.output

    def backward(
        self, dL_du: np.ndarray, learning_rate: float, momentum_coeff: float = 0.9
    ):
        """
        Backward pass for BLSTM.

        Args:
            dL_du: gradient of loss w.r.t. output layer
            learning_rate: learning rate
            momentum_coeff: momentum coefficient
        """
        T = dL_du.shape[1]

        # output layer gradient computation
        dW_ho = dL_du @ self.combined_h.T
        db_ho = np.sum(dL_du, axis=1, keepdims=True)
        dL_dcombined_h = self.W.T @ dL_du

        # split gradients for forward and backward LSTMs
        dL_dh_f = dL_dcombined_h[: self.hidden_dim, :]
        dL_dh_b = dL_dcombined_h[self.hidden_dim :, :]

        # backward through LSTM layers
        self.lstm_f.backward(dL_dh_f, self.cache_f, learning_rate, momentum_coeff)
        self.lstm_b.backward(dL_dh_b, self.cache_b, learning_rate, momentum_coeff)

        # clip gradients
        np.clip(dW_ho, -config.GRADIENT_CLIP, config.GRADIENT_CLIP, out=dW_ho)
        np.clip(db_ho, -config.GRADIENT_CLIP, config.GRADIENT_CLIP, out=db_ho)

        # update weights with momentum
        self.mom_W = momentum_coeff * self.mom_W - learning_rate * dW_ho
        self.W += self.mom_W
        self.mom_b = momentum_coeff * self.mom_b - learning_rate * db_ho
        self.b += self.mom_b


# --- Decoding ---
def log_add_exp(a: float, b: float) -> float:
    """
    Numerically stable log addition of two exponentials.
    """
    if a == -float("inf"):
        return b
    if b == -float("inf"):
        return a
    max_val = max(a, b)
    return max_val + np.log(np.exp(a - max_val) + np.exp(b - max_val))


def ctc_beam_search_decoder(
    probs: np.ndarray,
    phoneme_list: List[str],
    blank_symbol_idx: int,
    beam_width: int,
) -> List[str]:
    """
    Beam search decoding for CTC.

    Args:
        probs: Softmax probabilities (num_classes, T)
        phoneme_list: List of phoneme strings
        blank_symbol_idx: Index of the blank symbol
        beam_width: Beam width for pruning

    Returns:
        Decoded phoneme sequence
    """
    num_classes, T = probs.shape
    log_probs = np.log(np.maximum(probs, config.EPSILON))

    log_prob_b_prev = {tuple(): 0.0}
    log_prob_nb_prev = {}

    scored_prefixes = []

    for t in range(T):
        log_prob_b_cur = {}
        log_prob_nb_cur = {}

        # extract current time step probabilities
        current_log_probs = log_probs[:, t]
        blank_log_prob = current_log_probs[blank_symbol_idx]

        # process paths ending in blank from previous step
        for prefix, lpb_prev in log_prob_b_prev.items():
            # blank transition
            new_score = lpb_prev + blank_log_prob
            log_prob_b_cur[prefix] = log_add_exp(
                log_prob_b_cur.get(prefix, -float("inf")), new_score
            )

            # non-blank transitions
            for c_idx in range(num_classes):
                if c_idx != blank_symbol_idx:
                    new_prefix = prefix + (c_idx,)
                    new_score = lpb_prev + current_log_probs[c_idx]
                    log_prob_nb_cur[new_prefix] = log_add_exp(
                        log_prob_nb_cur.get(new_prefix, -float("inf")), new_score
                    )

        # process paths ending in non-blank from previous step
        for prefix, lpnb_prev in log_prob_nb_prev.items():
            # blank transition
            new_score = lpnb_prev + blank_log_prob
            log_prob_b_cur[prefix] = log_add_exp(
                log_prob_b_cur.get(prefix, -float("inf")), new_score
            )

            # non-blank transitions
            last_char = prefix[-1] if prefix else None
            for c_idx in range(num_classes):
                if c_idx != blank_symbol_idx:
                    new_score = lpnb_prev + current_log_probs[c_idx]
                    next_prefix = prefix if c_idx == last_char else prefix + (c_idx,)
                    log_prob_nb_cur[next_prefix] = log_add_exp(
                        log_prob_nb_cur.get(next_prefix, -float("inf")), new_score
                    )

        # pruning step
        all_prefixes = set(log_prob_b_cur.keys()) | set(log_prob_nb_cur.keys())

        scored_prefixes.clear()
        scored_prefixes.extend(
            [
                (
                    p,
                    log_add_exp(
                        log_prob_b_cur.get(p, -float("inf")),
                        log_prob_nb_cur.get(p, -float("inf")),
                    ),
                )
                for p in all_prefixes
            ]
        )

        scored_prefixes = sorted(scored_prefixes, key=lambda x: x[1], reverse=True)[
            :beam_width
        ]

        # rebuild previous dictionaries
        log_prob_b_prev = {
            p: log_prob_b_cur.get(p, -float("inf")) for p, _ in scored_prefixes
        }
        log_prob_nb_prev = {
            p: log_prob_nb_cur.get(p, -float("inf")) for p, _ in scored_prefixes
        }

    # find best final path
    all_final_prefixes = set(log_prob_b_prev.keys()) | set(log_prob_nb_prev.keys())

    if not all_final_prefixes:
        return []

    best_prefix_indices = max(
        all_final_prefixes,
        key=lambda p: log_add_exp(
            log_prob_b_prev.get(p, -float("inf")),
            log_prob_nb_prev.get(p, -float("inf")),
        ),
    )

    return [phoneme_list[idx] for idx in best_prefix_indices]


def best_path_decode(
    probs: np.ndarray, char_list: List[str], blank_symbol_idx: int
) -> List[str]:
    """
    Greedy decoding for CTC.

    Args:
        probs: Softmax probabilities (num_classes, T)
        char_list: List of character strings
        blank_symbol_idx: Index of the blank symbol

    Returns:
        Decoded character sequence
    """
    decoded_elements = []
    for idx in filter(lambda idx: idx != blank_symbol_idx, np.argmax(probs, axis=0)):
        if not decoded_elements or char_list[idx] != decoded_elements[-1]:
            decoded_elements.append(char_list[idx])
    return decoded_elements


# --- Data Handling and Training ---
def preprocess_audio(
    file_path: pathlib.Path, sr, n_mfcc, hop_length, win_length
) -> Optional[np.ndarray]:
    """
    Preprocess audio file to extract MFCC features.

    Args:
        file_path: Path to audio file
        sr: Sample rate
        n_mfcc: Number of MFCC coefficients
        hop_length: Hop length for STFT
        win_length: Window length for STFT

    Returns:
        MFCC features (n_mfcc * 3, T) or None if error occurs
    """
    try:
        # load audio with parameters
        y, _ = librosa.load(file_path, sr=sr, mono=True)

        # extract mfcc features
        mfccs = librosa.feature.mfcc(
            y=y, sr=sr, n_mfcc=n_mfcc, hop_length=hop_length, win_length=win_length
        )

        # compute deltas
        delta_mfccs = librosa.feature.delta(mfccs)
        delta2_mfccs = librosa.feature.delta(mfccs, order=2)
        return np.concatenate((mfccs, delta_mfccs, delta2_mfccs), axis=0)
    except Exception as e:
        print(f"Error processing {file_path}: {e}")
        return None


def build_label_map_and_list(
    label_sequences: List[List[str]], blank_label: str = "_"
) -> Tuple[Dict[str, int], List[str]]:
    """
    Build label map and list from sequences (e.g., phonemes).

    Args:
        label_sequences: List of label sequences
        blank_label: Blank label symbol

    Returns:
        label_map: Dictionary mapping labels to indices
        label_list: List of labels
    """
    label_list = [blank_label] + sorted(
        {label for sequence in label_sequences for label in sequence} - {blank_label}
    )
    label_map = {label: i for i, label in enumerate(label_list)}
    return label_map, label_list


def load_timit_subset(
    data_dir: pathlib.Path, max_samples: Optional[int] = None
) -> List[Tuple[pathlib.Path, List[str], str]]:
    """
    Load a subset of TIMIT dataset.

    Args:
        data_dir: Path to TIMIT data directory
        max_samples: Maximum number of samples to load

    Returns:
        List of tuples (wav_path, phoneme_sequence, text_content)
    """
    data_pairs = []
    if not data_dir.exists():
        print(f"Data directory {data_dir} not found.")
        return data_pairs

    wav_files = sorted(data_dir.rglob("*.WAV"))
    if not wav_files:
        print(f"No .WAV files found in {data_dir}.")
        return data_pairs

    files_to_load = wav_files
    if max_samples is not None and max_samples > 0:
        files_to_load = wav_files[:max_samples]
        print(f"Loading a subset of {len(files_to_load)} samples.")

    for wav_file in files_to_load:
        phn_file = wav_file.with_suffix(".PHN")
        txt_file = wav_file.with_suffix(".TXT")

        if phn_file.exists() and txt_file.exists():
            try:
                phoneme_sequence = []
                with open(phn_file, "r") as f_phn:
                    phoneme_sequence = [
                        parts[2]
                        for line in f_phn.readlines()
                        if len(parts := line.strip().split()) == 3
                    ]

                with open(txt_file, "r") as f_txt:
                    text_content = " ".join(f_txt.readline().strip().split()[2:])

                # validate extracted data before adding
                if phoneme_sequence and text_content:
                    data_pairs.append((wav_file, phoneme_sequence, text_content))
                else:
                    print(
                        f"Warning: Empty phoneme sequence or text for {wav_file.name}"
                    )

            except (IOError, OSError) as e:
                print(f"Error processing file {wav_file.name}: {e}")
            except Exception as e:
                print(f"Unexpected error processing file {wav_file.name}: {e}")

    return data_pairs


def normalize_features(
    features_list: List[np.ndarray],
) -> Tuple[List[np.ndarray], Optional[np.ndarray], Optional[np.ndarray]]:
    """
    Normalize features by computing mean and std deviation.

    Args:
        features_list: List of feature matrices

    Returns:
        normalized_features_list: List of normalized feature matrices
        mean: Mean of features
        std: Standard deviation of features
    """
    valid_features = [
        f_matrix
        for f_matrix in features_list
        if f_matrix.ndim == 2 and f_matrix.shape[1] > 0
    ]

    if not valid_features:
        print("Warning: No valid features found to compute normalization stats.")
        return features_list, None, None

    # concatenate all features along time axis for normalization computation
    concatenated_features = np.concatenate([f.T for f in valid_features], axis=0)

    mean = np.mean(concatenated_features, axis=0)
    std = np.maximum(np.std(concatenated_features, axis=0), config.EPSILON)

    normalized_features_list = [
        (
            (f_matrix - mean[:, np.newaxis]) / std[:, np.newaxis]
            if f_matrix.ndim == 2 and f_matrix.shape[1] > 0
            else f_matrix
        )
        for f_matrix in features_list
    ]

    return normalized_features_list, mean, std


def apply_normalization(
    features: np.ndarray, mean: Optional[np.ndarray], std: Optional[np.ndarray]
) -> np.ndarray:
    """
    Apply normalization to features using precomputed mean and std deviation.

    Args:
        features: Feature matrix
        mean: Mean of features
        std: Standard deviation of features

    Returns:
        Normalized feature matrix
    """
    # early return for invalid inputs
    if features.size == 0 or mean is None or std is None:
        return features

    return (features - mean[:, np.newaxis]) / std[:, np.newaxis]


def evaluate_sample(
    model: BLSTM,
    features: np.ndarray,
    target_labels: List[str],
    target_text: str,
    label_list: List[str],
    blank_label_index: int,
    beam_width: int = 5,
    print_prefix: str = "  ",
) -> Optional[Tuple[float, float]]:
    """
    Evaluate a single sample and print results.

    Args:
        model: BLSTM model instance
        features: Input feature matrix (n_mfcc * 3, T)
        target_labels: List of target phoneme labels
        target_text: Target text content for logging
        label_list: List of all phoneme labels
        blank_label_index: Index of the blank label in label_list
        beam_width: Beam width for beam search decoding
        print_prefix: Prefix for printed output

    Returns:
        Tuple of (ler_greedy, ler_beam) or None if evaluation fails
    """
    if features is None or features.shape[1] == 0:
        print(f"{print_prefix}Warning: Sample has empty features. Skipping evaluation.")
        return None

    # forward pass
    test_probs, _ = model.forward(features)

    # pre-compute target labels string
    target_labels_str = " ".join(target_labels)
    target_len = len(target_labels_str)

    print(f"{print_prefix}Target Text             : {target_text}")
    print(f"{print_prefix}Target Labels           : {target_labels_str}")

    # greedy (best path) decoding
    decoded_labels_greedy = best_path_decode(test_probs, label_list, blank_label_index)
    decoded_greedy_str = " ".join(decoded_labels_greedy)
    ler_greedy = (
        levenshtein_distance(target_labels_str, decoded_greedy_str) / target_len
        if target_len > 0
        else 0.0
    )
    print(f"{print_prefix}Predicted (best path)   : {decoded_greedy_str}")
    print(f"{print_prefix}LER (best path)         : {ler_greedy:.4f}")

    # beam search decoding
    decoded_labels_beam = ctc_beam_search_decoder(
        test_probs, label_list, blank_label_index, beam_width
    )
    decoded_beam_str = " ".join(decoded_labels_beam)
    ler_beam = (
        levenshtein_distance(target_labels_str, decoded_beam_str) / target_len
        if target_len > 0
        else 0.0
    )
    print(f"{print_prefix}Predicted (beam search) : {decoded_beam_str}")
    print(f"{print_prefix}LER (beam search)       : {ler_beam:.4f}")

    return ler_greedy, ler_beam


if __name__ == "__main__":
    data_directory = pathlib.Path("TIMIT")
    if not data_directory.exists():
        print(
            f"TIMIT directory {data_directory} not found. Please create it or adjust the path."
        )

    all_data = load_timit_subset(data_directory, max_samples=config.MAX_SAMPLES_TO_LOAD)
    if not all_data:
        print("No data loaded. Exiting.")
        exit()

    random.seed(9120)
    np.random.seed(9120)
    random.shuffle(all_data)
    split_idx = int(len(all_data) * 0.8)
    train_data_tuples = all_data[:split_idx]
    eval_data_tuples = all_data[split_idx:]

    if not train_data_tuples:
        print("No training data available after split. Exiting.")
        exit()

    train_phoneme_sequences = [p_seq for _, p_seq, _ in train_data_tuples]
    train_label_map, train_label_list = build_label_map_and_list(
        train_phoneme_sequences
    )
    blank_label_index = train_label_map["_"]
    num_classes = len(train_label_list)

    print(f"Phoneme Vocabulary ({num_classes} symbols): {train_label_list}")

    train_features_list = []
    train_targets_list = []
    print("Preprocessing training data...")
    for wav_path, phoneme_sequence, _ in train_data_tuples:
        features = preprocess_audio(
            wav_path,
            config.SAMPLE_RATE,
            config.N_MFCC,
            config.HOP_LENGTH,
            config.WIN_LENGTH,
        )
        if features is not None and features.shape[1] > 0:
            train_features_list.append(features)
            target_indices = prepare_target_sequence(phoneme_sequence, train_label_map)
            train_targets_list.append(target_indices)
        else:
            print(
                f"Skipping {wav_path.name} due to preprocessing error or empty features."
            )

    if not train_features_list:
        print("No features extracted for training. Exiting.")
        exit()

    train_features_list, train_mean, train_std = normalize_features(train_features_list)
    if train_mean is None or train_std is None:
        print("Normalization failed. Exiting.")
        exit()

    print("Training data preprocessed and normalized.")

    # prepare a sample from eval_data_tuples for epoch evaluation
    eval_epoch_features = None
    eval_epoch_target_phonemes = None
    eval_epoch_target_text = None
    eval_sample_name_for_log = "N/A"

    if eval_data_tuples:
        print(
            f"Preparing one evaluation sample from the {len(eval_data_tuples)} available eval samples..."
        )
        eval_wav_path, eval_phns, eval_text_content = eval_data_tuples[0]
        eval_sample_name_for_log = eval_wav_path.stem

        eval_epoch_features_raw = preprocess_audio(
            eval_wav_path,
            config.SAMPLE_RATE,
            config.N_MFCC,
            config.HOP_LENGTH,
            config.WIN_LENGTH,
        )
        if eval_epoch_features_raw is not None and eval_epoch_features_raw.shape[1] > 0:
            eval_epoch_features = apply_normalization(
                eval_epoch_features_raw, train_mean, train_std
            )
            eval_epoch_target_phonemes = eval_phns
            eval_epoch_target_text = eval_text_content
            print(f"Evaluation sample '{eval_sample_name_for_log}' prepared.")
        else:
            print(
                f"Warning: Could not preprocess evaluation sample {eval_wav_path}. Epoch evaluation will be skipped."
            )
    else:
        print(
            "No evaluation data available. Skipping setup of epoch evaluation sample."
        )

    input_dim = train_features_list[0].shape[0]

    blstm_ctc_model = BLSTM(input_dim, config.HIDDEN_DIM, num_classes)

    print(f"Starting BLSTM training with {len(train_features_list)} samples...")
    losses = []

    # pre-compute constants for training loop
    num_samples = len(train_features_list)
    update_interval = 10  # progress update frequency

    for epoch in range(config.EPOCHS):
        total_epoch_loss = 0.0

        # shuffle training data indices for each epoch
        shuffled_indices = np.random.permutation(num_samples)

        progress_bar = tqdm(
            shuffled_indices, desc=f"Epoch {epoch + 1}/{config.EPOCHS}", unit=" sample"
        )

        for i_idx, data_idx in enumerate(progress_bar):
            X_sample = train_features_list[data_idx]
            y_target_sample = train_targets_list[data_idx]

            # forward pass
            probs, _ = blstm_ctc_model.forward(X_sample)

            # CTC loss and gradient computation
            loss, grad = ctc_loss_and_gradient(
                probs, y_target_sample, blank_label_index
            )
            total_epoch_loss += loss

            # backward pass
            blstm_ctc_model.backward(grad, config.LEARNING_RATE, config.MOMENTUM)

            # update progress
            if (i_idx + 1) % update_interval == 0:
                progress_bar.set_postfix(loss=f"{loss:.4f}")

        # average loss computation
        avg_epoch_loss = total_epoch_loss / num_samples
        losses.append(avg_epoch_loss)
        progress_bar.close()
        print(f"Epoch {epoch + 1}/{config.EPOCHS}, Avg Loss: {avg_epoch_loss:.4f}")

        # epoch evaluation
        if (
            eval_epoch_features is not None
            and eval_epoch_target_phonemes is not None
            and eval_epoch_target_text is not None
        ):
            print(
                f"--- Evaluating on eval sample '{eval_sample_name_for_log}' after Epoch {epoch + 1} ---"
            )
            try:
                evaluate_sample(
                    blstm_ctc_model,
                    eval_epoch_features,
                    eval_epoch_target_phonemes,
                    eval_epoch_target_text,
                    train_label_list,
                    blank_label_index,
                    config.BEAM_WIDTH,
                    print_prefix="  ",
                )
            except Exception as e:
                print(f"  Error during evaluation sample processing: {e}")
            print("-----------------------------------------------------------")

    print("BLSTM Training finished.")

    model_save_path = "blstm_ctc_final_model.npz"
    np.savez(
        model_save_path,
        weights=pickle.dumps(blstm_ctc_model.__dict__),
        phoneme_map=pickle.dumps(train_label_map),
        phoneme_list=train_label_list,
        train_mean=train_mean,
        train_std=train_std,
        input_dim=input_dim,
        hidden_dim=config.HIDDEN_DIM,
        output_dim=num_classes,
        blank_symbol_idx=blank_label_index,
    )
    print(f"Saved final model to {model_save_path}")

    # evaluation loop
    if eval_data_tuples:
        print("\n--- Starting Evaluation ---")
        total_phoneme_ler = 0.0
        num_eval_samples = 0

        for i, (wav_path, target_phoneme_list, target_text) in enumerate(
            eval_data_tuples
        ):
            print(f"\nEvaluating Sample {i+1}/{len(eval_data_tuples)}: {wav_path.name}")
            eval_features = preprocess_audio(
                wav_path,
                config.SAMPLE_RATE,
                config.N_MFCC,
                config.HOP_LENGTH,
                config.WIN_LENGTH,
            )

            # validate features before processing
            if eval_features is not None and eval_features.shape[1] > 0:
                # apply normalization
                normalized_eval_features = apply_normalization(
                    eval_features, train_mean, train_std
                )

                # evaluate sample with error handling
                eval_result = evaluate_sample(
                    blstm_ctc_model,
                    normalized_eval_features,
                    target_phoneme_list,
                    target_text,
                    train_label_list,
                    blank_label_index,
                    config.BEAM_WIDTH,
                    print_prefix="  ",
                )

                # accumulate results
                if eval_result is not None:
                    ler_greedy, ler_beam = eval_result
                    total_phoneme_ler += ler_greedy
                    num_eval_samples += 1
            else:
                print(
                    f"  Skipping {wav_path.name}: preprocessing error or empty features"
                )

        if num_eval_samples > 0:
            avg_phoneme_ler = total_phoneme_ler / num_eval_samples
            print(f"\n--- Evaluation Summary ---")
            print(f"Average Phoneme LER: {avg_phoneme_ler:.4f}")
        else:
            print("\nNo samples were evaluated.")
    else:
        print("\nNo evaluation data available.")

    plt.figure(figsize=(10, 6))
    plt.plot(losses)
    plt.title("CTC Loss During BLSTM Training")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.grid(True)
    plt.savefig("blstm_training_history.png")
    print("Saved training history plot to blstm_training_history.png")
