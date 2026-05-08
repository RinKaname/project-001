# QUANTCONNECT.COM - Democratizing Finance, Empowering Individuals.
# Lean Algorithmic Trading Engine v2.0. Copyright 2014 QuantConnect Corporation.
#
# Zero-Autograd Predictive Coding Model Baseline

try:
    from AlgorithmImports import *
except ImportError:
    # Handle local syntax testing outside of QuantConnect
    pass

import torch
import torch.nn.functional as F
import math
import numpy as np

class PCFinanceLayer:
    """
    A single Predictive Coding layer adapted for continuous time series data.
    """
    def __init__(self, hidden_size: int, eta_x: float = 0.1, eta_w: float = 0.001):
        self.hidden_size = hidden_size
        self.eta_x = eta_x
        self.eta_w = eta_w

        bound = 1.0 / math.sqrt(self.hidden_size)
        self.W = torch.empty(self.hidden_size, self.hidden_size).uniform_(-bound, bound)
        self.b = torch.zeros(self.hidden_size)

    def activation(self, x):
        return F.silu(x)

    def activation_derivative(self, x):
        sig = torch.sigmoid(x)
        return sig * (1 + x * (1 - sig))

    def forward_predict(self, x_higher):
        proj = torch.matmul(x_higher, self.W.T) + self.b
        proj = F.layer_norm(proj, (self.hidden_size,))
        return self.activation(proj), proj

    def update_weights(self, error_local, pre_act_proj, x_higher):
        f_prime = self.activation_derivative(pre_act_proj)
        local_signal = error_local * f_prime

        local_signal_flat = local_signal.view(-1, self.hidden_size)
        x_higher_flat = x_higher.view(-1, self.hidden_size)

        local_signal_flat_norm = F.normalize(local_signal_flat, p=2, dim=-1)
        x_higher_flat_norm = F.normalize(x_higher_flat, p=2, dim=-1)

        delta_W = torch.matmul(local_signal_flat_norm.T, x_higher_flat_norm) / local_signal_flat.shape[0]
        delta_b = local_signal_flat.mean(dim=0)

        self.W.add_(delta_W, alpha=self.eta_w)
        self.b.add_(delta_b, alpha=self.eta_w)


class PCQuantConnectModel:
    """
    Stateful Zero-Autograd Predictive Coding model.
    Outputs a single weight between 0.0 and 1.0 representing portfolio allocation.
    """
    def __init__(self, input_dim=1, hidden_size=64, num_layers=2):
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.t_infer = 5
        self.eta_x = 0.1
        self.eta_w = 0.001

        # Input projection
        self.input_proj = torch.empty(input_dim, hidden_size)
        torch.nn.init.normal_(self.input_proj, std=0.02)

        self.layers = [PCFinanceLayer(hidden_size, self.eta_x, self.eta_w) for _ in range(num_layers)]

        # Output Readout (Hidden Dimension -> Predicted Return)
        self.output_head = torch.empty(1, hidden_size)
        torch.nn.init.normal_(self.output_head, std=0.02)
        self.output_bias = torch.zeros(1)

    def online_step(self, x_seq, prev_y_target):
        B, S, _ = x_seq.shape

        # --- PHASE 1: PREDICTION (Forward Sweep) ---
        states_fwd = []
        x_0_fwd = torch.matmul(x_seq, self.input_proj)
        states_fwd.append(x_0_fwd)

        current_x = x_0_fwd
        for layer in self.layers:
            proj = torch.matmul(current_x, layer.W) + layer.b
            proj = F.layer_norm(proj, (self.hidden_size,))
            current_x = layer.activation(proj)
            states_fwd.append(current_x)

        top_state_fwd = states_fwd[-1][:, -1, :]
        raw_prediction = torch.matmul(top_state_fwd, self.output_head.T) + self.output_bias

        # Long-Only Filter: Use Sigmoid to output an allocation between 0.0 (Cash) and 1.0 (Crypto)
        predicted_weight = torch.sigmoid(raw_prediction)

        # --- PHASE 2: ONLINE LEARNING (Inference & Weight Update) ---
        if prev_y_target is not None:
            x_seq_learn = x_seq[:, :-1, :] # T-1

            states = []
            x_0 = torch.matmul(x_seq_learn, self.input_proj)
            states.append(x_0.clone())

            current_x = x_0
            for layer in self.layers:
                proj = torch.matmul(current_x, layer.W) + layer.b
                proj = F.layer_norm(proj, (self.hidden_size,))
                current_x = layer.activation(proj)
                states.append(current_x.clone())

            next_states = [torch.empty_like(s) for s in states]
            next_states[0].copy_(states[0])

            for t in range(self.t_infer):
                for l in range(1, len(self.layers) + 1):
                    layer = self.layers[l-1]
                    pred_prev, pre_act = layer.forward_predict(states[l])
                    err_prev = states[l-1] - pred_prev

                    f_prime = layer.activation_derivative(pre_act)
                    grad_bottom_up = -torch.matmul(err_prev * f_prime, layer.W)

                    grad_top_down = torch.zeros_like(states[l])
                    if l < len(self.layers):
                        next_layer = self.layers[l]
                        pred_curr, _ = next_layer.forward_predict(states[l+1])
                        err_curr = states[l] - pred_curr
                        grad_top_down = err_curr

                    total_grad = torch.clamp_(grad_bottom_up + grad_top_down, -1.0, 1.0)
                    next_states[l].copy_(states[l])
                    next_states[l].sub_(total_grad, alpha=self.eta_x)
                    next_states[l] = F.normalize(next_states[l], p=2, dim=-1)

                states, next_states = next_states, states

            for l in range(1, len(self.layers) + 1):
                layer = self.layers[l-1]
                pred_prev, pre_act = layer.forward_predict(states[l])
                err_prev = states[l-1] - pred_prev
                layer.update_weights(err_prev, pre_act, states[l])

            top_state_last = states[-1][:, -1, :]
            prediction = torch.matmul(top_state_last, self.output_head.T) + self.output_bias

            error = prediction - prev_y_target
            grad_pred = 2.0 * error / B
            grad_pred = torch.clamp(grad_pred, -5.0, 5.0)

            delta_head = torch.matmul(grad_pred.T, top_state_last)
            delta_bias = grad_pred.sum(dim=0)

            weight_decay = 0.001
            self.output_head.sub_(delta_head + weight_decay * self.output_head, alpha=self.eta_w)
            self.output_bias.sub_(delta_bias + weight_decay * self.output_bias, alpha=self.eta_w)
            self.output_head.data = F.normalize(self.output_head.data, p=2, dim=-1)

            pred_l1, _ = self.layers[0].forward_predict(states[1])
            err_l0 = states[0] - pred_l1

            err_l0_flat = err_l0.view(-1, self.hidden_size)
            x_seq_flat = x_seq_learn.view(-1, 1)
            delta_input = torch.matmul(x_seq_flat.T, err_l0_flat)

            self.input_proj.add_(delta_input - weight_decay * self.input_proj, alpha=self.eta_w)
            self.input_proj.data = F.normalize(self.input_proj.data, p=2, dim=-1)

        return predicted_weight


# Only define the class if QCAlgorithm is available (in QuantConnect) or mock it
class MockQCAlgorithm:
    pass

QCAlgorithm = globals().get('QCAlgorithm', MockQCAlgorithm)

class PredictiveCodingAlgorithm(QCAlgorithm):

    def Initialize(self):
        self.SetStartDate(2018, 1, 1)
        self.SetCash(100000)

        # Request Crypto Data
        self.symbol = self.AddCrypto("BTCUSD", Resolution.Daily).Symbol

        # Configuration
        self.seq_len = 10
        self.window = RollingWindow[float](self.seq_len + 1)

        # Initialize biological brain
        # We disable autograd completely for extreme efficiency
        torch.set_grad_enabled(False)
        self.model = PCQuantConnectModel(input_dim=1, hidden_size=128, num_layers=2)

        # Warm up the window so the model can trade on Day 1
        history = self.History(self.symbol, self.seq_len + 1, Resolution.Daily)
        if not history.empty:
            for index, row in history.iterrows():
                self.window.Add(float(row["close"]))

    def OnData(self, data):
        if not data.ContainsKey(self.symbol):
            return

        # Add new daily price
        self.window.Add(float(data[self.symbol].Close))

        if not self.window.IsReady:
            return

        # Extract the prices from the rolling window (it stores newest first, so we reverse it)
        prices = np.array([self.window[i] for i in range(self.window.Count - 1, -1, -1)])

        # Strict zero look-ahead bias normalization
        window_min = np.min(prices)
        window_max = np.max(prices)

        if window_max == window_min:
            window_max += 1e-5

        normalized_prices = (prices - window_min) / (window_max - window_min)

        # The sequence today (for prediction)
        x_seq_tensor = torch.tensor(normalized_prices[1:], dtype=torch.float32).view(1, self.seq_len, 1)

        # The target from yesterday (for learning)
        prev_y_target_tensor = torch.tensor(normalized_prices[-1], dtype=torch.float32).view(1, 1)

        # Online Learning Step
        with torch.no_grad():
            predicted_weight_tensor = self.model.online_step(x_seq_tensor, prev_y_target_tensor)

        # Extract allocation between 0.0 and 1.0
        allocation = predicted_weight_tensor.item()

        # Issue order to QuantConnect Engine
        self.SetHoldings(self.symbol, allocation)

        # Optional: Plot the allocation weight for visual debugging in QuantConnect Console
        self.Plot("Predictive Coding", "Portfolio Weight", allocation)

if __name__ == "__main__":
    print("Zero-Autograd Predictive Coding Model wrapped for QuantConnect / LEAN.")
    print("To test, copy and paste this code into the QuantConnect IDE.")
