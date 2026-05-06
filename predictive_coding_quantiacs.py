import torch
import torch.nn.functional as F
import math
import xarray as xr
import numpy as np
import logging

# Quantiacs specific imports (Mocked locally for syntax clarity, but intended for real Quantiacs Env)
try:
    import qnt.data as qndata
    import qnt.backtester as qnbt
    import qnt.output as qnout
except ImportError:
    logging.warning("Quantiacs 'qnt' library not found. Script is intended to be run in the Quantiacs environment.")


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


class PCQuantiacsModel:
    """
    Stateful Zero-Autograd Predictive Coding model for Quantiacs Crypto Long Contest.
    Takes N assets as continuous inputs, outputs N portfolio weights.
    """
    def __init__(self, num_assets: int, hidden_size=64, num_layers=2):
        self.num_assets = num_assets
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.t_infer = 5
        self.eta_x = 0.1
        self.eta_w = 0.001

        # Input projection (N Assets -> Hidden Dimension)
        self.input_proj = torch.empty(num_assets, hidden_size)
        torch.nn.init.normal_(self.input_proj, std=0.02)

        self.layers = [PCFinanceLayer(hidden_size, self.eta_x, self.eta_w) for _ in range(num_layers)]

        # Output Readout (Hidden Dimension -> Predicted Return for N assets)
        self.output_head = torch.empty(num_assets, hidden_size)
        torch.nn.init.normal_(self.output_head, std=0.02)
        self.output_bias = torch.zeros(num_assets)

    def online_step(self, x_seq, prev_y_target):
        """
        Executes a single online learning step.
        x_seq: The normalized price sequence [1, seq_len, num_assets]
        prev_y_target: The actual prices of the *previous* day to calculate error [1, num_assets]

        Returns:
            predicted_weights: [1, num_assets] Portfolio allocations between 0.0 and 1.0
        """
        B, S, _ = x_seq.shape

        # --- PHASE 1: PREDICTION (Forward Sweep) ---
        # We predict tomorrow's allocations purely based on today's sequence
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

        # Crypto Long Contest Rule: Long positions only (Weights between 0 and 1)
        # We apply softmax or sigmoid. We'll use Softmax to distribute a 100% portfolio allocation.
        predicted_weights = F.softmax(raw_prediction, dim=-1)

        # --- PHASE 2: ONLINE LEARNING (Inference & Weight Update) ---
        # We can only learn if we have a target from the PREVIOUS prediction cycle
        if prev_y_target is not None:
            # We treat the sequence up to T-1 as the input, and T as the target
            x_seq_learn = x_seq[:, :-1, :] # [1, seq_len-1, num_assets]

            # Bottom-Up Init
            states = []
            x_0 = torch.matmul(x_seq_learn, self.input_proj)
            states.append(x_0.clone())

            current_x = x_0
            for layer in self.layers:
                proj = torch.matmul(current_x, layer.W) + layer.b
                proj = F.layer_norm(proj, (self.hidden_size,))
                current_x = layer.activation(proj)
                states.append(current_x.clone())

            # t_infer relaxation
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

            # Hebbian Weight Updates
            for l in range(1, len(self.layers) + 1):
                layer = self.layers[l-1]
                pred_prev, pre_act = layer.forward_predict(states[l])
                err_prev = states[l-1] - pred_prev
                layer.update_weights(err_prev, pre_act, states[l])

            # Output Head Update (MSE to predict continuous returns/prices)
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

            # Input Projection Update
            pred_l1, _ = self.layers[0].forward_predict(states[1])
            err_l0 = states[0] - pred_l1

            err_l0_flat = err_l0.view(-1, self.hidden_size)
            x_seq_flat = x_seq_learn.view(-1, self.num_assets)
            delta_input = torch.matmul(x_seq_flat.T, err_l0_flat)

            self.input_proj.add_(delta_input - weight_decay * self.input_proj, alpha=self.eta_w)
            self.input_proj.data = F.normalize(self.input_proj.data, p=2, dim=-1)

        return predicted_weights

# =====================================================================
# Quantiacs Strategy and Backtest Wrapper
# =====================================================================

def strategy(data, state):
    """
    Quantiacs Stateful Strategy.
    data: xarray containing the sequence history up to the current day.
    state: The PCQuantiacsModel instance (holds memory of the biological weights).
    """
    import numpy as np

    # Configuration
    seq_len = 10

    # We are operating in the crypto_daily_long environment.
    # Data shape is typically (field, time, asset)
    close_prices = data.sel(field='close').values # [time, asset]
    assets = data.asset.values
    num_assets = len(assets)

    # Initialize the model on the very first day
    if state is None:
        # Use our efficient Goldilocks zone (128 hidden)
        state = PCQuantiacsModel(num_assets=num_assets, hidden_size=128, num_layers=2)

    model = state

    # We need at least seq_len + 1 days of data to make predictions and calculate previous targets
    if close_prices.shape[0] < seq_len + 1:
        return xr.DataArray(np.zeros(num_assets), dims=["asset"], coords=dict(asset=assets)), model

    # --- Data Prep & Rolling Normalization ---
    # We grab the window including today
    window = close_prices[-(seq_len+1):]

    # We normalize each asset independently using its own local rolling min/max
    window_min = np.min(window, axis=0, keepdims=True)
    window_max = np.max(window, axis=0, keepdims=True)

    # Avoid zero division
    range_diff = window_max - window_min
    range_diff[range_diff == 0] = 1e-5

    window_norm = (window - window_min) / range_diff

    # The sequence today (for prediction)
    x_seq_tensor = torch.tensor(window_norm[1:], dtype=torch.float32).unsqueeze(0) # [1, seq_len, num_assets]

    # The target from yesterday (for learning)
    prev_y_target_tensor = torch.tensor(window_norm[-1], dtype=torch.float32).unsqueeze(0) # [1, num_assets]

    # Execute the biological online step
    with torch.no_grad():
        predicted_weights_tensor = model.online_step(x_seq_tensor, prev_y_target_tensor)

    weights_np = predicted_weights_tensor.numpy()[0]

    # Filter for liquidity (standard Quantiacs practice)
    # If the asset wasn't liquid yesterday, don't invest in it today
    if 'is_liquid' in data.coords['field'].values:
        liquidity = data.sel(field='is_liquid')[-1].values
        weights_np = weights_np * liquidity

    # Re-normalize weights so they sum to 1.0
    weight_sum = np.sum(abs(weights_np))
    if weight_sum > 0:
        weights_np = weights_np / weight_sum

    # Package into xarray for Quantiacs
    weights = xr.DataArray(weights_np, dims=["asset"], coords=dict(asset=assets))

    return weights, model


if __name__ == "__main__":
    print("Zero-Autograd Predictive Coding Model wrapped for Quantiacs Crypto Long Contest.")
    print("To test locally in Quantiacs environment, run:")
    print("qnbt.backtest(competition_type='crypto_daily_long', lookback_period=15, start_date='2014-01-01', strategy=strategy)")

    # If ran locally, verify syntax compiles
    print("\nScript compiled successfully. Ready for Quantiacs Jupyter execution.")
