import torch
import torch.nn.functional as F
import math

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

class PCFinanceModel:
    """
    A lightweight, zero-autograd Predictive Coding model for stock market prediction.
    Takes 1 continuous input (price) and outputs 1 continuous prediction (next price).
    """
    def __init__(self, input_dim=1, hidden_size=64, num_layers=2):
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.t_infer = 5
        self.eta_x = 0.1
        self.eta_w = 0.001

        # Input projection (Price -> Hidden Dimension)
        self.input_proj = torch.empty(input_dim, hidden_size)
        torch.nn.init.normal_(self.input_proj, std=0.02)

        self.layers = [PCFinanceLayer(hidden_size, self.eta_x, self.eta_w) for _ in range(num_layers)]

        # Output Readout (Hidden Dimension -> Predicted Price)
        self.output_head = torch.empty(1, hidden_size)
        torch.nn.init.normal_(self.output_head, std=0.02)
        self.output_bias = torch.zeros(1)

    def train_step(self, x_seq, y_target):
        """
        x_seq: [batch, seq_len, 1] (historical prices)
        y_target: [batch, 1] (the actual price of the next day)
        """
        B, S, _ = x_seq.shape

        # 1. Initialize States (Bottom-Up)
        states = []
        x_0 = torch.matmul(x_seq, self.input_proj) # Project continuous input to hidden size
        states.append(x_0.clone())

        current_x = x_0
        for layer in self.layers:
            proj = torch.matmul(current_x, layer.W) + layer.b
            proj = F.layer_norm(proj, (self.hidden_size,))
            current_x = layer.activation(proj)
            states.append(current_x.clone())

        # 2. State Inference Dynamics (t_infer loop)
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

            states, next_states = next_states, states

        # 3. Local Weight Updates
        total_energy = 0.0
        for l in range(1, len(self.layers) + 1):
            layer = self.layers[l-1]
            pred_prev, pre_act = layer.forward_predict(states[l])
            err_prev = states[l-1] - pred_prev
            total_energy += (err_prev ** 2).sum(dim=-1).mean().item()
            layer.update_weights(err_prev, pre_act, states[l])

        # 4. Final Output Prediction (Continuous)
        # We only care about the last time step for predicting tomorrow's price
        top_state_last = states[-1][:, -1, :] # [batch, hidden]

        # Predict the continuous price
        prediction = torch.matmul(top_state_last, self.output_head.T) + self.output_bias # [batch, 1]

        # Loss is Mean Squared Error (MSE), NOT Cross-Entropy
        error = prediction - y_target # The simple distance between guess and reality
        mse_loss = (error ** 2).mean().item()

        # Update Output Head via continuous gradient
        # d_loss/d_pred = 2 * (pred - target)
        grad_pred = 2.0 * error / B

        # d_W = grad_pred^T @ top_state_last
        delta_head = torch.matmul(grad_pred.T, top_state_last)
        delta_bias = grad_pred.sum(dim=0)

        self.output_head.sub_(delta_head, alpha=self.eta_w * 5)
        self.output_bias.sub_(delta_bias, alpha=self.eta_w * 5)

        # Update Input Projection (Sensory error)
        pred_l1, _ = self.layers[0].forward_predict(states[1])
        err_l0 = states[0] - pred_l1

        # d_input_proj = input^T @ err_l0
        err_l0_flat = err_l0.view(-1, self.hidden_size)
        x_seq_flat = x_seq.view(-1, 1)
        delta_input = torch.matmul(x_seq_flat.T, err_l0_flat)
        self.input_proj.add_(delta_input, alpha=self.eta_w)

        return mse_loss, total_energy


import yfinance as yf
import numpy as np

def get_stock_data(ticker="AAPL", history="5y", seq_len=10):
    """
    Downloads historical stock data and prepares sliding windows for sequence prediction.
    """
    print(f"Downloading {ticker} stock data...")
    stock = yf.Ticker(ticker)
    df = stock.history(period=history)

    # We will just predict the 'Close' price
    prices = df['Close'].values

    # Mathematical Guardrail: We MUST normalize the dollar values.
    # Raw prices like $250.00 will explode the un-normalized input layer instantly.
    # We will use simple min-max scaling to bound prices between 0 and 1.
    price_min = np.min(prices)
    price_max = np.max(prices)
    normalized_prices = (prices - price_min) / (price_max - price_min)

    # Create sliding windows: look at `seq_len` days to predict day `seq_len + 1`
    X, Y = [], []
    for i in range(len(normalized_prices) - seq_len - 1):
        # Shape: [seq_len, 1]
        x_seq = normalized_prices[i : i + seq_len].reshape(-1, 1)
        # Shape: [1]
        y_target = normalized_prices[i + seq_len].reshape(1)

        X.append(x_seq)
        Y.append(y_target)

    # Convert to PyTorch Tensors
    X_tensor = torch.tensor(np.array(X), dtype=torch.float32)
    Y_tensor = torch.tensor(np.array(Y), dtype=torch.float32)

    return X_tensor, Y_tensor, price_min, price_max


def run_finance_training():
    print("="*50)
    print("Predictive Coding for Time Series (Zero-Autograd)")
    print("="*50)

    # 1. Fetch Data
    seq_len = 10
    X, Y, price_min, price_max = get_stock_data(ticker="AAPL", history="2y", seq_len=seq_len)

    # 2. Initialize Model
    # input_dim = 1 (just the price), hidden_size = 32 (tiny because it's just 1 number)
    model = PCFinanceModel(input_dim=1, hidden_size=32, num_layers=2)

    print(f"Data shape: {X.shape}")
    print("Starting Zero-Autograd Training...")

    # 3. Training Loop
    batch_size = 16
    epochs = 3 # Small number of epochs for demonstration

    with torch.no_grad():
        for epoch in range(epochs):
            total_mse = 0.0
            steps = 0

            # Simple batching
            for i in range(0, len(X) - batch_size, batch_size):
                x_batch = X[i : i + batch_size]
                y_batch = Y[i : i + batch_size]

                mse_loss, energy = model.train_step(x_batch, y_batch)

                total_mse += mse_loss
                steps += 1

            avg_mse = total_mse / steps
            print(f"Epoch {epoch+1}/{epochs} | Avg MSE Loss: {avg_mse:.6f}")

    # 4. Final Prediction Test
    with torch.no_grad():
        test_seq = X[-1:] # The very last sequence available
        actual_normalized = Y[-1:]

        # Bottom-up pass for prediction
        x_0 = torch.matmul(test_seq, model.input_proj)
        current_x = x_0
        for layer in model.layers:
            proj = torch.matmul(current_x, layer.W) + layer.b
            proj = F.layer_norm(proj, (model.hidden_size,))
            current_x = layer.activation(proj)

        top_state_last = current_x[:, -1, :]
        prediction_normalized = torch.matmul(top_state_last, model.output_head.T) + model.output_bias

        # De-normalize to get real dollar values
        predicted_price = prediction_normalized.item() * (price_max - price_min) + price_min
        actual_price = actual_normalized.item() * (price_max - price_min) + price_min

        print("\n" + "="*50)
        print("Final Sequence Prediction Test")
        print("="*50)
        print(f"Predicted AAPL Price (Next Day): ${predicted_price:.2f}")
        print(f"Actual AAPL Price: ${actual_price:.2f}")
        print(f"Absolute Error: ${abs(predicted_price - actual_price):.2f}")
        print("="*50)

if __name__ == "__main__":
    torch.set_grad_enabled(False)
    run_finance_training()
