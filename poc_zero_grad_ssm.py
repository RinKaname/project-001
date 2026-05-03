import torch
import torch.nn as nn
import torch.nn.functional as F

class LocalSelectiveSSMLayer(nn.Module):
    """
    A single Selective SSM layer that trains strictly using a local greedy objective,
    without relying on backpropagation from subsequent layers.
    """
    def __init__(self, d_model, vocab_size, state_size=16):
        super().__init__()
        self.d_model = d_model
        self.state_size = state_size
        self.vocab_size = vocab_size

        # --- SSM Architecture Parameters ---
        # Instead of self-attention, we use selective state space parameters
        self.A_log = nn.Parameter(torch.log(torch.rand(d_model, state_size) * 0.9 + 0.1))
        self.D = nn.Parameter(torch.ones(d_model))

        # Projections to make parameters input-dependent (Selective)
        self.proj_delta = nn.Linear(d_model, d_model)
        self.proj_B = nn.Linear(d_model, state_size)
        self.proj_C = nn.Linear(d_model, state_size)

        # --- Local Learning Architecture ---
        # A small classification head attached ONLY to this layer for local training
        self.local_head = nn.Linear(d_model, vocab_size)

        # We manually manage the optimizer for this specific layer
        self.learning_rate = 0.001

    def forward_ssm(self, x):
        """
        Executes the forward pass of the continuous-time sequence model.
        x: (batch, length, d_model)
        """
        B_batch, L, D = x.shape

        # Compute input-dependent parameters
        # delta: step size, B: input projection, C: output projection
        delta = F.softplus(self.proj_delta(x)) # (Batch, Length, D)
        B_matrix = self.proj_B(x)              # (Batch, Length, State)
        C_matrix = self.proj_C(x)              # (Batch, Length, State)

        # For simplicity in this PoC, we implement a slow, explicit loop over time
        # In a highly optimized version, this would be a parallel scan algorithm
        h = torch.zeros(B_batch, D, self.state_size, device=x.device)
        A = -torch.exp(self.A_log) # Ensure A is negative for stability

        y_out = []
        for t in range(L):
            x_t = x[:, t, :] # (Batch, D)
            delta_t = delta[:, t, :] # (Batch, D)
            B_t = B_matrix[:, t, :] # (Batch, State)
            C_t = C_matrix[:, t, :] # (Batch, State)

            # Discretize continuous parameters (Zero-Order Hold approximation)
            # bar_A = exp(delta * A)
            bar_A = torch.exp(delta_t.unsqueeze(-1) * A) # (Batch, D, State)

            # bar_B = (delta * B) -> simplified from strict ZOH for speed in PoC
            bar_B = delta_t.unsqueeze(-1) * B_t.unsqueeze(1) # (Batch, D, State)

            # Update hidden state: h_t = A * h_{t-1} + B * x_t
            h = bar_A * h + bar_B * x_t.unsqueeze(-1)

            # Compute output: y_t = C * h_t
            # Sum over the state dimension
            y_t = torch.sum(h * C_t.unsqueeze(1), dim=-1) + x_t * self.D
            y_out.append(y_t)

        return torch.stack(y_out, dim=1) # (Batch, Length, D)

    def local_forward_and_update(self, x, targets):
        """
        The core logic to bypass the global chain rule.
        1. Pass input through SSM.
        2. Make local prediction.
        3. Compute loss locally.
        4. Calculate gradients manually and update weights INSTANTLY.
        5. Return detached output.
        """
        # Ensure x does not bring gradients from previous layers
        x = x.detach()
        x.requires_grad_(True) # If we wanted to pass gradients *back* to previous layer, but we don't.

        # 1. SSM Forward Pass
        y = self.forward_ssm(x)

        # 2. Local Prediction
        # Flatten for classification: (Batch * Length, Vocab)
        logits = self.local_head(y.view(-1, self.d_model))
        flat_targets = targets.view(-1)

        # 3. Local Loss Calculation
        # This loss NEVER travels to layer l+1 or l-1. It is trapped here.
        local_loss = F.cross_entropy(logits, flat_targets)

        # 4. Zero-Gradient Weight Update (The Hackathon Requirement)
        # We compute gradients only for this layer's parameters
        layer_params = list(self.parameters())

        # Clear any existing gradients
        for p in layer_params:
            if p.grad is not None:
                p.grad.zero_()

        # Compute gradients purely locally
        grads = torch.autograd.grad(local_loss, layer_params, retain_graph=False)

        # Apply manual SGD update (raw tensor operations, avoiding global optimizers)
        with torch.no_grad():
            for param, grad in zip(layer_params, grads):
                param.sub_(self.learning_rate * grad)

        # 5. Detach output to prevent global backpropagation from subsequent layers
        return y.detach(), local_loss.item()

# ==========================================
# Proof of Concept Execution
# ==========================================
if __name__ == "__main__":
    torch.manual_seed(42) # Strict determinism requirement

    V = 100 # Vocab size
    D = 64  # Model dimension
    B = 2   # Batch size
    L = 10  # Sequence length

    print("Initializing Local Selective SSM PoC...")
    layer1 = LocalSelectiveSSMLayer(d_model=D, vocab_size=V)
    layer2 = LocalSelectiveSSMLayer(d_model=D, vocab_size=V)

    # Dummy embedding layer (in reality, this would also be locally trained)
    embed = nn.Embedding(V, D)

    # Dummy sequence data
    input_ids = torch.randint(0, V, (B, L))

    # Target is predicting the next token (shifted by 1)
    # Pad the end with 0 for the last token prediction
    targets = torch.cat([input_ids[:, 1:], torch.zeros((B, 1), dtype=torch.long)], dim=1)

    print("\nStarting Zero-Gradient Training Loop...")
    for epoch in range(5):
        # 1. Embed input
        x = embed(input_ids).detach()

        # 2. Train Layer 1 locally
        out1, loss1 = layer1.local_forward_and_update(x, targets)

        # 3. Train Layer 2 locally using Layer 1's DETACHED output
        out2, loss2 = layer2.local_forward_and_update(out1, targets)

        print(f"Epoch {epoch+1} | Layer 1 Local Loss: {loss1:.4f} | Layer 2 Local Loss: {loss2:.4f}")

    print("\nProof of Concept Complete. Layers updated successfully without global loss.backward().")
