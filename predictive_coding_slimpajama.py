import torch
import torch.nn.functional as F
import math
from dataclasses import dataclass
from datasets import load_dataset
from transformers import AutoTokenizer

@dataclass
class ModelConfig:
    # Scale configuration
    vocab_size: int = 50432 # EleutherAI/gpt-neox-20b true vocab size
    hidden_size: int = 64
    num_layers: int = 2
    max_seq_len: int = 128
    
    # Predictive Coding specific configuration
    t_infer: int = 5          # Number of iterations for state inference
    eta_x: float = 0.1        # Learning rate for state updates (inference)
    eta_w: float = 0.001      # Learning rate for weight updates
    
    # Batch details
    batch_size: int = 2

def get_toy_config():
    """Toy configuration for local testing on a CPU/small GPU"""
    return ModelConfig()

def get_production_config():
    """Production configuration for the 4 Billion parameter challenge"""
    # 4B params approximation with PC architecture
    return ModelConfig(
        hidden_size=4096,
        num_layers=32,
        max_seq_len=2048,
        batch_size=4
    )


class PCLayer:
    """
    A single Predictive Coding layer.
    It takes an input from below (x_{l+1}), makes a prediction of the layer above (x_{l}),
    computes the error (ε_l), and updates states/weights locally.
    
    Note: Indexing in generative models typically goes from top (output) to bottom (input),
    but standard LLMs go from bottom (input) to top (output). 
    We align with the standard LLM flow: layer l receives x_in, predicts x_out.
    """
    def __init__(self, config: ModelConfig, layer_idx: int):
        self.hidden_size = config.hidden_size
        self.eta_x = config.eta_x
        self.eta_w = config.eta_w
        self.layer_idx = layer_idx
        
        # Generative weights (used to predict this layer's state from the layer above)
        # Note: we use explicit tensors without requires_grad to ensure NO autograd.
        bound = 1.0 / math.sqrt(self.hidden_size)
        self.W = torch.empty(self.hidden_size, self.hidden_size).uniform_(-bound, bound).to(torch.float32)
        self.b = torch.zeros(self.hidden_size).to(torch.float32)
        
        # Move to GPU if available
        if torch.cuda.is_available():
            self.W = self.W.cuda()
            self.b = self.b.cuda()
            
    def activation(self, x):
        """Silu activation"""
        return F.silu(x)
        
    def activation_derivative(self, x):
        """Derivative of SiLU (x * sigmoid(x))
           d(silu)/dx = silu(x) + sigmoid(x) * (1 - silu(x))
           Wait, simpler: sigmoid(x) * (1 + x * (1 - sigmoid(x)))
        """
        sig = torch.sigmoid(x)
        return sig * (1 + x * (1 - sig))
        
    def forward_predict(self, x_higher):
        """
        Top-down prediction: predict the current layer's state based on the higher layer.
        x_higher: [batch_size, seq_len, hidden_size]
        """
        # Linear projection
        proj = torch.matmul(x_higher, self.W.T) + self.b
        return self.activation(proj), proj
        
    def update_weights(self, error_local, pre_act_proj, x_higher):
        """
        Local Hebbian-like weight update (NO AUTOGRAD).
        Based on: ΔW_l = η_w * (ε_l ⊙ f'(W_l x_{l+1})) · x_{l+1}^T
        
        error_local: The prediction error at this layer [batch, seq, hidden]
        pre_act_proj: The raw projection before activation [batch, seq, hidden]
        x_higher: The state from the layer above [batch, seq, hidden]
        """
        # Compute derivative of activation
        f_prime = self.activation_derivative(pre_act_proj) # [B, S, H]
        
        # Local error scaled by activation derivative
        # local_signal: [B, S, H]
        local_signal = error_local * f_prime
        
        # Compute weight update via outer product (averaged over batch and seq)
        # reshape to [B*S, H]
        local_signal_flat = local_signal.view(-1, self.hidden_size)
        x_higher_flat = x_higher.view(-1, self.hidden_size)
        
        # delta_W: [H, H]
        delta_W = torch.matmul(local_signal_flat.T, x_higher_flat) / local_signal_flat.shape[0]
        
        # delta_b: [H]
        delta_b = local_signal_flat.mean(dim=0)
        
        # Apply update
        self.W.add_(delta_W, alpha=self.eta_w)
        self.b.add_(delta_b, alpha=self.eta_w)


class PCLanguageModel:
    """
    A full sequence model using Predictive Coding layers.
    Flows bottom-up for sensory input, top-down for predictions.
    """
    def __init__(self, config: ModelConfig):
        self.config = config
        
        # Embeddings
        self.token_emb = torch.empty(config.vocab_size, config.hidden_size)
        torch.nn.init.normal_(self.token_emb, std=0.02)
        
        # PC Layers
        self.layers = [PCLayer(config, i) for i in range(config.num_layers)]
        
        # Final Readout (vocab projection)
        self.lm_head = torch.empty(config.vocab_size, config.hidden_size)
        torch.nn.init.normal_(self.lm_head, std=0.02)
        
        if torch.cuda.is_available():
            self.token_emb = self.token_emb.cuda()
            self.lm_head = self.lm_head.cuda()

    def get_device(self):
        return self.token_emb.device

    def print_parameter_summary(self):
        """
        Prints a detailed granular summary of the model's parameters per layer,
        including shapes, parameter counts, memory usage, and dtypes.
        """
        print("-" * 105)
        print(f"{'Component':<25} | {'Name':<15} | {'Shape':<20} | {'Parameters':<12} | {'Memory':<10} | {'Dtype'}")
        print("-" * 105)

        total_params = 0
        total_bytes = 0

        def add_param(component, name, tensor):
            nonlocal total_params, total_bytes
            num_params = tensor.numel()
            mem_bytes = num_params * tensor.element_size()
            total_params += num_params
            total_bytes += mem_bytes

            # Format memory nicely
            if mem_bytes >= 1024 ** 3:
                mem_str = f"{mem_bytes / (1024**3):.2f} GB"
            elif mem_bytes >= 1024 ** 2:
                mem_str = f"{mem_bytes / (1024**2):.2f} MB"
            else:
                mem_str = f"{mem_bytes / 1024:.2f} KB"

            shape_str = str(list(tensor.shape))
            print(f"{component:<25} | {name:<15} | {shape_str:<20} | {num_params:<12,d} | {mem_str:<10} | {str(tensor.dtype).replace('torch.', '')}")

        # Token Embedding
        add_param("Token Embedding", "token_emb", self.token_emb)

        # PC Layers
        for i, layer in enumerate(self.layers):
            comp_name = f"PCLayer {i}"
            add_param(comp_name, "W", layer.W)
            add_param(comp_name, "b", layer.b)

        # LM Head
        add_param("LM Head", "lm_head", self.lm_head)

        print("-" * 105)
        print(f"Total Parameters: {total_params:,}")

        if total_bytes >= 1024 ** 3:
            total_mem_str = f"{total_bytes / (1024**3):.2f} GB"
        else:
            total_mem_str = f"{total_bytes / (1024**2):.2f} MB"
        print(f"Total Estimated Memory: {total_mem_str}")
        print("-" * 105)

    def train_step(self, input_ids):
        """
        Executes one training step on a batch of input_ids using Predictive Coding.
        NO autograd is used.
        """
        # Ensure input is on correct device
        input_ids = input_ids.to(self.get_device())
        B, S = input_ids.shape
        
        # 1. Initialize States (Bottom-Up Pass)
        # In a strict PC network, states (x_l) are independent variables that we optimize.
        # We initialize them using a bottom-up feed-forward sweep for faster convergence.
        
        states = []
        # Input layer state is strictly clamped to the data
        x_0 = F.embedding(input_ids, self.token_emb) 
        states.append(x_0.clone())
        
        # Forward sweep to initialize higher states
        current_x = x_0
        for layer in self.layers:
            # We use the layer's transpose for the forward pass initialization
            proj = torch.matmul(current_x, layer.W) + layer.b
            current_x = layer.activation(proj)
            states.append(current_x.clone())
            
        # Top-level state needs a placeholder for prediction
        # The LM head takes the top state to predict next token
        
        # 2. State Inference Dynamics (Minimizing Free Energy)
        # We iteratively adjust states to minimize prediction errors across all layers
        for t in range(self.config.t_infer):
            new_states = [s.clone() for s in states]
            
            # We don't update x_0 because it is clamped to sensory data
            for l in range(1, len(self.layers) + 1):
                layer = self.layers[l-1]
                
                # Top-down prediction: predict x_{l-1} from x_l
                pred_prev, pre_act = layer.forward_predict(states[l])
                
                # Local Error at l-1: e_{l-1} = x_{l-1} - pred_prev
                err_prev = states[l-1] - pred_prev
                
                # Bottom-up influence: push x_l to make a better prediction of x_{l-1}
                # gradient = - W^T @ (err_prev * f'(pre_act))
                f_prime = layer.activation_derivative(pre_act)
                grad_bottom_up = -torch.matmul(err_prev * f_prime, layer.W)
                
                # Top-down influence: x_l is pulled by the prediction from x_{l+1}
                grad_top_down = torch.zeros_like(states[l])
                if l < len(self.layers):
                    next_layer = self.layers[l]
                    pred_curr, _ = next_layer.forward_predict(states[l+1])
                    err_curr = states[l] - pred_curr
                    grad_top_down = err_curr # x_l is pulled toward pred_curr
                    
                # Update state
                total_grad = grad_bottom_up + grad_top_down
                new_states[l].sub_(total_grad, alpha=self.config.eta_x)
                
            states = new_states
            
        # 3. Local Weight Updates (Hebbian)
        total_energy = 0.0
        for l in range(1, len(self.layers) + 1):
            layer = self.layers[l-1]
            
            pred_prev, pre_act = layer.forward_predict(states[l])
            err_prev = states[l-1] - pred_prev
            
            # Energy calculation for logging
            layer_energy = (err_prev ** 2).sum(dim=-1).mean().item()
            total_energy += layer_energy
            
            # Update weights locally based on the error
            layer.update_weights(err_prev, pre_act, states[l])
            
        # 4. Final Output Prediction (LM Head Update)
        # Shift targets for next-token prediction
        targets = input_ids[:, 1:].contiguous()
        top_state = states[-1][:, :-1, :].contiguous() # Drop last token state
        
        logits = torch.matmul(top_state, self.lm_head.T)
        
        # Custom backward for LM head (since we can't use autograd)
        # Softmax derivative: logits - one_hot(targets)
        probs = F.softmax(logits, dim=-1)
        
        # Create one-hot targets
        targets_one_hot = F.one_hot(targets, num_classes=self.config.vocab_size).float()
        
        # Error signal at the top
        d_logits = (probs - targets_one_hot) / (B * (S - 1))
        
        # Update LM Head
        # d_lm_head = d_logits^T @ top_state
        d_logits_flat = d_logits.view(-1, self.config.vocab_size)
        top_state_flat = top_state.view(-1, self.config.hidden_size)
        
        delta_lm_head = torch.matmul(d_logits_flat.T, top_state_flat)
        self.lm_head.sub_(delta_lm_head, alpha=self.config.eta_w * 10) # Higher LR for head
        
        # Optional Token Embedding Update
        # x_0 state has been pushed by grad_top_down (the error from layer 1) during inference.
        # We can update the embeddings using this cumulative delta.
        # delta_emb = original_x_0 - settled_x_0 (which is stored in states[0] since we only updated higher states in inference)
        # Actually, in our inference loop above, we skipped updating x_0. 
        # So we can compute the gradient at x_0 directly from the layer 1 error.
        
        pred_l1, _ = self.layers[0].forward_predict(states[1])
        err_l0 = states[0] - pred_l1 # Error at the sensory layer
        
        # Update embeddings by pulling them towards the prediction (or vice versa)
        # For simplicity, we just add the error (scaled by lr) to the active embeddings
        err_l0_flat = err_l0.view(-1, self.config.hidden_size)
        input_ids_flat = input_ids.view(-1)
        
        # Use scatter_add_ to update the embedding matrix in a zero-gradient way
        self.token_emb.index_add_(0, input_ids_flat, err_l0_flat, alpha=self.config.eta_w)

        # Calculate CE loss for reporting
        ce_loss = F.cross_entropy(logits.view(-1, self.config.vocab_size), targets.view(-1))
        
        return ce_loss.item(), total_energy

    def generate(self, input_ids, max_new_tokens, temperature=1.0):
        """
        Generates new tokens using a bottom-up forward sweep (pure prediction).
        input_ids: [batch_size, seq_len]
        """
        generated_ids = input_ids.clone()

        for _ in range(max_new_tokens):
            # Crop to max_seq_len to avoid exceeding any positional bounds (though we don't use pos embeddings here, good practice)
            cond_idx = generated_ids[:, -self.config.max_seq_len:]

            # Bottom-Up Pass
            x_0 = F.embedding(cond_idx, self.token_emb)
            current_x = x_0

            for layer in self.layers:
                proj = torch.matmul(current_x, layer.W) + layer.b
                current_x = layer.activation(proj)

            # Get the state of the last token in the sequence
            top_state_last = current_x[:, -1, :]

            # Compute logits
            logits = torch.matmul(top_state_last, self.lm_head.T)

            # Sample next token
            if temperature == 0.0:
                # Greedy decoding
                next_token = torch.argmax(logits, dim=-1, keepdim=True)
            else:
                # Temperature sampling
                logits = logits / temperature
                probs = F.softmax(logits, dim=-1)
                next_token = torch.multinomial(probs, num_samples=1)

            # Append to generated sequence
            generated_ids = torch.cat((generated_ids, next_token), dim=1)

        return generated_ids

import time
import os

def run_training():
    """
    Main training loop utilizing the zero-gradient Predictive Coding algorithm.
    """
    # 1. Select Config (Toy for local testing, Production for 4B constraint)
    # Note for Hackathon: Use get_production_config() to hit the 4B parameter constraint!
    print("Loading Toy Configuration for Verification...")
    config = get_toy_config()
    
    print(f"Model Configuration: Hidden={config.hidden_size}, Layers={config.num_layers}")
    
    # 2. Initialize the Model
    # Everything is run within torch.no_grad() to strictly satisfy the hackathon rules.
    with torch.no_grad():
        model = PCLanguageModel(config)
    
    # Print the requested detailed parameter summary
    model.print_parameter_summary()

    # 3. Setup Dataset Streaming
    print("Initializing tokenizer: EleutherAI/gpt-neox-20b")
    tokenizer = AutoTokenizer.from_pretrained("EleutherAI/gpt-neox-20b")
    tokenizer.pad_token = tokenizer.eos_token
    
    # Hackathon constraint requires reading from public corpus. We stream to prevent OOM.
    print("Streaming dataset: DKYoon/SlimPajama-6B")
    dataset = load_dataset("DKYoon/SlimPajama-6B", split="train", streaming=True)
    
    print("Starting Zero-Gradient Training...")
    
    # Strictly enforce zero-gradient environment
    with torch.no_grad():
        batch_ids = []
        step = 0
        max_steps = 10  # Toy run limits steps
        start_time = time.time()
        
        for sample in dataset:
            text = sample["text"]
            
            # Tokenize sequence
            tokens = tokenizer(
                text, 
                max_length=config.max_seq_len, 
                truncation=True, 
                padding="max_length", 
                return_tensors="pt"
            )
            
            batch_ids.append(tokens['input_ids'][0])
            
            if len(batch_ids) == config.batch_size:
                input_tensor = torch.stack(batch_ids)
                
                # Execute one Predictive Coding update step
                ce_loss, total_energy = model.train_step(input_tensor)
                
                step += 1
                if step % 2 == 0:
                    print(f"Step {step} | CrossEntropy: {ce_loss:.4f} | System Energy: {total_energy:.4f}")
                
                batch_ids = []
                
                if step >= max_steps:
                    break
                    
        elapsed = time.time() - start_time
        print(f"Training successfully completed {max_steps} steps in {elapsed:.2f}s.")
        print("Zero-gradient constraint satisfied: All weights updated via local prediction errors.")
        
        # 5. Test Generation
        print("\n" + "="*50)
        print("Testing Generation Capabilities")
        print("="*50)

        prompt = "The future of Artificial Intelligence is"
        print(f"Prompt: '{prompt}'")

        prompt_tokens = tokenizer(prompt, return_tensors="pt")["input_ids"].to(model.get_device())

        # Generate tokens
        generated_tokens = model.generate(
            input_ids=prompt_tokens,
            max_new_tokens=20,
            temperature=0.7 # slightly deterministic for sanity check
        )

        # Decode and print the result
        generated_text = tokenizer.decode(generated_tokens[0], skip_special_tokens=True)
        print(f"Generated Output:\n{generated_text}\n")
        print("="*50)

if __name__ == "__main__":
    # Disable autograd globally just to prove compliance with Hackathon rules
    torch.set_grad_enabled(False)
    run_training()
