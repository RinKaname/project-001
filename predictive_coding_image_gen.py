import torch
import torch.nn.functional as F
import math
import os
from torchvision import transforms
from datasets import load_dataset
from PIL import Image

class PCConvLayer:
    """
    A single Predictive Coding Convolutional layer.
    It takes an input from below (x_{l-1}), a prediction from above (x_{l+1}),
    and updates its weights locally using F.unfold (Zero Autograd).
    """
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int = 3, stride: int = 1, padding: int = 1, eta_x: float = 0.1, eta_w: float = 0.001):
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding

        self.eta_x = eta_x
        self.eta_w = eta_w

        # Initialize Convolutional Weights manually
        # Shape: [out_channels, in_channels, kernel_size, kernel_size]
        bound = 1.0 / math.sqrt(in_channels * kernel_size * kernel_size)
        self.W = torch.empty(out_channels, in_channels, kernel_size, kernel_size).uniform_(-bound, bound)
        self.b = torch.zeros(out_channels)

        if torch.cuda.is_available():
            self.W = self.W.cuda()
            self.b = self.b.cuda()

    def activation(self, x):
        """SiLU Activation"""
        return F.silu(x)

    def activation_derivative(self, x):
        """Derivative of SiLU"""
        sig = torch.sigmoid(x)
        return sig * (1 + x * (1 - sig))

    def forward_predict(self, x_higher):
        """
        Top-down prediction: predict the current layer's state based on the higher layer.
        Because information flows from abstract (small spatial, many channels) to concrete (large spatial, few channels),
        this uses a transposed convolution (or standard conv if spatial size is kept same).
        For simplicity in this experimental script, we maintain spatial dimensions across layers.
        x_higher: [Batch, out_channels, H, W]
        returns predicted x_lower: [Batch, in_channels, H, W]
        """
        # Linear prediction downwards
        # W shape is [out, in, K, K]. conv_transpose2d expects weight shape [in_channels, out_channels, K, K]
        # Wait, if x_higher is [B, out_channels, H, W], and we want [B, in_channels, H, W]

        # Actually, let's treat the generative direction (top-down) as the primary forward pass
        # The abstract layer (in_channels) generates the concrete layer (out_channels).
        # We will use standard conv2d for the generative top-down prediction.

        proj = F.conv2d(x_higher, self.W, bias=self.b, padding=self.padding, stride=self.stride)

        # Apply InstanceNorm as a guardrail
        proj = F.instance_norm(proj)
        return self.activation(proj), proj

    def update_weights(self, error_local, pre_act_proj, x_higher):
        """
        Local Convolutional Hebbian Weight Update (NO AUTOGRAD).
        Uses F.unfold to extract patches and compute outer products.

        error_local: The prediction error at the LOWER layer [B, out_channels, H, W]
        pre_act_proj: Raw projection before activation [B, out_channels, H, W]
        x_higher: The state from the HIGHER layer [B, in_channels, H, W]
        """
        B, C_out, H, W = error_local.shape

        # 1. Local error scaled by activation derivative
        f_prime = self.activation_derivative(pre_act_proj) # [B, out, H, W]
        local_signal = error_local * f_prime # [B, out, H, W]

        # Flatten the local signal spatially
        # Shape: [B, out, H*W] -> [B, out, L]
        local_signal_flat = local_signal.view(B, self.out_channels, -1)

        # 2. Extract patches from the higher state using F.unfold
        # Unfold extracts sliding local blocks from a batched input tensor.
        # Shape: [B, in_channels * kernel_size * kernel_size, L]
        patches = F.unfold(x_higher, kernel_size=self.kernel_size, padding=self.padding, stride=self.stride)

        # 3. Compute outer product over the spatial dimensions
        # We want delta_W of shape [out_channels, in_channels * K * K]
        # local_signal_flat: [B, out, L]
        # patches: [B, in*K*K, L] -> transpose to [B, L, in*K*K]

        # bmm gives [B, out, in*K*K]
        delta_W_batch = torch.bmm(local_signal_flat, patches.transpose(1, 2))

        # Average over the batch
        delta_W_flat = delta_W_batch.mean(dim=0) / (H * W) # Average over spatial locations too

        # Reshape to original weight shape [out_channels, in_channels, K, K]
        delta_W = delta_W_flat.view_as(self.W)

        # Update Bias
        delta_b = local_signal.mean(dim=(0, 2, 3))

        # Apply update
        self.W.add_(delta_W, alpha=self.eta_w)
        self.b.add_(delta_b, alpha=self.eta_w)


class PCImageModel:
    """
    A multi-layer Hierarchical Predictive Coding Model for images.
    """
    def __init__(self, image_size=64, channels=3, hidden_dim=64, num_layers=3):
        self.image_size = image_size
        self.channels = channels
        self.t_infer = 5
        self.eta_x = 0.05

        self.layers = []

        # Layer 0 (Pixels) to Layer 1
        self.layers.append(PCConvLayer(in_channels=hidden_dim, out_channels=channels))

        # Hidden Layers
        for _ in range(num_layers - 1):
            self.layers.append(PCConvLayer(in_channels=hidden_dim, out_channels=hidden_dim))

    def get_device(self):
        return self.layers[0].W.device

    def train_step(self, image_batch):
        """
        image_batch: [Batch, 3, H, W]
        """
        image_batch = image_batch.to(self.get_device())
        B = image_batch.size(0)

        # 1. Initialize States
        # Normally we'd do a bottom-up pass, but for simplicity we initialize top states randomly
        states = []

        # L0 is clamped to the actual image pixels
        states.append(image_batch.clone())

        # Initialize hidden spatial states
        for layer in self.layers:
            # We initialize hidden states with random noise
            h_state = torch.randn(B, layer.in_channels, self.image_size, self.image_size, device=self.get_device()) * 0.1
            states.append(h_state)

        next_states = [torch.empty_like(s) for s in states]
        next_states[0].copy_(states[0]) # Clamp L0

        # 2. Inference Relaxation Loop (Minimizing Error)
        for t in range(self.t_infer):
            for l in range(1, len(self.layers) + 1):
                layer = self.layers[l-1]

                # Top-down prediction: predict L_{l-1} from L_l
                pred_prev, pre_act = layer.forward_predict(states[l])

                # Local Error at L_{l-1}
                err_prev = states[l-1] - pred_prev

                # Bottom-up influence (Gradient of error with respect to states[l])
                # grad_bottom_up = d(error)/d(x_l)
                # Since prediction = conv(x_l, W), d(pred)/dx_l = conv_transpose(err * f_prime, W)
                f_prime = layer.activation_derivative(pre_act)
                grad_bottom_up = -F.conv_transpose2d(err_prev * f_prime, layer.W, padding=layer.padding, stride=layer.stride)

                # Top-down influence (pulled by prediction from above)
                grad_top_down = torch.zeros_like(states[l])
                if l < len(self.layers):
                    next_layer = self.layers[l]
                    pred_curr, _ = next_layer.forward_predict(states[l+1])
                    grad_top_down = states[l] - pred_curr

                total_grad = torch.clamp_(grad_bottom_up + grad_top_down, -0.1, 0.1)

                next_states[l].copy_(states[l])
                next_states[l].sub_(total_grad, alpha=self.eta_x)

            states, next_states = next_states, states

        # 3. Weight Updates
        total_mse = 0.0
        for l in range(1, len(self.layers) + 1):
            layer = self.layers[l-1]
            pred_prev, pre_act = layer.forward_predict(states[l])
            err_prev = states[l-1] - pred_prev

            total_mse += (err_prev ** 2).mean().item()
            layer.update_weights(err_prev, pre_act, states[l])

        return total_mse

    def generate(self, batch_size=1):
        """
        Unconditional Generation ("Dreaming")
        Inject random noise at the top layer, and let the network generate pixels at L0.
        """
        # Initialize States
        states = []

        # L0 is un-clamped! We start with random gray pixels
        x_0 = torch.zeros(batch_size, self.channels, self.image_size, self.image_size, device=self.get_device())
        states.append(x_0)

        for layer in self.layers:
            states.append(torch.zeros(batch_size, layer.in_channels, self.image_size, self.image_size, device=self.get_device()))

        # Clamp the VERY TOP layer with random abstract noise
        states[-1] = torch.randn_like(states[-1]) * 1.0

        next_states = [torch.empty_like(s) for s in states]
        next_states[-1].copy_(states[-1]) # Lock the top layer

        # Relaxation Loop (Let the dream cascade downwards)
        # We run it longer to allow pixels to form
        for t in range(self.t_infer * 4):
            # We now update ALL layers below the top, including L0 (the pixels)
            for l in range(0, len(self.layers)):

                if l == 0:
                    # Update pixels based on top-down prediction
                    layer_above = self.layers[0]
                    pred_pixels, _ = layer_above.forward_predict(states[1])
                    err_pixels = states[0] - pred_pixels

                    next_states[0].copy_(states[0])
                    next_states[0].sub_(err_pixels, alpha=self.eta_x * 2) # Move pixels toward prediction

                else:
                    layer = self.layers[l-1]
                    pred_prev, pre_act = layer.forward_predict(states[l])
                    err_prev = states[l-1] - pred_prev

                    f_prime = layer.activation_derivative(pre_act)
                    grad_bottom_up = -F.conv_transpose2d(err_prev * f_prime, layer.W, padding=layer.padding, stride=layer.stride)

                    grad_top_down = torch.zeros_like(states[l])
                    if l < len(self.layers) - 1: # Don't calculate top-down for the locked top layer
                        next_layer = self.layers[l]
                        pred_curr, _ = next_layer.forward_predict(states[l+1])
                        grad_top_down = states[l] - pred_curr

                    total_grad = torch.clamp_(grad_bottom_up + grad_top_down, -0.1, 0.1)

                    next_states[l].copy_(states[l])
                    next_states[l].sub_(total_grad, alpha=self.eta_x)

            states, next_states = next_states, states

        generated_pixels = torch.clamp(states[0], 0.0, 1.0)
        return generated_pixels


def test_training():
    print("Initializing Zero-Gradient PC Image Generator...")
    # Keep it tiny (32x32) so it runs fast and doesn't OOM on CPU
    IMG_SIZE = 32

    with torch.no_grad():
        model = PCImageModel(image_size=IMG_SIZE, channels=3, hidden_dim=16, num_layers=2)

    print("Loading Dataset: BangumiBase/princessconnectredive (Streaming mode)")
    try:
        # User requested alternative script-free dataset
        dataset = load_dataset("BangumiBase/princessconnectredive", split="train", streaming=True)
    except Exception as e:
        print(f"Failed to load dataset: {e}")
        return

    transform = transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.ToTensor()
    ])

    print("Starting Training Loop...")
    batch_size = 4
    batch_tensors = []

    with torch.no_grad():
        step = 0
        max_steps = 5 # Just a quick sanity check

        for sample in dataset:
            img = sample['image']
            if img.mode != 'RGB':
                img = img.convert('RGB')

            img_tensor = transform(img)
            batch_tensors.append(img_tensor)

            if len(batch_tensors) == batch_size:
                x_batch = torch.stack(batch_tensors)

                mse = model.train_step(x_batch)

                step += 1
                print(f"Step {step}/{max_steps} | System Energy (MSE): {mse:.4f}")

                batch_tensors = []

                if step >= max_steps:
                    break

        print("Training complete! Zero autograd verified for Convolutional Hebbian updates.")

        print("\nTesting Unconditional Generation ('Dreaming')...")
        generated_images = model.generate(batch_size=1)
        print(f"Successfully generated image tensor of shape: {generated_images.shape}")

        # Save output image
        import torchvision
        torchvision.utils.save_image(generated_images, "pc_dream.png")
        print("Saved generated image to pc_dream.png")

if __name__ == "__main__":
    # Strictly disable autograd globally
    torch.set_grad_enabled(False)
    test_training()