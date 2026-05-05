# Mathematical Foundation: Predictive Coding as a Local Learning Rule

## 1. Core Philosophy

Predictive Coding (PC) posits that the brain (and by extension, an artificial neural network) is a hierarchical generative model. The goal is not merely to map inputs to outputs, but to **minimize the prediction error** at every level of the hierarchy.

Unlike Backpropagation (BP), which requires a distinct backward pass to transport global error gradients, PC operates as a **continuous dynamical system** where inference (state update) and learning (weight update) occur simultaneously using only local information.

## 2. The Generative Model

Consider a hierarchical network with layers $l = 0, \dots, L$.
- $x_l$: The state (activation) of layer $l$.
- $W_l$: The weights connecting layer $l$ to $l+1$.
- $f(\cdot)$: A non-linear activation function.

The **generative model** predicts the state of a lower layer based on the state of the layer above it:
$$ \hat{x}_l = f(W_l x_{l+1}) $$
*(Note: In some formulations, prediction flows top-down, while sensory data flows bottom-up. Here we define the prediction error based on the discrepancy between the actual state and the top-down prediction.)*

## 3. The Energy Function (Variational Free Energy)

The core of PC is the minimization of an energy function $E$, often equivalent to Variational Free Energy. This energy is the sum of squared prediction errors ($\epsilon$) across all layers:

$$ E = \sum_{l=0}^{L} \frac{1}{2} \| \epsilon_l \|^2 $$

Where the prediction error $\epsilon_l$ at layer $l$ is defined as the difference between the actual state $x_l$ and the predicted state from the layer above:

$$ \epsilon_l = x_l - f(W_l x_{l+1}) $$

**Boundary Conditions:**
- **Input Layer ($l=0$):** $x_0$ is clamped to the input data $d$. The error $\epsilon_0 = d - \hat{x}_0$ drives the inference.
- **Output Layer ($l=L$):** If labels are present, $x_L$ can be clamped to the target $y$, or a specific loss term can be added. For unsupervised learning, the top layer prior acts as the constraint.

## 4. Dynamics: Inference vs. Learning

PC separates the process into two interacting dynamics, both derived from the gradient of the Energy function $\nabla E$.

### A. Inference (State Update)
The network states $x_l$ evolve to minimize $E$ while weights $W$ are fixed. This is an iterative process (often running for $T$ steps per batch) until the network reaches an equilibrium where prediction errors are minimized.

Using gradient descent on states:
$$ \Delta x_l \propto -\frac{\partial E}{\partial x_l} $$

Applying the chain rule:
$$ \frac{\partial E}{\partial x_l} = \epsilon_l - W_{l-1}^T \left( \epsilon_{l-1} \odot f'(W_{l-1} x_l) \right) $$

Thus, the update rule for state $x_l$ is:
$$ x_l \leftarrow x_l + \eta_x \left( \epsilon_l - W_{l-1}^T (\epsilon_{l-1} \odot f'(W_{l-1} x_l)) \right) $$

*Interpretation:* The state $x_l$ is pulled in two directions:
1.  **Bottom-up:** To match the prediction from above ($\epsilon_l$).
2.  **Top-down:** To help the layer below make a better prediction ($W^T \epsilon_{l-1}$).

### B. Learning (Weight Update)
Once the states have settled (or concurrently), the weights $W_l$ are updated to minimize $E$.

$$ \Delta W_l \propto -\frac{\partial E}{\partial W_l} $$

Deriving the gradient:
$$ \frac{\partial E}{\partial W_l} = \frac{\partial}{\partial W_l} \frac{1}{2} \| x_l - f(W_l x_{l+1}) \|^2 $$
$$ = -\epsilon_l \odot f'(W_l x_{l+1}) \cdot x_{l+1}^T $$

Thus, the local learning rule is:
$$ \Delta W_l = \eta_w \left( (\epsilon_l \odot f'(W_l x_{l+1})) \cdot x_{l+1}^T \right) $$

**Crucial Property:** This update depends **only** on:
1.  The local prediction error $\epsilon_l$ (available at the synapse).
2.  The presynaptic activity $x_{l+1}$.
3.  The derivative of the activation function.

No global loss signal or backward transport of errors from the output layer is required. This strictly satisfies the **Local Learning Rule** constraint of the hackathon.

## 5. Comparison with Backpropagation

| Feature | Backpropagation (BP) | Predictive Coding (PC) |
| :--- | :--- | :--- |
| **Error Signal** | Global loss gradient $\frac{\partial L}{\partial W}$ | Local prediction error $\epsilon_l$ |
| **Memory** | Must store all activations for backward pass | Can be implemented equilibrium-based; lower memory footprint |
| **Update Timing** | Strict Forward Pass $\to$ Backward Pass | Continuous / Simultaneous Inference & Learning |
| **Biological Plausibility** | Low (requires symmetric weight transport) | High (local Hebbian-like updates) |
| **Convergence** | Fast per step, high memory cost | Slower per step (requires inference iterations), but highly parallelizable |

## 6. Implementation Strategy for the Hackathon

To surpass BP in the context of the "Post-Backprop Challenge," our implementation will leverage:

1.  **Equilibrium Propagation Approximation:** Instead of running full inference to convergence for every batch (which is slow), we use a few inference steps ($T_{infer} \approx 5-10$) initialized from the previous batch's state. This creates a "momentum" effect in the latent space.
2.  **Hybrid Architecture:** We combine PC with efficient architectures like State Space Models (SSM) or Mixture of Experts (MoE). The PC rule handles the weight updates, while the architecture handles long-range dependencies efficiently.
3.  **Memory Efficiency:** By avoiding the storage of the full computational graph for autograd (calculating gradients manually via the PC equations), we reduce VRAM usage significantly, allowing larger models (approaching the 4B parameter limit) on the provided T4 GPUs.
4.  **Stability:** We employ RMSNorm and careful initialization to ensure the iterative inference dynamics do not diverge.

## 7. Conclusion

Predictive Coding offers a mathematically rigorous, biologically plausible alternative to Backpropagation. By framing learning as energy minimization through local prediction errors, it bypasses the need for non-local weight transport. While traditionally slower to converge per epoch, its memory efficiency and parallelizability make it a strong candidate for scaling to large parameter counts under strict hardware constraints, potentially outperforming BP in the specific metric of **parameters-per-byte-of-memory** and **training stability** in low-precision environments.
