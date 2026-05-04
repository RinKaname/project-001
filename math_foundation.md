# Mathematical Foundation: Zero-Gradient State Space Model (SSM)

This document formalizes the architecture and learning rule for a custom, zero-gradient sequence model. This approach is designed to satisfy the strict constraints of "The Post-Backprop Challenge," specifically bypassing the global chain rule (Backpropagation Through Time) while maintaining extreme computational efficiency.

## 1. The Architecture: Custom Selective State Space Model

Standard Transformer architectures utilize Self-Attention, which suffers from an $\mathcal{O}(N^2)$ memory bottleneck with respect to sequence length $N$. To adhere to the strict 16GB VRAM constraint for a 4 Billion parameter model, we define a continuous-time State Space Model (SSM), discretized for token processing.

Given an input sequence $x \in \mathbb{R}^{B \times L \times D}$ (Batch, Length, Dimension), a discrete SSM operates sequentially over each time step $t$:

$$ h_t = \bar{A} h_{t-1} + \bar{B} x_t $$
$$ y_t = C h_t + D x_t $$

Where:
*   $h_t \in \mathbb{R}^N$ is the hidden state.
*   $\bar{A}, \bar{B}$ are discretized state matrices derived from continuous parameters $(\Delta, A, B)$ using a discretization rule (e.g., Zero-Order Hold):
    *   $\bar{A} = \exp(\Delta A)$
    *   $\bar{B} = (\Delta A)^{-1}(\exp(\Delta A) - I) \cdot \Delta B$
*   $C, D$ are projection matrices.

**The "Selective" Mechanism (Crucial for language):**
Unlike linear time-invariant systems, language requires context-dependent routing. We make $\Delta, B,$ and $C$ functions of the input $x_t$ (Selective SSM, inspired by Mamba):

$$ s_t = \text{Linear}(x_t) $$
$$ \Delta_t = \text{Softplus}(\text{Linear}_{\Delta}(s_t)) $$
$$ B_t = \text{Linear}_B(s_t) $$
$$ C_t = \text{Linear}_C(s_t) $$

The discrete update becomes:
$$ h_t = \bar{A}_t h_{t-1} + \bar{B}_t x_t $$

## 2. The Problem: The Banned Global Chain Rule (BPTT)

In standard training, given a final target $T_t$ and final network output $\hat{Y}_t$, the global loss is $\mathcal{L} = \text{CrossEntropy}(\hat{Y}_t, T_t)$.

To update the parameters $W^{(l)}$ of an intermediate layer $l$, standard Deep Learning relies on the chain rule, passing gradients backwards through all subsequent layers:

$$ \frac{\partial \mathcal{L}}{\partial W^{(l)}} = \frac{\partial \mathcal{L}}{\partial Y^{(L)}} \frac{\partial Y^{(L)}}{\partial Y^{(L-1)}} \dots \frac{\partial Y^{(l+1)}}{\partial Y^{(l)}} \frac{\partial Y^{(l)}}{\partial W^{(l)}} $$

**This exact mathematical formulation is banned.** It requires storing the activation graph of the entire network in memory, violating both the hackathon rules and the core principles of biological plausibility.

## 3. The Solution: The Forward-Forward Algorithm

To strictly adhere to the "Zero Existing Optimizers" and "Zero torch.autograd" rules, we completely abandon automatic differentiation. We mathematically sever the dependency between layers using a Hebbian-inspired learning rule known as the **Forward-Forward (FF)** algorithm.

### The Goodness Objective
Instead of minimizing a Cross-Entropy loss via gradients, we train each layer to classify data as either "Real" (Positive) or "Corrupted" (Negative).

For any arbitrary layer $l$, receiving input $X^{(l)}$, its output is $Y^{(l)}$. We define the "Goodness" $\mathcal{G}$ of the layer as the sum of squared activations:

$$ \mathcal{G}(Y^{(l)}) = \sum_{j} (Y^{(l)}_j)^2 $$

The learning objective for layer $l$ is to push the Goodness well above a threshold $\theta$ for positive data, and well below $\theta$ for negative data. Let $\sigma$ be the logistic sigmoid function:

$$ p(\text{positive}) = \sigma(\mathcal{G}(Y_{pos}^{(l)}) - \theta) $$
$$ p(\text{negative}) = 1 - \sigma(\mathcal{G}(Y_{neg}^{(l)}) - \theta) $$

### The Raw Tensor Hebbian Update (Zero Autograd)

To optimize this objective **without utilizing the global chain rule or any automatic differentiation**, we derive the raw tensor update explicitly. The weight update $\Delta W^{(l)}$ for a linear projection within the layer is defined purely by the outer product of the inputs and the downstream error scalar:

Let the scalar error for a positive pass be $e_{pos} = 1 - p(\text{positive})$
Let the scalar error for a negative pass be $e_{neg} = 0 - (1 - p(\text{negative}))$

The exact explicit weight update matrix applied to $W^{(l)}$ is calculated via pure matrix multiplication:

$$ \Delta W^{(l)} = \alpha \left( e_{pos} \cdot (X_{pos}^{(l)})^T Y_{pos}^{(l)} + e_{neg} \cdot (X_{neg}^{(l)})^T Y_{neg}^{(l)} \right) $$

**The Mathematical Guarantee:**
1. The update $\Delta W^{(l)}$ relies exclusively on $X^{(l)}$ (the input to the layer) and $Y^{(l)}$ (the output of the layer).
2. The term $\frac{\partial Y^{(l+1)}}{\partial Y^{(l)}}$ (the backward error signal from the next layer) does not exist in the formulation.
3. The calculation is executed using primitive `torch.matmul` operations, bypassing `torch.autograd` completely and satisfying the strictest constraints of the Post-Backprop Challenge.

### Practical Execution (The 3-Hour Constraint)
1.  **Generate Negatives:** We generate $X_{neg}$ by randomly permuting the token sequences of the real batch $X_{pos}$.
2.  **Forward Pass 1:** $X_{pos}$ moves through layer $l$. We compute Goodness and $e_{pos}$.
3.  **Forward Pass 2:** $X_{neg}$ moves through layer $l$. We compute Goodness and $e_{neg}$.
4.  **Raw Tensor Update:** We compute $\Delta W^{(l)}$ using primitive matrix operations and subtract it directly from the layer's parameters.
5.  **Detach and Forward:** The normalized positive output $Y_{pos}^{(l)}$ is mathematically detached to destroy the computation graph, and passed as input to layer $l+1$.

By discarding the computation graph after every single layer, memory consumption is $\mathcal{O}(1)$ with respect to network depth, easily satisfying the 50% peak VRAM reduction constraint and enabling massive throughput.

## 4. Scaling Efficiency: Zero-Gradient Mixture of Experts (MoE)

To meet the 4 Billion parameter architectural requirement while strictly adhering to the 180-minute training limit on a T4 GPU, we augment the SSM architecture with a locally-trained Mixture of Experts (MoE) mechanism. This decouples parameter count from active computational FLOPs.

### Local Routing Mechanism

For layer $l$, the output of the SSM block $Y_{ssm}^{(l)}$ is passed to a router network $R^{(l)}$:

$$ \text{logits}^{(l)} = Y_{ssm}^{(l)} \cdot W_r^{(l)} $$
$$ g^{(l)} = \text{Softmax}(\text{logits}^{(l)}) $$

Where $W_r^{(l)}$ is the router projection matrix, and $g^{(l)} \in \mathbb{R}^E$ represents the routing probabilities across $E$ experts. We select the top-$k$ experts (e.g., $k=1$ for maximum speed).

The output of the MoE layer is the weighted sum of the selected experts' Feed-Forward Networks (FFN):

$$ Y_{moe}^{(l)} = \sum_{i=1}^k g^{(l)}_{idx_i} \cdot \text{FFN}_{idx_i}(Y_{ssm}^{(l)}) $$

### Localized Router Optimization

In standard architectures, the router $W_r^{(l)}$ is trained via global backpropagation from the final loss. In our zero-gradient paradigm, the router is optimized strictly via the **Local Loss Function** $\mathcal{L}^{(l)}$ defined in Section 3.

Because our fundamental constraint forbids the calculus-based chain rule entirely, we derive the router and expert updates utilizing the same **Hebbian Outer-Product** objective defined in Section 3.

The positive and negative downstream error scalars ($e_{pos}$ and $e_{neg}$) drive the optimization.
For the MoE Router $W_r^{(l)}$, the surrogate algebraic update correlates the input sequence to the desired expert projection:

$$ \Delta W_r^{(l)} = \alpha \left( e_{pos} \cdot (X_{pos}^{(l)})^T R(X_{pos}^{(l)}) + e_{neg} \cdot (X_{neg}^{(l)})^T R(X_{neg}^{(l)}) \right) $$

For a routed expert $k$ with an up-projection $W_{up,k}^{(l)}$ and down-projection $W_{down,k}^{(l)}$, the explicit inner hidden state $H_k$ is recreated during the update step to form the localized outer products:

$$ \Delta W_{down,k}^{(l)} = \alpha \left( e_{pos} \cdot (H_{k, pos}^{(l)})^T Y_{pos}^{(l)} + e_{neg} \cdot (H_{k, neg}^{(l)})^T Y_{neg}^{(l)} \right) $$
$$ \Delta W_{up,k}^{(l)} = \alpha \left( e_{pos} \cdot (X_{pos}^{(l)})^T H_{k, pos}^{(l)} + e_{neg} \cdot (X_{neg}^{(l)})^T H_{k, neg}^{(l)} \right) $$

This ensures that the MoE layers expand to 4 Billion parameters and specialize for token clustering, all while rigorously satisfying the strict `torch.autograd` ban.
