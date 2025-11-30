# Linear Attention & Kimi Linear Research Course Plan

## Course Goal
To guide viewers from the basics of attention mechanisms to the cutting-edge "Kimi Linear" architecture, enabling them to understand, implement, and conduct research on efficient attention models.

## Target Audience
AI researchers, engineers, and advanced students familiar with Transformers and PyTorch.

---

## Module 1: The Foundation - From Softmax to Linear Attention

### Concepts
1.  **The Quadratic Bottleneck**: Why Standard Attention is $O(N^2)$.
    *   $\text{Attention}(Q, K, V) = \text{softmax}(\frac{QK^T}{\sqrt{d_k}})V$
    *   The $N \times N$ matrix is the problem.
2.  **The Kernel Trick**: Katharopoulos et al. (2020).
    *   $\phi(x)$ feature map.
    *   Associative property of matrix multiplication: $(QK^T)V \rightarrow Q(K^TV)$.
    *   Complexity becomes $O(N \cdot d^2)$ (linear in sequence length).
3.  **Recurrent View (RNN)**:
    *   Linear attention can be computed as an RNN.
    *   $S_t = S_{t-1} + \phi(k_t)\phi(v_t)^T$
    *   $o_t = \phi(q_t)^T S_t$

### Math
*   Standard Attention Equation.
*   Linear Attention Recurrence Equation.

### Key Papers
*   *Transformers are RNNs: Fast Autoregressive Transformers with Linear Attention* (Katharopoulos et al., 2020)

---

## Module 2: The Delta Rule - Attention as Online Learning

### Concepts
1.  **Linear Attention as Gradient Descent**:
    *   Standard Linear Attention optimizes an unbounded correlation objective.
    *   Problem: Memory $S_t$ grows indefinitely, leading to interference.
2.  **DeltaNet**:
    *   Reinterprets recurrence as online gradient descent on a **reconstruction objective**: $\mathcal{L}_t = \frac{1}{2} \| S^T k_t - v_t \|^2$.
    *   **The Delta Rule**: $S_t = S_{t-1} - \beta \nabla \mathcal{L}$.
    *   Update: $S_t = (I - \beta k_t k_t^T)S_{t-1} + \beta k_t v_t^T$.
    *   This is a "write" operation that also "erases" conflicting information (Householder transformation).

### Math
*   Reconstruction Loss Objective.
*   Delta Rule Update Equation.

### Key Papers
*   *Linear Transformers Are Secretly Fast Weight Programmers* (Schlag et al., 2021)

---

## Module 3: Gating Mechanisms - Controlling Memory

### Concepts
1.  **The Need for Forgetting**:
    *   DeltaNet is better, but still retains outdated info too long.
2.  **Gated Linear Attention (GLA)**:
    *   Introduces a decay factor $\alpha_t$.
    *   $S_t = \alpha_t S_{t-1} + k_t v_t^T$.
3.  **Gated DeltaNet (GDN)**:
    *   Combines Delta Rule with Gating.
    *   $S_t = \alpha_t (I - \beta k_t k_t^T)S_{t-1} + \beta k_t v_t^T$.
    *   $\alpha_t$ acts as **weight decay** on the fast weights.

### Math
*   Gated Recurrence Equations.
*   Comparison of scalar vs. vector gating.

### Key Papers
*   *Gated Linear Attention Transformers with Hardware-Efficient Training* (Yang et al., 2024)
*   *Gated Delta Networks* (Yang et al., 2025)

---

## Module 4: Kimi Delta Attention (KDA) - The Core Architecture

### Concepts
1.  **Fine-Grained Gating**:
    *   GDN uses a scalar gate (per head).
    *   **KDA** uses a **diagonal matrix gate** (per channel/feature).
    *   Allows independent forgetting rates for each feature dimension.
    *   $S_t = (I - \beta k_t k_t^T) \text{Diag}(\alpha_t) S_{t-1} + \beta k_t v_t^T$.
2.  **Relation to RoPE**:
    *   KDA's decay can be seen as a **learnable, data-dependent positional encoding**.
    *   Contrasts with RoPE's fixed rotation frequencies.
3.  **Chunk-wise Parallelization**:
    *   Training linear attention sequentially is slow ($O(N)$ but non-parallel).
    *   **Chunking**: Split sequence into chunks of size $C$.
    *   **Intra-chunk**: Compute parallelly (like Attention).
    *   **Inter-chunk**: Recurrent update (like RNN).
    *   **WY Representation**: Efficiently packing rank-1 updates.

### Math
*   KDA Recurrence Equation (Eq. 1 in paper).
*   Chunk-wise formulation (Eq. 3 in paper).
*   WY Representation (Eq. 5 in paper).

### Key Papers
*   *Kimi Linear: An Expressive, Efficient Attention Architecture* (The current paper)

---

## Module 5: Implementation Deep Dive

### Code Walkthrough (`fla/layers/kda.py`)
1.  **`KimiDeltaAttention` Class**:
    *   Initialization: `q_proj`, `k_proj`, `v_proj`, `f_proj` (for gate), `b_proj` (for beta).
    *   `ShortConvolution`: Local mixing before the linear attention.
2.  **Forward Pass**:
    *   Projections & Activation (SiLU).
    *   Gate computation: `g = f_proj(x)`, `beta = b_proj(x).sigmoid()`.
    *   Mode selection: `chunk` (training) vs `fused_recurrent` (inference/generation).
3.  **Kernels (`fla/ops/kda`)**:
    *   `chunk_kda`: The heavy lifter for training.
    *   `fused_recurrent_kda`: Optimized for generation.

### Practical Exercise
*   Run the `flash-linear-attention` benchmark.
*   Implement a simple version of the KDA recurrence in pure PyTorch to verify understanding.

---

## Module 6: Research Frontiers & Questions

### Research Questions
1.  **Hybrid Architectures**:
    *   Kimi Linear uses a 3:1 ratio (3 KDA layers, 1 Full Attention).
    *   *Question*: Can we dynamically determine where to place Full Attention layers?
    *   *Question*: Can KDA completely replace Full Attention with better gating?
2.  **Gating vs. RoPE**:
    *   The paper argues KDA handles position via gating.
    *   *Question*: How does KDA perform on "Needle in a Haystack" tasks compared to RoPE-based models without explicit position embeddings?
3.  **RL Test-Time Scaling**:
    *   The paper mentions KDA is good for RL.
    *   *Question*: Why specifically? Is it the memory efficiency allowing more rollouts, or the "forgetting" allowing better state tracking?
4.  **Hardware Efficiency**:
    *   KDA simplifies DPLR.
    *   *Question*: Can we further quantize the state $S_t$ for even lower memory usage?

### Suggestions for Viewers
*   **Replicate**: Try to train a small Kimi Linear model on a dataset like Wikitext-103.
*   **Ablate**: Remove the fine-grained gating and see how performance drops.
*   **Extend**: Apply KDA to Vision Transformers (ViT).
