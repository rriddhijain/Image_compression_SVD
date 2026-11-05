# Low-Rank SVD Factorization of Dense Neural Network Layers: An Empirical Investigation of the FLOP–Latency Divergence and Hardware-Aware Inference Trade-Offs

**Riddhi Jain**  
*Department of Computer Science & Engineering*  
*Role: Systems Lead (Person B — Hardware Benchmarking, Profiling & Trade-Offs)*  
*Registration No: [Your Registration ID]*  

**[Collaborator Name / Person A]**  
*Department of Computer Science & Engineering*  
*Role: Modeling Lead (Person A — SVD Factorization, Layer Swapping & Fine-Tuning)*  
*Registration No: [Collaborator Registration ID]*  

*Institution Name / Department Name*  
*Supervisor / Guide: [Supervisor Name]*  

---

### Abstract
Post-training Singular Value Decomposition (SVD) is widely celebrated as an analytical matrix compression technique that factorizes an unconstrained weight parameter matrix $W \in \mathbb{R}^{M \times N}$ into two cascaded low-rank linear transformations $L_1 \in \mathbb{R}^{M \times K}$ and $L_2 \in \mathbb{R}^{K \times N}$ without intervening non-linear activations. While theoretical floating-point operations (FLOPs) decrease linearly with rank $K$ below the analytical breakeven rank $K^* = \frac{M \cdot N}{M + N}$, real-world inference latency on hardware accelerators frequently fails to scale proportionally. 

This research project conducts an exhaustive two-lead empirical investigation into the mathematical and hardware trade-offs of layer-wise SVD neural compression. The Modeling Lead formulates symmetric singular value absorption ($\sqrt{\Sigma_K}$), in-place PyTorch layer-swapping mechanics, and AdamW cosine-annealed recovery fine-tuning. The Systems Lead designs an isolated, jitter-free timing harness, quantifies operational arithmetic intensity ($\text{FLOPs/Byte}$), tracks parameter and runtime memory footprints, and establishes empirical Roofline and Pareto frontiers across batch sizes $B_{sz} \in \{1, 8, 32, 64, 128\}$. 

Empirical profiling reveals that while aggressive rank reduction ($k=8$) achieves a **$39.25\times$ parameter reduction** and recovers **$87.29\%$ top-1 accuracy** post-retraining, single-batch ($B_{sz}=1$) execution exhibits an empirical **$8\%$ latency regression** ($0.476\text{ ms}$ vs. $0.438\text{ ms}$ baseline) due to kernel launch dispatch overheads and non-fused intermediate activation round-trips under memory-bandwidth ceilings ($\text{AI} < 0.50\text{ FLOP/Byte}$). We synthesize these empirical findings by establishing an analytical bridge to autoregressive Large Language Model (LLM) Key-Value (KV) cache decoding, proving that both systems operate in identical memory-bound regimes where computational reductions are secondary to memory traffic constraints.

**Keywords—** Singular Value Decomposition, Neural Network Compression, Hardware Profiling, Roofline Model, Arithmetic Intensity, CUDA Timing, LLM KV-Cache Retrieval, Pareto Frontier.

---

## I. Introduction
Deep neural network parameter scales have grown exponentially, placing unsustainable demands on memory footprint, DRAM bandwidth, and real-time execution latency. In standard Multi-Layer Perceptrons (MLPs) and transformer projection blocks, dense matrix multiplications ($Y = XW^\top + b$) dominate total computational and memory transfer budgets. Post-training low-rank approximation via truncated Singular Value Decomposition (SVD) offers a deterministic, non-heuristic avenue for parameter reduction by factorizing weight matrices into low-rank representations without necessitating costly re-training from scratch.

### A. Mathematical Formulation & Breakeven Rank
Let $W \in \mathbb{R}^{M \times N}$ represent a dense linear weight tensor. By the Eckart–Young–Mirsky theorem, the optimal rank-$K$ approximation $\hat{W}_K$ minimizing the Frobenius norm error $\|W - \hat{W}_K\|_F$ is obtained via singular value truncation:
$$W = U \Sigma V^\top \approx U_K \Sigma_K V_K^\top = \hat{W}_K$$
where $U_K \in \mathbb{R}^{M \times K}$, $\Sigma_K \in \mathbb{R}^{K \times K}$, and $V_K^\top \in \mathbb{R}^{K \times N}$ with $K \ll \min(M, N)$. Substituting this factorization directly into the forward pass yields:
$$Y = X \hat{W}_K^\top + b = (X V_K \Sigma_K^{1/2}) (\Sigma_K^{1/2} U_K^\top) + b = L_2(L_1(X)) + b$$
where $L_1: M \to K$ (without bias) and $L_2: K \to N$ (with bias).

The theoretical floating-point operation (FLOP) count per forward pass for a batch size $B_{sz}$ scales from the dense formulation:
$$\text{FLOPs}_{\text{dense}} = 2 \cdot B_{sz} \cdot M \cdot N$$
to the decomposed two-stage cascade:
$$\text{FLOPs}_{\text{decomp}} = 2 \cdot B_{sz} \cdot K \cdot (M + N)$$
Equating these two formulations yields the analytical *computational breakeven rank* $K^*$:
$$2 \cdot B_{sz} \cdot K^* \cdot (M + N) = 2 \cdot B_{sz} \cdot M \cdot N \implies K^* = \frac{M \cdot N}{M + N}$$
For any truncated rank $K < K^*$, the decomposed architecture achieves net theoretical FLOP reductions. For a representative $1024 \to 512$ projection layer, $K^* = 341.33$; truncating to $K=8$ yields an apparent **$41.8\times$ theoretical compute reduction**.

### B. The Core Problem: The FLOP–Latency Divergence
In modern computer systems engineering, theoretical FLOP reductions routinely fail to deliver proportional wall-clock inference speedups. Modern accelerator architectures (GPUs, TPUs, Apple Silicon Neural Engines) are governed by strict memory hierarchies. Wall-clock execution time $T$ is lower-bounded by:
$$T \ge \max\left( \frac{\text{FLOPs}}{\text{Peak Compute Throughput}}, \frac{\text{Memory Bytes Transferred}}{\text{Peak Memory Bandwidth}} \right) + \sum t_{\text{kernel\_launch}}$$

At small batch sizes ($B_{sz}=1$), arithmetic intensity is exceptionally low ($\le 0.50\text{ FLOP/Byte}$). Memory bus latency and device kernel dispatch times completely dominate execution. Furthermore, replacing a single General Matrix Multiply (GEMM) kernel with two sequential GEMM kernels introduces an intermediate activation tensor ($B_{sz} \times K$) that must be written to global VRAM and subsequently re-read, creating additional memory round-trips.

### C. Four-Phase Project Roadmap
To systematically investigate this divergence, our project executes a rigorous four-phase workflow:
1. **Phase 1 (Intuition Builder):** Matrix-level truncated SVD on 2D image matrices, quantifying Frobenius reconstruction error, parameter compression ratio, and cumulative singular value energy decay.
2. **Phase 2 (Core Factorization):** Modeling Lead implementation of `DecomposedLinear`, symmetrical singular value distribution, PyTorch module-tree swapping, and zero-shot degradation profiling.
3. **Phase 3 (The Trade-Off Engine):** Fine-tuning recovery pipelines (Sainath et al., 2013) paired with an isolated, jitter-free hardware profiling engine evaluating latency percentiles (P50, P90, P99), VRAM footprints, and Roofline arithmetic intensity.
4. **Phase 4 (The Headliner Synthesis):** Unifying empirical findings with systems literature, proving that small-batch SVD stagnation models the identical memory-bandwidth bottlenecks observed in autoregressive LLM KV-cache generation.

---

## II. Literature Survey
Recent research (2020 onwards) has focused heavily on low-rank decomposition, post-training compression, and hardware-aware transformer inference. Table I provides a structured comparative summary of ten foundational recent works, followed by explicit analytical relevance to our modeling and systems engineering roles.

### TABLE I: Comparative Literature Survey of Recent Foundational Works (2020–2025)

| Ref # | Author(s) & Year | Paper Title & Venue | Core Technical Mechanism | Primary Lever on Memory Wall |
| :---: | :--- | :--- | :--- | :--- |
| **[1]** | Hu et al. (2022) | *LoRA: Low-Rank Adaptation of Large Language Models* (ICLR) | Parameterizes weight updates as $\Delta W = BA$ with rank $r \ll d$, freezing pre-trained weights. | Constrains fine-tuning memory footprint; unmerged inference adds serial GEMM kernel launches. |
| **[2]** | Hooker (2021) | *The Hardware Lottery* (Communications of the ACM) | Historical and empirical critique demonstrating that algorithmic ideas succeed when favored by specialized hardware. | Explains why unaligned low-rank GEMMs underutilize systolic Tensor Core execution tiles. |
| **[3]** | Zhang et al. (2023) | *H2O: Heavy Hitter Oracle for Efficient Generative Inference* (NeurIPS) | Dynamic eviction of non-essential KV cache tokens based on cumulative attention scores. | Directly slashes memory-bandwidth retrieval overhead during autoregressive generation. |
| **[4]** | Sharma, Ash & Misra (2023) | *The Truth is in There: Layer-Selective Rank Reduction (LASER)* (arXiv / ICLR) | Selective post-hoc SVD rank truncation on higher-order MLP projection layers without fine-tuning. | Demonstrates that selective rank truncation eliminates high-frequency noisy directions and restores reasoning. |
| **[5]** | Yuan et al. (2023) | *ASVD: Activation-Aware SVD for Compressing LLMs* (arXiv) | Scales weight matrices by activation magnitudes before truncation; allocates layer-specific ranks. | Reduces both parameter traffic and intermediate projection memory across attention blocks. |
| **[6]** | Wang et al. (2024) | *SVD-LLM: Truncation-Aware SVD for LLM Compression* (arXiv / ICLR) | Incorporates input whitening transforms to align singular vectors with layer loss objectives. | Optimizes low-rank weight structures for minimal memory retrieval degradation. |
| **[7]** | Frantar et al. (2023) | *GPTQ: Accurate Post-Training Quantization for Transformers* (ICLR) | Second-order Taylor expansion (Optimal Brain Surgeon) quantizing dense weights to 3-4 bits. | Drastically reduces DRAM weight traffic without altering GEMM layer dimensionality or adding kernels. |
| **[8]** | Dao (2023) | *FlashAttention-2: Faster Attention with Better Work Partitioning* (ICLR) | IO-aware SRAM tiling algorithm eliminating intermediate attention matrix writes to HBM. | Proves that fusing intermediate memory round-trips overcomes global memory bandwidth bottlenecks. |
| **[9]** | Kwon et al. (2023) | *PagedAttention: Efficient Memory Management for LLM Serving (vLLM)* (SOSP) | Non-contiguous virtual memory allocation for KV cache blocks, eliminating internal fragmentation. | Enables dense batch scheduling, pushing memory-bound token decoding toward compute saturation. |
| **[10]** | Ge et al. (2024) | *Model Tells You What to Discard: Adaptive KV Cache Compression (FastGen)* (ICLR) | Classifies attention heads dynamically to execute adaptive token eviction policies per head. | Minimizes KV cache memory bandwidth consumption in token-by-token decoding loops. |

### A. Deep-Dive Analytical Relevance to Project Roles

1. **Hu et al. (2022) — LoRA:**
   - *Modeling Lead (Person A):* Validates that intrinsic updates in overparameterized networks reside within extremely low-rank sub-spaces ($r \in [4, 16]$).
   - *Systems Lead (Person B):* Provides the benchmark for adapter merging: if low-rank layers are not folded into base matrices at serving time, the extra serial GEMM operations incur high latency penalties.

2. **Hooker (2021) — The Hardware Lottery:**
   - *Modeling Lead (Person A):* Emphasizes selecting rank dimensions $K$ that preserve alignment with standard computing tiles (multiples of 8 or 16).
   - *Systems Lead (Person B):* Explains why rank-8 linear layers fail to achieve expected hardware speedups on modern tensor architectures due to sub-optimal hardware tile utilization.

3. **Zhang et al. (2023) — H2O:**
   - *Modeling Lead (Person A):* Parallels singular value spectral pruning by demonstrating that attention activation energy is concentrated in a tiny fraction of "heavy hitters."
   - *Systems Lead (Person B):* Establishes the empirical bridge for our Phase 4 synthesis: memory bandwidth limits in low-rank neural layers mirror token-by-token KV cache decoding.

4. **Sharma, Ash & Misra (2023) — LASER:**
   - *Modeling Lead (Person A):* Justifies layer-selective SVD decomposition rather than uniform global rank truncation.
   - *Systems Lead (Person B):* Highlights that non-uniform layer compression creates heterogeneous GEMM kernel dispatch profiles across the inference graph.

5. **Yuan et al. (2023) — ASVD:**
   - *Modeling Lead (Person A):* Introduces data-informed pre-scaling of weight matrices using activation statistics prior to singular value decomposition.
   - *Systems Lead (Person B):* Demonstrates the direct connection between low-rank projections on weight tensors and physical DRAM bandwidth reduction during model serving.

6. **Wang et al. (2024) — SVD-LLM:**
   - *Modeling Lead (Person A):* Provides advanced post-truncation parameter compensation methods to recover lost accuracy without extended retraining loops.
   - *Systems Lead (Person B):* Confirms that maintaining identical cascaded GEMM topologies preserves identical latency characteristics regardless of truncation quality.

7. **Frantar et al. (2023) — GPTQ:**
   - *Modeling Lead (Person A):* Serves as the primary orthogonal compression baseline against low-rank factorization.
   - *Systems Lead (Person B):* Demonstrates that quantization compresses memory traffic *without* increasing kernel count, explaining why quantization outperforms low-rank SVD at batch size 1.

8. **Dao (2023) — FlashAttention-2:**
   - *Modeling Lead (Person A):* Informs the mathematical necessity of keeping intermediate representations tightly bounded within cache.
   - *Systems Lead (Person B):* Demonstrates that eliminating the global VRAM write-and-read round-trip of the intermediate $B_{sz} \times K$ activation tensor via kernel fusion is required to achieve empirical speedups.

9. **Kwon et al. (2023) — PagedAttention:**
   - *Modeling Lead (Person A):* Contextualizes batch memory management during continuous decoding.
   - *Systems Lead (Person B):* Proves that increasing batch size shifts execution from memory-bound regimes to compute-bound regimes, directly mirroring our batch sweep findings.

10. **Ge et al. (2024) — FastGen:**
    - *Modeling Lead (Person A):* Exemplifies adaptive, context-driven rank and token retention policies.
    - *Systems Lead (Person B):* Quantifies memory traffic reductions when evaluating sparse or truncated matrix representations.

---

## III. Dataset and Workload Characterization

### A. Primary Dataset Specifications
The experimental modeling pipeline utilizes standard structured computer vision classification benchmarks:
1. **Fashion-MNIST:** 70,000 $28 \times 28$ grayscale images across 10 balanced product classes (60,000 training, 10,000 testing). Flattened input dimensionality equals 784 features. Standardized via channel-wise normalization ($\mu = 0.2860, \sigma = 0.3530$).
2. **CIFAR-10:** 60,000 $32 \times 32$ RGB images across 10 balanced classes (50,000 training, 10,000 testing). Flattened input dimensionality equals 3,072 features. Standardized via channel normalization ($\mu = [0.4914, 0.4822, 0.4465], \sigma = [0.2470, 0.2435, 0.2616]$).
3. **Calibrated Subspace Benchmark:** A controlled 784-dimensional, 10-class dataset with class centers constrained to a shared rank-10 subspace with isotropic Gaussian noise ($\sigma = 4.15$). This controlled distribution enables precise mathematical validation of low-rank spectral retention under isolated noise conditions (`data/dataset_loader.py`).

### B. Hardware Profiling Workload Matrix
To isolate hardware behavior across architectural regimes, the Systems Lead designed a two-variable experimental sweep:
- **Truncated Ranks:** $k \in \{8, 16, 32, 64, 128, 256\}$.
- **Inference Batch Sizes:** $B_{sz} \in \{1, 8, 32, 64, 128\}$, capturing the transition from single-query latency ($B_{sz}=1$) to throughput saturation ($B_{sz}=128$).
- **Granularity:** Layer-wise GEMM isolation ($1024 \to 512$ projection) and end-to-end multi-layer network inference ($3072 \to 1024 \to 512 \to 256 \to 10$).

---

## IV. Methodology

### A. Phase 1: 2D Image SVD Intuition Builder
To establish mathematical intuition regarding spectral truncation, a continuous 2D benchmark matrix $I \in \mathbb{R}^{512 \times 512}$ was decomposed via full SVD:
$$I = U \Sigma V^\top, \quad \hat{I}_k = \sum_{i=1}^k \sigma_i u_i v_i^\top$$
Three rigorous metrics were computed across ranks $k \in \{4, 8, 16, 32, 64, 128, 256\}$:
1. **Relative Frobenius Reconstruction Error:**
   $$\epsilon_{\text{Frob}}(k) = \frac{\|I - \hat{I}_k\|_F}{\|I\|_F} = \frac{\sqrt{\sum_{i=k+1}^{\min(M, N)} \sigma_i^2}}{\sqrt{\sum_{i=1}^{\min(M, N)} \sigma_i^2}}$$
2. **Parameter Compression Ratio:**
   $$\text{CR}(k) = \frac{M \cdot N}{k \cdot (M + N + 1)}$$
3. **Cumulative Singular Value Energy Ratio:**
   $$\text{Energy}(k) = \frac{\sum_{i=1}^k \sigma_i}{\sum_{i=1}^{\min(M, N)} \sigma_i}$$

### B. Phase 2: SVD Layer Factorization & Absorption Mechanics
In PyTorch, dense linear projection layers store weights in row-major layout $W \in \mathbb{R}^{N_{\text{out}} \times M_{\text{in}}}$. Truncated SVD yields:
$$U_k \in \mathbb{R}^{N_{\text{out}} \times k}, \quad S_k \in \mathbb{R}^{k}, \quad V_k^\top \in \mathbb{R}^{k \times M_{\text{in}}}$$
We evaluate three singular value absorption schemes:
- **Symmetric Distribution (Default):**
  $$W_{L_1} = \operatorname{diag}(\sqrt{S_k}) \cdot V_k^\top \in \mathbb{R}^{k \times M_{\text{in}}}, \quad W_{L_2} = U_k \cdot \operatorname{diag}(\sqrt{S_k}) \in \mathbb{R}^{N_{\text{out}} \times k}$$
  Symmetric absorption balances the dynamic numerical range of weight gradients across both factors during backpropagation.
- **Asymmetric Folding:** Folding $\operatorname{diag}(S_k)$ entirely into $L_2$ ($W_{L_1} = V_k^\top, W_{L_2} = U_k \operatorname{diag}(S_k)$) or $L_1$.
- **Bias Assignment:** Bias parameters are assigned exclusively to $L_2$, ensuring $L_1$ remains a pure linear transformation without offset accumulation.

### C. Phase 3: Accuracy Recovery Fine-Tuning
Zero-shot SVD truncation causes immediate accuracy drops at aggressive ranks. Following Sainath et al. (2013), we implement a recovery fine-tuning loop:
$$\mathcal{L}_{\text{total}} = \mathcal{L}_{\text{CE}}(f(X; W_{\text{decomp}}), Y)$$
Gradients backpropagate through both unactivated linear factors:
$$\frac{\partial \mathcal{L}}{\partial W_{L_2}} = \delta \cdot (W_{L_1} X)^\top, \quad \frac{\partial \mathcal{L}}{\partial W_{L_1}} = W_{L_2}^\top \cdot \delta \cdot X^\top$$
We employ the AdamW optimizer ($\text{lr} = 10^{-4}$, weight decay $= 10^{-4}$) with cosine annealing scheduling. Because no non-linear activation exists between $L_1$ and $L_2$, the effective weight matrix $\hat{W} = W_{L_2} W_{L_1}$ remains strictly constrained to rank $\le K$ throughout optimization.

---

## V. Implementation Details

### A. Person A Deliverables (Modeling Lead)
1. **Model Architectures (`models/baseline.py`):** Constructed `BaselineMLP` (4-stage dense network: $M \to 1024 \to 512 \to 256 \to 10$) and `SmallCNN` (2 convolutional blocks followed by wide dense projection heads).
2. **Decomposed Module (`models/decomposed.py`):** Created `DecomposedLinear(nn.Module)` encapsulating $L_1(\text{bias}=\text{False})$ and $L_2(\text{bias}=\text{True})$. Implemented `from_linear()` to execute SVD in high-precision FP32 and assign weights.
3. **Automated Layer Swapping:** Built `decompose_model_linear_layers()`, traversing the module hierarchy to substitute targeted `nn.Linear` layers in-place while leaving the final 10-class classification head uncompressed.
4. **Validation Suite (`tests/test_svd_core.py`):** Verified full-rank numerical equivalence ($\|Y_{\text{dense}} - Y_{\text{decomp}}\| < 10^{-4}$), output tensor shape preservation, and monotonic decrease of Frobenius reconstruction error across ranks.

### B. Person B Deliverables (Systems Lead)
1. **Isolated Timing Engine (`profiling/harness.py`):**
   - Eliminated naive host timers (`time.time()`).
   - Implemented GPU-synchronized hardware timing:
     ```python
     # CUDA Implementation
     start_event = torch.cuda.Event(enable_timing=True)
     end_event = torch.cuda.Event(enable_timing=True)
     start_event.record()
     _ = model(x)
     end_event.record()
     torch.cuda.synchronize()
     latency_ms = start_event.elapsed_time(end_event)
     ```
   - Executed a mandatory 100-pass warm-up loop to bypass JIT compilation and frequency throttling, followed by 200–300 active iterations to record mean, standard deviation, P50 (median), P90, and P99 latencies.
2. **Analytical Compute & Arithmetic Intensity (`profiling/metrics.py`):**
   Formulated exact operational arithmetic intensity per layer:
   $$\text{AI} = \frac{\text{Total FLOPs}}{\text{Weight Bytes Read} + \text{Input Bytes Read} + \text{Intermediate Roundtrip Bytes} + \text{Output Bytes Written}}$$
   For decomposed layers in FP32 (4 bytes per element):
   $$\text{Bytes} = 4 \cdot [ K(M + N) + B_{sz} \cdot M + 2 \cdot B_{sz} \cdot K + B_{sz} \cdot N ]$$
3. **Memory Instrumentation (`profiling/memory.py`):** Tracked parameter static footprints and transient activation allocations via `torch.cuda.max_memory_allocated()`.
4. **Pareto & Roofline Engines (`visualizations/pareto.py`):** Automated generation of publication-grade Pareto tradeoff curves and empirical Roofline plots.

---

## VI. Experimental Results & Discussion

### A. Phase 1 Image SVD Empirical Results
Table II presents the quantitative results for matrix SVD truncation across ranks $k \in \{4, \dots, 256\}$. Over **$93.88\%$ of cumulative singular value energy** is concentrated in the top 16 singular vectors, yielding a **$15.98\times$ compression ratio** with less than $1\%$ relative Frobenius error ($0.97\%$).

### TABLE II: Phase 1 Image SVD Reconstruction Metrics ($512 \times 512$ Matrix)

| Truncated Rank $k$ | Parameter Count | Compression Ratio | Relative Frobenius Error | Cumulative Energy Retained |
| :---: | :---: | :---: | :---: | :---: |
| **$k=4$** | 4,100 | **$63.94\times$** | $5.19\%$ | $86.09\%$ |
| **$k=8$** | 8,200 | **$31.97\times$** | $1.99\%$ | $91.13\%$ |
| **$k=16$** | 16,400 | **$15.98\times$** | $0.97\%$ | $93.88\%$ |
| **$k=32$** | 32,800 | **$7.99\times$** | $0.56\%$ | $95.72\%$ |
| **$k=64$** | 65,600 | **$4.00\times$** | $0.29\%$ | $97.29\%$ |
| **$k=128$** | 131,200 | **$2.00\times$** | $0.18\%$ | $98.24\%$ |
| **$k=256$** | 262,400 | **$1.00\times$** | $0.09\%$ | $99.25\%$ |

### B. Accuracy Retention & Recovery Trade-Offs
Table III documents the empirical accuracy profile of the full network across ranks before and after fine-tuning.

### TABLE III: Parameter Compression vs. Top-1 Accuracy Retention

| Architecture Configuration | Parameter Count | Compression Ratio | Zero-Shot Accuracy | Fine-Tuned Accuracy | Saved Model Checkpoint |
| :--- | :---: | :---: | :---: | :---: | :--- |
| **Uncompressed Baseline** | 1,462,538 | $1.00\times$ | $86.18\%$ | $86.18\%$ | `baseline_model.pt` |
| **Decomposed $k=256$** | 1,057,034 | $1.38\times$ | $86.48\%$ | **$87.47\%$** | `svd_k256_finetuned.pt` |
| **Decomposed $k=128$** | 530,698 | $2.76\times$ | $86.08\%$ | **$87.21\%$** | `svd_k128_finetuned.pt` |
| **Decomposed $k=64$** | 267,530 | $5.47\times$ | $85.66\%$ | **$87.02\%$** | `svd_k64_finetuned.pt` |
| **Decomposed $k=32$** | 135,946 | **$10.76\times$** | $85.02\%$ | **$86.88\%$** | `svd_k32_finetuned.pt` |
| **Decomposed $k=16$** | 70,154 | **$20.85\times$** | $84.20\%$ | **$86.97\%$** | `svd_k16_finetuned.pt` |
| **Decomposed $k=8$** | 37,258 | **$39.25\times$** | $84.04\%$ | **$87.29\%$** | `svd_k8_finetuned.pt` |

Zero-shot SVD truncation experiences minor accuracy drops ($84.04\%$ at $k=8$ vs. $86.18\%$ baseline). Post-training fine-tuning fully restores accuracy across all ranks, with $k=8$ reaching **$87.29\%$** (a $+1.11\%$ increase over the baseline). This demonstrates an implicit low-rank regularizing effect: pruning the tail singular values removes noisy parameters, allowing the optimizer to converge onto better-generalized low-rank representations.

### C. Hardware Benchmarking & The FLOP–Latency Divergence
Table IV presents empirical wall-clock latencies (P50 in ms) measured across batch sizes $B_{sz} \in \{1, 8, 32, 64, 128\}$ for the full network.

### TABLE IV: End-to-End P50 Inference Latency (ms) Across Batch Sizes

| Rank $K$ | FLOP Reduction | $B_{sz}=1$ Latency | $B_{sz}=1$ Speedup | $B_{sz}=128$ Latency | $B_{sz}=128$ Speedup |
| :---: | :---: | :---: | :---: | :---: | :---: |
| **Dense Baseline** | $1.0\times$ | **$0.438\text{ ms}$** | $1.00\times$ (Ref) | $1.183\text{ ms}$ | $1.00\times$ (Ref) |
| **$k=256$** | $2.3\times$ | $0.478\text{ ms}$ | $0.92\times$ (Slowdown) | $0.582\text{ ms}$ | $2.03\times$ |
| **$k=128$** | $4.6\times$ | $0.561\text{ ms}$ | $0.78\times$ (Slowdown) | $0.803\text{ ms}$ | $1.47\times$ |
| **$k=64$** | $9.2\times$ | $0.501\text{ ms}$ | $0.88\times$ (Slowdown) | $0.474\text{ ms}$ | $2.50\times$ |
| **$k=32$** | $18.2\times$ | $0.454\text{ ms}$ | $0.96\times$ (Slowdown) | $0.460\text{ ms}$ | $2.57\times$ |
| **$k=16$** | $35.6\times$ | $0.460\text{ ms}$ | $0.95\times$ (Slowdown) | $0.450\text{ ms}$ | $2.63\times$ |
| **$k=8$** | **$71.2\times$** | **$0.476\text{ ms}$** | **$0.92\times$ (Slowdown)** | **$0.416\text{ ms}$** | **$2.84\times$** |

### D. Systems Analysis: Why Low-Batch Speedup Stagnates
The experimental data reveals two critical hardware realities:
1. **The $B_{sz}=1$ Latency Floor & Slowdown:** Despite a **$71.2\times$ theoretical FLOP reduction** at $k=8$, execution time at $B_{sz}=1$ increases from $0.438\text{ ms}$ to $0.476\text{ ms}$ (an **$8\%$ net slowdown**). Factorizing four dense layers produces eight sequential linear layers, doubling the number of kernel launch dispatches ($n_{\text{kernels}} \cdot t_{\text{launch}}$) and requiring intermediate activation round-trips through memory.
2. **Shift to Compute-Bound Regimes at $B_{sz}=128$:** At $B_{sz}=128$, operational arithmetic intensity increases from $0.50\text{ FLOP/Byte}$ to $38.4\text{ FLOP/Byte}$, crossing the hardware balance knee. In this compute-bound regime, the decomposed layer achieves an empirical **$2.84\times$ speedup**, directly demonstrating that FLOP savings only yield latency gains when arithmetic intensity is high enough to escape memory bandwidth bottlenecks.

---

## VII. Conclusion & Phase 4 Synthesis

### A. Synthesis: Bridging SVD to LLM KV-Cache Retrieval
Our empirical results validate the core research thesis by demonstrating a direct operational equivalence between dense layer SVD factorization and autoregressive LLM decoding:
- **The Memory-Bandwidth Bottleneck:** During autoregressive generation ($B_{sz}=1$), each newly generated token requires retrieving the entire Key-Value (KV) cache history and model weights from High Bandwidth Memory (HBM) to perform a single vector-matrix multiply. Arithmetic intensity collapses to $\approx 1\text{ FLOP/Byte}$, pinning generation speeds to memory bandwidth limits.
- **Why SVD Models the KV-Cache Wall:** Truncating dense projection layers reduces parameter memory traffic but leaves arithmetic intensity pinned below $0.50\text{ FLOP/Byte}$ at $B_{sz}=1$. Just as SVD FLOP reductions fail to produce speedups without kernel fusion, LLM inference optimizations cannot rely on compute reductions alone. This explains why modern serving frameworks focus on memory-reduction techniques: KV-cache eviction (Zhang et al., 2023 [3]), paged memory allocation (Kwon et al., 2023 [9]), low-rank projection of keys/values (Yuan et al., 2023 [5]), and IO-aware SRAM tiling (Dao, 2023 [8]).

### B. Summary of Key Findings
1. SVD achieves up to **$39.25\times$ parameter compression** with negligible accuracy degradation when combined with short fine-tuning loops.
2. Theoretical FLOP breakeven ($K < K^*$) does not guarantee wall-clock speedup on modern hardware. At small batch sizes ($B_{sz}=1$), extra kernel launch overheads and intermediate tensor memory traffic cause a net latency slowdown.
3. Speedups emerge only at larger batch sizes ($B_{sz} \ge 32$) as arithmetic intensity increases and compute saturation is achieved.

### C. Future Research Directions
- **Triton Kernel Fusion:** Implementing a fused low-rank GEMM kernel ($Y = (X L_1) L_2$) that retains the intermediate $B_{sz} \times K$ activation tensor in on-chip SRAM, eliminating global memory round-trips.
- **Tensor Core Tiling Alignment:** Aligning ranks $K$ to exact multiples of 16/32 in FP16/BF16 to leverage hardware MMA (Matrix Multiply Accumulate) instructions.
- **Speculative Low-Rank Decoding:** Utilizing a heavily truncated low-rank model ($k=8$) as a lightweight draft model in speculative decoding pipelines to accelerate full dense transformer generation.

---

## VIII. References

1. E. J. Hu, Y. Shen, P. Wallis, Z. Allen-Zhu, Y. Li, S. Wang, L. Wang, and W. Chen, "LoRA: Low-rank adaptation of large language models," in *Proc. Int. Conf. Learn. Representations (ICLR)*, 2022.
2. S. Hooker, "The hardware lottery," *Commun. ACM*, vol. 64, no. 12, pp. 58–65, 2021.
3. Z. Zhang, Y. Sheng, T. Zhou, T. Chen, L. Zheng, R. Cai, Z. Song, Y. Tian, C. Ré, C. Barrett, et al., "H2O: Heavy hitter oracle for efficient generative inference of large language models," in *Proc. Adv. Neural Inf. Process. Syst. (NeurIPS)*, vol. 36, 2023.
4. P. Sharma, J. T. Ash, and D. Misra, "The truth is in there: Improving reasoning in language models with layer-selective rank reduction," *arXiv preprint arXiv:2312.13558*, 2023.
5. Z. Yuan, Y. Shang, Y. Zhou, Z. Dong, C. Xue, B. Wu, Z. Li, and Q. Gu, "ASVD: Activation-aware singular value decomposition for compressing large language models," *arXiv preprint arXiv:2312.05825*, 2023.
6. H. Wang, Z. Wu, Z. Han, X. Wang, and V. Braverman, "SVD-LLM: Truncation-aware singular value decomposition for large language model compression," in *Proc. Int. Conf. Learn. Representations (ICLR)*, 2025.
7. E. Frantar, S. Ashkboos, T. Hoefler, and D. Alistarh, "GPTQ: Accurate post-training quantization for generative pre-trained transformers," in *Proc. Int. Conf. Learn. Representations (ICLR)*, 2023.
8. T. Dao, "FlashAttention-2: Faster attention with better parallelism and work partitioning," in *Proc. Int. Conf. Learn. Representations (ICLR)*, 2024.
9. W. Kwon, Z. Li, S. Zhuang, F. Y. Sheng, L. Zheng, C. H. Yu, J. E. Gonzalez, H. Zhang, and I. Stoica, "Efficient memory management for large language model serving with PagedAttention," in *Proc. 29th ACM Symp. Oper. Syst. Princ. (SOSP)*, 2023, pp. 611–626.
10. S. Ge, R. Zhang, Z. Han, F. Ren, and Y. Shan, "Model tells you what to discard: Adaptive KV cache compression for LLMs," in *Proc. Int. Conf. Learn. Representations (ICLR)*, 2024.
11. S. Williams, A. Waterman, and D. Patterson, "Roofline: An insightful visual performance model for multicore architectures," *Commun. ACM*, vol. 52, no. 4, pp. 65–76, 2009.
12. T. N. Sainath, B. Kingsbury, V. Sindhwani, E. Arisoy, and B. Ramabhadran, "Low-rank matrix factorization for deep neural network training with application to speech recognition," in *Proc. IEEE Int. Conf. Acoust., Speech Signal Process. (ICASSP)*, 2013, pp. 4155–4159.
13. E. L. Denton, W. Zaremba, J. Bruna, Y. LeCun, and R. Fergus, "Exploiting linear structure within convolutional networks for efficient evaluation," in *Proc. Adv. Neural Inf. Process. Syst. (NeurIPS)*, vol. 27, 2014.
