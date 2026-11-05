# Student Guide & Defense Playbook: Low-Rank SVD Neural Compression

**Project:** Low-Rank SVD Factorization of Dense Neural Network Layers & Hardware Profiling  
**Team:**  
- **Person A (Modeling Lead):** Mathematical Factorization, PyTorch Layers, SVD Layer-Swapping, Fine-Tuning Accuracy Recovery  
- **Person B (Systems Lead — Riddhi Jain):** Hardware Benchmarking, CUDA/MPS Timing Integrity, FLOP & Roofline Math, Pareto Analysis  

---

## 1. The Entire Project Explained in Plain English (5-Minute Conceptual Masterclass)

### What is the basic idea?
In a standard deep neural network, dense layers do matrix multiplication:
$$\text{Output} = X \cdot W^\top + b$$
If $W$ is a big matrix of size $1024 \times 512$, it has $524,288$ numbers (parameters).
Every time an input passes through, the computer must multiply and add all these numbers.

### How does SVD compress it?
Linear algebra tells us that any matrix $W$ can be broken down using **Singular Value Decomposition (SVD)** into three matrices:
$$W = U \cdot \Sigma \cdot V^\top$$
- $U$ and $V^\top$ represent directional patterns (singular vectors).
- $\Sigma$ (Sigma) is a diagonal list of numbers (singular values, $\sigma_1 \ge \sigma_2 \ge \dots$) telling you **how important** each direction is.

In real-world networks, most information is concentrated in the top few singular values! The rest is mostly high-frequency noise or redundant redundancy.  
If we keep only the top $K$ singular values (e.g., $K = 8$ or $16$ instead of $512$), we throw away the tail.
We then fold the matrix into **two small, back-to-back linear layers** with no activation function between them:
- $L_1$: maps $1024 \to K$ (size $1024 \times K$)
- $L_2$: maps $K \to 512$ (size $K \times 512$)

For $K=8$:
- **Original parameters:** $1024 \times 512 = 524,288$
- **Decomposed parameters:** $(1024 \times 8) + (8 \times 512) = 8,192 + 4,096 = 12,288$  
That is a **$41.8\times$ reduction in parameters and theoretical math operations (FLOPs)!**

### The Big Catch (The Core Capstone Thesis):
If you reduce calculations by $41\times$, does the model run $41\times$ faster on a GPU?  
**NO. In fact, at Batch Size = 1, it actually runs slightly SLOWER.**

Why? Because modern GPUs are not bottlenecked by raw math at small batch sizes—they are bottlenecked by **memory bandwidth** and **overhead**:
1. **Kernel Launch Overhead:** Instead of telling the GPU to launch 1 matrix multiplication kernel, you now tell it to launch 2 kernels in a row.
2. **Intermediate Tensor Traffic:** $L_1$ produces an intermediate tensor of size $(B_{sz} \times K)$ that has to be written to GPU memory and then read back into the chip for $L_2$.
3. **Low Arithmetic Intensity:** At Batch Size = 1, the GPU spends 95% of its time fetching weights from memory and only 5% actually doing math.

Only when batch sizes are large ($B_{sz} = 128$) does the GPU have enough parallel math to hide the memory latency and finally show a real speedup ($2.84\times$).

### How does this connect to Large Language Models (LLMs)?
When ChatGPT or an LLM generates text, it generates **one token at a time (Batch Size = 1)**.  
To generate just one word, the GPU has to pull gigabytes of Key-Value (KV) cache and weights from memory just to do a tiny vector multiplication. It is completely memory-bandwidth bound.  
Our SVD experiment on dense layers proves the exact same hardware memory-wall phenomenon!

---

## 2. Who Did What: Clear Division of Roles

### Person A: Modeling Lead
**Primary Focus:** Mathematics, PyTorch Architecture, Weight Factorization, Accuracy Recovery.

* **What Person A built & owns:**
  1. `models/baseline.py`: Defined the dense neural network (`BaselineMLP`: 4 layers, $784 \to 1024 \to 512 \to 256 \to 10$) and `SmallCNN`.
  2. `models/decomposed.py`: Created `DecomposedLinear(nn.Module)`. Built the SVD splitting algorithm that takes an existing layer, runs SVD, truncates to rank $K$, and splits singular values symmetrically:
     $$W_{L_1} = \sqrt{\Sigma_K} \cdot V_K^\top, \quad W_{L_2} = U_K \cdot \sqrt{\Sigma_K}$$
  3. Layer-swapping logic (`decompose_model_linear_layers`): An automatic PyTorch tree-traversal script that finds all dense projection layers in the trained model and replaces them with `DecomposedLinear` modules while preserving the final 10-class classifier.
  4. Accuracy recovery fine-tuning (`scripts/train_finetune.py`): Built the training loop that trains the baseline, records zero-shot accuracy drops across ranks ($k \in \{8, 16, 32, 64, 128, 256\}$), and fine-tunes the factorized layers using AdamW with Cosine Annealing to recover lost accuracy.

* **Key numbers Person A must know:**
  - Baseline model parameter count: **1,462,538 parameters** (~1.46M).
  - Baseline accuracy: **86.18%**.
  - Rank $k=8$ compressed parameter count: **37,258 parameters** (**$39.25\times$ smaller!**).
  - Fine-tuned accuracy recovered: **87.29%** (actually beats baseline by +1.11% because low-rank SVD cuts out noisy tail singular values, acting as an implicit regularizer!).

---

### Person B: Systems Lead (Riddhi Jain)
**Primary Focus:** Hardware Performance, CUDA/MPS Benchmarking, FLOP Counting, Memory Tracking, Roofline & Pareto Plots.

* **What Person B built & owns:**
  1. `profiling/harness.py`: The isolated, jitter-free timing harness. Bans naive Python `time.time()` because GPU execution is asynchronous! Uses hardware-synchronized timing (`torch.cuda.Event` with stream synchronization `torch.cuda.synchronize()`, and high-resolution synchronized fallback for MPS/CPU). Implemented a **100-iteration warmup** to stabilize GPU clocks followed by **200–300 active iterations** to compute percentiles: **Mean, Std Dev, P50 (median), P90, P99**.
  2. `profiling/metrics.py`: The analytical compute module.
     - Theoretical Dense FLOPs: $2 \cdot B_{sz} \cdot M \cdot N$
     - Theoretical Decomposed FLOPs: $2 \cdot B_{sz} \cdot K \cdot (M + N)$
     - Breakeven Rank formula: $K^* = \frac{M \cdot N}{M + N}$ ($K^* = 341.3$ for $1024 \to 512$).
     - Operational Arithmetic Intensity (FLOPs/Byte transferred).
  3. `profiling/memory.py`: Tracks physical parameter storage footprint (MB) and peak runtime activation VRAM allocation.
  4. `scripts/run_sweep.py`: Built the multi-variable experimental matrix sweeping ranks $k \in \{8, 16, 32, 64, 128, 256\}$ across batch sizes $B_{sz} \in \{1, 8, 32, 64, 128\}$ across 70 configurations, outputting `results/benchmark_data.csv`.
  5. `visualizations/pareto.py`: Plotted the 3 publication figures:
     - Plot 1: Pareto frontier (Compression Ratio vs Top-1 Accuracy).
     - Plot 2: Theoretical FLOPs vs Measured Wall-Clock Latency (proving latency stagnation at $B=1$).
     - Plot 3: Empirical Roofline model (Memory-bound vs Compute-bound regimes).

* **Key numbers Person B must know:**
  - Full model FLOP reduction at $k=8$: **$71.2\times$ fewer FLOPs**.
  - Measured latency at $B_{sz}=1$: Baseline is **$0.438\text{ ms}$**, decomposed $k=8$ is **$0.476\text{ ms}$** (**8% slower!**).
  - Measured latency at $B_{sz}=128$: Baseline is **$1.183\text{ ms}$**, decomposed $k=8$ is **$0.416\text{ ms}$** (**$2.84\times$ faster!**).
  - Arithmetic intensity at $B=1$: **$< 0.50\text{ FLOP/Byte}$** (severely memory-bandwidth bound under the Roofline).

---

### Joint Work (Pair Programming)
- **Phase 1 Image SVD (`scripts/image_svd.py`):** Both teammates worked together to build the 2D image matrix SVD intuition on a $512 \times 512$ image across ranks $k \in \{4, 8, 16, 32, 64, 128, 256\}$. Proved that $k=16$ preserves **$93.88\%$ of cumulative singular value energy** with **$<1\%$ relative Frobenius error** at **$15.98\times$ compression**.
- **Phase 4 Final Synthesis:** Jointly authored the synthesis connecting low-batch SVD latency stagnation to the memory-bandwidth wall of token-by-token LLM KV-cache retrieval.

---

## 3. How to Present (Slide-by-Slide Speaker Allocation)

When standing in front of the professors, do not read bullet points. Talk smoothly, look confident, and pass the baton cleanly between each other.

| Slide # | Slide Title | Speaker | What to Say & Transition |
| :---: | :--- | :---: | :--- |
| **1** | Title Slide | **Person A** | *"Good morning professors. Today we present our capstone research on Low-Rank SVD Neural Compression & Hardware Profiling. I am Person A, the Modeling Lead, and my co-researcher Riddhi Jain is the Systems Lead."* |
| **2** | Abstract & Core Thesis | **Person B (Riddhi)** | *"Our central research question is: When theoretical FLOPs drop by up to 71×, why doesn't wall-clock inference speed up on real GPUs? We show that at small batch sizes, memory bandwidth and kernel overhead create a hard latency floor, modeling the exact memory wall seen in LLM text generation."* |
| **3–4** | Mathematical Formulation & The Core Problem | **Person A** | Walk through SVD decomposition $W = U \Sigma V^\top \approx L_2 \cdot L_1$. Explain the Breakeven Rank formula $K^* = \frac{MN}{M+N}$. Explain why ranks below $K^*$ have fewer FLOPs. |
| **5** | Four-Phase Roadmap | **Person A** | Briefly explain: Phase 1 (Image intuition) $\to$ Phase 2 (Layer factorization) $\to$ Phase 3 (Tradeoff engine) $\to$ Phase 4 (Synthesis). Hand over: *"Now Riddhi will walk through the literature survey."* |
| **6–8** | Literature Survey (10 Papers) | **Person B (Riddhi)** | Highlight 3 key papers from Table I: **LoRA** (low-rank structure), **The Hardware Lottery** (Hooker — hardware favors large dense matrices over thin unaligned ones), and **H2O** (Zhang — memory-bound KV-cache). |
| **9** | Dataset & Workload Characterization | **Person A** | Explain the dataset structure (784 features, 10 classes, balanced splits) and batch sizes evaluated ($B \in \{1, 8, 32, 64, 128\}$). |
| **10** | Methodology (Phases 1, 2, 3) | **Person A** | Explain the symmetric singular value split ($\sqrt{\Sigma}$ on both sides), why bias stays on $L_2$, and how AdamW fine-tuning works. |
| **11** | Implementation: Modeling Lead (Person A) | **Person A** | Show the `DecomposedLinear` PyTorch implementation, test cases (all 5 tests passing), and the accuracy recovery table (86.18% $\to$ 87.29% at $k=8$). |
| **12** | Implementation: Systems Lead (Person B) | **Person B (Riddhi)** | Explain the CUDA Event timing harness, why `time.time()` is banned, 100-pass warmup, stream synchronization, and Roofline arithmetic intensity math. |
| **13** | Results: The FLOP–Latency Divergence | **Person B (Riddhi)** | **THE CLIMAX SLIDE:** Point at Table IV. Show that at $B=1$, rank 8 has 71× fewer FLOPs but is 8% slower. Show that speedup only happens at $B=128$ ($2.84\times$). |
| **14** | Pareto Frontier & Empirical Roofline | **Person B (Riddhi)** | Explain the 3 plots: Plot 1 shows the knee at $k=32..64$; Plot 2 shows latency stagnation at $B=1$; Plot 3 shows points stuck under the memory bandwidth ceiling. |
| **15** | Conclusion & Synthesis (The Headliner) | **Both** | **Person A:** Summarizes that SVD compresses weights by up to 39× with zero accuracy loss after fine-tuning.  <br>**Person B:** Delivers the punchline: SVD at $B=1$ simulates the LLM token generation bottleneck. Both are trapped by memory bytes, not compute FLOPs! |
| **16** | Future Work & Q&A | **Both** | Mention Triton fused kernels, FP16 Tensor Core alignment, and invite questions: *"Thank you. We are now open for questions."* |

---

## 4. How to Show the Live Demo to the Professor

If the professor asks: *"Can you run your system or show me the code working?"*, here is your exact 2-minute demonstration plan.

Open your terminal in `/Users/riddhijain/Desktop/Image_compression_SVD`.

### Step 1: Prove Mathematical Correctness (5 seconds)
Run the automated test suite:
```bash
PYTHONPATH=. .venv/bin/python -m unittest discover tests
```
**What to say to the professor:**  
*"First, we verify our mathematical unit tests. In 0.5 seconds, all 5 unit tests pass: verifying full-rank numerical equivalence where the decomposed layer matches the dense layer to within $10^{-4}$ precision, verifying output tensor shapes, and proving monotonic error reduction as rank increases."*

### Step 2: Show Phase 1 Image SVD (15 seconds)
Run the image decomposition script:
```bash
PYTHONPATH=. .venv/bin/python scripts/image_svd.py
```
Open the generated plot:
`results/plots/svd_image_reconstruction_grid.png`  
**What to say to the professor:**  
*"Here is Phase 1. We decomposed a $512 \times 512$ matrix. Notice that with rank $k=16$, we achieve a $15.98\times$ compression ratio while retaining $93.88\%$ of total singular value energy and less than $1\%$ Frobenius reconstruction error. This proves visually that low-frequency semantic structures are preserved by top singular vectors."*

### Step 3: Show the Saved Model Weights & Checkpoints (15 seconds)
List the saved checkpoints in `results/checkpoints/`:
```bash
ls -lh results/checkpoints/
```
Point out:
- `baseline_model.pt`: **5.9 MB** (uncompressed)
- `svd_k8_finetuned.pt`: **154 KB** (**$38\times$ smaller on disk!**)

### Step 4: Show the FLOP vs Latency Divergence Live (30 seconds)
Run the hardware sweep:
```bash
PYTHONPATH=. .venv/bin/python scripts/run_sweep.py
```
Open `results/plots/theoretical_flops_vs_measured_latency.png`.  
**What to say to the professor (Person B):**  
*"Professors, look at this plot. The dashed black line is theoretical speedup based on FLOPs—it shows that rank 8 should be over $20\times$ faster. But the red line is our empirical wall-clock speedup measured with our synchronized hardware harness at Batch Size = 1. It sits below 1.0x—it is actually a slight slowdown! This exposes the memory bandwidth wall and kernel dispatch overhead."*

### Step 5: Launch the Interactive Results Dashboard (15 seconds)
Run the unified results script:
```bash
PYTHONPATH=. .venv/bin/python scripts/show_results.py --open
```
Opens `results/dashboard.html` in the browser with live KPI cards, interactive tabs for all 5 publication plots, accuracy recovery tables, hardware sweep matrices, and viva defense cheat-sheets.

---

## 5. Professor Viva / Q&A Defense Guide: The Tough Questions & Perfect Answers

Below are the exact tough questions professors ask during project evaluations, along with the precise technical answers you should give.

---

### Question 1: "Why did you use SVD instead of simple post-training quantization like INT8 or INT4?"
**Answer (Person A / Person B):**  
*"Quantization and low-rank SVD address orthogonal dimensions of compression:
1. **Quantization (e.g., INT8/FP4)** reduces the bit-width per parameter (from 32 bits to 8 or 4 bits) while leaving the matrix dimensions $M \times N$ intact. It cuts memory traffic by $4\times$, but still executes the exact same number of matrix multiplications.
2. **Low-rank SVD** changes the fundamental computational complexity from $\mathcal{O}(MN)$ to $\mathcal{O}(K(M+N))$. At larger batch sizes ($B=128$), SVD delivers genuine algorithmic arithmetic speedups ($2.84\times$) that quantization alone cannot achieve without specialized hardware.
3. Furthermore, modern state-of-the-art compression frameworks (like ASVD and Palu in 2024) combine both: they perform SVD factorization first to reduce parameter rank, and then quantize the resulting $L_1$ and $L_2$ factors to 4 bits!"*

---

### Question 2: "In your decomposition $W = L_2 \cdot L_1$, where did you put the singular values $\Sigma$? Why not keep $\Sigma$ separate as a third diagonal layer?"
**Answer (Person A - Modeling Lead):**  
*"If we kept $\Sigma$ as a third layer, the forward pass would be $X \cdot V \cdot \Sigma \cdot U^\top$. That would require three matrix multiplications and three separate kernel dispatches, adding even more kernel launch latency!  
Instead, we fold $\Sigma$ into the weight matrices. We chose the **symmetric absorption scheme**:
$$W_{L_1} = \sqrt{\Sigma_K} \cdot V_K^\top, \quad W_{L_2} = U_K \cdot \sqrt{\Sigma_K}$$
By taking the square root of the singular values and distributing them equally to both $L_1$ and $L_2$, we balance the numerical magnitude and condition numbers of both layers. This ensures that during recovery fine-tuning, the gradients flowing through $L_1$ and $L_2$ have similar scales, preventing vanishing or exploding gradients in either factor."*

---

### Question 3: "Why did the latency get worse at Batch Size = 1 despite 71x fewer FLOPs? Explain the exact hardware reason."
**Answer (Person B - Systems Lead):**  
*"At Batch Size = 1, execution is governed by two hardware constraints:
1. **Low Arithmetic Intensity under the Roofline Model:** Arithmetic intensity is calculated as $\frac{\text{Total FLOPs}}{\text{Total Bytes Transferred}}$. At $B=1$, arithmetic intensity is less than $0.50\text{ FLOP/Byte}$. That places the layer deep in the memory-bandwidth-bound slope of the Roofline model. The GPU's compute cores spend most of their time idle, waiting for bytes to travel across the memory bus. Cutting FLOPs does not reduce execution time when compute cores are already idle.
2. **Kernel Launch and Memory Round-Trip Overhead:** A dense layer executes 1 GEMM kernel. The decomposed layer executes 2 GEMM kernels sequentially. Each kernel incurs CPU-to-GPU dispatch latency ($t_{\text{launch}} \approx 5\text{–}10\ \mu\text{s}$). In addition, because PyTorch runs un-fused kernels, the intermediate activation tensor of size $(B_{sz} \times K)$ must be written out to VRAM by $L_1$ and then read back into on-chip cache by $L_2$. The cost of this intermediate memory round-trip and extra kernel launch exceeds the tiny mathematical savings at $B=1$."*

---

### Question 4: "What is the Breakeven Rank $K^*$, and how did you derive it mathematically?"
**Answer (Person A or Person B):**  
*"The Breakeven Rank $K^*$ is the rank at which the decomposed linear layers have exactly the same number of theoretical FLOPs as the original dense layer.
- Dense layer FLOPs: $2 \cdot B_{sz} \cdot M \cdot N$
- Decomposed layer FLOPs: $2 \cdot B_{sz} \cdot K \cdot M + 2 \cdot B_{sz} \cdot K \cdot N = 2 \cdot B_{sz} \cdot K \cdot (M + N)$

Setting them equal to each other:
$$2 \cdot B_{sz} \cdot K^* \cdot (M + N) = 2 \cdot B_{sz} \cdot M \cdot N$$
Canceling $2 \cdot B_{sz}$ from both sides:
$$K^* \cdot (M + N) = M \cdot N \implies K^* = \frac{M \cdot N}{M + N}$$
For our $1024 \to 512$ projection layer:
$$K^* = \frac{1024 \times 512}{1024 + 512} = \frac{524,288}{1,536} \approx 341.33$$
If we pick any rank $K < 341$, theoretical FLOPs are reduced. If we pick $K > 341$, the decomposed model would theoretically require *more* calculations than the original dense layer."*

---

### Question 5: "Why did fine-tuned accuracy at $k=8$ (87.29%) exceed the original baseline accuracy (86.18%)? Did SVD create new information?"
**Answer (Person A - Modeling Lead):**  
*"No, SVD cannot create new information. However, SVD acts as an **implicit low-rank regularizer**.  
In high-dimensional dense layers ($1024 \times 512$), many parameters capture random noise or overfit to subtle variance in the training distribution. The tail singular values represent tiny, noisy directions in weight space.  
By truncating the matrix to rank $K=8$, we explicitly filter out the noisy tail of the spectrum and restrict the network's hypothesis space to the dominant low-rank subspace. When we fine-tune with AdamW and cosine annealing, the optimizer only tunes parameters within this clean subspace, preventing overfitting and resulting in slightly better test-set generalization."*

---

### Question 6: "Why did you ban `time.time()` in your benchmarking harness? What does `torch.cuda.synchronize()` do?"
**Answer (Person B - Systems Lead):**  
*"Python's `time.time()` or `time.perf_counter()` measures CPU host time. But CUDA calls in PyTorch are **asynchronous non-blocking calls**!  
When you call `model(x)` in PyTorch, the CPU queues the kernel instruction into the CUDA stream and returns almost immediately, while the GPU is still executing the kernel in the background. If you use `time.time()`, you are measuring how fast the CPU queued the command, not how long the GPU took to compute it!  
To fix this, we:
1. Use `torch.cuda.Event(enable_timing=True)`, which places high-precision timestamps directly onto the GPU hardware execution stream.
2. Call `torch.cuda.synchronize()`, which halts the CPU until the GPU finishes all queued operations before recording the elapsed time.
3. Run a **100-iteration warmup** before taking measurements to ensure GPU clocks have ramped up to maximum boost frequency and driver caches are pre-warmed."*

---

### Question 7: "Explain your Phase 4 Synthesis: How does compressing a small MLP relate to LLM KV-Cache retrieval?"
**Answer (Person B - Systems Lead):**  
*"In Large Language Models, text generation happens token-by-token (autoregressive decoding).  
At each generation step, the batch size for that single user is $B_{sz} = 1$. To predict the next single token, the transformer must load the entire multi-gigabyte Key-Value (KV) cache of past context and all model weights from High Bandwidth Memory (HBM) into on-chip SRAM just to compute a single vector-matrix dot product.  
Its arithmetic intensity drops to $\approx 1\text{ FLOP/Byte}$—the exact same memory-bandwidth ceiling we observed at $B_{sz}=1$ in our SVD experiments!  
This proves why LLM serving cannot be accelerated by reducing FLOPs alone. It explains why all recent landmark LLM systems papers (like H2O [Zhang 2023], FlashAttention-2 [Dao 2023], and PagedAttention [Kwon 2023]) focus exclusively on reducing memory traffic: evicting KV cache tokens, paging memory, and fusing kernels to stop intermediate memory round-trips."*

---

### Question 8: "What are the limitations of your current setup, and what would you do next?"
**Answer (Both):**  
*"We have two key next steps planned:
1. **Fused Triton Kernel:** Right now, PyTorch executes $L_1$ and $L_2$ as two separate GEMM kernels. We plan to write a custom fused GPU kernel in Triton or CUDA that computes $Y = (X \cdot L_1) \cdot L_2$ entirely within the GPU's on-chip SRAM tiles, eliminating the intermediate global memory round-trip and cutting kernel launch overhead in half.
2. **Tensor Core Tile Alignment:** Modern NVIDIA Tensor Cores require matrix dimensions to be multiples of 16 or 32 in FP16/BF16. In future sweeps, we will constrain truncated ranks $K$ to exact multiples of 16 ($k \in \{16, 32, 64, 128\}$) to evaluate hardware-native MMA (Matrix Multiply Accumulate) instructions."*

---

## 6. Quick Cheat-Sheet Table to Keep in Your Pocket

| Concept | Key Formula / Metric | Value in Project |
| :--- | :--- | :--- |
| **Breakeven Rank** | $K^* = \frac{M \cdot N}{M + N}$ | $341.33$ for $1024 \to 512$ layer |
| **Symmetric Split** | $L_1 = \sqrt{\Sigma} V^\top, \quad L_2 = U \sqrt{\Sigma}$ | Eliminates gradient imbalance |
| **Baseline Accuracy** | Top-1 Test Accuracy | **86.18%** (1.46M parameters) |
| **Rank $k=8$ Compression** | Baseline Params / Decomposed Params | **$39.25\times$ smaller** (37.2K parameters) |
| **Rank $k=8$ Accuracy** | Post-fine-tuning recovery | **87.29%** (+1.11% recovery) |
| **$B=1$ Latency** | Baseline vs $k=8$ (MPS/CUDA) | $0.438\text{ ms} \to 0.476\text{ ms}$ (**8% slower**) |
| **$B=128$ Latency** | Baseline vs $k=8$ (MPS/CUDA) | $1.183\text{ ms} \to 0.416\text{ ms}$ (**$2.84\times$ faster**) |
| **Arithmetic Intensity** | $\text{Total FLOPs} / \text{Total Bytes Transferred}$ | $<0.50\text{ FLOP/B}$ at $B=1$; $38.4\text{ FLOP/B}$ at $B=128$ |
| **Timing Mechanism** | `torch.cuda.Event(enable_timing=True)` | Jitter-free GPU hardware timestamps |
