// Real empirical benchmark data and metadata directly collected from hardware profiling sweeps
export const ACCURACY_DATA = [
  { rank: "Baseline", isDecomposed: false, compressionRatio: 1.0, params: 1462538, zeroShotAcc: 86.18, finetunedAcc: 86.18, delta: "+0.00%" },
  { rank: 8, isDecomposed: true, compressionRatio: 39.25, params: 37258, zeroShotAcc: 84.04, finetunedAcc: 87.29, delta: "+1.11%" },
  { rank: 16, isDecomposed: true, compressionRatio: 20.85, params: 70154, zeroShotAcc: 84.20, finetunedAcc: 86.97, delta: "+0.79%" },
  { rank: 32, isDecomposed: true, compressionRatio: 10.76, params: 135946, zeroShotAcc: 85.02, finetunedAcc: 86.88, delta: "+0.70%" },
  { rank: 64, isDecomposed: true, compressionRatio: 5.47, params: 267530, zeroShotAcc: 85.66, finetunedAcc: 87.02, delta: "+0.84%" },
  { rank: 128, isDecomposed: true, compressionRatio: 2.76, params: 530698, zeroShotAcc: 86.08, finetunedAcc: 87.21, delta: "+1.03%" },
  { rank: 256, isDecomposed: true, compressionRatio: 1.38, params: 1057034, zeroShotAcc: 86.48, finetunedAcc: 87.47, delta: "+1.29%" }
];

export const END_TO_END_LATENCY = [
  {
    batchSize: 1,
    baselineMs: 0.444,
    k8: { ms: 0.498, speedup: 0.89, status: "slowdown" },
    k16: { ms: 0.501, speedup: 0.89, status: "slowdown" },
    k32: { ms: 0.500, speedup: 0.89, status: "slowdown" },
    k64: { ms: 0.528, speedup: 0.84, status: "slowdown" },
    k128: { ms: 0.517, speedup: 0.86, status: "slowdown" },
    k256: { ms: 0.530, speedup: 0.84, status: "slowdown" }
  },
  {
    batchSize: 8,
    baselineMs: 0.452,
    k8: { ms: 0.421, speedup: 1.07, status: "modest" },
    k16: { ms: 0.404, speedup: 1.12, status: "modest" },
    k32: { ms: 0.406, speedup: 1.11, status: "modest" },
    k64: { ms: 0.409, speedup: 1.11, status: "modest" },
    k128: { ms: 0.417, speedup: 1.08, status: "modest" },
    k256: { ms: 0.435, speedup: 1.04, status: "modest" }
  },
  {
    batchSize: 32,
    baselineMs: 0.568,
    k8: { ms: 0.407, speedup: 1.39, status: "speedup" },
    k16: { ms: 0.501, speedup: 1.13, status: "modest" },
    k32: { ms: 0.438, speedup: 1.30, status: "speedup" },
    k64: { ms: 0.441, speedup: 1.29, status: "speedup" },
    k128: { ms: 0.449, speedup: 1.26, status: "speedup" },
    k256: { ms: 0.465, speedup: 1.22, status: "speedup" }
  },
  {
    batchSize: 64,
    baselineMs: 0.735,
    k8: { ms: 0.410, speedup: 1.79, status: "speedup" },
    k16: { ms: 0.439, speedup: 1.67, status: "speedup" },
    k32: { ms: 0.441, speedup: 1.67, status: "speedup" },
    k64: { ms: 0.457, speedup: 1.61, status: "speedup" },
    k128: { ms: 0.538, speedup: 1.37, status: "speedup" },
    k256: { ms: 0.511, speedup: 1.44, status: "speedup" }
  },
  {
    batchSize: 128,
    baselineMs: 1.184,
    k8: { ms: 0.421, speedup: 2.81, status: "major_speedup" },
    k16: { ms: 0.457, speedup: 2.59, status: "major_speedup" },
    k32: { ms: 0.466, speedup: 2.54, status: "major_speedup" },
    k64: { ms: 0.480, speedup: 2.47, status: "major_speedup" },
    k128: { ms: 0.784, speedup: 1.51, status: "speedup" },
    k256: { ms: 0.586, speedup: 2.02, status: "major_speedup" }
  }
];

export const IMAGE_SVD_METRICS = [
  { rank: 4, compression: 63.94, frobError: 5.19, energyRetained: 86.09 },
  { rank: 8, compression: 31.97, frobError: 1.99, energyRetained: 91.13 },
  { rank: 16, compression: 15.98, frobError: 0.97, energyRetained: 93.88 },
  { rank: 32, compression: 7.99, frobError: 0.56, energyRetained: 95.72 },
  { rank: 64, compression: 4.00, frobError: 0.29, energyRetained: 97.29 },
  { rank: 128, compression: 2.00, frobError: 0.18, energyRetained: 98.24 },
  { rank: 256, compression: 1.00, frobError: 0.09, energyRetained: 99.25 }
];

export const CHECKPOINTS_INFO = [
  { name: "baseline_model.pt", size: "5.58 MB", type: "Uncompressed Dense Baseline (4-layer MLP)", rank: "Full", compression: "1.0x" },
  { name: "svd_k256_finetuned.pt", size: "4.04 MB", type: "Low-Rank SVD (Cosine Annealing AdamW)", rank: "k=256", compression: "1.38x" },
  { name: "svd_k128_finetuned.pt", size: "2.03 MB", type: "Low-Rank SVD (Cosine Annealing AdamW)", rank: "k=128", compression: "2.76x" },
  { name: "svd_k64_finetuned.pt", size: "1.03 MB", type: "Low-Rank SVD (Cosine Annealing AdamW)", rank: "k=64", compression: "5.47x" },
  { name: "svd_k32_finetuned.pt", size: "535.6 KB", type: "Low-Rank SVD (Cosine Annealing AdamW)", rank: "k=32", compression: "10.76x" },
  { name: "svd_k16_finetuned.pt", size: "278.6 KB", type: "Low-Rank SVD (Cosine Annealing AdamW)", rank: "k=16", compression: "20.85x" },
  { name: "svd_k8_finetuned.pt", size: "150.1 KB", type: "Low-Rank SVD (Cosine Annealing AdamW)", rank: "k=8", compression: "39.25x" }
];

export const VIVA_QUESTIONS = [
  {
    q: "Why did you choose SVD factorization instead of post-training quantization like INT8 or INT4?",
    speaker: "Person A & Person B (Joint)",
    answer: "Quantization and low-rank SVD address orthogonal compression axes. Quantization reduces bit-width (32-bit to 8-bit or 4-bit) while leaving the matrix dimensions intact; it reduces memory traffic but requires the same arithmetic MAC count. Low-rank SVD reduces the algebraic complexity from O(M*N) to O(K*(M+N)), giving real algorithmic speedups at compute-bound batch sizes (up to 2.81x). State-of-the-art 2024 methods (like ASVD and Palu) actually combine both: SVD factorization first, followed by 4-bit quantization."
  },
  {
    q: "In your factorized layer W ≈ L₂ · L₁, where did you place the singular values Σ? Why not keep Σ as a separate layer?",
    speaker: "Person A (Modeling Lead)",
    answer: "If kept as a separate diagonal matrix, the forward pass would require 3 matrix multiplications and 3 separate kernel launches, worsening kernel dispatch latency. Instead, we use a symmetric absorption scheme: W_L1 = sqrt(Σ_k) · V_k^T and W_L2 = U_k · sqrt(Σ_k). Symmetrically splitting sqrt(Σ) balances the spectral norm and condition number of both layers, which stabilizes gradient propagation during post-SVD recovery fine-tuning."
  },
  {
    q: "Why does rank k=8 at Batch Size = 1 show an 8-11% latency SLOWDOWN despite reducing FLOPs by 71x?",
    speaker: "Person B (Systems Lead — Riddhi Jain)",
    answer: "At Batch Size = 1, inference is strictly memory-bandwidth bound and dispatch-latency bound under the Roofline model. Arithmetic intensity is below 0.50 FLOP/Byte. Splitting one dense layer into two introduces a second GPU kernel launch, plus intermediate tensor write/read round-trips to GPU DRAM. When memory latency dominates math execution time, reducing FLOPs cannot decrease runtime."
  },
  {
    q: "Why did fine-tuning recovered accuracy to 87.29%, actually beating the uncompressed baseline (86.18%)?",
    speaker: "Person A (Modeling Lead)",
    answer: "In deep networks, weight matrices are full-rank numerically but have low effective rank. Tail singular values represent high-frequency noise and overfit patterns. Truncating the tail acts as an explicit low-rank regularizer (Sainath et al., 2013). When we fine-tune with AdamW and cosine annealing, the network optimizes in this low-rank subspace, improving generalization on test data by +1.11%."
  },
  {
    q: "Why is Python time.time() banned for GPU benchmarking in your harness?",
    speaker: "Person B (Systems Lead — Riddhi Jain)",
    answer: "PyTorch GPU execution (CUDA and Apple MPS) is asynchronous! Python's CPU thread returns immediately after queuing GPU commands. Naive time.time() merely measures CPU dispatch latency, not actual GPU kernel execution. Our harness uses hardware-synchronized timing (torch.cuda.Event with synchronize() or high-res MPS stream barriers) with a 100-iteration warmup to stabilize GPU frequency clocks, followed by 200 iterations reporting mean, std dev, and percentiles (P50, P90, P99)."
  },
  {
    q: "How does this experiment connect to Large Language Models (LLMs) and ChatGPT?",
    speaker: "Person B (Systems Lead — Riddhi Jain)",
    answer: "Autoregressive LLM token generation is inherently Batch Size = 1. Every generated token requires loading gigabytes of model weights and KV-cache tensors from memory just to perform vector-matrix multiplication. This pins LLM serving in the memory-bound regime (<0.50 FLOP/Byte), exactly like our Batch Size = 1 SVD layer experiment."
  },
  {
    q: "What is the Breakeven Rank (K*) formula and what does it mean?",
    speaker: "Person A (Modeling Lead)",
    answer: "The theoretical FLOP breakeven rank K* = (M · N) / (M + N). For our 1024 -> 512 dense projection layer, K* = (1024 * 512) / (1024 + 512) = 341.33. If rank k < 341, decomposed FLOPs are strictly lower than dense FLOPs. If rank k > 341, decomposed FLOPs actually EXCEED the original dense layer because the two smaller matrix multiplications sum to more operations than the single large matrix."
  },
  {
    q: "How would you solve the Batch Size = 1 latency overhead in production?",
    speaker: "Person B (Systems Lead — Riddhi Jain)",
    answer: "Two main techniques: 1) Kernel Fusion via OpenAI Triton or CUDA C++ to fuse L1 and L2 into a single kernel, keeping the intermediate k-dimensional tensor in high-speed SRAM registers and eliminating DRAM round-trips. 2) Dimension alignment to multiples of 8 or 16 to utilize Tensor Core MMA instructions."
  }
];

export const DEMO_SCRIPT_STEPS = [
  {
    step: 1,
    duration: "5 sec",
    title: "Mathematical Correctness & Tests",
    command: ".venv/bin/python3 -m unittest discover tests",
    speech: "Professors, we first verify our mathematical integrity: all 5 unit tests pass in under 0.5s, verifying full-rank numerical equivalence to 10^-4 precision, output shape conservation, and monotonic Frobenius error reduction."
  },
  {
    step: 2,
    duration: "15 sec",
    title: "Phase 1 Image SVD Visual Intuition",
    command: ".venv/bin/python3 scripts/image_svd.py",
    speech: "Here is Phase 1. On a 512x512 matrix, rank k=16 achieves 15.98x compression while retaining 93.88% of total singular value energy with under 1% reconstruction error. This visually proves that top singular vectors capture semantic content."
  },
  {
    step: 3,
    duration: "15 sec",
    title: "Physical Checkpoint Storage Footprint",
    command: "ls -lh results/checkpoints/",
    speech: "Notice our saved PyTorch weights: the baseline model is 5.58 MB. Our rank k=8 model is only 150.1 KB—that is a physical 39.25x storage reduction on disk."
  },
  {
    step: 4,
    duration: "30 sec",
    title: "The Climax: The FLOP vs Latency Divergence",
    command: ".venv/bin/python3 scripts/show_results.py",
    speech: "Look at the empirical hardware divergence. At Batch Size = 1, rank 8 has 71x fewer FLOPs but runs 11% slower (0.444ms vs 0.498ms) due to kernel launch overhead and intermediate tensor memory traffic. Real 2.81x speedup only happens at Batch Size = 128."
  },
  {
    step: 5,
    duration: "15 sec",
    title: "Interactive Web Dashboard & Viva Defense",
    command: "open results/dashboard.html",
    speech: "We built this unified interactive research dashboard so you can explore the Pareto frontier, simulate custom matrix dimensions, and inspect empirical Roofline bounds."
  }
];
