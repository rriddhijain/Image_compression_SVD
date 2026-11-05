import React, { useState } from 'react';
import { Brain, Cpu, Layers, BookOpen, AlertCircle, ArrowRight, Zap, Target } from 'lucide-react';

export default function TheoryAndLLMThesis() {
  const [activeTab, setActiveTab] = useState('llm_thesis'); // 'llm_thesis', 'math_theory', 'roofline'

  // Roofline interactive calculation
  const [peakTflops, setPeakTflops] = useState(10.4); // Apple M-series GPU
  const [peakBandwidthGbs, setPeakBandwidthGbs] = useState(150); // GB/s unified memory
  
  // Ridge point: FLOP/Byte = (TFLOPs * 1e12) / (Bandwidth * 1e9) = (TFLOPs * 1000) / Bandwidth
  const ridgePoint = ((peakTflops * 1000) / peakBandwidthGbs).toFixed(2);

  return (
    <div className="demo-section">
      <div style={{ marginBottom: 20 }}>
        <h2 style={{ fontSize: '1.4rem', fontWeight: 700, color: '#fff', display: 'flex', alignItems: 'center', gap: 10 }}>
          <span style={{ color: 'var(--accent-purple)' }}>Capstone Thesis:</span> The LLM Memory Wall & Math Rigor
        </h2>
        <p style={{ color: 'var(--text-muted)', fontSize: '0.88rem' }}>
          Unpacks the fundamental physics of GPU hardware constraints and mathematical proofs underlying low-rank decomposition.
        </p>
      </div>

      {/* Navigation Subtabs */}
      <div style={{ display: 'flex', gap: 8, marginBottom: 24 }}>
        <button
          id="btn-subtab-llm"
          className={`btn btn-pill ${activeTab === 'llm_thesis' ? 'btn-primary' : 'btn-secondary'}`}
          onClick={() => setActiveTab('llm_thesis')}
        >
          <Brain size={14} /> The LLM KV-Cache Connection
        </button>
        <button
          id="btn-subtab-math"
          className={`btn btn-pill ${activeTab === 'math_theory' ? 'btn-primary' : 'btn-secondary'}`}
          onClick={() => setActiveTab('math_theory')}
        >
          <BookOpen size={14} /> Mathematical Derivations
        </button>
        <button
          id="btn-subtab-roofline"
          className={`btn btn-pill ${activeTab === 'roofline' ? 'btn-primary' : 'btn-secondary'}`}
          onClick={() => setActiveTab('roofline')}
        >
          <Cpu size={14} /> Roofline Model Explorer
        </button>
      </div>

      {/* Subtab 1: LLM Memory Wall Connection */}
      {activeTab === 'llm_thesis' && (
        <div>
          <div className="glass-panel glass-panel-glow" style={{ padding: 24, marginBottom: 24 }}>
            <h3 style={{ fontSize: '1.2rem', color: '#fff', marginBottom: 12, display: 'flex', alignItems: 'center', gap: 10 }}>
              <Brain size={22} color="var(--accent-cyan)" />
              Why Batch Size = 1 Models the LLM Generative Wall
            </h3>
            <p style={{ color: '#d1d5db', lineHeight: 1.7, fontSize: '0.92rem', marginBottom: 18 }}>
              In autoregressive Large Language Models (ChatGPT, LLaMA, Claude), tokens are generated sequentially <strong>one at a time (B = 1)</strong>.
              For every single token generated, the accelerator must transfer tens of gigabytes of weights and Key-Value (KV) cache tensors from High Bandwidth Memory (HBM) into on-chip SRAM registers just to perform a single vector-matrix multiplication (y = x · W).
            </p>

            <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(280px, 1fr))', gap: 16 }}>
              
              <div style={{ background: 'rgba(0, 240, 255, 0.05)', padding: 18, borderRadius: 12, border: '1px solid rgba(0, 240, 255, 0.2)' }}>
                <div style={{ fontSize: '0.8rem', color: 'var(--accent-cyan)', fontWeight: 600, textTransform: 'uppercase', marginBottom: 6 }}>
                  Our SVD Experiment (B=1)
                </div>
                <div style={{ fontSize: '1.05rem', fontWeight: 600, color: '#fff', marginBottom: 8 }}>
                  Memory-Bandwidth Bound
                </div>
                <p style={{ fontSize: '0.84rem', color: 'var(--text-muted)', lineHeight: 1.5 }}>
                  Cutting FLOPs by 71x yields zero speedup because the GPU spends 95% of its execution cycle idling on DRAM bus transfers. Adding a second kernel increases dispatch latency.
                </p>
              </div>

              <div style={{ background: 'rgba(139, 92, 246, 0.05)', padding: 18, borderRadius: 12, border: '1px solid rgba(139, 92, 246, 0.2)' }}>
                <div style={{ fontSize: '0.8rem', color: '#c4b5fd', fontWeight: 600, textTransform: 'uppercase', marginBottom: 6 }}>
                  LLM Token Generation (B=1)
                </div>
                <div style={{ fontSize: '1.05rem', fontWeight: 600, color: '#fff', marginBottom: 8 }}>
                  KV-Cache Memory Wall
                </div>
                <p style={{ fontSize: '0.84rem', color: 'var(--text-muted)', lineHeight: 1.5 }}>
                  Modern LLMs hit an identical memory bandwidth wall. Peak FP16 compute FLOPS are idle; throughput is strictly dictated by memory bandwidth (GB/s), not TFLOPs.
                </p>
              </div>
            </div>
          </div>

          {/* Solutions in Modern Production */}
          <div className="glass-panel" style={{ padding: 20 }}>
            <h4 style={{ fontSize: '1rem', fontWeight: 600, color: '#fff', marginBottom: 12, display: 'flex', alignItems: 'center', gap: 8 }}>
              <Zap size={16} color="var(--accent-emerald)" />
              How Modern Production Engines Overcome This Bottleneck
            </h4>
            <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(260px, 1fr))', gap: 14 }}>
              <div style={{ background: 'rgba(0,0,0,0.25)', padding: 14, borderRadius: 10, border: '1px solid var(--border-subtle)' }}>
                <strong style={{ color: 'var(--accent-cyan)', display: 'block', marginBottom: 4 }}>1. Triton / CUDA Kernel Fusion</strong>
                <span style={{ fontSize: '0.84rem', color: 'var(--text-muted)' }}>
                  Fuse L₁ and L₂ into a single GPU kernel. The intermediate K-dimensional tensor is kept entirely in fast SRAM registers, bypassing DRAM write/reads.
                </span>
              </div>
              <div style={{ background: 'rgba(0,0,0,0.25)', padding: 14, borderRadius: 10, border: '1px solid var(--border-subtle)' }}>
                <strong style={{ color: 'var(--accent-emerald)', display: 'block', marginBottom: 4 }}>2. Continuous Batching (vLLM)</strong>
                <span style={{ fontSize: '0.84rem', color: 'var(--text-muted)' }}>
                  Batch multiple requests together dynamically to push operational batch size from B=1 up to B ≥ 64, moving into compute-bound speedup territory.
                </span>
              </div>
              <div style={{ background: 'rgba(0,0,0,0.25)', padding: 14, borderRadius: 10, border: '1px solid var(--border-subtle)' }}>
                <strong style={{ color: 'var(--accent-purple)', display: 'block', marginBottom: 4 }}>3. Hybrid SVD + 4-bit Quantization</strong>
                <span style={{ fontSize: '0.84rem', color: 'var(--text-muted)' }}>
                  Palu & ASVD (2024): SVD factorizes low-rank structure first, then weights are quantized to INT4, slashing both FLOPs and DRAM bytes simultaneously.
                </span>
              </div>
            </div>
          </div>
        </div>
      )}

      {/* Subtab 2: Mathematical Derivations */}
      {activeTab === 'math_theory' && (
        <div style={{ display: 'flex', flexDirection: 'column', gap: 20 }}>
          
          {/* SVD & Eckart-Young */}
          <div className="glass-panel" style={{ padding: 24 }}>
            <h3 style={{ fontSize: '1.15rem', color: '#fff', marginBottom: 12 }}>
              1. Singular Value Decomposition & Eckart-Young-Mirsky Theorem
            </h3>
            <p style={{ color: 'var(--text-muted)', fontSize: '0.9rem', lineHeight: 1.6, marginBottom: 14 }}>
              Any real matrix W ∈ ℝ^(M × N) can be factorized into three matrices:
            </p>
            <div className="code-box" style={{ marginBottom: 14 }}>
              W = U · Σ · Vᵀ = ∑(i=1..r) σ_i · u_i · v_iᵀ
            </div>
            <p style={{ color: 'var(--text-muted)', fontSize: '0.88rem', lineHeight: 1.6 }}>
              Where U ∈ ℝ^(M × M) and V ∈ ℝ^(N × N) are orthogonal matrices containing left and right singular vectors, and Σ ∈ ℝ^(M × N) contains singular values σ₁ ≥ σ₂ ≥ ... ≥ σ_r ≥ 0.
              According to the <strong>Eckart-Young-Mirsky Theorem (1936)</strong>, the truncated rank-K approximation W_K = ∑(i=1..K) σ_i u_i v_iᵀ is the provably optimal rank-K approximation minimizing the Frobenius norm error:
            </p>
            <div className="code-box" style={{ marginTop: 10 }}>
              min_{'{rank(B) ≤ K}'} ||W - B||_F = ||W - W_K||_F = √( ∑(j=K+1..r) σ_j² )
            </div>
          </div>

          {/* Breakeven Rank Formula */}
          <div className="glass-panel" style={{ padding: 24 }}>
            <h3 style={{ fontSize: '1.15rem', color: '#fff', marginBottom: 12 }}>
              2. Analytical Breakeven Rank (K*) Derivation
            </h3>
            <p style={{ color: 'var(--text-muted)', fontSize: '0.9rem', lineHeight: 1.6, marginBottom: 14 }}>
              When is low-rank factorization computationally beneficial? We equate the theoretical FLOP count of the dense layer to the factorized cascaded layers:
            </p>
            <div className="code-box" style={{ marginBottom: 14 }}>
              FLOPs_dense = 2 · B · M · N<br/>
              FLOPs_decomp = 2 · B · K · (M + N)<br/><br/>
              Setting FLOPs_decomp = FLOPs_dense:<br/>
              2 · B · K* · (M + N) = 2 · B · M · N<br/>
              ⟹ K* = (M · N) / (M + N)
            </div>
            <p style={{ color: 'var(--text-muted)', fontSize: '0.88rem', lineHeight: 1.6 }}>
              For our primary layer (1024 → 512):<br/>
              K* = (1024 × 512) / (1024 + 512) = 524288 / 1536 ≈ <strong>341.33</strong>.<br/>
              Any rank K &lt; 341 guarantees theoretical FLOP savings; any rank K &gt; 341 causes arithmetic expansion.
            </p>
          </div>
        </div>
      )}

      {/* Subtab 3: Roofline Model Explorer */}
      {activeTab === 'roofline' && (
        <div className="glass-panel" style={{ padding: 24 }}>
          <h3 style={{ fontSize: '1.15rem', color: '#fff', marginBottom: 12, display: 'flex', alignItems: 'center', gap: 10 }}>
            <Cpu size={20} color="var(--accent-cyan)" />
            Williams et al. (2009) Roofline Model Interactive Simulator
          </h3>
          <p style={{ color: 'var(--text-muted)', fontSize: '0.9rem', lineHeight: 1.6, marginBottom: 20 }}>
            The Roofline model defines attainable performance as Performance = min(Peak FLOP/s, Operational Intensity × Memory Bandwidth).
          </p>

          <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: 20, marginBottom: 24 }}>
            <div>
              <label style={{ display: 'block', fontSize: '0.85rem', color: 'var(--text-muted)', marginBottom: 6 }}>
                Hardware Peak Compute (TFLOP/s):
              </label>
              <input 
                id="input-peak-tflops"
                type="number" 
                value={peakTflops} 
                onChange={(e) => setPeakTflops(parseFloat(e.target.value) || 1)}
                style={{
                  width: '100%',
                  padding: '10px 14px',
                  borderRadius: 8,
                  background: 'var(--bg-tertiary)',
                  border: '1px solid var(--border-subtle)',
                  color: '#fff',
                  fontFamily: 'var(--font-mono)'
                }}
              />
            </div>

            <div>
              <label style={{ display: 'block', fontSize: '0.85rem', color: 'var(--text-muted)', marginBottom: 6 }}>
                Memory Bandwidth (GB/s):
              </label>
              <input 
                id="input-bandwidth-gbs"
                type="number" 
                value={peakBandwidthGbs} 
                onChange={(e) => setPeakBandwidthGbs(parseFloat(e.target.value) || 1)}
                style={{
                  width: '100%',
                  padding: '10px 14px',
                  borderRadius: 8,
                  background: 'var(--bg-tertiary)',
                  border: '1px solid var(--border-subtle)',
                  color: '#fff',
                  fontFamily: 'var(--font-mono)'
                }}
              />
            </div>
          </div>

          <div style={{ background: 'rgba(0,0,0,0.3)', padding: 20, borderRadius: 12, border: '1px solid var(--border-subtle)', marginBottom: 20 }}>
            <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
              <div>
                <span style={{ fontSize: '0.8rem', color: 'var(--text-muted)', textTransform: 'uppercase' }}>
                  CALCULATED MACHINE RIDGE POINT (I*)
                </span>
                <div style={{ fontSize: '1.8rem', fontWeight: 700, color: 'var(--accent-cyan)', fontFamily: 'var(--font-mono)' }}>
                  {ridgePoint} FLOP/Byte
                </div>
              </div>
              <span className="badge badge-purple">Transition Threshold</span>
            </div>
            <p style={{ fontSize: '0.85rem', color: 'var(--text-muted)', marginTop: 8 }}>
              Any workload with Arithmetic Intensity I &lt; {ridgePoint} FLOP/Byte (like our SVD layer at B=1 with I ≈ 0.44) is strictly <strong>Memory-Bandwidth Bound</strong>.
              Only workloads with I ≥ {ridgePoint} FLOP/Byte can achieve peak arithmetic execution speedups.
            </p>
          </div>
        </div>
      )}
    </div>
  );
}
