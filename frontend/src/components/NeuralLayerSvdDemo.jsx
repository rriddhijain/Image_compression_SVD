import React, { useState } from 'react';
import { Layers, ArrowRight, Cpu, AlertTriangle, CheckCircle, Calculator, Zap } from 'lucide-react';

export default function NeuralLayerSvdDemo() {
  const [inFeatures, setInFeatures] = useState(1024);
  const [outFeatures, setOutFeatures] = useState(512);
  const [rank, setRank] = useState(16);
  const [batchSize, setBatchSize] = useState(1);

  // Math derivations
  const denseParams = inFeatures * outFeatures;
  const denseFlops = 2 * batchSize * inFeatures * outFeatures;

  const decompParams = rank * (inFeatures + outFeatures);
  const decompFlops = 2 * batchSize * rank * (inFeatures + outFeatures);

  const compressionRatio = (denseParams / decompParams).toFixed(2);
  const flopRatio = (denseFlops / decompFlops).toFixed(2);

  // Breakeven rank K* = (M * N) / (M + N)
  const breakevenRank = Math.floor((inFeatures * outFeatures) / (inFeatures + outFeatures));
  const isBelowBreakeven = rank < breakevenRank;

  // Preset dimension choices
  const PRESETS = [
    { label: 'Project MLP (1024 → 512)', inF: 1024, outF: 512, defaultK: 16 },
    { label: 'Input Projection (784 → 1024)', inF: 784, outF: 1024, defaultK: 32 },
    { label: 'LLaMA / Mistral MLP (4096 → 4096)', inF: 4096, outF: 4096, defaultK: 128 },
    { label: 'Output Classifier (256 → 10)', inF: 256, outF: 10, defaultK: 8 }
  ];

  return (
    <div className="demo-section">
      <div style={{ marginBottom: 20 }}>
        <h2 style={{ fontSize: '1.4rem', fontWeight: 700, color: '#fff', display: 'flex', alignItems: 'center', gap: 10 }}>
          <span style={{ color: 'var(--accent-indigo)' }}>Phase 2:</span> Neural Linear Layer SVD Factorization
        </h2>
        <p style={{ color: 'var(--text-muted)', fontSize: '0.88rem' }}>
          Deconstructs a dense weight matrix W ∈ ℝ^(M × N) into cascaded low-rank linear layers L₁ ∈ ℝ^(M × K) and L₂ ∈ ℝ^(K × N) without an intermediate activation function.
        </p>
      </div>

      {/* Preset Buttons */}
      <div style={{ display: 'flex', gap: 8, flexWrap: 'wrap', marginBottom: 20 }}>
        <span style={{ fontSize: '0.85rem', color: 'var(--text-muted)', display: 'flex', alignItems: 'center' }}>
          Architecture Presets:
        </span>
        {PRESETS.map((p) => (
          <button
            key={p.label}
            className={`btn btn-pill ${inFeatures === p.inF && outFeatures === p.outF ? 'btn-primary' : 'btn-secondary'}`}
            onClick={() => {
              setInFeatures(p.inF);
              setOutFeatures(p.outF);
              setRank(p.defaultK);
            }}
          >
            {p.label}
          </button>
        ))}
      </div>

      {/* Control Sliders & Inputs */}
      <div className="glass-panel" style={{ padding: 24, marginBottom: 24 }}>
        <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(260px, 1fr))', gap: 24 }}>
          
          {/* Rank Slider */}
          <div className="slider-container">
            <div className="slider-label">
              <span><strong>Truncated Rank ($K$):</strong></span>
              <span className="slider-value">K = {rank}</span>
            </div>
            <input 
              id="slider-neural-rank"
              type="range" 
              min="2" 
              max={Math.min(inFeatures, outFeatures, 256)} 
              value={rank} 
              onChange={(e) => setRank(parseInt(e.target.value))}
            />
            <div style={{ display: 'flex', justifyContent: 'space-between', fontSize: '0.75rem', color: 'var(--text-dim)', marginTop: 4 }}>
              <span>Min: k=2</span>
              <span style={{ color: 'var(--accent-amber)' }}>Breakeven K* = {breakevenRank}</span>
              <span>Max: {Math.min(inFeatures, outFeatures, 256)}</span>
            </div>
          </div>

          {/* Batch Size Selector */}
          <div className="slider-container">
            <div className="slider-label">
              <span><strong>Evaluation Batch Size (B_sz):</strong></span>
              <span className="slider-value">B = {batchSize}</span>
            </div>
            <div style={{ display: 'flex', gap: 8, marginTop: 4 }}>
              {[1, 8, 32, 64, 128].map((b) => (
                <button
                  key={b}
                  id={`btn-batch-${b}`}
                  onClick={() => setBatchSize(b)}
                  style={{
                    flex: 1,
                    padding: '8px 0',
                    borderRadius: 8,
                    fontSize: '0.85rem',
                    fontFamily: 'var(--font-mono)',
                    border: '1px solid',
                    borderColor: batchSize === b ? 'var(--accent-indigo)' : 'var(--border-subtle)',
                    background: batchSize === b ? 'rgba(99, 102, 241, 0.25)' : 'transparent',
                    color: batchSize === b ? '#fff' : 'var(--text-muted)',
                    cursor: 'pointer'
                  }}
                >
                  B={b}
                </button>
              ))}
            </div>
          </div>
        </div>
      </div>

      {/* Visual Architectural Decomposition Diagram */}
      <div className="glass-panel" style={{ padding: 24, marginBottom: 24 }}>
        <h3 style={{ fontSize: '1rem', fontWeight: 600, marginBottom: 16, color: 'var(--text-muted)' }}>
          Weight Matrix Transformation & Tensor Pipeline
        </h3>

        <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'center', flexWrap: 'wrap', gap: 24, padding: '20px 0' }}>
          
          {/* Dense Layer Block */}
          <div style={{
            background: 'linear-gradient(135deg, rgba(239, 68, 68, 0.15), rgba(17, 24, 39, 0.8))',
            border: '1px solid rgba(239, 68, 68, 0.4)',
            borderRadius: 14,
            padding: 20,
            textAlign: 'center',
            minWidth: 220
          }}>
            <div style={{ fontSize: '0.78rem', color: '#fca5a5', fontWeight: 600, textTransform: 'uppercase', marginBottom: 6 }}>
              Uncompressed Dense
            </div>
            <div style={{ fontSize: '1.25rem', fontWeight: 700, color: '#fff', fontFamily: 'var(--font-mono)' }}>
              W ∈ ℝ^{inFeatures} × {outFeatures}
            </div>
            <div style={{ fontSize: '0.85rem', color: 'var(--text-muted)', marginTop: 8 }}>
              {denseParams.toLocaleString()} Parameters
            </div>
            <div style={{ fontSize: '0.8rem', color: 'var(--text-dim)', marginTop: 4 }}>
              1 GEMM Kernel Launch
            </div>
          </div>

          {/* SVD Decomposition Arrow */}
          <div style={{ display: 'flex', flexDirection: 'column', alignItems: 'center', gap: 6 }}>
            <span className="badge badge-purple">SVD Factorization</span>
            <ArrowRight size={28} color="var(--accent-cyan)" />
            <span style={{ fontSize: '0.75rem', fontFamily: 'var(--font-mono)', color: 'var(--text-dim)' }}>
              W ≈ L₂ · L₁
            </span>
          </div>

          {/* Cascaded Decomposed Layers */}
          <div style={{ display: 'flex', alignItems: 'center', gap: 12, flexWrap: 'wrap' }}>
            
            {/* L1 */}
            <div style={{
              background: 'linear-gradient(135deg, rgba(0, 240, 255, 0.12), rgba(17, 24, 39, 0.8))',
              border: '1px solid rgba(0, 240, 255, 0.4)',
              borderRadius: 14,
              padding: 20,
              textAlign: 'center',
              minWidth: 180
            }}>
              <div style={{ fontSize: '0.78rem', color: 'var(--accent-cyan)', fontWeight: 600, textTransform: 'uppercase', marginBottom: 6 }}>
                L₁ (Down-Projection)
              </div>
              <div style={{ fontSize: '1.15rem', fontWeight: 700, color: '#fff', fontFamily: 'var(--font-mono)' }}>
                ℝ^{inFeatures} × {rank}
              </div>
              <div style={{ fontSize: '0.8rem', color: 'var(--text-muted)', marginTop: 6 }}>
                {(inFeatures * rank).toLocaleString()} Params
              </div>
              <div style={{ fontSize: '0.75rem', color: 'var(--text-dim)', marginTop: 2 }}>
                W_L1 = √(Σ_k) · V_kᵀ
              </div>
            </div>

            <div style={{ color: 'var(--text-dim)', fontWeight: 700, fontSize: '1.2rem' }}>×</div>

            {/* L2 */}
            <div style={{
              background: 'linear-gradient(135deg, rgba(99, 102, 241, 0.12), rgba(17, 24, 39, 0.8))',
              border: '1px solid rgba(99, 102, 241, 0.4)',
              borderRadius: 14,
              padding: 20,
              textAlign: 'center',
              minWidth: 180
            }}>
              <div style={{ fontSize: '0.78rem', color: '#c4b5fd', fontWeight: 600, textTransform: 'uppercase', marginBottom: 6 }}>
                L₂ (Up-Projection)
              </div>
              <div style={{ fontSize: '1.15rem', fontWeight: 700, color: '#fff', fontFamily: 'var(--font-mono)' }}>
                ℝ^{rank} × {outFeatures}
              </div>
              <div style={{ fontSize: '0.8rem', color: 'var(--text-muted)', marginTop: 6 }}>
                {(outFeatures * rank).toLocaleString()} Params
              </div>
              <div style={{ fontSize: '0.75rem', color: 'var(--text-dim)', marginTop: 2 }}>
                W_L2 = U_k · √(Σ_k) + b
              </div>
            </div>
          </div>
        </div>
      </div>

      {/* KPI Cards: Parameters, FLOPs, Breakeven Analysis */}
      <div className="kpi-grid">
        
        {/* Parameter Compression */}
        <div className="glass-panel kpi-card">
          <div className="kpi-header">
            <span>PARAMETER COMPRESSION</span>
            <Cpu size={16} color="var(--accent-cyan)" />
          </div>
          <div className="kpi-value" style={{ color: 'var(--accent-cyan)' }}>
            {compressionRatio}x
          </div>
          <div className="kpi-footer">
            {decompParams.toLocaleString()} vs {denseParams.toLocaleString()} parameters ({(((denseParams - decompParams) / denseParams) * 100).toFixed(1)}% reduction)
          </div>
        </div>

        {/* Theoretical FLOPs */}
        <div className="glass-panel kpi-card">
          <div className="kpi-header">
            <span>THEORETICAL FLOP SAVINGS</span>
            <Calculator size={16} color="var(--accent-emerald)" />
          </div>
          <div className="kpi-value" style={{ color: isBelowBreakeven ? 'var(--accent-emerald)' : 'var(--accent-rose)' }}>
            {flopRatio}x
          </div>
          <div className="kpi-footer">
            {decompFlops.toLocaleString()} vs {denseFlops.toLocaleString()} FLOPs @ B={batchSize}
          </div>
        </div>

        {/* Breakeven Rank Check */}
        <div className="glass-panel kpi-card">
          <div className="kpi-header">
            <span>BREAKEVEN RANK (K*)</span>
            {isBelowBreakeven ? (
              <CheckCircle size={16} color="var(--accent-emerald)" />
            ) : (
              <AlertTriangle size={16} color="var(--accent-rose)" />
            )}
          </div>
          <div className="kpi-value">
            K* = {breakevenRank}
          </div>
          <div className="kpi-footer">
            {isBelowBreakeven ? (
              <span style={{ color: 'var(--accent-emerald)' }}>✓ rank {rank} &lt; {breakevenRank} (True FLOP reduction)</span>
            ) : (
              <span style={{ color: 'var(--accent-rose)' }}>⚠ rank {rank} &ge; {breakevenRank} (FLOP expansion!)</span>
            )}
          </div>
        </div>
      </div>

      {/* Symmetric Singular Value Scheme Derivation Callout */}
      <div className="glass-panel" style={{ padding: 20 }}>
        <h4 style={{ fontSize: '0.95rem', fontWeight: 600, color: '#fff', marginBottom: 8, display: 'flex', alignItems: 'center', gap: 8 }}>
          <Zap size={16} color="var(--accent-cyan)" />
          Symmetric Singular Value Allocation & Bias Placement
        </h4>
        <p style={{ fontSize: '0.88rem', color: 'var(--text-muted)', lineHeight: 1.6, marginBottom: 12 }}>
          Singular values Σ_K are split equally via square root between both factors:
          W_L1 = √(Σ_K) · V_Kᵀ and W_L2 = U_K · √(Σ_K).
          This ensures equal spectral norm balance between both matrices, avoiding gradient vanishing in L₁ and explosion in L₂ during fine-tuning.
          The bias vector b ∈ ℝ^N is assigned solely to L₂ so y = (x W_L1) W_L2 + b strictly preserves the affine transformation.
        </p>
        <div className="code-box">
          # PyTorch Equivalent Forward Pass:<br/>
          h_intermediate = torch.matmul(x, W_L1.t())   # Shape: (B, K)<br/>
          output = torch.matmul(h_intermediate, W_L2.t()) + bias # Shape: (B, N)
        </div>
      </div>
    </div>
  );
}
