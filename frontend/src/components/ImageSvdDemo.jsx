import React, { useState, useEffect, useRef } from 'react';
import { Sliders, Image as ImageIcon, Zap, Info, Upload, CheckCircle2 } from 'lucide-react';
import { computeTopKSVD, reconstructFromSVD, calculateSVDMetrics } from '../utils/svdMath';
import { IMAGE_SVD_METRICS } from '../data/benchmarkData';

export default function ImageSvdDemo() {
  const [rank, setRank] = useState(8);
  const [imageSource, setImageSource] = useState('cameraman'); // 'cameraman', 'synthetic', 'custom'
  const [matrixData, setMatrixData] = useState(null);
  const [svdResult, setSvdResult] = useState(null);
  const [metrics, setMetrics] = useState(null);
  const [isComputing, setIsComputing] = useState(false);

  const origCanvasRef = useRef(null);
  const reconCanvasRef = useRef(null);
  const diffCanvasRef = useRef(null);

  const MATRIX_SIZE = 72; // Optimized size for instantaneous interactive 60fps client-side SVD

  // Generate synthetic multi-frequency test matrix
  const generateSyntheticMatrix = (size) => {
    const mat = [];
    for (let i = 0; i < size; i++) {
      const row = [];
      for (let j = 0; j < size; j++) {
        // Multi-frequency wave pattern + circular feature
        const wave1 = Math.sin((i / size) * Math.PI * 4);
        const wave2 = Math.cos((j / size) * Math.PI * 6);
        const distCenter = Math.sqrt(Math.pow(i - size / 2, 2) + Math.pow(j - size / 2, 2));
        const circle = distCenter < size / 4 ? 0.8 : 0.1;
        const val = Math.max(0, Math.min(1, 0.4 + 0.3 * wave1 * wave2 + 0.3 * circle));
        row.push(val);
      }
      mat.push(row);
    }
    return mat;
  };

  // Load image from URL or synthesized
  useEffect(() => {
    setIsComputing(true);
    if (imageSource === 'synthetic') {
      const mat = generateSyntheticMatrix(MATRIX_SIZE);
      setMatrixData(mat);
      const svd = computeTopKSVD(mat, 48);
      setSvdResult(svd);
      setIsComputing(false);
    } else {
      const img = new Image();
      img.crossOrigin = 'anonymous';
      img.src = imageSource === 'cameraman' ? '/cameraman.png' : imageSource;
      img.onload = () => {
        const offCanvas = document.createElement('canvas');
        offCanvas.width = MATRIX_SIZE;
        offCanvas.height = MATRIX_SIZE;
        const ctx = offCanvas.getContext('2d');
        ctx.drawImage(img, 0, 0, MATRIX_SIZE, MATRIX_SIZE);
        const imgData = ctx.getImageData(0, 0, MATRIX_SIZE, MATRIX_SIZE);
        
        const mat = [];
        for (let i = 0; i < MATRIX_SIZE; i++) {
          const row = [];
          for (let j = 0; j < MATRIX_SIZE; j++) {
            const idx = (i * MATRIX_SIZE + j) * 4;
            // Grayscale luminance
            const lum = (0.299 * imgData.data[idx] + 0.587 * imgData.data[idx + 1] + 0.114 * imgData.data[idx + 2]) / 255.0;
            row.push(lum);
          }
          mat.push(row);
        }

        setMatrixData(mat);
        const svd = computeTopKSVD(mat, 48);
        setSvdResult(svd);
        setIsComputing(false);
      };
      img.onerror = () => {
        // Fallback to synthetic if image fails loading
        const mat = generateSyntheticMatrix(MATRIX_SIZE);
        setMatrixData(mat);
        const svd = computeTopKSVD(mat, 48);
        setSvdResult(svd);
        setIsComputing(false);
      };
    }
  }, [imageSource]);

  // Handle custom image upload
  const handleFileUpload = (e) => {
    const file = e.target.files[0];
    if (file) {
      const reader = new FileReader();
      reader.onload = (event) => {
        setImageSource(event.target.result);
      };
      reader.readAsDataURL(file);
    }
  };

  // Reconstruct on rank change
  useEffect(() => {
    if (!matrixData || !svdResult) return;

    const reconMat = reconstructFromSVD(svdResult.U, svdResult.S, svdResult.V, rank);
    const m = calculateSVDMetrics(matrixData, reconMat, svdResult.S, rank);
    setMetrics(m);

    // Draw original
    if (origCanvasRef.current) {
      drawMatrixToCanvas(origCanvasRef.current, matrixData);
    }
    // Draw reconstructed
    if (reconCanvasRef.current) {
      drawMatrixToCanvas(reconCanvasRef.current, reconMat);
    }
    // Draw error heatmap
    if (diffCanvasRef.current) {
      drawDiffToCanvas(diffCanvasRef.current, matrixData, reconMat);
    }
  }, [rank, matrixData, svdResult]);

  const drawMatrixToCanvas = (canvas, mat) => {
    const ctx = canvas.getContext('2d');
    const size = mat.length;
    canvas.width = size;
    canvas.height = size;
    const imgData = ctx.createImageData(size, size);

    for (let i = 0; i < size; i++) {
      for (let j = 0; j < size; j++) {
        const idx = (i * size + j) * 4;
        const val = Math.max(0, Math.min(255, Math.round(mat[i][j] * 255)));
        imgData.data[idx] = val;
        imgData.data[idx + 1] = val;
        imgData.data[idx + 2] = val;
        imgData.data[idx + 3] = 255;
      }
    }
    ctx.putImageData(imgData, 0, 0);
  };

  const drawDiffToCanvas = (canvas, orig, recon) => {
    const ctx = canvas.getContext('2d');
    const size = orig.length;
    canvas.width = size;
    canvas.height = size;
    const imgData = ctx.createImageData(size, size);

    for (let i = 0; i < size; i++) {
      for (let j = 0; j < size; j++) {
        const idx = (i * size + j) * 4;
        const diff = Math.abs(orig[i][j] - recon[i][j]);
        // Heatmap: dark blue -> bright red for high error
        imgData.data[idx] = Math.min(255, Math.round(diff * 4 * 255)); // Red
        imgData.data[idx + 1] = Math.round(diff * 1.5 * 255);          // Green
        imgData.data[idx + 2] = Math.max(0, 180 - Math.round(diff * 300)); // Blue
        imgData.data[idx + 3] = 255;
      }
    }
    ctx.putImageData(imgData, 0, 0);
  };

  const QUICK_RANKS = [2, 4, 8, 16, 24, 32, 48];

  return (
    <div className="demo-section">
      <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', flexWrap: 'wrap', gap: 16, marginBottom: 20 }}>
        <div>
          <h2 style={{ fontSize: '1.4rem', fontWeight: 700, color: '#fff', display: 'flex', alignItems: 'center', gap: 10 }}>
            <span style={{ color: 'var(--accent-cyan)' }}>Phase 1:</span> 2D Matrix SVD Reconstruction Lab
          </h2>
          <p style={{ color: 'var(--text-muted)', fontSize: '0.88rem' }}>
            Visualizes low-rank spectral decay: reconstruct any 2D image matrix using only top-K singular triplets W ≈ ∑ σᵢ uᵢ vᵢᵀ.
          </p>
        </div>

        {/* Source Picker */}
        <div style={{ display: 'flex', gap: 8, alignItems: 'center' }}>
          <button 
            id="btn-source-cameraman"
            className={`btn btn-pill ${imageSource === 'cameraman' ? 'btn-primary' : 'btn-secondary'}`}
            onClick={() => setImageSource('cameraman')}
          >
            Cameraman
          </button>
          <button 
            id="btn-source-synthetic"
            className={`btn btn-pill ${imageSource === 'synthetic' ? 'btn-primary' : 'btn-secondary'}`}
            onClick={() => setImageSource('synthetic')}
          >
            Multi-Freq Wave
          </button>
          <label 
            id="btn-source-upload"
            className="btn btn-pill btn-secondary" 
            style={{ cursor: 'pointer', display: 'inline-flex', alignItems: 'center', gap: 6 }}
          >
            <Upload size={14} /> Upload Custom
            <input type="file" accept="image/*" onChange={handleFileUpload} style={{ display: 'none' }} />
          </label>
        </div>
      </div>

      {/* Main Interactive Controls & KPIs */}
      <div className="glass-panel" style={{ padding: 24, marginBottom: 24 }}>
        <div style={{ display: 'grid', gridTemplateColumns: '1fr auto', gap: 24, alignItems: 'center', flexWrap: 'wrap' }}>
          
          <div className="slider-container">
            <div className="slider-label">
              <span style={{ display: 'flex', alignItems: 'center', gap: 8 }}>
                <Sliders size={16} color="var(--accent-cyan)" />
                <strong>Truncated Rank ($K$):</strong>
              </span>
              <span className="slider-value">K = {rank}</span>
            </div>
            <input 
              id="slider-svd-rank"
              type="range" 
              min="1" 
              max="48" 
              value={rank} 
              onChange={(e) => setRank(parseInt(e.target.value))}
            />
            
            {/* Quick Rank Selector Pills */}
            <div style={{ display: 'flex', gap: 6, marginTop: 8 }}>
              {QUICK_RANKS.map((kVal) => (
                <button
                  key={kVal}
                  id={`btn-rank-${kVal}`}
                  onClick={() => setRank(kVal)}
                  style={{
                    padding: '4px 10px',
                    borderRadius: 6,
                    fontSize: '0.78rem',
                    fontFamily: 'var(--font-mono)',
                    border: '1px solid',
                    borderColor: rank === kVal ? 'var(--accent-cyan)' : 'var(--border-subtle)',
                    background: rank === kVal ? 'rgba(0, 240, 255, 0.15)' : 'transparent',
                    color: rank === kVal ? 'var(--accent-cyan)' : 'var(--text-muted)',
                    cursor: 'pointer'
                  }}
                >
                  k={kVal}
                </button>
              ))}
            </div>
          </div>

          {/* Quick Metrics HUD */}
          {metrics && (
            <div style={{ display: 'flex', gap: 16, flexWrap: 'wrap' }}>
              <div style={{ background: 'rgba(0,0,0,0.3)', padding: '12px 18px', borderRadius: 12, border: '1px solid var(--border-subtle)', textAlign: 'center' }}>
                <div style={{ fontSize: '0.75rem', color: 'var(--text-muted)' }}>COMPRESSION</div>
                <div style={{ fontSize: '1.4rem', fontWeight: 700, color: 'var(--accent-cyan)', fontFamily: 'var(--font-mono)' }}>
                  {metrics.compressionRatio}x
                </div>
              </div>

              <div style={{ background: 'rgba(0,0,0,0.3)', padding: '12px 18px', borderRadius: 12, border: '1px solid var(--border-subtle)', textAlign: 'center' }}>
                <div style={{ fontSize: '0.75rem', color: 'var(--text-muted)' }}>ENERGY RETAINED</div>
                <div style={{ fontSize: '1.4rem', fontWeight: 700, color: 'var(--accent-emerald)', fontFamily: 'var(--font-mono)' }}>
                  {metrics.energyPercent}%
                </div>
              </div>

              <div style={{ background: 'rgba(0,0,0,0.3)', padding: '12px 18px', borderRadius: 12, border: '1px solid var(--border-subtle)', textAlign: 'center' }}>
                <div style={{ fontSize: '0.75rem', color: 'var(--text-muted)' }}>FROBENIUS ERROR</div>
                <div style={{ fontSize: '1.4rem', fontWeight: 700, color: metrics.relError < 2 ? 'var(--accent-emerald)' : 'var(--accent-amber)', fontFamily: 'var(--font-mono)' }}>
                  {metrics.relError}%
                </div>
              </div>
            </div>
          )}
        </div>
      </div>

      {/* Visual Canvas Comparison Grid */}
      <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(280px, 1fr))', gap: 20, marginBottom: 24 }}>
        
        {/* Original */}
        <div className="glass-panel" style={{ padding: 20, display: 'flex', flexDirection: 'column', alignItems: 'center' }}>
          <div style={{ width: '100%', display: 'flex', justifyContent: 'space-between', marginBottom: 12 }}>
            <span style={{ fontWeight: 600, fontSize: '0.9rem' }}>Original 2D Matrix (I)</span>
            <span className="badge badge-purple">Full Rank ({MATRIX_SIZE}x{MATRIX_SIZE})</span>
          </div>
          <div style={{ width: 220, height: 220, background: '#000', borderRadius: 8, overflow: 'hidden', display: 'flex', alignItems: 'center', justifyContent: 'center' }}>
            <canvas ref={origCanvasRef} style={{ width: '100%', height: '100%', imageRendering: 'pixelated' }} />
          </div>
          <div style={{ fontSize: '0.8rem', color: 'var(--text-dim)', marginTop: 12, textAlign: 'center' }}>
            Stored as uncompressed dense array ({MATRIX_SIZE * MATRIX_SIZE} values)
          </div>
        </div>

        {/* Reconstructed */}
        <div className="glass-panel glass-panel-glow" style={{ padding: 20, display: 'flex', flexDirection: 'column', alignItems: 'center' }}>
          <div style={{ width: '100%', display: 'flex', justifyContent: 'space-between', marginBottom: 12 }}>
            <span style={{ fontWeight: 600, fontSize: '0.9rem', color: 'var(--accent-cyan)' }}>
              Truncated SVD (Î_k)
            </span>
            <span className="badge badge-cyan">Rank k={rank}</span>
          </div>
          <div style={{ width: 220, height: 220, background: '#000', borderRadius: 8, overflow: 'hidden', display: 'flex', alignItems: 'center', justifyContent: 'center' }}>
            <canvas ref={reconCanvasRef} style={{ width: '100%', height: '100%', imageRendering: 'pixelated' }} />
          </div>
          <div style={{ fontSize: '0.8rem', color: 'var(--accent-cyan)', marginTop: 12, textAlign: 'center' }}>
            Approximated from {rank} singular vectors + values
          </div>
        </div>

        {/* Residual Difference Heatmap */}
        <div className="glass-panel" style={{ padding: 20, display: 'flex', flexDirection: 'column', alignItems: 'center' }}>
          <div style={{ width: '100%', display: 'flex', justifyContent: 'space-between', marginBottom: 12 }}>
            <span style={{ fontWeight: 600, fontSize: '0.9rem' }}>Error Residual (|I - Î_k|)</span>
            <span className="badge badge-amber">Heatmap</span>
          </div>
          <div style={{ width: 220, height: 220, background: '#000', borderRadius: 8, overflow: 'hidden', display: 'flex', alignItems: 'center', justifyContent: 'center' }}>
            <canvas ref={diffCanvasRef} style={{ width: '100%', height: '100%', imageRendering: 'pixelated' }} />
          </div>
          <div style={{ fontSize: '0.8rem', color: 'var(--text-dim)', marginTop: 12, textAlign: 'center' }}>
            Red = high discrepancy, Dark = near-zero error
          </div>
        </div>
      </div>

      {/* Published 512x512 Empirical Benchmark Comparison Table */}
      <div className="glass-panel" style={{ padding: 20, marginBottom: 20 }}>
        <h3 style={{ fontSize: '1.05rem', marginBottom: 12, display: 'flex', alignItems: 'center', gap: 8 }}>
          <Info size={16} color="var(--accent-indigo)" />
          Empirical 512×512 Image SVD Benchmark (From Project Reproduction)
        </h3>
        
        <div className="custom-table-container">
          <table className="custom-table">
            <thead>
              <tr>
                <th>Truncated Rank (k)</th>
                <th>Compression Ratio</th>
                <th>Frobenius Error</th>
                <th>Cumulative Energy</th>
                <th>Visual Quality Retention</th>
              </tr>
            </thead>
            <tbody>
              {IMAGE_SVD_METRICS.map((row) => (
                <tr key={row.rank} style={{ background: row.rank === 16 ? 'rgba(0, 240, 255, 0.08)' : 'transparent' }}>
                  <td>
                    <strong style={{ color: row.rank === 16 ? 'var(--accent-cyan)' : 'inherit' }}>
                      k = {row.rank} {row.rank === 16 ? ' (Sweet Spot)' : ''}
                    </strong>
                  </td>
                  <td>{row.compression}x</td>
                  <td>{row.frobError}%</td>
                  <td>
                    <div style={{ display: 'flex', alignItems: 'center', gap: 8 }}>
                      <div style={{ flex: 1, background: 'var(--bg-tertiary)', height: 6, borderRadius: 3, overflow: 'hidden' }}>
                        <div style={{ width: `${row.energyRetained}%`, height: '100%', background: 'var(--accent-emerald)' }} />
                      </div>
                      <span style={{ fontFamily: 'var(--font-mono)', fontSize: '0.8rem' }}>{row.energyRetained}%</span>
                    </div>
                  </td>
                  <td>
                    <span className={`badge ${row.rank >= 16 ? 'badge-emerald' : 'badge-amber'}`}>
                      {row.rank >= 64 ? 'Perceptually Identical' : row.rank >= 16 ? 'Near Lossless' : 'Coarse Silhouette'}
                    </span>
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </div>

      {/* Teacher Talking Script Callout */}
      <div className="alert-box alert-cyan">
        <CheckCircle2 size={20} color="var(--accent-cyan)" style={{ flexShrink: 0, marginTop: 2 }} />
        <div>
          <strong>Key Takeaway:</strong> Demonstrates the Eckart-Young-Mirsky theorem: truncating singular values yields the optimal rank-k approximation under the Frobenius norm. Notice that rank k=16 preserves over <strong>93.8% of energy</strong> with <strong>&lt;1% error</strong> while compressing parameters by <strong>15.98x</strong>.
        </div>
      </div>
    </div>
  );
}
