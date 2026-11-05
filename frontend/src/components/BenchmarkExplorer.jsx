import React, { useState } from 'react';
import { Database, Image as ImageIcon, HardDrive, Filter, ExternalLink } from 'lucide-react';
import { CHECKPOINTS_INFO } from '../data/benchmarkData';

export default function BenchmarkExplorer() {
  const [selectedPlot, setSelectedPlot] = useState('flops_latency');

  const PLOTS = [
    {
      id: 'flops_latency',
      title: 'Theoretical FLOPs vs Measured Latency (The Memory Wall)',
      filename: '/plots/theoretical_flops_vs_measured_latency.png',
      caption: 'Proves the divergence between theoretical math reductions (22.7x) and empirical wall-clock latency (0.89x speedup, 11% slowdown) at B=1 due to memory bandwidth limits.'
    },
    {
      id: 'pareto',
      title: 'Pareto Frontier: Parameter Compression vs Top-1 Accuracy',
      filename: '/plots/pareto_compression_vs_accuracy.png',
      caption: 'Shows zero-shot SVD degradation and post-SVD AdamW recovery exceeding the uncompressed baseline (+1.11% boost at k=8).'
    },
    {
      id: 'roofline',
      title: 'Empirical Roofline Model (Operational Arithmetic Intensity)',
      filename: '/plots/empirical_roofline.png',
      caption: 'Visualizes the operational trajectory from memory-bound (<0.50 FLOP/Byte @ B=1) to compute-bound (>38.4 FLOP/Byte @ B=128).'
    },
    {
      id: 'image_grid',
      title: 'Phase 1: 2D Image SVD Reconstruction Grid',
      filename: '/plots/svd_image_reconstruction_grid.png',
      caption: 'Visual comparison of 512x512 matrix reconstruction across ranks k in {4, 8, 16, 32, 64, 128, 256}.'
    },
    {
      id: 'image_metrics',
      title: 'Phase 1: Spectral Decay & Frobenius Error Tradeoff',
      filename: '/plots/svd_image_metrics_tradeoff.png',
      caption: 'Cumulative singular value energy curve and relative Frobenius error decay curves.'
    }
  ];

  const activePlot = PLOTS.find(p => p.id === selectedPlot) || PLOTS[0];

  return (
    <div className="demo-section">
      <div style={{ marginBottom: 24 }}>
        <h2 style={{ fontSize: '1.4rem', fontWeight: 700, color: '#fff', display: 'flex', alignItems: 'center', gap: 10 }}>
          <span style={{ color: 'var(--accent-indigo)' }}>Artifacts:</span> Empirical Figures & Storage Footprint
        </h2>
        <p style={{ color: 'var(--text-muted)', fontSize: '0.88rem' }}>
          Inspect high-resolution publication plots and verified PyTorch checkpoint weight storage on disk.
        </p>
      </div>

      {/* Plot Viewer Tabs */}
      <div className="glass-panel" style={{ padding: 20, marginBottom: 24 }}>
        <div style={{ display: 'flex', gap: 8, overflowX: 'auto', marginBottom: 20, paddingBottom: 4 }}>
          {PLOTS.map((plot) => (
            <button
              key={plot.id}
              id={`btn-plot-${plot.id}`}
              className={`btn btn-pill ${selectedPlot === plot.id ? 'btn-primary' : 'btn-secondary'}`}
              onClick={() => setSelectedPlot(plot.id)}
            >
              <ImageIcon size={14} />
              {plot.title.split(':')[0]}
            </button>
          ))}
        </div>

        {/* Display Active Plot */}
        <div style={{ background: '#070a12', borderRadius: 14, overflow: 'hidden', border: '1px solid var(--border-subtle)', padding: 16 }}>
          <div style={{ marginBottom: 12, display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
            <h3 style={{ fontSize: '1.05rem', color: '#fff' }}>{activePlot.title}</h3>
            <a 
              href={activePlot.filename} 
              target="_blank" 
              rel="noreferrer"
              className="badge badge-cyan"
              style={{ textDecoration: 'none' }}
            >
              Open Full Res <ExternalLink size={12} />
            </a>
          </div>

          <div style={{ width: '100%', minHeight: 320, display: 'flex', justifyContent: 'center', alignItems: 'center', background: '#000', borderRadius: 8, overflow: 'hidden' }}>
            <img 
              src={activePlot.filename} 
              alt={activePlot.title} 
              style={{ maxWidth: '100%', maxHeight: '550px', objectFit: 'contain' }}
              onError={(e) => {
                e.target.style.display = 'none';
                e.target.parentNode.innerHTML = `<div style="padding: 40px; text-align: center; color: var(--text-muted)">Plot rendered at: <code>results/plots/${activePlot.filename.split('/').pop()}</code></div>`;
              }}
            />
          </div>

          <div style={{ marginTop: 12, fontSize: '0.88rem', color: 'var(--text-muted)' }}>
            <strong>Analysis:</strong> {activePlot.caption}
          </div>
        </div>
      </div>

      {/* Checkpoint Disk Storage Footprint Table */}
      <div className="glass-panel" style={{ padding: 24 }}>
        <h3 style={{ fontSize: '1.15rem', color: '#fff', marginBottom: 16, display: 'flex', alignItems: 'center', gap: 8 }}>
          <HardDrive size={18} color="var(--accent-emerald)" />
          Persisted PyTorch Checkpoints on Disk (results/checkpoints/)
        </h3>

        <div className="custom-table-container">
          <table className="custom-table">
            <thead>
              <tr>
                <th>Checkpoint File (.pt)</th>
                <th>File Size on Disk</th>
                <th>Model Architecture & Rank</th>
                <th>Compression Factor</th>
              </tr>
            </thead>
            <tbody>
              {CHECKPOINTS_INFO.map((ckpt) => (
                <tr key={ckpt.name} style={{ background: ckpt.rank === 'k=8' ? 'rgba(0, 240, 255, 0.08)' : 'transparent' }}>
                  <td>
                    <code style={{ fontFamily: 'var(--font-mono)', color: ckpt.rank === 'k=8' ? 'var(--accent-cyan)' : 'inherit' }}>
                      {ckpt.name}
                    </code>
                  </td>
                  <td>
                    <strong style={{ color: ckpt.size.includes('KB') ? 'var(--accent-emerald)' : 'inherit' }}>
                      {ckpt.size}
                    </strong>
                  </td>
                  <td>{ckpt.type}</td>
                  <td>
                    <span className="badge badge-purple">{ckpt.compression}</span>
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </div>
    </div>
  );
}
