import React, { useState } from 'react';
import { Activity, AlertTriangle, CheckCircle, TrendingDown, Cpu, BarChart3, HelpCircle } from 'lucide-react';
import { END_TO_END_LATENCY } from '../data/benchmarkData';

export default function HardwareLatencyProfiler() {
  const [selectedBatch, setSelectedBatch] = useState(1);
  const [selectedRank, setSelectedRank] = useState(8);

  const currentBatchData = END_TO_END_LATENCY.find(b => b.batchSize === selectedBatch) || END_TO_END_LATENCY[0];
  const rankKey = `k${selectedRank}`;
  const currentRankData = currentBatchData[rankKey] || currentBatchData.k8;

  const baselineMs = currentBatchData.baselineMs;
  const decompMs = currentRankData.ms;
  const speedup = currentRankData.speedup;
  const isSlowdown = speedup < 1.0;

  // Arithmetic Intensity estimation based on batch size
  const getArithmeticIntensity = (b) => {
    switch (b) {
      case 1: return { val: 0.44, regime: 'Memory-Bandwidth Bound', color: 'var(--accent-amber)' };
      case 8: return { val: 3.51, regime: 'Transition Knee', color: 'var(--accent-purple)' };
      case 32: return { val: 10.37, regime: 'Balanced Regime', color: 'var(--accent-indigo)' };
      case 64: return { val: 20.26, regime: 'Compute Saturated', color: 'var(--accent-emerald)' };
      case 128: return { val: 38.40, regime: 'Compute-Bound Parallel Matrix Math', color: 'var(--accent-cyan)' };
      default: return { val: 0.5, regime: 'Memory Bound', color: 'var(--accent-amber)' };
    }
  };

  const aiInfo = getArithmeticIntensity(selectedBatch);

  return (
    <div className="demo-section">
      <div style={{ marginBottom: 20 }}>
        <h2 style={{ fontSize: '1.4rem', fontWeight: 700, color: '#fff', display: 'flex', alignItems: 'center', gap: 10 }}>
          <span style={{ color: 'var(--accent-rose)' }}>The Climax:</span> Hardware Profiler & The Memory Wall
        </h2>
        <p style={{ color: 'var(--text-muted)', fontSize: '0.88rem' }}>
          Empirically characterizes the divergence between theoretical FLOP reductions and actual wall-clock execution time on hardware accelerators.
        </p>
      </div>

      {/* Interactive Controls Bar */}
      <div className="glass-panel" style={{ padding: 20, marginBottom: 24 }}>
        <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: 24, flexWrap: 'wrap' }}>
          
          {/* Batch Size Selector */}
          <div>
            <label style={{ display: 'block', fontSize: '0.85rem', fontWeight: 600, color: 'var(--text-muted)', marginBottom: 8 }}>
              Select Batch Size (B_sz):
            </label>
            <div style={{ display: 'flex', gap: 8 }}>
              {[1, 8, 32, 64, 128].map((b) => (
                <button
                  key={b}
                  id={`btn-hw-batch-${b}`}
                  className={`btn btn-pill ${selectedBatch === b ? 'btn-primary' : 'btn-secondary'}`}
                  style={{ flex: 1, padding: '8px 0' }}
                  onClick={() => setSelectedBatch(b)}
                >
                  B={b}
                </button>
              ))}
            </div>
          </div>

          {/* Rank Selector */}
          <div>
            <label style={{ display: 'block', fontSize: '0.85rem', fontWeight: 600, color: 'var(--text-muted)', marginBottom: 8 }}>
              Select Truncated Rank ($K$):
            </label>
            <div style={{ display: 'flex', gap: 8 }}>
              {[8, 16, 32, 64, 128, 256].map((k) => (
                <button
                  key={k}
                  id={`btn-hw-rank-${k}`}
                  className={`btn btn-pill ${selectedRank === k ? 'btn-primary' : 'btn-secondary'}`}
                  style={{ flex: 1, padding: '8px 0' }}
                  onClick={() => setSelectedRank(k)}
                >
                  k={k}
                </button>
              ))}
            </div>
          </div>
        </div>
      </div>

      {/* Headliner Climax Visualizer */}
      <div className="glass-panel glass-panel-glow" style={{ padding: 24, marginBottom: 24 }}>
        <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', flexWrap: 'wrap', gap: 16, marginBottom: 20 }}>
          <div>
            <span className="badge badge-cyan" style={{ marginBottom: 6 }}>
              Synchronized Apple MPS / CUDA Benchmark
            </span>
            <h3 style={{ fontSize: '1.2rem', color: '#fff' }}>
              Batch Size B={selectedBatch} | Rank k={selectedRank} Speedup Analysis
            </h3>
          </div>

          {/* Speedup Badge Indicator */}
          <div style={{
            display: 'flex',
            alignItems: 'center',
            gap: 12,
            padding: '10px 20px',
            borderRadius: 14,
            background: isSlowdown ? 'rgba(245, 158, 11, 0.15)' : 'rgba(16, 185, 129, 0.15)',
            border: `1px solid ${isSlowdown ? 'rgba(245, 158, 11, 0.4)' : 'rgba(16, 185, 129, 0.4)'}`
          }}>
            {isSlowdown ? (
              <AlertTriangle size={24} color="var(--accent-amber)" />
            ) : (
              <CheckCircle size={24} color="var(--accent-emerald)" />
            )}
            <div>
              <div style={{ fontSize: '0.75rem', color: 'var(--text-muted)', textTransform: 'uppercase' }}>
                WALL-CLOCK SPEEDUP
              </div>
              <div style={{
                fontSize: '1.5rem',
                fontWeight: 700,
                color: isSlowdown ? 'var(--accent-amber)' : 'var(--accent-emerald)',
                fontFamily: 'var(--font-mono)'
              }}>
                {speedup.toFixed(2)}x {isSlowdown ? '(Slowdown)' : '(Speedup)'}
              </div>
            </div>
          </div>
        </div>

        {/* Latency Comparison Visual Bars */}
        <div style={{ display: 'flex', flexDirection: 'column', gap: 16, padding: '10px 0' }}>
          
          {/* Baseline Bar */}
          <div>
            <div style={{ display: 'flex', justifyContent: 'space-between', fontSize: '0.85rem', marginBottom: 6 }}>
              <span style={{ fontWeight: 600 }}>Uncompressed BaselineMLP</span>
              <span style={{ fontFamily: 'var(--font-mono)', color: 'var(--text-muted)' }}>{baselineMs.toFixed(3)} ms</span>
            </div>
            <div style={{ background: 'var(--bg-tertiary)', height: 24, borderRadius: 6, overflow: 'hidden', position: 'relative' }}>
              <div style={{
                width: `${Math.min(100, (baselineMs / Math.max(baselineMs, decompMs)) * 100)}%`,
                height: '100%',
                background: 'linear-gradient(90deg, #64748b, #94a3b8)',
                borderRadius: 6
              }} />
            </div>
          </div>

          {/* Decomposed Bar */}
          <div>
            <div style={{ display: 'flex', justifyContent: 'space-between', fontSize: '0.85rem', marginBottom: 6 }}>
              <span style={{ fontWeight: 600, color: isSlowdown ? 'var(--accent-amber)' : 'var(--accent-cyan)' }}>
                DecomposedMLP (k={selectedRank})
              </span>
              <span style={{ fontFamily: 'var(--font-mono)', color: isSlowdown ? 'var(--accent-amber)' : 'var(--accent-cyan)' }}>
                {decompMs.toFixed(3)} ms
              </span>
            </div>
            <div style={{ background: 'var(--bg-tertiary)', height: 24, borderRadius: 6, overflow: 'hidden', position: 'relative' }}>
              <div style={{
                width: `${Math.min(100, (decompMs / Math.max(baselineMs, decompMs)) * 100)}%`,
                height: '100%',
                background: isSlowdown 
                  ? 'linear-gradient(90deg, #f59e0b, #d97706)' 
                  : 'linear-gradient(90deg, var(--accent-cyan), var(--accent-indigo))',
                borderRadius: 6
              }} />
            </div>
          </div>
        </div>

        {/* Contextual Warning / Insight Alert */}
        <div style={{ marginTop: 20 }}>
          {selectedBatch === 1 ? (
            <div className="alert-box alert-amber">
              <AlertTriangle size={22} color="var(--accent-amber)" style={{ flexShrink: 0, marginTop: 2 }} />
              <div>
                <strong>The Memory-Bandwidth Wall in Action:</strong> Even though rank k=8 cuts theoretical FLOPs by <strong>71.2x</strong>, wall-clock latency regresses from <strong>0.444 ms to 0.498 ms (11% slowdown)</strong>!
                At B_sz = 1, arithmetic intensity is only <strong>0.44 FLOP/Byte</strong>. The GPU is idling, waiting on DRAM memory transfers and paying overhead for launching two sequential GEMM kernels.
              </div>
            </div>
          ) : selectedBatch >= 64 ? (
            <div className="alert-box alert-emerald">
              <CheckCircle size={22} color="var(--accent-emerald)" style={{ flexShrink: 0, marginTop: 2 }} />
              <div>
                <strong>Compute-Bound Parallel Scaling:</strong> At batch size B_sz = {selectedBatch}, operational arithmetic intensity rises to <strong>{aiInfo.val} FLOP/Byte</strong>, crossing into the compute-bound zone under the Roofline model. High batch parallelism hides memory fetch latency, translating arithmetic reductions into a massive <strong>{speedup.toFixed(2)}x wall-clock speedup</strong>!
              </div>
            </div>
          ) : (
            <div className="alert-box alert-cyan">
              <Activity size={22} color="var(--accent-cyan)" style={{ flexShrink: 0, marginTop: 2 }} />
              <div>
                <strong>Transition Knee:</strong> At B_sz = {selectedBatch}, the system is transitioning away from the memory wall. Kernel overhead is partially amortized across batch items, yielding a modest {speedup.toFixed(2)}x speedup.
              </div>
            </div>
          )}
        </div>
      </div>

      {/* Latency Matrix Across All Batch Sizes Table */}
      <div className="glass-panel" style={{ padding: 20, marginBottom: 20 }}>
        <h3 style={{ fontSize: '1.05rem', marginBottom: 14, display: 'flex', alignItems: 'center', gap: 8 }}>
          <BarChart3 size={18} color="var(--accent-indigo)" />
          Complete Empirical Latency Matrix (Wall-Clock ms & Speedup)
        </h3>

        <div className="custom-table-container">
          <table className="custom-table">
            <thead>
              <tr>
                <th>Batch Size</th>
                <th>Baseline Latency</th>
                <th>k=8 Speedup</th>
                <th>k=16 Speedup</th>
                <th>k=32 Speedup</th>
                <th>k=64 Speedup</th>
                <th>k=128 Speedup</th>
              </tr>
            </thead>
            <tbody>
              {END_TO_END_LATENCY.map((row) => (
                <tr key={row.batchSize} style={{ background: row.batchSize === selectedBatch ? 'rgba(0, 240, 255, 0.08)' : 'transparent' }}>
                  <td>
                    <strong>B = {row.batchSize}</strong>
                  </td>
                  <td>{row.baselineMs.toFixed(3)} ms</td>
                  <td>
                    <span className={`badge ${row.k8.speedup < 1.0 ? 'badge-amber' : 'badge-emerald'}`}>
                      {row.k8.ms.toFixed(3)} ms ({row.k8.speedup}x)
                    </span>
                  </td>
                  <td>
                    <span className={`badge ${row.k16.speedup < 1.0 ? 'badge-amber' : 'badge-emerald'}`}>
                      {row.k16.ms.toFixed(3)} ms ({row.k16.speedup}x)
                    </span>
                  </td>
                  <td>
                    <span className={`badge ${row.k32.speedup < 1.0 ? 'badge-amber' : 'badge-emerald'}`}>
                      {row.k32.ms.toFixed(3)} ms ({row.k32.speedup}x)
                    </span>
                  </td>
                  <td>
                    <span className={`badge ${row.k64.speedup < 1.0 ? 'badge-amber' : 'badge-emerald'}`}>
                      {row.k64.ms.toFixed(3)} ms ({row.k64.speedup}x)
                    </span>
                  </td>
                  <td>
                    <span className={`badge ${row.k128.speedup < 1.0 ? 'badge-amber' : 'badge-emerald'}`}>
                      {row.k128.ms.toFixed(3)} ms ({row.k128.speedup}x)
                    </span>
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
