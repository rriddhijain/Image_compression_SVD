import React, { useState } from 'react';
import { Target, Award, Sparkles, TrendingUp, ShieldCheck, CheckCircle2 } from 'lucide-react';
import confetti from 'canvas-confetti';
import { ACCURACY_DATA } from '../data/benchmarkData';

export default function AccuracyParetoDemo() {
  const [selectedRank, setSelectedRank] = useState(8);

  const selectedData = ACCURACY_DATA.find(d => d.rank === selectedRank) || ACCURACY_DATA[1];
  const baseline = ACCURACY_DATA[0];

  const triggerConfetti = () => {
    confetti({
      particleCount: 80,
      spread: 70,
      origin: { y: 0.6 }
    });
  };

  return (
    <div className="demo-section">
      <div style={{ marginBottom: 20 }}>
        <h2 style={{ fontSize: '1.4rem', fontWeight: 700, color: '#fff', display: 'flex', alignItems: 'center', gap: 10 }}>
          <span style={{ color: 'var(--accent-emerald)' }}>Phase 3:</span> Accuracy Recovery & Pareto Frontier
        </h2>
        <p style={{ color: 'var(--text-muted)', fontSize: '0.88rem' }}>
          Demonstrates post-SVD recovery fine-tuning (Sainath et al., 2013): recovering lost zero-shot accuracy and achieving +1.11% accuracy boost via low-rank regularization.
        </p>
      </div>

      {/* Rank Selector */}
      <div className="glass-panel" style={{ padding: 20, marginBottom: 24 }}>
        <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', flexWrap: 'wrap', gap: 16 }}>
          <div>
            <label style={{ display: 'block', fontSize: '0.85rem', fontWeight: 600, color: 'var(--text-muted)', marginBottom: 8 }}>
              Select Truncated Rank to Inspect Recovery:
            </label>
            <div style={{ display: 'flex', gap: 8, flexWrap: 'wrap' }}>
              {ACCURACY_DATA.filter(d => d.isDecomposed).map((d) => (
                <button
                  key={d.rank}
                  id={`btn-acc-rank-${d.rank}`}
                  className={`btn btn-pill ${selectedRank === d.rank ? 'btn-primary' : 'btn-secondary'}`}
                  onClick={() => setSelectedRank(d.rank)}
                >
                  k = {d.rank} ({d.compressionRatio.toFixed(1)}x)
                </button>
              ))}
            </div>
          </div>

          <button
            id="btn-celebrate-recovery"
            className="btn btn-secondary"
            onClick={triggerConfetti}
            style={{ display: 'flex', alignItems: 'center', gap: 8 }}
          >
            <Sparkles size={16} color="var(--accent-cyan)" />
            Celebrate Recovery Boost
          </button>
        </div>
      </div>

      {/* Before / After Accuracy Recovery Hero */}
      <div className="glass-panel glass-panel-glow" style={{ padding: 28, marginBottom: 24 }}>
        <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(260px, 1fr))', gap: 24, alignItems: 'center' }}>
          
          {/* Uncompressed Baseline */}
          <div style={{
            background: 'rgba(255, 255, 255, 0.03)',
            borderRadius: 14,
            padding: 20,
            border: '1px solid var(--border-subtle)',
            textAlign: 'center'
          }}>
            <div style={{ fontSize: '0.78rem', color: 'var(--text-muted)', textTransform: 'uppercase', marginBottom: 6 }}>
              Baseline Model (Dense)
            </div>
            <div style={{ fontSize: '2.2rem', fontWeight: 700, color: '#fff', fontFamily: 'var(--font-mono)' }}>
              {baseline.finetunedAcc}%
            </div>
            <div style={{ fontSize: '0.85rem', color: 'var(--text-dim)', marginTop: 4 }}>
              1,462,538 Parameters (1.0x)
            </div>
          </div>

          {/* Zero-Shot Post-SVD Drop */}
          <div style={{
            background: 'rgba(245, 158, 11, 0.08)',
            borderRadius: 14,
            padding: 20,
            border: '1px solid rgba(245, 158, 11, 0.3)',
            textAlign: 'center'
          }}>
            <div style={{ fontSize: '0.78rem', color: 'var(--accent-amber)', textTransform: 'uppercase', marginBottom: 6 }}>
              Zero-Shot SVD Drop (k={selectedRank})
            </div>
            <div style={{ fontSize: '2.2rem', fontWeight: 700, color: 'var(--accent-amber)', fontFamily: 'var(--font-mono)' }}>
              {selectedData.zeroShotAcc}%
            </div>
            <div style={{ fontSize: '0.85rem', color: '#fef3c7', marginTop: 4 }}>
              -{(baseline.finetunedAcc - selectedData.zeroShotAcc).toFixed(2)}% drop before fine-tuning
            </div>
          </div>

          {/* Fine-Tuned Recovery (The Winner) */}
          <div style={{
            background: 'rgba(16, 185, 129, 0.12)',
            borderRadius: 14,
            padding: 20,
            border: '1px solid rgba(16, 185, 129, 0.4)',
            textAlign: 'center'
          }}>
            <div style={{ fontSize: '0.78rem', color: 'var(--accent-emerald)', textTransform: 'uppercase', marginBottom: 6 }}>
              Fine-Tuned Accuracy (AdamW)
            </div>
            <div style={{ fontSize: '2.2rem', fontWeight: 700, color: 'var(--accent-emerald)', fontFamily: 'var(--font-mono)' }}>
              {selectedData.finetunedAcc}%
            </div>
            <div style={{ fontSize: '0.85rem', color: '#a7f3d0', marginTop: 4 }}>
              <strong>{selectedData.delta}</strong> net improvement vs baseline!
            </div>
          </div>
        </div>

        {/* Regularization Insight Callout */}
        <div style={{ marginTop: 24, background: 'rgba(0,0,0,0.3)', padding: 18, borderRadius: 12, border: '1px solid var(--border-subtle)' }}>
          <div style={{ display: 'flex', alignItems: 'center', gap: 10, marginBottom: 6 }}>
            <Award size={18} color="var(--accent-cyan)" />
            <strong style={{ color: '#fff' }}>Why does rank k=8 beat the uncompressed baseline (+1.11%)?</strong>
          </div>
          <p style={{ fontSize: '0.88rem', color: 'var(--text-muted)', lineHeight: 1.6 }}>
            In overparameterized deep networks, tail singular values represent high-frequency noise and sample-specific memorization.
            Truncating the tail to rank $k=8$ forces representations into a low-dimensional subspace, providing an <strong>implicit low-rank regularizer</strong> (Sainath et al., 2013).
            When the factorized layers $L_1$ and $L_2$ are fine-tuned with AdamW and cosine learning rate decay, the model optimizes solely within this clean semantic subspace, preventing overfitting and yielding higher test generalization.
          </p>
        </div>
      </div>

      {/* Accuracy Pareto Table Across All Ranks */}
      <div className="glass-panel" style={{ padding: 20, marginBottom: 24 }}>
        <h3 style={{ fontSize: '1.05rem', marginBottom: 14, display: 'flex', alignItems: 'center', gap: 8 }}>
          <Target size={18} color="var(--accent-indigo)" />
          Complete Accuracy Retention & Parameter Compression Table
        </h3>

        <div className="custom-table-container">
          <table className="custom-table">
            <thead>
              <tr>
                <th>Model Architecture</th>
                <th>Parameter Count</th>
                <th>Compression Ratio</th>
                <th>Zero-Shot Acc</th>
                <th>Fine-Tuned Acc</th>
                <th>Net Accuracy Delta</th>
              </tr>
            </thead>
            <tbody>
              {ACCURACY_DATA.map((row) => (
                <tr key={row.rank} style={{ background: row.rank === selectedRank ? 'rgba(0, 240, 255, 0.08)' : 'transparent' }}>
                  <td>
                    <strong>{row.rank === 'Baseline' ? 'Uncompressed BaselineMLP' : `DecomposedMLP (k=${row.rank})`}</strong>
                  </td>
                  <td>{row.params.toLocaleString()}</td>
                  <td>
                    <span className="badge badge-purple">{row.compressionRatio.toFixed(2)}x</span>
                  </td>
                  <td>{row.zeroShotAcc}%</td>
                  <td>
                    <strong style={{ color: 'var(--accent-emerald)' }}>{row.finetunedAcc}%</strong>
                  </td>
                  <td>
                    <span className={`badge ${row.delta.startsWith('+') && row.rank !== 'Baseline' ? 'badge-emerald' : 'badge-cyan'}`}>
                      {row.delta}
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
