import React, { useState } from 'react';
import { 
  Cpu, 
  Layers, 
  Sliders, 
  Target, 
  BookOpen, 
  Activity, 
  Sparkles,
  Zap,
  HardDrive,
  Shield
} from 'lucide-react';

import ImageSvdDemo from './components/ImageSvdDemo';
import NeuralLayerSvdDemo from './components/NeuralLayerSvdDemo';
import HardwareLatencyProfiler from './components/HardwareLatencyProfiler';
import AccuracyParetoDemo from './components/AccuracyParetoDemo';
import BenchmarkExplorer from './components/BenchmarkExplorer';
import TheoryAndLLMThesis from './components/TheoryAndLLMThesis';
import TeacherVivaDefense from './components/TeacherVivaDefense';

export default function App() {
  const [activeTab, setActiveTab] = useState('hardware_climax');

  const NAV_ITEMS = [
    { id: 'hardware_climax', label: 'Hardware Climax (Memory Wall)', icon: Activity, badge: 'Crucial' },
    { id: 'image_svd', label: 'Phase 1: Image SVD Lab', icon: Sliders },
    { id: 'neural_layer', label: 'Phase 2: Layer Factorization', icon: Layers },
    { id: 'accuracy_pareto', label: 'Phase 3: Accuracy Recovery', icon: Target },
    { id: 'benchmark_artifacts', label: 'Empirical Figures & Plots', icon: HardDrive },
    { id: 'theory_thesis', label: 'LLM Thesis & Math Theory', icon: BookOpen },
    { id: 'viva_guide', label: 'Viva & Demo Defense Playbook', icon: Shield, badge: 'Defense' }
  ];

  return (
    <div className="app-container">
      
      {/* Top Header & HUD */}
      <header className="header-hud glass-panel">
        <div className="brand-wrapper">
          <div className="brand-icon-box">
            <Cpu size={28} />
          </div>
          <div className="brand-titles">
            <h1>Low-Rank SVD Neural Compression & Hardware Profiling</h1>
            <p className="brand-subtitle">
              Post-Training Truncated SVD on Dense Projection Layers • Hardware Profiling & LLM Memory-Wall Synthesis
            </p>
          </div>
        </div>
      </header>

      {/* Global Quick KPI Summary Banner */}
      <section className="kpi-grid">
        <div className="glass-panel kpi-card">
          <div className="kpi-header">
            <span>MAX PARAMETER COMPRESSION</span>
            <Sparkles size={16} color="var(--accent-cyan)" />
          </div>
          <div className="kpi-value" style={{ color: 'var(--accent-cyan)' }}>
            39.25x
          </div>
          <div className="kpi-footer">
            1.46M down to 37.3K parameters at rank k=8
          </div>
        </div>

        <div className="glass-panel kpi-card">
          <div className="kpi-header">
            <span>ACCURACY RECOVERY BOOST</span>
            <Target size={16} color="var(--accent-emerald)" />
          </div>
          <div className="kpi-value" style={{ color: 'var(--accent-emerald)' }}>
            +1.11%
          </div>
          <div className="kpi-footer">
            Baseline 86.18% → Fine-tuned 87.29% via low-rank regularizer
          </div>
        </div>

        <div className="glass-panel kpi-card">
          <div className="kpi-header">
            <span>LATENCY AT B=1 (MEMORY WALL)</span>
            <Activity size={16} color="var(--accent-amber)" />
          </div>
          <div className="kpi-value" style={{ color: 'var(--accent-amber)' }}>
            0.89x
          </div>
          <div className="kpi-footer">
            11% slowdown (0.444ms → 0.498ms) due to kernel launch & DRAM round-trips
          </div>
        </div>

        <div className="glass-panel kpi-card">
          <div className="kpi-header">
            <span>COMPUTE SPEEDUP AT B=128</span>
            <Zap size={16} color="var(--accent-cyan)" />
          </div>
          <div className="kpi-value" style={{ color: 'var(--accent-cyan)' }}>
            2.81x
          </div>
          <div className="kpi-footer">
            1.184ms down to 0.421ms in compute-bound parallel math
          </div>
        </div>
      </section>

      {/* Navigation Pills Bar */}
      <nav className="nav-bar">
        {NAV_ITEMS.map((item) => {
          const Icon = item.icon;
          const isActive = activeTab === item.id;
          return (
            <button
              key={item.id}
              id={`nav-tab-${item.id}`}
              className={`nav-item ${isActive ? 'active' : ''}`}
              onClick={() => setActiveTab(item.id)}
            >
              <Icon size={16} />
              <span>{item.label}</span>
              {item.badge && (
                <span style={{
                  fontSize: '0.68rem',
                  padding: '2px 6px',
                  borderRadius: 4,
                  background: isActive ? 'rgba(255,255,255,0.2)' : 'rgba(0, 240, 255, 0.15)',
                  color: isActive ? '#fff' : 'var(--accent-cyan)',
                  marginLeft: 4
                }}>
                  {item.badge}
                </span>
              )}
            </button>
          );
        })}
      </nav>

      {/* Main Tab Content View */}
      <main style={{ minHeight: '600px' }}>
        {activeTab === 'hardware_climax' && <HardwareLatencyProfiler />}
        {activeTab === 'image_svd' && <ImageSvdDemo />}
        {activeTab === 'neural_layer' && <NeuralLayerSvdDemo />}
        {activeTab === 'accuracy_pareto' && <AccuracyParetoDemo />}
        {activeTab === 'benchmark_artifacts' && <BenchmarkExplorer />}
        {activeTab === 'theory_thesis' && <TheoryAndLLMThesis />}
        {activeTab === 'viva_guide' && <TeacherVivaDefense />}
      </main>

      {/* Footer */}
      <footer style={{
        marginTop: 48,
        padding: '24px 0',
        borderTop: '1px solid var(--border-subtle)',
        display: 'flex',
        justifyContent: 'space-between',
        alignItems: 'center',
        flexWrap: 'wrap',
        gap: 16,
        color: 'var(--text-dim)',
        fontSize: '0.85rem'
      }}>
        <div>
          Capstone Research & Evaluation Platform • Verified against 70 experimental hardware profiling runs.
        </div>
      </footer>
    </div>
  );
}
