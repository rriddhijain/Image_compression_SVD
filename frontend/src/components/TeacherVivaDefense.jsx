import React, { useState } from 'react';
import { HelpCircle, ChevronDown, ChevronUp, Copy, Check, Terminal, Clock, Mic, Users, BookOpen, Shield } from 'lucide-react';
import { VIVA_QUESTIONS, DEMO_SCRIPT_STEPS } from '../data/benchmarkData';

export default function TeacherVivaDefense() {
  const [activeStep, setActiveStep] = useState(1);
  const [copiedIndex, setCopiedIndex] = useState(null);
  const [openQuestionIndex, setOpenQuestionIndex] = useState(0);

  const copyToClipboard = (text, idx) => {
    navigator.clipboard.writeText(text);
    setCopiedIndex(idx);
    setTimeout(() => setCopiedIndex(null), 2000);
  };

  return (
    <div className="demo-section">
      <div style={{ marginBottom: 24 }}>
        <h2 style={{ fontSize: '1.4rem', fontWeight: 700, color: '#fff', display: 'flex', alignItems: 'center', gap: 10 }}>
          <span style={{ color: 'var(--accent-cyan)' }}>Viva Playbook:</span> Teacher Demo & Defense Guide
        </h2>
        <p style={{ color: 'var(--text-muted)', fontSize: '0.88rem' }}>
          Equips you with a 2-minute live demo protocol, speaker transitions, and professor-grade answers for high-stakes capstone evaluation.
        </p>
      </div>

      {/* 2-Minute Live Demo Protocol */}
      <div className="glass-panel glass-panel-glow" style={{ padding: 24, marginBottom: 28 }}>
        <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', flexWrap: 'wrap', gap: 12, marginBottom: 16 }}>
          <h3 style={{ fontSize: '1.15rem', color: '#fff', display: 'flex', alignItems: 'center', gap: 8 }}>
            <Terminal size={18} color="var(--accent-cyan)" />
            The 2-Minute Professor Live Demonstration Protocol
          </h3>
          <span className="badge badge-cyan">
            <Clock size={12} /> Total Duration: ~1 min 20 sec
          </span>
        </div>

        {/* Step Selector Pills */}
        <div style={{ display: 'flex', gap: 8, overflowX: 'auto', marginBottom: 20, paddingBottom: 4 }}>
          {DEMO_SCRIPT_STEPS.map((s) => (
            <button
              key={s.step}
              id={`btn-demo-step-${s.step}`}
              className={`btn btn-pill ${activeStep === s.step ? 'btn-primary' : 'btn-secondary'}`}
              onClick={() => setActiveStep(s.step)}
            >
              Step {s.step} ({s.duration})
            </button>
          ))}
        </div>

        {/* Active Step Card */}
        {(() => {
          const s = DEMO_SCRIPT_STEPS.find(item => item.step === activeStep) || DEMO_SCRIPT_STEPS[0];
          return (
            <div style={{ background: 'rgba(0,0,0,0.3)', padding: 20, borderRadius: 14, border: '1px solid var(--border-subtle)' }}>
              <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: 12 }}>
                <strong style={{ fontSize: '1.05rem', color: '#fff' }}>
                  Step {s.step}: {s.title}
                </strong>
                <span className="badge badge-purple">{s.duration}</span>
              </div>

              {/* Terminal Command */}
              <div style={{ marginBottom: 16 }}>
                <div style={{ fontSize: '0.78rem', color: 'var(--text-muted)', marginBottom: 4, textTransform: 'uppercase' }}>
                  Execute in Terminal:
                </div>
                <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', background: '#070b12', padding: '10px 16px', borderRadius: 8, border: '1px solid var(--border-subtle)' }}>
                  <code style={{ fontFamily: 'var(--font-mono)', color: 'var(--accent-cyan)', fontSize: '0.88rem' }}>
                    {s.command}
                  </code>
                  <button
                    id={`btn-copy-step-${s.step}`}
                    onClick={() => copyToClipboard(s.command, s.step)}
                    style={{ background: 'transparent', border: 'none', color: copiedIndex === s.step ? 'var(--accent-emerald)' : 'var(--text-muted)', cursor: 'pointer' }}
                    title="Copy command"
                  >
                    {copiedIndex === s.step ? <Check size={16} /> : <Copy size={16} />}
                  </button>
                </div>
              </div>

              {/* Speech Cue */}
              <div style={{ background: 'rgba(99, 102, 241, 0.1)', borderLeft: '4px solid var(--accent-indigo)', padding: '12px 16px', borderRadius: '0 8px 8px 0' }}>
                <div style={{ display: 'flex', alignItems: 'center', gap: 6, fontSize: '0.8rem', color: '#c4b5fd', fontWeight: 600, marginBottom: 4 }}>
                  <Mic size={14} /> WHAT TO SAY TO THE PROFESSOR:
                </div>
                <p style={{ fontSize: '0.9rem', color: '#e0e7ff', fontStyle: 'italic', lineHeight: 1.5 }}>
                  "{s.speech}"
                </p>
              </div>
            </div>
          );
        })()}
      </div>

      {/* Role Allocation Breakdown */}
      <div className="glass-panel" style={{ padding: 24, marginBottom: 28 }}>
        <h3 style={{ fontSize: '1.15rem', color: '#fff', marginBottom: 16, display: 'flex', alignItems: 'center', gap: 8 }}>
          <Users size={18} color="var(--accent-purple)" />
          Clear Division of Roles (Who Built What)
        </h3>

        <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(300px, 1fr))', gap: 20 }}>
          
          {/* Person A */}
          <div style={{ background: 'rgba(255,255,255,0.02)', padding: 18, borderRadius: 12, border: '1px solid var(--border-subtle)' }}>
            <span className="badge badge-purple" style={{ marginBottom: 10 }}>Modeling Lead: Person A</span>
            <ul style={{ paddingLeft: 18, fontSize: '0.85rem', color: 'var(--text-muted)', display: 'flex', flexDirection: 'column', gap: 6 }}>
              <li><strong>Baseline Model Definition:</strong> 4-layer BaselineMLP (1.46M params) & SmallCNN.</li>
              <li><strong>DecomposedLinear Layer:</strong> PyTorch SVD layer-swapping module.</li>
              <li><strong>Symmetric Singular Splitting:</strong> Equal sqrt(Σ) allocation between L1 and L2.</li>
              <li><strong>Accuracy Recovery Loop:</strong> Post-SVD AdamW cosine fine-tuning (86.18% → 87.29%).</li>
            </ul>
          </div>

          {/* Person B */}
          <div style={{ background: 'rgba(0, 240, 255, 0.03)', padding: 18, borderRadius: 12, border: '1px solid rgba(0, 240, 255, 0.2)' }}>
            <span className="badge badge-cyan" style={{ marginBottom: 10 }}>Systems Lead: Person B (Riddhi Jain)</span>
            <ul style={{ paddingLeft: 18, fontSize: '0.85rem', color: 'var(--text-muted)', display: 'flex', flexDirection: 'column', gap: 6 }}>
              <li><strong>Hardware Timing Engine:</strong> Jitter-free CUDA Events & MPS stream barriers (banned naive time.time()).</li>
              <li><strong>Analytical Metrics:</strong> Breakeven rank (K* = 341.33) and arithmetic intensity math.</li>
              <li><strong>Experimental Sweep:</strong> 70-configuration parameter sweep across ranks & batch sizes.</li>
              <li><strong>Memory & Roofline Profiling:</strong> Uncovered the B=1 memory wall & 2.81x speedup at B=128.</li>
            </ul>
          </div>
        </div>
      </div>

      {/* 10 Tough Viva Defense Q&A Accordion */}
      <div className="glass-panel" style={{ padding: 24 }}>
        <h3 style={{ fontSize: '1.15rem', color: '#fff', marginBottom: 16, display: 'flex', alignItems: 'center', gap: 8 }}>
          <Shield size={18} color="var(--accent-cyan)" />
          Tough Professor Viva Questions & Model Answers
        </h3>

        <div>
          {VIVA_QUESTIONS.map((item, idx) => (
            <div key={idx} className="accordion-item">
              <button 
                id={`accordion-btn-viva-${idx}`}
                className="accordion-header"
                onClick={() => setOpenQuestionIndex(openQuestionIndex === idx ? null : idx)}
              >
                <div style={{ display: 'flex', alignItems: 'center', gap: 10 }}>
                  <span className="badge badge-cyan" style={{ fontSize: '0.72rem' }}>Q{idx + 1}</span>
                  <span>{item.q}</span>
                </div>
                {openQuestionIndex === idx ? <ChevronUp size={18} /> : <ChevronDown size={18} />}
              </button>

              {openQuestionIndex === idx && (
                <div className="accordion-body">
                  <div style={{ fontSize: '0.75rem', color: 'var(--accent-purple)', fontWeight: 600, textTransform: 'uppercase', marginBottom: 6 }}>
                    SPEAKER: {item.speaker}
                  </div>
                  <p style={{ lineHeight: 1.6, color: '#e5e7eb' }}>
                    {item.answer}
                  </p>
                </div>
              )}
            </div>
          ))}
        </div>
      </div>
    </div>
  );
}
