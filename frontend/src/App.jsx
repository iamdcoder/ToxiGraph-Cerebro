import AudioRecorder from "./components/AudioRecorder";
import LiveEmotionPanel from "./components/LiveEmotionPanel";
import LiveMultiSpeakerPanel from "./components/LiveMultiSpeakerPanel";
import RecordingComparisonLab from "./components/RecordingComparisonLab";
import EvaluationDashboard from "./components/EvaluationDashboard";
import RobustnessDashboard from "./components/RobustnessDashboard";
import ModelOptimizationPanel from "./components/ModelOptimizationPanel";
import SpeakerAwareAnalysis from "./components/SpeakerAwareAnalysis";

export default function App() {
  return (
    <div className="app-shell">
      <header className="topbar">
        <div className="brand">
          <div className="brand-mark" aria-hidden="true">
            <span />
            <span />
            <span />
            <span />
          </div>
          <div>
            <div className="brand-name">CEREBRO</div>
            <div className="brand-subtitle">SPEECH EMOTION INTELLIGENCE</div>
          </div>
        </div>

        <div className="topbar-meta">
          <span className="build-chip">ALGOTHON'26</span>
          <span className="build-chip muted">LIVE MULTI-SPEAKER + LONGITUDINAL INTELLIGENCE</span>
        </div>
      </header>

      <main className="page-shell">
        <section className="hero">
          <div>
            <p className="eyebrow">CEREBRO · SPEECH EMOTION INTELLIGENCE</p>
            <h1>Understand emotion <span>through the voice.</span></h1>
            <p className="hero-copy">
              CEREBRO analyzes how something is spoken, not just what was said.
              It combines acoustic and deep speech models, calibrated fusion, temporal affect analysis,
              speaker-aware conversation intelligence, personal voice history, and real-time multi-speaker inference.
            </p>
          </div>
          <div className="architecture-card">
            <span className="eyebrow">CURRENT PIPELINE</span>
            <div className="pipeline-line"><span>MIC</span><i>→</i><span>WAV</span><i>→</i><span>FUSION</span><i>→</i><span>TIMELINE</span><i>→</i><span>COMPARE</span></div>
            <p>A calibrated acoustic + deep speech fusion now feeds dimensional affect estimates, evidence-backed explanations, uncertainty analysis, overlapping-window temporal emotion analysis, local ASR content verification, controlled comparison workflows, speaker-aware interaction graphs, conversation state analysis, longitudinal voice intelligence, and real-data model optimization workflows.</p>
          </div>
        </section>

        <LiveEmotionPanel />
        <LiveMultiSpeakerPanel />
        <AudioRecorder />
        <RecordingComparisonLab />
        <SpeakerAwareAnalysis />
        <EvaluationDashboard />
        <ModelOptimizationPanel />
        <RobustnessDashboard />
      </main>
    </div>
  );
}
