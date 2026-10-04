import { useMemo } from "react";

function titleCase(value) {
  return String(value || "").replace(/\b\w/g, (letter) => letter.toUpperCase());
}

function signed(value) {
  if (value == null) return "—";
  const number = Number(value);
  return `${number >= 0 ? "+" : ""}${number.toFixed(2)}`;
}

function nodeColor(index) {
  const colors = ["#9a7cff", "#45d8ff", "#45e0ae", "#f7cf55", "#ff7fa0", "#b69cff"];
  return colors[index % colors.length];
}

function layoutNodes(nodes) {
  const count = nodes.length;
  if (count === 1) return [{ ...nodes[0], x: 50, y: 50 }];
  const radius = count <= 4 ? 31 : 36;
  return nodes.map((node, index) => {
    const angle = (-Math.PI / 2) + (index / count) * Math.PI * 2;
    return {
      ...node,
      x: 50 + Math.cos(angle) * radius,
      y: 50 + Math.sin(angle) * radius,
    };
  });
}

function edgePath(source, target) {
  const dx = target.x - source.x;
  const dy = target.y - source.y;
  const distance = Math.max(1, Math.hypot(dx, dy));
  const curve = Math.min(13, Math.max(3, distance * 0.10));
  const mx = (source.x + target.x) / 2;
  const my = (source.y + target.y) / 2;
  const nx = -dy / distance;
  const ny = dx / distance;
  return `M ${source.x} ${source.y} Q ${mx + nx * curve} ${my + ny * curve} ${target.x} ${target.y}`;
}

export default function ConversationInteractionGraph({ graph }) {
  const positionedNodes = useMemo(() => layoutNodes(graph?.nodes || []), [graph]);
  const nodeBySpeaker = useMemo(
    () => new Map(positionedNodes.map((node) => [node.speaker, node])),
    [positionedNodes],
  );
  if (!graph?.nodes?.length) return null;

  const metrics = graph.metrics || {};
  const maxSpeaking = Math.max(...positionedNodes.map((node) => node.speaking_share || 0), 0.01);

  return (
    <section className="interaction-graph-card">
      <div className="section-header-row compact">
        <div>
          <p className="eyebrow">CONVERSATION INTERACTION GRAPH</p>
          <h3>Observed response pathways between speakers</h3>
          <p className="muted-copy">
            Nodes represent speakers. Directed edges represent observed turn-to-turn responses; edge affect changes describe the sequence, not causation.
          </p>
        </div>
        <div className="graph-metric-strip">
          <span><strong>{metrics.node_count ?? 0}</strong> speakers</span>
          <span><strong>{metrics.edge_count ?? 0}</strong> pathways</span>
          <span><strong>{metrics.total_interactions ?? 0}</strong> responses</span>
        </div>
      </div>

      <div className="interaction-graph-layout">
        <div className="interaction-graph-canvas" aria-label="Conversation interaction graph">
          <svg viewBox="0 0 100 100" role="img">
            <defs>
              <marker id="cerebro-arrow" viewBox="0 0 10 10" refX="8.5" refY="5" markerWidth="3.2" markerHeight="3.2" orient="auto-start-reverse">
                <path d="M 0 0 L 10 5 L 0 10 z" fill="rgba(255,255,255,.68)" />
              </marker>
            </defs>
            <circle cx="50" cy="50" r="46" fill="rgba(255,255,255,.012)" stroke="rgba(255,255,255,.06)" strokeDasharray="1 2" />
            {graph.edges.map((edge) => {
              const source = nodeBySpeaker.get(edge.from_speaker);
              const target = nodeBySpeaker.get(edge.to_speaker);
              if (!source || !target) return null;
              const width = 0.7 + Number(edge.interaction_weight || 0) * 2.8;
              const strokeOpacity = 0.22 + Number(edge.interaction_weight || 0) * 0.48;
              return (
                <path
                  key={`${edge.from_speaker}-${edge.to_speaker}`}
                  d={edgePath(source, target)}
                  className="interaction-graph-edge"
                  strokeWidth={width}
                  strokeOpacity={strokeOpacity}
                  markerEnd="url(#cerebro-arrow)"
                >
                  <title>{`${edge.from_speaker} → ${edge.to_speaker}: ${edge.interaction_count} observed responses`}</title>
                </path>
              );
            })}
            {positionedNodes.map((node, index) => {
              const nodeRadius = 5 + ((node.speaking_share || 0) / maxSpeaking) * 4.5;
              return (
                <g key={node.speaker}>
                  <circle
                    cx={node.x}
                    cy={node.y}
                    r={nodeRadius}
                    fill={nodeColor(index)}
                    fillOpacity=".22"
                    stroke={nodeColor(index)}
                    strokeWidth=".7"
                  />
                  <text x={node.x} y={node.y - nodeRadius - 2.2} textAnchor="middle" className="interaction-graph-node-label">
                    {node.speaker}
                  </text>
                  <text x={node.x} y={node.y + 1} textAnchor="middle" className="interaction-graph-node-emotion">
                    {titleCase(node.dominant_emotion).slice(0, 9)}
                  </text>
                </g>
              );
            })}
          </svg>
          <div className="graph-axis-label graph-axis-top">HIGH ACTIVE SHARE</div>
          <div className="graph-center-note">response network</div>
        </div>

        <div className="interaction-graph-side">
          <div className="graph-stat-grid">
            <div><span>Density</span><strong>{Math.round(Number(metrics.density || 0) * 100)}%</strong></div>
            <div><span>Reciprocity</span><strong>{Math.round(Number(metrics.reciprocity || 0) * 100)}%</strong></div>
            <div><span>Connected</span><strong>{metrics.connected_speakers ?? 0}</strong></div>
            <div><span>Active pairs</span><strong>{metrics.active_pairs ?? 0}</strong></div>
          </div>

          <div className="graph-edge-list">
            <p className="eyebrow">RESPONSE PATHWAYS</p>
            {graph.edges.length === 0 ? (
              <p className="muted-copy">Only one speaker produced usable turns, so there are no cross-speaker edges.</p>
            ) : (
              graph.edges
                .slice()
                .sort((a, b) => b.interaction_count - a.interaction_count || a.from_speaker.localeCompare(b.from_speaker))
                .map((edge) => (
                  <article className="graph-edge-card" key={`${edge.from_speaker}-${edge.to_speaker}`}>
                    <div className="graph-edge-heading">
                      <strong>{edge.from_speaker} → {edge.to_speaker}</strong>
                      <span>{edge.interaction_count} response{edge.interaction_count === 1 ? "" : "s"}</span>
                    </div>
                    <div className="graph-edge-meta">
                      <span>avg gap {Number(edge.mean_gap_seconds || 0).toFixed(1)}s</span>
                      <span>emotion shifts {edge.emotion_transition_count}</span>
                      <span>activated {edge.activated_shift_count}</span>
                    </div>
                    {(edge.mean_valence_change != null || edge.mean_arousal_change != null) && (
                      <div className="graph-edge-affect">
                        <span>ΔV {signed(edge.mean_valence_change)}</span>
                        <span>ΔA {signed(edge.mean_arousal_change)}</span>
                      </div>
                    )}
                  </article>
                ))
            )}
          </div>
        </div>
      </div>
    </section>
  );
}
