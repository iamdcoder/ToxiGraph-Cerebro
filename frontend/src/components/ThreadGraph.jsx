import { useMemo } from "react";

function sentimentColor(sentiment) {
  const value = Math.max(
    -1,
    Math.min(1, Number(sentiment || 0))
  );

  const hue =
    value >= 0
      ? 150 + value * 45
      : 8 + (value + 1) * 35;

  return `hsl(${hue} 82% 58%)`;
}

function shortText(text) {
  if (!text) return "";
  return text.length <= 42
    ? text
    : `${text.slice(0, 42)}…`;
}

function buildTreeLayout(nodes) {
  const ordered = [...nodes].sort(
    (a, b) => new Date(a.timestamp) - new Date(b.timestamp)
  );

  if (!ordered.length) {
    return {
      nodes: [],
      edges: [],
      roots: [],
      width: 1600,
      height: 820,
    };
  }

  const lookup = Object.fromEntries(
    ordered.map((node) => [node.id, node])
  );

  const children = {};

  for (const node of ordered) {
    if (!node.parent_id || !lookup[node.parent_id]) continue;
    if (!children[node.parent_id]) children[node.parent_id] = [];
    children[node.parent_id].push(node);
  }

  for (const parentId of Object.keys(children)) {
    children[parentId].sort(
      (a, b) => new Date(a.timestamp) - new Date(b.timestamp)
    );
  }

  const roots = ordered.filter(
    (node) => !node.parent_id || !lookup[node.parent_id]
  );

  const leafWidth = 168;
  const rootGap = 96;
  const branchGap = 72;
  const horizontalPadding = 120;
  const levelGap = 126;
  const topPadding = 72;
  const bottomPadding = 88;

  const countLeavesMemo = new Map();
  const visiting = new Set();

  function countLeaves(nodeId) {
    if (countLeavesMemo.has(nodeId)) {
      return countLeavesMemo.get(nodeId);
    }

    if (visiting.has(nodeId)) {
      return 1;
    }

    visiting.add(nodeId);
    const direct = children[nodeId] || [];

    let result = 1;
    if (direct.length) {
      result = direct.reduce(
        (total, child) => total + countLeaves(child.id),
        0
      );
    }

    visiting.delete(nodeId);
    countLeavesMemo.set(nodeId, result);
    return result;
  }

  const rootSpans = roots.map((root) => countLeaves(root.id));
  const totalLeaves = rootSpans.reduce(
    (total, value) => total + value,
    0
  );

  const minimumWidth = 1600;
  const contentWidth =
    horizontalPadding * 2 +
    Math.max(totalLeaves, 1) * leafWidth +
    Math.max(roots.length - 1, 0) * rootGap +
    Math.max(ordered.length - 1, 0) * 4;
  const width = Math.max(minimumWidth, contentWidth);

  const maxDepth = Math.max(
    ...ordered.map((node) => Number(node.depth || 0)),
    0
  );

  const height = Math.max(
    720,
    topPadding +
      (maxDepth + 1) * levelGap +
      bottomPadding
  );

  const positions = {};

  function placeSubtree(node, startX, leafSpan) {
    const direct = children[node.id] || [];
    const depth = Number(node.depth || 0);

    if (!direct.length) {
      positions[node.id] = {
        x: startX + (leafSpan * leafWidth) / 2,
        y: topPadding + depth * levelGap,
      };
      return;
    }

    let childX = startX;

    for (const child of direct) {
      const childSpan = countLeaves(child.id);
      placeSubtree(child, childX, childSpan);
      childX += childSpan * leafWidth;
    }

    const childPositions = direct
      .map((child) => positions[child.id])
      .filter(Boolean);

    const averageX =
      childPositions.reduce(
        (sum, position) => sum + position.x,
        0
      ) / Math.max(childPositions.length, 1);

    positions[node.id] = {
      x: averageX,
      y: topPadding + depth * levelGap,
    };
  }

  let cursorX = horizontalPadding;

  for (let index = 0; index < roots.length; index += 1) {
    const root = roots[index];
    const rootSpan = rootSpans[index];
    placeSubtree(root, cursorX, rootSpan);
    cursorX += rootSpan * leafWidth + rootGap;
  }

  const positionedNodes = ordered.map((node) => ({
    ...node,
    x: positions[node.id]?.x ?? horizontalPadding,
    y: positions[node.id]?.y ?? topPadding,
  }));

  const edges = [];

  for (const node of positionedNodes) {
    if (!node.parent_id || !positions[node.parent_id]) continue;

    edges.push({
      source: {
        ...lookup[node.parent_id],
        x: positions[node.parent_id].x,
        y: positions[node.parent_id].y,
      },
      target: {
        ...node,
        x: positions[node.id].x,
        y: positions[node.id].y,
      },
    });
  }

  const branchRoots = roots.flatMap((root) =>
    (children[root.id] || []).map((child, index) => ({
      ...child,
      branchIndex: index + 1,
      rootId: root.id,
    }))
  );

  return {
    nodes: positionedNodes,
    edges,
    roots,
    branchRoots,
    width,
    height,
  };
}

export default function ThreadGraph({
  thread,
  selectedId,
  onSelect,
}) {
  const layout = useMemo(
    () => buildTreeLayout(thread?.nodes || []),
    [thread]
  );

  if (!thread) {
    return (
      <div className="empty-visual">
        No thread loaded.
      </div>
    );
  }

  const fractureId =
    thread?.drift_analysis?.global_timeline?.fracture_point?.comment_id;

  const triggerId = thread?.attribution?.primary_trigger_id;

  const branchLabels = layout.branchRoots || [];

  return (
    <div className="graph-scroll">
      <svg
        className="thread-svg"
        viewBox={`0 0 ${layout.width} ${layout.height}`}
        preserveAspectRatio="xMinYMin meet"
        width={layout.width}
        height={layout.height}
        style={{
          width: `${layout.width}px`,
          height: `${layout.height}px`,
          minWidth: `${layout.width}px`,
          display: "block",
        }}
        role="img"
        aria-label="Interactive conversation thread graph"
      >
        <defs>
          <filter
            id="triggerGlow"
            x="-100%"
            y="-100%"
            width="300%"
            height="300%"
          >
            <feGaussianBlur
              stdDeviation="6"
              result="blur"
            />
            <feMerge>
              <feMergeNode in="blur" />
              <feMergeNode in="SourceGraphic" />
            </feMerge>
          </filter>

          <linearGradient
            id="edgeGradient"
            x1="0"
            y1="0"
            x2="0"
            y2="1"
          >
            <stop
              offset="0%"
              stopColor="#64748b"
              stopOpacity="0.40"
            />
            <stop
              offset="100%"
              stopColor="#64748b"
              stopOpacity="0.10"
            />
          </linearGradient>
        </defs>

        {branchLabels.map((branch) => (
          <g
            key={`branch-label-${branch.id}`}
            pointerEvents="none"
          >
            <text
              x={branch.x}
              y={branch.y - 44}
              textAnchor="middle"
              style={{
                fill: "#697386",
                fontSize: "9px",
                fontWeight: 800,
                letterSpacing: "0.17em",
              }}
            >
              BRANCH {branch.branchIndex}
            </text>
          </g>
        ))}

        {layout.edges.map((edge) => {
          const source = edge.source;
          const target = edge.target;
          const targetFeatures = target.features || {};
          const isTrigger = target.id === triggerId;
          const isFracture = target.id === fractureId;

          const influence = Number(
            targetFeatures.trigger_contribution_score ||
              targetFeatures.drift_score ||
              0.20
          );

          const distance = Math.max(
            target.y - source.y,
            1
          );
          const curve = Math.min(distance * 0.38, 84);

          return (
            <path
              key={`${source.id}-${target.id}`}
              d={`M ${source.x} ${source.y}
                  C ${source.x} ${source.y + curve},
                    ${target.x} ${target.y - curve},
                    ${target.x} ${target.y}`}
              fill="none"
              stroke={
                isTrigger
                  ? "#ff4d6d"
                  : isFracture
                  ? "#fbbf24"
                  : "url(#edgeGradient)"
              }
              strokeWidth={
                isTrigger
                  ? 4.5
                  : isFracture
                  ? 3
                  : 1.5 + Math.min(influence, 1) * 3
              }
              strokeLinecap="round"
            />
          );
        })}

        {layout.nodes.map((node) => {
          const features = node.features || {};
          const analysis = node.analysis;

          const sentiment = Number(
            analysis?.sentiment?.sentiment_score ??
              features.sentiment_score ??
              0
          );

          const toxicity = Number(
            analysis?.toxicity?.toxicity ??
              features.toxicity_score ??
              0
          );

          const drift = Number(
            features.drift_score || 0
          );

          const forecast = Number(
            features.forecast_probability || 0
          );

          const isSelected = selectedId === node.id;
          const isFracture = node.id === fractureId;
          const isTrigger = node.id === triggerId;

          const radius =
            15 +
            Math.min(
              13,
              drift * 11 + toxicity * 7 + forecast * 3
            );

          const chronologicalIndex = Number(
            features.chronological_index ?? 0
          );

          return (
            <g
              key={node.id}
              className={`graph-node ${
                isSelected ? "selected" : ""
              } ${isFracture ? "fracture" : ""} ${
                isTrigger ? "trigger" : ""
              }`}
              transform={`translate(${node.x}, ${node.y})`}
              onClick={() => onSelect(node.id)}
              onKeyDown={(event) => {
                if (
                  event.key === "Enter" ||
                  event.key === " "
                ) {
                  event.preventDefault();
                  onSelect(node.id);
                }
              }}
              tabIndex={0}
            >
              {isFracture && (
                <circle
                  r={radius + 10}
                  className="fracture-ring"
                />
              )}

              {isTrigger && (
                <circle
                  r={radius + 14}
                  className="trigger-glow"
                  filter="url(#triggerGlow)"
                />
              )}

              <circle
                r={radius}
                fill={sentimentColor(sentiment)}
                stroke={
                  isTrigger
                    ? "#ff4d6d"
                    : isFracture
                    ? "#fbbf24"
                    : "rgba(255,255,255,0.35)"
                }
                strokeWidth={
                  isTrigger
                    ? 3.5
                    : isFracture
                    ? 2.5
                    : 1.2
                }
              />

              <text
                x="0"
                y="5"
                textAnchor="middle"
                className="node-index"
                style={{
                  fontSize: "10px",
                  fontWeight: 900,
                }}
              >
                {chronologicalIndex + 1}
              </text>

              <text
                x="0"
                y={radius + 18}
                textAnchor="middle"
                className="node-author"
                style={{
                  fontSize: "9px",
                  fontWeight: 700,
                }}
              >
                {node.author}
              </text>

              {(isFracture || isTrigger) && (
                <text
                  x="0"
                  y={radius + 34}
                  textAnchor="middle"
                  style={{
                    fill: isTrigger ? "#ff8297" : "#fbbf24",
                    fontSize: "7px",
                    fontWeight: 900,
                    letterSpacing: "0.12em",
                  }}
                >
                  {isTrigger ? "TRIGGER" : "FRACTURE"}
                </text>
              )}

              <title>
                {`#${chronologicalIndex + 1} @${node.author}: ${shortText(
                  node.text
                )}`}
              </title>
            </g>
          );
        })}
      </svg>

      <div className="graph-legend">
        <span>
          <i className="legend-dot positive" />
          Positive
        </span>
        <span>
          <i className="legend-dot negative" />
          Negative
        </span>
        <span>
          <i className="legend-dot fracture" />
          Fracture
        </span>
        <span>
          <i className="legend-dot trigger" />
          Trigger
        </span>
      </div>
    </div>
  );
}
