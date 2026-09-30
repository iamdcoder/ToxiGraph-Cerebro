function clamp(value, min, max) {
  return Math.max(min, Math.min(max, value));
}

function buildPath(points, x, y) {
  return points
    .map(
      (point, index) =>
        `${index === 0 ? "M" : "L"} ${x(point)} ${y(point)}`
    )
    .join(" ");
}

export default function SentimentTimeline({ thread }) {
  const timeline =
    thread?.drift_analysis?.global_timeline?.timeline || [];

  const forecast =
    thread?.early_warning?.timeline || [];

  if (!timeline.length) {
    return (
      <div className="empty-visual">
        No drift timeline available.
      </div>
    );
  }

  const width = Math.max(980, timeline.length * 44);
  const height = 290;

  const left = 48;
  const right = 20;
  const top = 26;

  const chartWidth = width - left - right;
  const chartHeight = 108;

  const x = (point) =>
    left +
    (point.chronological_index /
      Math.max(timeline.length - 1, 1)) *
      chartWidth;

  const sentimentY = (value) =>
    top +
    (1 - (clamp(value, -1, 1) + 1) / 2) *
      chartHeight;

  const riskY = (value) =>
    172 + (1 - clamp(value, 0, 1)) * 82;

  const sentimentPath = buildPath(
    timeline,
    (point) => x(point),
    (point) => sentimentY(point.sentiment_score)
  );

  const riskPath = forecast.length
    ? buildPath(
        forecast,
        (point) => x(point),
        (point) => riskY(point.forecast_probability)
      )
    : "";

  const fracture =
    thread?.drift_analysis?.global_timeline?.fracture_point;

  const fractureX = fracture
    ? left +
      (fracture.chronological_index /
        Math.max(timeline.length - 1, 1)) *
        chartWidth
    : null;

  return (
    <div className="timeline-wrap">
      <svg
        className="timeline-svg"
        viewBox={`0 0 ${width} ${height}`}
        width={width}
        height={height}
        style={{
          width: `${width}px`,
          minWidth: `${width}px`,
          height: `${height}px`,
          display: "block",
        }}
        role="img"
        aria-label="Sentiment drift and toxicity forecast timeline"
      >
        <defs>
          <linearGradient
            id="sentimentFill"
            x1="0"
            y1="0"
            x2="0"
            y2="1"
          >
            <stop
              offset="0%"
              stopColor="#7c3aed"
              stopOpacity="0.22"
            />
            <stop
              offset="100%"
              stopColor="#7c3aed"
              stopOpacity="0"
            />
          </linearGradient>
        </defs>

        <text
          x={left}
          y="16"
          className="chart-label"
        >
          SENTIMENT ARC
        </text>

        <line
          x1={left}
          x2={width - right}
          y1={sentimentY(0)}
          y2={sentimentY(0)}
          className="chart-zero"
        />

        <text
          x={left - 8}
          y={sentimentY(1) + 4}
          textAnchor="end"
          className="axis-label"
        >
          +1
        </text>
        <text
          x={left - 8}
          y={sentimentY(0) + 4}
          textAnchor="end"
          className="axis-label"
        >
          0
        </text>
        <text
          x={left - 8}
          y={sentimentY(-1) + 4}
          textAnchor="end"
          className="axis-label"
        >
          −1
        </text>

        {fractureX !== null && (
          <>
            <line
              x1={fractureX}
              x2={fractureX}
              y1={top}
              y2={height - 14}
              className="fracture-line"
            />
            <text
              x={fractureX + 7}
              y={top + 8}
              className="fracture-label"
            >
              FRACTURE
            </text>
          </>
        )}

        <path
          d={`${sentimentPath} L ${x(
            timeline[timeline.length - 1]
          )} ${sentimentY(0)} L ${left} ${sentimentY(0)} Z`}
          fill="url(#sentimentFill)"
          opacity="0.55"
        />

        <path
          d={sentimentPath}
          className="sentiment-line"
        />

        {timeline.map((point) => (
          <circle
            key={`sent-${point.comment_id}`}
            cx={x(point)}
            cy={sentimentY(point.sentiment_score)}
            r={point.is_change_point ? 5 : 2.8}
            className={
              point.is_change_point
                ? "sentiment-point change"
                : "sentiment-point"
            }
          />
        ))}

        <text
          x={left}
          y="157"
          className="chart-label"
        >
          ESCALATION FORECAST
        </text>

        <line
          x1={left}
          x2={width - right}
          y1={riskY(0.55)}
          y2={riskY(0.55)}
          className="risk-threshold"
        />

        <text
          x={width - right - 2}
          y={riskY(0.55) - 5}
          textAnchor="end"
          className="threshold-label"
        >
          warning
        </text>

        {riskPath && (
          <path
            d={riskPath}
            className="risk-line"
          />
        )}

        {forecast.map((point) => (
          <circle
            key={`risk-${point.comment_id}`}
            cx={x(point)}
            cy={riskY(point.forecast_probability)}
            r="2.5"
            className="risk-point"
          />
        ))}

        <text
          x={left}
          y={height - 2}
          className="axis-label"
        >
          comment 1
        </text>

        <text
          x={width - right}
          y={height - 2}
          textAnchor="end"
          className="axis-label"
        >
          comment {timeline.length}
        </text>
      </svg>
    </div>
  );
}
