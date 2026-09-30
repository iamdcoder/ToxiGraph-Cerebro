export default function FracturePanel({
  thread,
  onSelect,
}) {
  const fracture =
    thread
      ?.drift_analysis
      ?.global_timeline
      ?.fracture_point;

  if (!fracture) {
    return (
      <section className="panel">
        <div className="panel-heading">
          <div>
            <span className="eyebrow">
              FRACTURE ANALYSIS
            </span>

            <h2>
              No major fracture detected
            </h2>
          </div>
        </div>

        <p className="muted-text">
          The current thread does not contain
          a sufficiently strong regime change.
        </p>
      </section>
    );
  }


  const node =
    thread.nodes.find(
      (item) =>
        item.id ===
        fracture.comment_id
    );


  return (
    <section className="panel fracture-panel">

      <div className="panel-heading">
        <div>
          <span className="eyebrow hot">
            FRACTURE POINT
          </span>

          <h2>
            Comment #
            {fracture.chronological_index +
              1}
          </h2>
        </div>

        <span className="confidence-badge">
          {Math.round(
            fracture.confidence *
              100
          )}
          % confidence
        </span>
      </div>


      {node && (
        <button
          className="fracture-comment"
          onClick={() =>
            onSelect(
              fracture.comment_id
            )
          }
        >
          <div className="comment-meta">
            <span>
              @{node.author}
            </span>

            <span>
              {new Date(
                node.timestamp
              ).toLocaleTimeString(
                [],
                {
                  hour: "2-digit",
                  minute: "2-digit",
                }
              )}
            </span>
          </div>

          <p>
            "{node.text}"
          </p>
        </button>
      )}


      <div className="phase-shift">

        <div>
          <span>
            Before
          </span>

          <strong>
            {fracture.phase_before}
          </strong>

          <em>
            {fracture.sentiment_before.toFixed(
              2
            )}
          </em>
        </div>

        <div className="shift-arrow">
          →
        </div>

        <div className="phase-current">
          <span>
            Fracture
          </span>

          <strong>
            {fracture.phase_at}
          </strong>

          <em>
            {fracture.sentiment_at.toFixed(
              2
            )}
          </em>
        </div>

        <div className="shift-arrow">
          →
        </div>

        <div>
          <span>
            After
          </span>

          <strong>
            {fracture.phase_after}
          </strong>

          <em>
            {fracture.sentiment_after.toFixed(
              2
            )}
          </em>
        </div>

      </div>


      <div className="signal-list">

        {fracture.signals.map(
          (signal) => (
            <div
              key={signal}
              className="signal-chip hot-chip"
            >
              <span>
                +
              </span>

              {signal}
            </div>
          )
        )}

      </div>


      <div className="metric-strip">

        <div>
          <span>
            Drift score
          </span>

          <strong>
            {Math.round(
              fracture.drift_score *
                100
            )}
          </strong>
        </div>

        <div>
          <span>
            Change probability
          </span>

          <strong>
            {Math.round(
              fracture.change_point_probability *
                100
            )}
            %
          </strong>
        </div>

        <div>
          <span>
            Toxicity at point
          </span>

          <strong>
            {Math.round(
              fracture.toxicity_at *
                100
            )}
            %
          </strong>
        </div>

      </div>

    </section>
  );
}