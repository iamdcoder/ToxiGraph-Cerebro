function percentage(
  value
) {
  return `${Math.round(
    Number(
      value || 0
    ) *
      100
  )}%`;
}


function metricClass(
  value,
  positive = false
) {
  if (positive) {
    return Number(
      value || 0
    ) >= 0.5
      ? "metric-good"
      : "";
  }

  return Number(
    value || 0
  ) >= 0.5
    ? "metric-hot"
    : "";
}


export default function CommentInspector({
  node,
  thread,
  onClose,
}) {
  if (!node) {
    return (
      <section className="panel inspector empty-inspector">

        <span className="eyebrow">
          COMMENT INSPECTOR
        </span>

        <h2>
          Select a node
        </h2>

        <p>
          Click any comment in the thread
          graph to inspect its semantic,
          structural, and escalation signals.
        </p>

      </section>
    );
  }


  const analysis =
    node.analysis ||
    {};

  const sentiment =
    analysis
      .sentiment
      ?.sentiment_score ??
    node.features
      ?.sentiment_score ??
    0;

  const toxicity =
    analysis
      .toxicity
      ?.toxicity ??
    node.features
      ?.toxicity_score ??
    0;

  const emotion =
    analysis.emotion ||
    {};

  const linguistic =
    analysis.linguistic ||
    {};

  const features =
    node.features ||
    {};


  const chronologicalIndex =
    Number(
      features
        .chronological_index ??
        0
    );


  const branchRootId =
    features
      .branch_root_id;


  const branchRoot =
    thread?.nodes?.find(
      (item) =>
        item.id ===
        branchRootId
    );


  const branchLabel =
    branchRootId &&
    branchRootId !==
      node.id
      ? `@${branchRoot?.author || "branch"}`
      : "root";


  return (
    <section className="panel inspector">

      <div className="panel-heading">

        <div>

          <span className="eyebrow">
            COMMENT INSPECTOR
          </span>

          <h2>
            @{node.author}
          </h2>

        </div>


        <button
          className="icon-button"
          onClick={onClose}
          aria-label="Close inspector"
        >
          ×
        </button>

      </div>


      <div className="inspector-comment">

        <p>
          "{node.text}"
        </p>

        <span>
          COMMENT #
          {chronologicalIndex +
            1}
          {" · "}
          {new Date(
            node.timestamp
          ).toLocaleTimeString(
            [],
            {
              hour: "2-digit",
              minute: "2-digit",
              second: "2-digit",
            }
          )}
        </span>

      </div>


      <div className="inspection-grid">

        <div className="inspection-card">

          <span>
            Sentiment
          </span>

          <strong
            className={
              Number(
                sentiment
              ) < 0
                ? "metric-hot"
                : "metric-good"
            }
          >
            {Number(
              sentiment
            ).toFixed(2)}
          </strong>

        </div>


        <div className="inspection-card">

          <span>
            Toxicity
          </span>

          <strong
            className={metricClass(
              toxicity
            )}
          >
            {percentage(
              toxicity
            )}
          </strong>

        </div>


        <div className="inspection-card">

          <span>
            Drift
          </span>

          <strong>
            {percentage(
              features
                .drift_score
            )}
          </strong>

        </div>


        <div className="inspection-card">

          <span>
            Forecast
          </span>

          <strong
            className={metricClass(
              features
                .forecast_probability
            )}
          >
            {percentage(
              features
                .forecast_probability
            )}
          </strong>

        </div>

      </div>


      <div className="inspection-section">

        <span className="eyebrow">
          GRAPH POSITION
        </span>


        <div className="detail-list">

          <div>
            <span>
              Depth
            </span>

            <strong>
              {node.depth}
            </strong>
          </div>


          <div>
            <span>
              Branch
            </span>

            <strong>
              {branchLabel}
            </strong>
          </div>


          <div>
            <span>
              Replies
            </span>

            <strong>
              {features
                .branching_factor ||
                0}
            </strong>
          </div>


          <div>
            <span>
              Reply velocity
            </span>

            <strong>
              {Number(
                features
                  .reply_velocity ||
                  0
              ).toFixed(
                1
              )}
              s
            </strong>
          </div>

        </div>

      </div>


      <div className="inspection-section">

        <span className="eyebrow">
          LINGUISTIC SIGNALS
        </span>


        <div className="signal-list compact">

          <div className="signal-chip">
            Pronoun shift{" "}
            {percentage(
              linguistic
                .pronoun_shift
            )}
          </div>


          <div className="signal-chip">
            Negation{" "}
            {percentage(
              linguistic
                .negation_density
            )}
          </div>


          <div className="signal-chip">
            Hedging{" "}
            {percentage(
              linguistic
                .hedging_score
            )}
          </div>


          <div className="signal-chip">
            Profanity{" "}
            {percentage(
              linguistic
                .profanity_score
            )}
          </div>

        </div>

      </div>


      <div className="inspection-section">

        <span className="eyebrow">
          EMOTION PROFILE
        </span>


        <div className="emotion-bars">

          {Object.entries(
            emotion.scores ||
              {}
          )
            .sort(
              (
                [, a],
                [, b]
              ) =>
                b - a
            )
            .slice(
              0,
              5
            )
            .map(
              ([
                name,
                value,
              ]) => (

                <div
                  className="emotion-bar"
                  key={name}
                >

                  <div>
                    <span>
                      {name}
                    </span>

                    <strong>
                      {percentage(
                        value
                      )}
                    </strong>
                  </div>


                  <span className="bar-track">

                    <span
                      className="bar-fill"
                      style={{
                        width: `${Math.min(
                          100,
                          Number(
                            value
                          ) *
                            100
                        )}%`,
                      }}
                    />

                  </span>

                </div>

              )
            )}

        </div>

      </div>


      {features
        .is_change_point && (
        <div className="primary-trigger-banner">

          <span>
            ◉
          </span>

          <div>

            <strong>
              Bayesian change point
            </strong>

            <span>
              This comment sits on a detected
              emotional regime boundary.
            </span>

          </div>

        </div>
      )}


      {features
        .is_primary_trigger && (
        <div className="primary-trigger-banner">

          <span>
            ⚡
          </span>

          <div>

            <strong>
              Primary contributing trigger
            </strong>

            <span>
              Counterfactual analysis ranked
              this comment first.
            </span>

          </div>

        </div>
      )}

    </section>
  );
}