export default function ForensicReport({
  thread,
}) {
  if (!thread) {
    return null;
  }


  const nodes =
    [
      ...(thread.nodes || []),
    ].sort(
      (a, b) =>
        new Date(
          a.timestamp
        ) -
        new Date(
          b.timestamp
        )
    );


  const fracture =
    thread
      ?.drift_analysis
      ?.global_timeline
      ?.fracture_point;


  const attribution =
    thread?.attribution;


  const earlyWarning =
    thread?.early_warning;


  const fractureNode =
    nodes.find(
      (node) =>
        node.id ===
        fracture?.comment_id
    );


  const triggerNode =
    nodes.find(
      (node) =>
        node.id ===
        attribution
          ?.primary_trigger_id
    );


  const rootIds = new Set(
    nodes
      .filter(
        (node) =>
          node.parent_id ===
          null
      )
      .map(
        (node) =>
          node.id
      )
  );


  const branchCount =
    new Set(
      nodes
        .map(
          (node) =>
            node
              ?.features
              ?.branch_root_id
        )
        .filter(
          (branchId) =>
            branchId &&
            !rootIds.has(
              branchId
            )
        )
    ).size ||
    rootIds.size;


  const first =
    nodes.slice(
      0,
      3
    );


  const earlySentiment =
    first.length
      ? first.reduce(
          (
            sum,
            node
          ) =>
            sum +
            Number(
              node
                ?.analysis
                ?.sentiment
                ?.sentiment_score ||
              0
            ),
          0
        ) /
        first.length
      : 0;


  const currentSentiment =
    nodes.length
      ? Number(
          nodes[
            nodes.length -
              1
          ]?.analysis
            ?.sentiment
            ?.sentiment_score ||
          0
        )
      : 0;


  return (
    <section className="panel report-panel">

      <div className="report-header">

        <div>

          <span className="eyebrow">
            FORENSIC REPORT
          </span>

          <h2>
            How the conversation changed
          </h2>

        </div>


        <span className="report-tag">
          TOXIGRAPH ANALYSIS
        </span>

      </div>


      <div className="report-body">

        <p>

          <strong>
            Thread summary.
          </strong>{" "}

          ToxiGraph analyzed{" "}

          <strong>
            {nodes.length}
          </strong>{" "}

          comments across approximately{" "}

          <strong>
            {branchCount}
          </strong>{" "}

          conversation branches.

          The thread moved from an
          initial sentiment of{" "}

          <strong>
            {earlySentiment.toFixed(
              2
            )}
          </strong>{" "}

          to{" "}

          <strong
            className={
              currentSentiment <
              0
                ? "report-negative"
                : "report-positive"
            }
          >
            {currentSentiment.toFixed(
              2
            )}
          </strong>
          .

        </p>


        {fracture && (
          <p>

            <strong>
              Fracture point.
            </strong>{" "}

            A significant emotional regime
            change was detected at{" "}

            <strong>
              Comment #
              {fracture
                .chronological_index +
                1}
            </strong>


            {fractureNode && (
              <>
                {" "}
                when{" "}

                <span className="quote-inline">
                  "
                  {
                    fractureNode.text
                  }
                  "
                </span>
              </>
            )}
            . The estimated change-point
            probability was{" "}

            <strong>
              {Math.round(
                fracture
                  .change_point_probability *
                  100
              )}
              %
            </strong>
            .

          </p>
        )}


        {triggerNode && (
          <p>

            <strong>
              Trigger attribution.
            </strong>{" "}

            Counterfactual analysis identified{" "}

            <strong>
              @{triggerNode.author}
            </strong>
            's reply as the strongest
            contributing trigger.

            {attribution
              ?.candidates?.[0]
              ?.drift_reduction !=
              null && (
              <>
                {" "}
                Removing that candidate
                reduced the local drift
                signal by{" "}

                <strong>
                  {Math.round(
                    Number(
                      attribution
                        .candidates?.[0]
                        ?.drift_reduction ||
                        0
                    ) *
                      100
                  )}
                  %
                </strong>
                .
              </>
            )}

          </p>
        )}


        {earlyWarning && (
          <p>

            <strong>
              Early warning.
            </strong>{" "}

            The forecast peaked at{" "}

            <strong>
              {Math.round(
                earlyWarning
                  .peak_forecast_probability *
                  100
              )}
              %
            </strong>


            {earlyWarning
              .warning_was_early &&
              earlyWarning
                .earliest_warning_lead_comments !=
                null && (
                <>
                  {" "}
                  and issued its first
                  warning{" "}

                  <strong>
                    {
                      earlyWarning
                        .earliest_warning_lead_comments
                    }{" "}
                    comments
                  </strong>{" "}

                  before the realized
                  toxicity event in this
                  thread.
                </>
              )}

          </p>
        )}


        {fracture
          ?.signals
          ?.length > 0 && (
          <div className="report-evidence">

            <span className="eyebrow">
              OBSERVED EVIDENCE
            </span>


            <div className="signal-list">

              {fracture.signals.map(
                (
                  signal
                ) => (

                  <div
                    className="signal-chip"
                    key={signal}
                  >
                    {signal}
                  </div>

                )
              )}

            </div>

          </div>
        )}

      </div>

    </section>
  );
}