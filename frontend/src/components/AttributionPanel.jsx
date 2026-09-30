export default function AttributionPanel({
  thread,
  onSelect,
}) {
  const attribution =
    thread?.attribution;

  if (!attribution) {
    return (
      <section className="panel">
        <div className="panel-heading">
          <div>
            <span className="eyebrow">
              TRIGGER ATTRIBUTION
            </span>

            <h2>
              No supported trigger
            </h2>
          </div>
        </div>

        <p className="muted-text">
          There is not enough evidence around
          the current fracture for trigger
          attribution.
        </p>
      </section>
    );
  }


  const triggerNode =
    thread.nodes.find(
      (node) =>
        node.id ===
        attribution.primary_trigger_id
    );


  return (
    <section className="panel attribution-panel">

      <div className="panel-heading">
        <div>
          <span className="eyebrow">
            COUNTERFACTUAL ATTRIBUTION
          </span>

          <h2>
            Strongest contributing reply
          </h2>
        </div>

        <span
          className={`confidence-badge ${attribution.confidence}`}
        >
          {attribution.confidence}
        </span>
      </div>


      {triggerNode && (
        <button
          className="trigger-card"
          onClick={() =>
            onSelect(
              triggerNode.id
            )
          }
        >

          <div className="trigger-top">
            <span className="trigger-mark">
              ⚡
            </span>

            <div>
              <strong>
                @{triggerNode.author}
              </strong>

              <span>
                {triggerNode.text}
              </span>
            </div>
          </div>

          <div className="trigger-score">
            <span>
              contribution score
            </span>

            <strong>
              {Math.round(
                Number(
                  attribution
                    ?.candidates?.[0]
                    ?.contribution_score ||
                    0
                ) * 100
              )}
            </strong>
          </div>
        </button>
      )}


      <p className="attribution-summary">
        {attribution.summary}
      </p>


      <div className="candidate-list">
        {attribution.candidates.map(
          (candidate) => {
            const node =
              thread.nodes.find(
                (item) =>
                  item.id ===
                  candidate.comment_id
              );

            return (
              <button
                key={
                  candidate.comment_id
                }
                className={`candidate-row ${
                  candidate.rank ===
                  1
                    ? "primary"
                    : ""
                }`}
                onClick={() =>
                  onSelect(
                    candidate.comment_id
                  )
                }
              >

                <span className="rank">
                  #{candidate.rank}
                </span>

                <div className="candidate-main">
                  <strong>
                    {node
                      ? `@${node.author}`
                      : candidate.comment_id}
                  </strong>

                  <span>
                    {node?.text ||
                      "Comment"}
                  </span>
                </div>

                <div className="candidate-score">
                  <strong>
                    {Math.round(
                      candidate.contribution_score *
                        100
                    )}
                  </strong>

                  <span>
                    contribution
                  </span>
                </div>

              </button>
            );
          }
        )}
      </div>


      {attribution.causal_chain?.length >
        0 && (
        <div className="chain-block">

          <span className="eyebrow">
            ESCALATION CHAIN
          </span>

          <div className="chain">

            {attribution.causal_chain.map(
              (link, index) => (
                <div
                  key={`${link.from_comment_id}-${link.to_comment_id}`}
                  className="chain-step"
                >

                  <button
                    onClick={() =>
                      onSelect(
                        link.from_comment_id
                      )
                    }
                  >
                    #{link.from_comment_id.replace(
                      "comment_",
                      ""
                    )}
                  </button>

                  <span>
                    →
                  </span>

                  <button
                    onClick={() =>
                      onSelect(
                        link.to_comment_id
                      )
                    }
                  >
                    #{link.to_comment_id.replace(
                      "comment_",
                      ""
                    )}
                  </button>

                  {index ===
                    attribution
                      .causal_chain
                      .length -
                      1 && (
                    <span className="chain-end">
                      FRACTURE
                    </span>
                  )}

                </div>
              )
            )}

          </div>
        </div>
      )}

    </section>
  );
}