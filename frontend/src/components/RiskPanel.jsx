function riskColor(
  level
) {
  if (level === "critical") {
    return "#ff4d6d";
  }

  if (level === "warning") {
    return "#fb923c";
  }

  if (level === "watch") {
    return "#facc15";
  }

  return "#34d399";
}


export default function RiskPanel({
  earlyWarning,
}) {
  const risk = Math.max(
    0,
    Math.min(
      1,
      Number(
        earlyWarning
          ?.current_risk ||
          0
      )
    )
  );

  const level =
    earlyWarning
      ?.current_risk_level ||
    "safe";

  const color =
    riskColor(level);

  const radius = 48;

  const circumference =
    2 *
    Math.PI *
    radius;

  const offset =
    circumference *
    (1 - risk);


  return (
    <section className="panel risk-panel">

      <div className="panel-heading">
        <div>
          <span className="eyebrow">
            EARLY WARNING
          </span>

          <h2>
            Thread health
          </h2>
        </div>

        <span
          className={`status-pill ${level}`}
        >
          {level.toUpperCase()}
        </span>
      </div>


      <div className="risk-main">

        <div className="gauge">

          <svg
            viewBox="0 0 120 120"
            className="gauge-svg"
          >

            <circle
              cx="60"
              cy="60"
              r={radius}
              className="gauge-track"
            />

            <circle
              cx="60"
              cy="60"
              r={radius}
              className="gauge-progress"
              stroke={color}
              strokeDasharray={
                circumference
              }
              strokeDashoffset={
                offset
              }
            />
          </svg>

          <div className="gauge-content">
            <strong>
              {Math.round(
                risk * 100
              )}
              %
            </strong>

            <span>
              risk
            </span>
          </div>
        </div>


        <div className="risk-copy">

          <div className="risk-big">
            {level === "critical"
              ? "Escalation is imminent."
              : level === "warning"
              ? "Conversation is becoming unstable."
              : level === "watch"
              ? "Early precursor signals detected."
              : "Conversation is currently stable."}
          </div>


          {earlyWarning
            ?.current_risk_level !==
            "safe" && (
            <div className="forecast-horizon">
              Estimated event horizon:{" "}
              <strong>
                {earlyWarning
                  ?.timeline
                  ?.at(-1)
                  ?.predicted_comments_to_event ||
                  3}{" "}
                comments
              </strong>
            </div>
          )}

        </div>
      </div>


      <div className="risk-footer">

        <div>
          <span>
            Peak forecast
          </span>

          <strong>
            {Math.round(
              Number(
                earlyWarning
                  ?.peak_forecast_probability ||
                  0
              ) * 100
            )}
            %
          </strong>
        </div>


        <div>
          <span>
            Forecast horizon
          </span>

          <strong>
            {earlyWarning
              ?.horizon_comments ||
              3}{" "}
            comments
          </strong>
        </div>


        <div>
          <span>
            Earliest warning
          </span>

          <strong>
            {earlyWarning
              ?.earliest_warning_index !=
            null
              ? `#${Number(
                  earlyWarning
                    .earliest_warning_index
                ) + 1}`
              : "—"}
          </strong>
        </div>

      </div>


      {earlyWarning
        ?.warning_was_early && (
        <div className="validation-banner">
          <span className="validation-icon">
            ✓
          </span>

          <div>
            <strong>
              Warning preceded the toxicity event
            </strong>

            <span>
              {earlyWarning
                ?.earliest_warning_lead_comments}{" "}
              comments of lead time in this
              analyzed thread.
            </span>
          </div>
        </div>
      )}

    </section>
  );
}