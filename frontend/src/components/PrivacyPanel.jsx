import { useEffect, useState } from "react";
import { deleteAllProfileData, getPrivacyProfile } from "../api";

export default function PrivacyPanel({ profileId }) {
  const [status, setStatus] = useState(null);
  const [busy, setBusy] = useState(false);
  const [message, setMessage] = useState("");
  const [error, setError] = useState("");

  useEffect(() => {
    if (!profileId) return undefined;
    let mounted = true;
    getPrivacyProfile(profileId)
      .then((payload) => {
        if (mounted) setStatus(payload);
      })
      .catch((err) => {
        if (mounted) setError(err?.message || "Privacy status is unavailable.");
      });
    return () => {
      mounted = false;
    };
  }, [profileId]);

  async function deleteEverything() {
    if (!profileId || busy) return;
    const confirmed = window.confirm(
      "Delete this browser profile's stored CEREBRO baseline, voice history, and personalization data? This cannot be undone."
    );
    if (!confirmed) return;
    setBusy(true);
    setError("");
    setMessage("");
    try {
      const payload = await deleteAllProfileData(profileId);
      setStatus(await getPrivacyProfile(profileId));
      setMessage(payload.message || "Stored profile data deleted.");
    } catch (err) {
      setError(err?.message || "Stored profile data could not be deleted.");
    } finally {
      setBusy(false);
    }
  }

  if (!profileId) return null;

  const categories = status?.categories || {};
  const stored = status?.stored_file_count || 0;

  return (
    <section className="privacy-panel">
      <div className="result-heading">
        <div>
          <p className="eyebrow">PRIVACY &amp; SECURITY</p>
          <h3>Your data, your controls</h3>
          <p className="muted-copy">
            CEREBRO does not persist raw microphone recordings through its profile services. Persistent profile data is limited to compact acoustic summaries and explicitly supplied calibration feedback.
          </p>
        </div>
        <span className="result-badge success">LOCAL-FIRST</span>
      </div>

      <div className="privacy-grid">
        <div className="privacy-card">
          <span className="eyebrow">STORAGE POSTURE</span>
          <div className="privacy-stat"><span>Raw audio stored</span><strong>NO</strong></div>
          <div className="privacy-stat"><span>Raw transcripts stored</span><strong>NO</strong></div>
          <div className="privacy-stat"><span>Stored profile files</span><strong>{stored}</strong></div>
          <div className="privacy-stat"><span>Automatic retention</span><strong>{status?.retention_enabled ? `${status.retention_days} DAYS` : "OFF"}</strong></div>
        </div>

        <div className="privacy-card">
          <span className="eyebrow">ACTIVE PROFILE</span>
          <strong className="privacy-profile-id">{profileId}</strong>
          <small>Anonymous browser profile identifier. It is not an account identity.</small>
          <div className="privacy-category-list">
            <span>Acoustic baseline <b>{categories.personal_baseline ? "STORED" : "EMPTY"}</b></span>
            <span>Voice history <b>{categories.voice_history ? "STORED" : "EMPTY"}</b></span>
            <span>Personal calibration <b>{categories.personalization ? "STORED" : "EMPTY"}</b></span>
          </div>
        </div>
      </div>

      <div className="privacy-actions">
        <button className="secondary-button danger-button" type="button" onClick={deleteEverything} disabled={busy || stored === 0}>
          {busy ? "Deleting stored data…" : "Delete all stored profile data"}
        </button>
        <span>Deletion covers the three CEREBRO profile stores for this browser profile.</span>
      </div>

      {message && <p className="baseline-message">{message}</p>}
      {error && <p className="baseline-error" role="alert">{error}</p>}
    </section>
  );
}
