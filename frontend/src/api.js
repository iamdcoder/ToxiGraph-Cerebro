const API_BASE = import.meta.env.VITE_API_BASE_URL || "/api/v1";
const API_KEY = import.meta.env.VITE_CEREBRO_API_KEY || "";

function cerebroFetch(url, options = {}) {
  const headers = new Headers(options.headers || {});
  if (API_KEY) headers.set("X-API-Key", API_KEY);
  return fetch(url, { ...options, headers });
}

function formatApiError(payload, fallback) {
  const detail = payload?.detail;

  if (typeof detail === "string" && detail.trim()) {
    return detail;
  }

  if (Array.isArray(detail)) {
    const messages = detail
      .map((item) => item?.msg || (typeof item === "string" ? item : ""))
      .filter(Boolean);
    if (messages.length) {
      return messages.join(" · ");
    }
  }

  return fallback;
}

export async function validateAudio(blob, filename = "cerebro-recording.wav") {
  const formData = new FormData();
  formData.append("file", blob, filename);

  let response;
  try {
    response = await cerebroFetch(`${API_BASE}/audio/validate`, {
      method: "POST",
      body: formData,
    });
  } catch {
    throw new Error(
      "Cannot reach the CEREBRO backend. Start FastAPI on port 8000 and try again."
    );
  }

  let payload;
  try {
    payload = await response.json();
  } catch {
    throw new Error(
      `The backend returned an invalid response (HTTP ${response.status}).`
    );
  }

  if (!response.ok) {
    throw new Error(
      formatApiError(payload, `Audio validation failed (HTTP ${response.status}).`)
    );
  }

  return payload;
}


export async function extractAcousticFeatures(blob, filename = "cerebro-recording.wav") {
  const formData = new FormData();
  formData.append("file", blob, filename);

  let response;
  try {
    response = await cerebroFetch(`${API_BASE}/audio/features`, {
      method: "POST",
      body: formData,
    });
  } catch {
    throw new Error(
      "Cannot reach the CEREBRO backend for acoustic analysis. Start FastAPI on port 8000 and try again."
    );
  }

  let payload;
  try {
    payload = await response.json();
  } catch {
    throw new Error(
      `The backend returned an invalid acoustic-analysis response (HTTP ${response.status}).`
    );
  }

  if (!response.ok) {
    throw new Error(
      formatApiError(payload, `Acoustic feature extraction failed (HTTP ${response.status}).`)
    );
  }

  return payload;
}


export async function getEmotionModelStatus() {
  let response;
  try {
    response = await cerebroFetch(`${API_BASE}/audio/emotion/status`);
  } catch {
    throw new Error("Cannot reach the CEREBRO emotion-model service.");
  }

  const payload = await response.json().catch(() => ({}));
  if (!response.ok) {
    throw new Error(formatApiError(payload, `Emotion model status failed (HTTP ${response.status}).`));
  }
  return payload;
}

export async function predictEmotion(blob, filename = "cerebro-recording.wav") {
  const formData = new FormData();
  formData.append("file", blob, filename);

  let response;
  try {
    response = await cerebroFetch(`${API_BASE}/audio/emotion`, {
      method: "POST",
      body: formData,
    });
  } catch {
    throw new Error("Cannot reach the CEREBRO emotion inference service.");
  }

  const payload = await response.json().catch(() => ({}));
  if (!response.ok) {
    throw new Error(formatApiError(payload, `Emotion inference failed (HTTP ${response.status}).`));
  }
  return payload;
}

export async function predictDeepEmotion(blob, filename = "cerebro-recording.wav") {
  const formData = new FormData();
  formData.append("file", blob, filename);

  let response;
  try {
    response = await cerebroFetch(`${API_BASE}/audio/emotion/deep`, {
      method: "POST",
      body: formData,
    });
  } catch {
    throw new Error("Cannot reach the CEREBRO deep speech emotion service.");
  }

  const payload = await response.json().catch(() => ({}));
  if (!response.ok) {
    throw new Error(formatApiError(payload, `Deep speech emotion inference failed (HTTP ${response.status}).`));
  }
  return payload;
}


export async function predictFusedEmotion(blob, filename = "cerebro-recording.wav") {
  const formData = new FormData();
  formData.append("file", blob, filename);

  let response;
  try {
    response = await cerebroFetch(`${API_BASE}/audio/emotion/fusion`, {
      method: "POST",
      body: formData,
    });
  } catch {
    throw new Error("Cannot reach the CEREBRO fusion inference service.");
  }

  const payload = await response.json().catch(() => ({}));
  if (!response.ok) {
    throw new Error(formatApiError(payload, `Fusion inference failed (HTTP ${response.status}).`));
  }
  return payload;
}


export async function predictTemporalEmotion(blob, filename = "cerebro-recording.wav") {
  const formData = new FormData();
  formData.append("file", blob, filename);

  let response;
  try {
    response = await cerebroFetch(`${API_BASE}/audio/emotion/temporal`, {
      method: "POST",
      body: formData,
    });
  } catch {
    throw new Error("Cannot reach the CEREBRO temporal emotion service.");
  }

  const payload = await response.json().catch(() => ({}));
  if (!response.ok) {
    throw new Error(formatApiError(payload, `Temporal emotion analysis failed (HTTP ${response.status}).`));
  }
  return payload;
}

export async function predictEmotionInsights(blob, filename = "cerebro-recording.wav", profileId = null) {
  const formData = new FormData();
  formData.append("file", blob, filename);
  if (profileId) {
    formData.append("profile_id", profileId);
  }

  let response;
  try {
    response = await cerebroFetch(`${API_BASE}/audio/emotion/insights`, {
      method: "POST",
      body: formData,
    });
  } catch {
    throw new Error("Cannot reach the CEREBRO explanation service.");
  }

  const payload = await response.json().catch(() => ({}));
  if (!response.ok) {
    throw new Error(formatApiError(payload, `Emotion insights failed (HTTP ${response.status}).`));
  }
  return payload;
}


export async function getPersonalBaselineStatus(profileId = "default") {
  let response;
  try {
    response = await cerebroFetch(`${API_BASE}/audio/baseline/${encodeURIComponent(profileId)}`);
  } catch {
    throw new Error("Cannot reach the CEREBRO personal-baseline service.");
  }
  const payload = await response.json().catch(() => ({}));
  if (!response.ok) {
    throw new Error(formatApiError(payload, `Personal baseline status failed (HTTP ${response.status}).`));
  }
  return payload;
}

export async function createPersonalBaseline(profileId, blobs) {
  const formData = new FormData();
  blobs.forEach((blob, index) => {
    formData.append("files", blob, `cerebro-baseline-${index + 1}.wav`);
  });

  let response;
  try {
    response = await cerebroFetch(`${API_BASE}/audio/baseline/${encodeURIComponent(profileId)}`, {
      method: "POST",
      body: formData,
    });
  } catch {
    throw new Error("Cannot reach the CEREBRO personal-baseline service.");
  }
  const payload = await response.json().catch(() => ({}));
  if (!response.ok) {
    throw new Error(formatApiError(payload, `Personal baseline creation failed (HTTP ${response.status}).`));
  }
  return payload;
}

export async function deletePersonalBaseline(profileId = "default") {
  let response;
  try {
    response = await cerebroFetch(`${API_BASE}/audio/baseline/${encodeURIComponent(profileId)}`, {
      method: "DELETE",
    });
  } catch {
    throw new Error("Cannot reach the CEREBRO personal-baseline service.");
  }
  const payload = await response.json().catch(() => ({}));
  if (!response.ok) {
    throw new Error(formatApiError(payload, `Personal baseline deletion failed (HTTP ${response.status}).`));
  }
  return payload;
}

export async function compareEmotionRecordings(recordingA, recordingB, profileId = null) {
  const formData = new FormData();
  formData.append("recording_a", recordingA, "cerebro-recording-a.wav");
  formData.append("recording_b", recordingB, "cerebro-recording-b.wav");
  if (profileId) {
    formData.append("profile_id", profileId);
  }

  let response;
  try {
    response = await cerebroFetch(`${API_BASE}/audio/emotion/compare`, {
      method: "POST",
      body: formData,
    });
  } catch {
    throw new Error("Cannot reach the CEREBRO recording-comparison service.");
  }

  const payload = await response.json().catch(() => ({}));
  if (!response.ok) {
    throw new Error(formatApiError(payload, `Recording comparison failed (HTTP ${response.status}).`));
  }
  return payload;
}


export async function transcribeAudio(blob, filename = "cerebro-recording.wav") {
  const formData = new FormData();
  formData.append("file", blob, filename);

  let response;
  try {
    response = await cerebroFetch(`${API_BASE}/audio/transcribe`, {
      method: "POST",
      body: formData,
    });
  } catch {
    throw new Error("Cannot reach the CEREBRO transcription service.");
  }

  const payload = await response.json().catch(() => ({}));
  if (!response.ok) {
    throw new Error(formatApiError(payload, `Speech transcription failed (HTTP ${response.status}).`));
  }
  return payload;
}


export async function getTranscriptionStatus() {
  let response;
  try {
    response = await cerebroFetch(`${API_BASE}/audio/transcribe/status`);
  } catch {
    throw new Error("Cannot reach the CEREBRO transcription service.");
  }

  const payload = await response.json().catch(() => ({}));
  if (!response.ok) {
    throw new Error(formatApiError(payload, `Transcription status failed (HTTP ${response.status}).`));
  }
  return payload;
}


export async function getEvaluationStatus() {
  let response;
  try {
    response = await cerebroFetch(`${API_BASE}/evaluation/status`);
  } catch {
    throw new Error("Cannot reach the CEREBRO evaluation service.");
  }
  const payload = await response.json().catch(() => ({}));
  if (!response.ok) {
    throw new Error(formatApiError(payload, `Evaluation status failed (HTTP ${response.status}).`));
  }
  return payload;
}


export async function getEvaluationReport() {
  let response;
  try {
    response = await cerebroFetch(`${API_BASE}/evaluation/report`);
  } catch {
    throw new Error("Cannot reach the CEREBRO evaluation service.");
  }
  const payload = await response.json().catch(() => ({}));
  if (!response.ok) {
    throw new Error(formatApiError(payload, `Evaluation report failed (HTTP ${response.status}).`));
  }
  return payload;
}


export async function getRobustnessStatus() {
  let response;
  try {
    response = await cerebroFetch(`${API_BASE}/robustness/status`);
  } catch {
    throw new Error("Cannot reach the CEREBRO robustness service.");
  }
  const payload = await response.json().catch(() => ({}));
  if (!response.ok) {
    throw new Error(formatApiError(payload, `Robustness status failed (HTTP ${response.status}).`));
  }
  return payload;
}


export async function getRobustnessReport() {
  let response;
  try {
    response = await cerebroFetch(`${API_BASE}/robustness/report`);
  } catch {
    throw new Error("Cannot reach the CEREBRO robustness service.");
  }
  const payload = await response.json().catch(() => ({}));
  if (!response.ok) {
    throw new Error(formatApiError(payload, `Robustness report failed (HTTP ${response.status}).`));
  }
  return payload;
}

export async function predictAffectDynamics(blob, filename = "cerebro-recording.wav") {
  const formData = new FormData();
  formData.append("file", blob, filename);

  let response;
  try {
    response = await cerebroFetch(`${API_BASE}/audio/emotion/affect/dynamics`, {
      method: "POST",
      body: formData,
    });
  } catch {
    throw new Error("Cannot reach the CEREBRO affect-dynamics service.");
  }

  const payload = await response.json().catch(() => ({}));
  if (!response.ok) {
    throw new Error(formatApiError(payload, `Affect-dynamics analysis failed (HTTP ${response.status}).`));
  }
  return payload;
}

export async function predictDimensionalAffect(blob, filename = "cerebro-recording.wav") {
  const formData = new FormData();
  formData.append("file", blob, filename);

  let response;
  try {
    response = await cerebroFetch(`${API_BASE}/audio/emotion/affect`, {
      method: "POST",
      body: formData,
    });
  } catch {
    throw new Error("Cannot reach the CEREBRO dimensional-affect service.");
  }

  const payload = await response.json().catch(() => ({}));
  if (!response.ok) {
    throw new Error(formatApiError(payload, `Dimensional affect analysis failed (HTTP ${response.status}).`));
  }
  return payload;
}

export async function getDimensionalAffectStatus() {
  let response;
  try {
    response = await cerebroFetch(`${API_BASE}/audio/emotion/affect/status`);
  } catch {
    throw new Error("Cannot reach the CEREBRO dimensional-affect service.");
  }
  const payload = await response.json().catch(() => ({}));
  if (!response.ok) {
    throw new Error(formatApiError(payload, `Dimensional affect status failed (HTTP ${response.status}).`));
  }
  return payload;
}


export async function getSpeakerAwareStatus() {
  let response;
  try {
    response = await cerebroFetch(`${API_BASE}/audio/speakers/status`);
  } catch {
    throw new Error("Cannot reach the CEREBRO speaker-aware analysis service.");
  }
  const payload = await response.json().catch(() => ({}));
  if (!response.ok) {
    throw new Error(formatApiError(payload, `Speaker-aware status failed (HTTP ${response.status}).`));
  }
  return payload;
}

export async function analyzeSpeakerAware(blob, filename = "cerebro-conversation.wav") {
  const formData = new FormData();
  formData.append("file", blob, filename);

  let response;
  try {
    response = await cerebroFetch(`${API_BASE}/audio/speakers`, {
      method: "POST",
      body: formData,
    });
  } catch {
    throw new Error("Cannot reach the CEREBRO speaker-aware analysis service.");
  }
  const payload = await response.json().catch(() => ({}));
  if (!response.ok) {
    throw new Error(formatApiError(payload, `Speaker-aware analysis failed (HTTP ${response.status}).`));
  }
  return payload;
}


export async function getVoiceHistory(profileId) {
  let response;
  try {
    response = await cerebroFetch(`${API_BASE}/audio/history/${encodeURIComponent(profileId)}`);
  } catch {
    throw new Error("Cannot reach the CEREBRO longitudinal voice service.");
  }
  const payload = await response.json().catch(() => ({}));
  if (!response.ok) {
    throw new Error(formatApiError(payload, `Voice history failed (HTTP ${response.status}).`));
  }
  return payload;
}

export async function saveVoiceHistorySession(profileId, session) {
  let response;
  try {
    response = await cerebroFetch(`${API_BASE}/audio/history/${encodeURIComponent(profileId)}/sessions`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(session),
    });
  } catch {
    throw new Error("Cannot reach the CEREBRO longitudinal voice service.");
  }
  const payload = await response.json().catch(() => ({}));
  if (!response.ok) {
    throw new Error(formatApiError(payload, `Voice-history save failed (HTTP ${response.status}).`));
  }
  return payload;
}

export async function deleteVoiceHistory(profileId) {
  let response;
  try {
    response = await cerebroFetch(`${API_BASE}/audio/history/${encodeURIComponent(profileId)}`, {
      method: "DELETE",
    });
  } catch {
    throw new Error("Cannot reach the CEREBRO longitudinal voice service.");
  }
  const payload = await response.json().catch(() => ({}));
  if (!response.ok) {
    throw new Error(formatApiError(payload, `Voice-history deletion failed (HTTP ${response.status}).`));
  }
  return payload;
}

export async function getVoiceChangeStatus() {
  let response;
  try {
    response = await cerebroFetch(`${API_BASE}/audio/voice-change/status`);
  } catch {
    throw new Error("Cannot reach the CEREBRO voice-change service.");
  }
  const payload = await response.json().catch(() => ({}));
  if (!response.ok) {
    throw new Error(formatApiError(payload, `Voice-change status failed (HTTP ${response.status}).`));
  }
  return payload;
}

export async function analyzeCurrentVoiceChange(profileId, session) {
  let response;
  try {
    response = await cerebroFetch(`${API_BASE}/audio/voice-change/${encodeURIComponent(profileId)}/change`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(session),
    });
  } catch {
    throw new Error("Cannot reach the CEREBRO voice-change service.");
  }
  const payload = await response.json().catch(() => ({}));
  if (!response.ok) {
    throw new Error(formatApiError(payload, `Voice-change analysis failed (HTTP ${response.status}).`));
  }
  return payload;
}

export async function getLatestVoiceChange(profileId) {
  let response;
  try {
    response = await cerebroFetch(`${API_BASE}/audio/voice-change/${encodeURIComponent(profileId)}/change`);
  } catch {
    throw new Error("Cannot reach the CEREBRO voice-change service.");
  }
  const payload = await response.json().catch(() => ({}));
  if (!response.ok) {
    throw new Error(formatApiError(payload, `Latest voice-change analysis failed (HTTP ${response.status}).`));
  }
  return payload;
}


export async function getCrossSessionPatternStatus() {
  let response;
  try {
    response = await cerebroFetch(`${API_BASE}/audio/history-patterns/status`);
  } catch {
    throw new Error("Cannot reach the CEREBRO cross-session pattern service.");
  }
  const payload = await response.json().catch(() => ({}));
  if (!response.ok) {
    throw new Error(formatApiError(payload, `Cross-session pattern status failed (HTTP ${response.status}).`));
  }
  return payload;
}

export async function getCrossSessionPatterns(profileId) {
  let response;
  try {
    response = await cerebroFetch(`${API_BASE}/audio/history-patterns/${encodeURIComponent(profileId)}`);
  } catch {
    throw new Error("Cannot reach the CEREBRO cross-session pattern service.");
  }
  const payload = await response.json().catch(() => ({}));
  if (!response.ok) {
    throw new Error(formatApiError(payload, `Cross-session pattern analysis failed (HTTP ${response.status}).`));
  }
  return payload;
}


export async function getPersonalizationStatus(profileId) {
  let response;
  try {
    response = await cerebroFetch(`${API_BASE}/audio/personalization/${encodeURIComponent(profileId)}`);
  } catch {
    throw new Error("Cannot reach the CEREBRO personalized-calibration service.");
  }
  const payload = await response.json().catch(() => ({}));
  if (!response.ok) {
    throw new Error(formatApiError(payload, `Personalized calibration status failed (HTTP ${response.status}).`));
  }
  return payload;
}

export async function submitPersonalizationFeedback(profileId, data) {
  let response;
  try {
    response = await cerebroFetch(`${API_BASE}/audio/personalization/${encodeURIComponent(profileId)}/feedback`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(data),
    });
  } catch {
    throw new Error("Cannot reach the CEREBRO personalized-calibration service.");
  }
  const payload = await response.json().catch(() => ({}));
  if (!response.ok) {
    throw new Error(formatApiError(payload, `Personalized feedback failed (HTTP ${response.status}).`));
  }
  return payload;
}

export async function predictPersonalizedEmotion(profileId, data) {
  let response;
  try {
    response = await cerebroFetch(`${API_BASE}/audio/personalization/${encodeURIComponent(profileId)}/predict`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(data),
    });
  } catch {
    throw new Error("Cannot reach the CEREBRO personalized-calibration service.");
  }
  const payload = await response.json().catch(() => ({}));
  if (!response.ok) {
    throw new Error(formatApiError(payload, `Personalized prediction failed (HTTP ${response.status}).`));
  }
  return payload;
}

export async function deletePersonalization(profileId) {
  let response;
  try {
    response = await cerebroFetch(`${API_BASE}/audio/personalization/${encodeURIComponent(profileId)}`, {
      method: "DELETE",
    });
  } catch {
    throw new Error("Cannot reach the CEREBRO personalized-calibration service.");
  }
  const payload = await response.json().catch(() => ({}));
  if (!response.ok) {
    throw new Error(formatApiError(payload, `Personalized calibration deletion failed (HTTP ${response.status}).`));
  }
  return payload;
}

export async function getOptimizationStatus() {
  let response;
  try {
    response = await cerebroFetch(`${API_BASE}/evaluation/optimization/status`);
  } catch {
    throw new Error("Cannot reach the CEREBRO model-optimization service.");
  }
  const payload = await response.json().catch(() => ({}));
  if (!response.ok) {
    throw new Error(formatApiError(payload, `Optimization status failed (HTTP ${response.status}).`));
  }
  return payload;
}

export async function getOptimizationReport() {
  let response;
  try {
    response = await cerebroFetch(`${API_BASE}/evaluation/optimization/report`);
  } catch {
    throw new Error("Cannot reach the CEREBRO model-optimization service.");
  }
  const payload = await response.json().catch(() => ({}));
  if (!response.ok) {
    throw new Error(formatApiError(payload, `Optimization report failed (HTTP ${response.status}).`));
  }
  return payload;
}

export async function getPrivacyStatus() {
  let response;
  try {
    response = await cerebroFetch(`${API_BASE}/privacy/status`);
  } catch {
    throw new Error("Cannot reach the CEREBRO privacy service.");
  }
  const payload = await response.json().catch(() => ({}));
  if (!response.ok) {
    throw new Error(formatApiError(payload, `Privacy status failed (HTTP ${response.status}).`));
  }
  return payload;
}

export async function getPrivacyProfile(profileId) {
  let response;
  try {
    response = await cerebroFetch(`${API_BASE}/privacy/${encodeURIComponent(profileId)}`);
  } catch {
    throw new Error("Cannot reach the CEREBRO privacy service.");
  }
  const payload = await response.json().catch(() => ({}));
  if (!response.ok) {
    throw new Error(formatApiError(payload, `Privacy profile failed (HTTP ${response.status}).`));
  }
  return payload;
}

export async function deleteAllProfileData(profileId) {
  let response;
  try {
    response = await cerebroFetch(`${API_BASE}/privacy/${encodeURIComponent(profileId)}`, {
      method: "DELETE",
    });
  } catch {
    throw new Error("Cannot reach the CEREBRO privacy service.");
  }
  const payload = await response.json().catch(() => ({}));
  if (!response.ok) {
    throw new Error(formatApiError(payload, `Profile deletion failed (HTTP ${response.status}).`));
  }
  return payload;
}
