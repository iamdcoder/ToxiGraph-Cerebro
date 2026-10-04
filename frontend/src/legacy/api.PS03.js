const API_BASE =
  import.meta.env.VITE_API_BASE_URL ||
  "/api/v1";

function formatApiError(payload) {
  const detail = payload?.detail;

  if (typeof detail === "string" && detail.trim()) {
    return detail;
  }

  if (Array.isArray(detail)) {
    const messages = detail
      .map((item) => {
        if (typeof item === "string") {
          return item;
        }

        const location = Array.isArray(item?.loc)
          ? item.loc.join(" → ")
          : "input";

        const message = item?.msg || "Invalid value";
        return `${location}: ${message}`;
      })
      .filter(Boolean);

    if (messages.length) {
      return messages.join(" · ");
    }
  }

  if (detail && typeof detail === "object") {
    try {
      return JSON.stringify(detail);
    } catch {
      return "The backend returned an unreadable error.";
    }
  }

  return "Thread analysis failed.";
}

export async function analyzeThread(thread) {
  let response;

  try {
    response = await fetch(
      `${API_BASE}/analysis/analyze`,
      {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
        },
        body: JSON.stringify(thread),
      }
    );
  } catch {
    throw new Error(
      "Cannot reach the ToxiGraph backend. Start FastAPI on port 8000 and try again."
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
    throw new Error(formatApiError(payload));
  }

  return payload;
}

export async function loadExampleThread() {
  let response;

  try {
    response = await fetch("/example_thread.json");
  } catch {
    throw new Error(
      "The example thread could not be reached. Check the Vite server."
    );
  }

  if (!response.ok) {
    throw new Error(
      `Example thread could not be loaded (HTTP ${response.status}).`
    );
  }

  try {
    return await response.json();
  } catch {
    throw new Error("The example thread file is not valid JSON.");
  }
}
