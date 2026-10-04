import { WORKLET_SOURCE } from "./recorder";

function buildWebSocketUrl(socketPath = "/audio/live/ws") {
  const configured = import.meta.env.VITE_API_BASE_URL;
  if (configured) {
    const base = new URL(configured, window.location.href);
    base.protocol = base.protocol === "https:" ? "wss:" : "ws:";
    base.pathname = `${base.pathname.replace(/\/$/, "")}${socketPath.startsWith("/") ? socketPath : `/${socketPath}`}`;
    base.search = "";
    return base.toString();
  }

  const protocol = window.location.protocol === "https:" ? "wss:" : "ws:";
  return `${protocol}//${window.location.host}/api/v1${socketPath}`;
}

function float32ToPcm16(samples) {
  const pcm = new Int16Array(samples.length);
  for (let index = 0; index < samples.length; index += 1) {
    const clamped = Math.max(-1, Math.min(1, samples[index]));
    pcm[index] = clamped < 0 ? clamped * 32768 : clamped * 32767;
  }
  return pcm;
}

function mergeFloat32(chunks, length) {
  const merged = new Float32Array(length);
  let offset = 0;
  for (const chunk of chunks) {
    merged.set(chunk, offset);
    offset += chunk.length;
  }
  return merged;
}

export class LiveAudioStreamer {
  constructor({ onMessage, onLevel, maxDurationSeconds = 120, chunkMilliseconds = 200, socketPath = "/audio/live/ws" } = {}) {
    this.onMessage = onMessage;
    this.onLevel = onLevel;
    this.maxDurationSeconds = maxDurationSeconds;
    this.chunkMilliseconds = chunkMilliseconds;
    this.socketPath = socketPath;
    this.stream = null;
    this.audioContext = null;
    this.source = null;
    this.processor = null;
    this.analyser = null;
    this.silentGain = null;
    this.workletUrl = null;
    this.socket = null;
    this.chunkQueue = [];
    this.queuedSamples = 0;
    this.startedAt = 0;
    this.levelTimer = null;
    this.timeoutId = null;
    this.state = "idle";
    this.stopPromise = null;
    this.stopResolve = null;
    this.stopReject = null;
  }

  async start() {
    if (this.state !== "idle") {
      throw new Error("The live recorder is already active.");
    }

    if (!window.isSecureContext && !["localhost", "127.0.0.1"].includes(window.location.hostname)) {
      throw new Error("Live microphone capture requires HTTPS or a local development origin.");
    }
    if (!navigator.mediaDevices?.getUserMedia) {
      throw new Error("This browser does not support microphone capture.");
    }

    this.state = "starting";
    this.stream = await navigator.mediaDevices.getUserMedia({
      audio: {
        channelCount: 1,
        echoCancellation: true,
        noiseSuppression: true,
        autoGainControl: true,
      },
    });

    try {
      this.audioContext = new AudioContext();
      await this.audioContext.resume();
      this.analyser = this.audioContext.createAnalyser();
      this.analyser.fftSize = 512;
      this.silentGain = this.audioContext.createGain();
      this.silentGain.gain.value = 0;

      const workletBlob = new Blob([WORKLET_SOURCE], { type: "application/javascript" });
      this.workletUrl = URL.createObjectURL(workletBlob);
      await this.audioContext.audioWorklet.addModule(this.workletUrl);

      this.source = this.audioContext.createMediaStreamSource(this.stream);
      this.processor = new AudioWorkletNode(
        this.audioContext,
        "cerebro-capture-processor",
        { numberOfInputs: 1, numberOfOutputs: 1, outputChannelCount: [1] },
      );

      this.source.connect(this.processor);
      this.source.connect(this.analyser);
      this.processor.connect(this.silentGain);
      this.analyser.connect(this.silentGain);
      this.silentGain.connect(this.audioContext.destination);

      this.chunkQueue = [];
      this.queuedSamples = 0;
      const targetChunkSamples = Math.max(
        1,
        Math.round(this.audioContext.sampleRate * (this.chunkMilliseconds / 1000)),
      );

      await this._connectSocket({ sampleRate: this.audioContext.sampleRate });

      this.processor.port.onmessage = (event) => {
        if (this.state !== "live" || !(event.data instanceof Float32Array)) {
          return;
        }

        const chunk = new Float32Array(event.data);
        this.chunkQueue.push(chunk);
        this.queuedSamples += chunk.length;

        while (this.queuedSamples >= targetChunkSamples) {
          const merged = mergeFloat32(this.chunkQueue, this.queuedSamples);
          const current = merged.slice(0, targetChunkSamples);
          const remainder = merged.slice(targetChunkSamples);
          this.chunkQueue = remainder.length ? [remainder] : [];
          this.queuedSamples = remainder.length;
          if (this.socket?.readyState === WebSocket.OPEN) {
            const pcm = float32ToPcm16(current);
            this.socket.send(pcm.buffer);
          }
        }
      };

      this.state = "live";
      this.startedAt = performance.now();
      this._startLevelLoop();
      this.timeoutId = window.setTimeout(() => {
        if (this.state === "live") {
          this.stop().catch(() => undefined);
        }
      }, this.maxDurationSeconds * 1000);
    } catch (error) {
      this.cancel();
      throw error;
    }
  }

  async stop() {
    if (this.state !== "live") {
      throw new Error("There is no active live recording.");
    }

    this.state = "stopping";
    window.clearTimeout(this.timeoutId);
    this.timeoutId = null;
    window.clearInterval(this.levelTimer);
    this.levelTimer = null;
    this.onLevel?.(0);

    if (this.socket?.readyState === WebSocket.OPEN) {
      this.stopPromise = new Promise((resolve, reject) => {
        this.stopResolve = resolve;
        this.stopReject = reject;
      });
      this.socket.send(JSON.stringify({ type: "stop" }));
      return this.stopPromise;
    }

    this._cleanup();
    this.state = "idle";
    return null;
  }

  cancel() {
    window.clearTimeout(this.timeoutId);
    this.timeoutId = null;
    window.clearInterval(this.levelTimer);
    this.levelTimer = null;
    this.onLevel?.(0);
    try {
      this.socket?.close(1000, "client_cancelled");
    } catch {
      // Ignore already-closed sockets.
    }
    this._cleanup();
    this.state = "idle";
    this.stopResolve?.(null);
    this.stopReject = null;
    this.stopResolve = null;
    this.stopPromise = null;
  }

  async _connectSocket({ sampleRate }) {
    const socket = new WebSocket(buildWebSocketUrl(this.socketPath));
    this.socket = socket;
    socket.binaryType = "arraybuffer";

    await new Promise((resolve, reject) => {
      const timeout = window.setTimeout(() => {
        reject(new Error("Timed out connecting to the CEREBRO live analysis service."));
        socket.close();
      }, 10000);

      socket.onopen = () => {
        window.clearTimeout(timeout);
        const token = import.meta.env.VITE_CEREBRO_WS_TOKEN || "";
        socket.send(JSON.stringify({
          type: "start",
          sample_rate: sampleRate,
          client: "cerebro-web",
          ...(token ? { token } : {}),
        }));
      };

      socket.onerror = () => {
        window.clearTimeout(timeout);
        reject(new Error("Cannot connect to the CEREBRO live WebSocket service."));
      };

      socket.onmessage = (event) => {
        const payload = JSON.parse(event.data);
        this.onMessage?.(payload);
        if (payload.type === "ready") {
          resolve();
        }
        if (payload.type === "error") {
          window.clearTimeout(timeout);
          reject(new Error(payload.message || "The live analysis service rejected the session."));
        }
      };

      socket.onclose = () => {
        window.clearTimeout(timeout);
        if (this.state === "live" || this.state === "stopping") {
          this.onMessage?.({
            type: "error",
            code: "socket_closed",
            message: "The live analysis connection closed unexpectedly.",
          });
        }
      };
    });

    socket.onmessage = (event) => {
      let payload;
      try {
        payload = JSON.parse(event.data);
      } catch {
        return;
      }
      this.onMessage?.(payload);
      if (payload.type === "complete") {
        this._cleanup();
        this.state = "idle";
        this.stopResolve?.(payload);
        this.stopResolve = null;
        this.stopReject = null;
        this.stopPromise = null;
      } else if (payload.type === "error" && this.state === "stopping") {
        this._cleanup();
        this.state = "idle";
        this.stopReject?.(new Error(payload.message || "Live analysis failed."));
        this.stopResolve = null;
        this.stopReject = null;
        this.stopPromise = null;
      }
    };
  }

  _startLevelLoop() {
    const values = new Uint8Array(this.analyser.fftSize);
    this.levelTimer = window.setInterval(() => {
      if (this.state !== "live" || !this.analyser) {
        return;
      }
      this.analyser.getByteTimeDomainData(values);
      let sum = 0;
      for (const value of values) {
        const centered = (value - 128) / 128;
        sum += centered * centered;
      }
      const rms = Math.sqrt(sum / values.length);
      this.onLevel?.(Math.min(1, rms * 4));
    }, 80);
  }

  _cleanup() {
    if (this.processor) {
      this.processor.disconnect();
      this.processor.port.onmessage = null;
    }
    if (this.source) {
      this.source.disconnect();
    }
    if (this.analyser) {
      this.analyser.disconnect();
    }
    if (this.silentGain) {
      this.silentGain.disconnect();
    }
    if (this.stream) {
      this.stream.getTracks().forEach((track) => track.stop());
    }
    if (this.audioContext && this.audioContext.state !== "closed") {
      this.audioContext.close().catch(() => undefined);
    }
    if (this.workletUrl) {
      URL.revokeObjectURL(this.workletUrl);
    }
    if (this.socket && this.socket.readyState !== WebSocket.CLOSED) {
      try {
        this.socket.close(1000, "session_complete");
      } catch {
        // Ignore socket close races.
      }
    }

    this.processor = null;
    this.source = null;
    this.analyser = null;
    this.silentGain = null;
    this.stream = null;
    this.audioContext = null;
    this.socket = null;
    this.workletUrl = null;
    this.chunkQueue = [];
    this.queuedSamples = 0;
  }
}
