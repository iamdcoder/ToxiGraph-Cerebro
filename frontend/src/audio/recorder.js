export const WORKLET_SOURCE = `
class CerebroCaptureProcessor extends AudioWorkletProcessor {
  process(inputs) {
    const input = inputs[0];
    if (!input || !input[0]) {
      return true;
    }

    this.port.postMessage(new Float32Array(input[0]));
    return true;
  }
}

registerProcessor("cerebro-capture-processor", CerebroCaptureProcessor);
`;

function writeAscii(view, offset, value) {
  for (let index = 0; index < value.length; index += 1) {
    view.setUint8(offset + index, value.charCodeAt(index));
  }
}

function encodeWav(samples, sampleRate) {
  const buffer = new ArrayBuffer(44 + samples.length * 2);
  const view = new DataView(buffer);

  writeAscii(view, 0, "RIFF");
  view.setUint32(4, 36 + samples.length * 2, true);
  writeAscii(view, 8, "WAVE");
  writeAscii(view, 12, "fmt ");
  view.setUint32(16, 16, true);
  view.setUint16(20, 1, true);
  view.setUint16(22, 1, true);
  view.setUint32(24, sampleRate, true);
  view.setUint32(28, sampleRate * 2, true);
  view.setUint16(32, 2, true);
  view.setUint16(34, 16, true);
  writeAscii(view, 36, "data");
  view.setUint32(40, samples.length * 2, true);

  let offset = 44;
  for (const sample of samples) {
    const clamped = Math.max(-1, Math.min(1, sample));
    const pcm = clamped < 0 ? clamped * 32768 : clamped * 32767;
    view.setInt16(offset, pcm, true);
    offset += 2;
  }

  return new Blob([buffer], { type: "audio/wav" });
}

export class BrowserAudioRecorder {
  constructor({ maxDurationSeconds = 60, onLevel } = {}) {
    this.maxDurationSeconds = maxDurationSeconds;
    this.onLevel = onLevel;
    this.stream = null;
    this.audioContext = null;
    this.source = null;
    this.processor = null;
    this.levelAnalyser = null;
    this.levelTimer = null;
    this.workletUrl = null;
    this.chunks = [];
    this.startedAt = 0;
    this.timeoutId = null;
    this.state = "idle";
  }

  async start() {
    if (this.state !== "idle") {
      throw new Error("The recorder is already active.");
    }

    if (!navigator.mediaDevices?.getUserMedia) {
      throw new Error("This browser does not support microphone capture.");
    }

    this.stream = await navigator.mediaDevices.getUserMedia({
      audio: {
        echoCancellation: true,
        noiseSuppression: true,
        autoGainControl: true,
      },
    });

    try {
      this.audioContext = new AudioContext();
      await this.audioContext.resume();

      const blob = new Blob([WORKLET_SOURCE], {
        type: "application/javascript",
      });
      this.workletUrl = URL.createObjectURL(blob);
      await this.audioContext.audioWorklet.addModule(this.workletUrl);

      this.source = this.audioContext.createMediaStreamSource(this.stream);
      this.processor = new AudioWorkletNode(
        this.audioContext,
        "cerebro-capture-processor",
        { numberOfInputs: 1, numberOfOutputs: 1, outputChannelCount: [1] }
      );
      this.levelAnalyser = this.audioContext.createAnalyser();
      this.levelAnalyser.fftSize = 512;

      this.chunks = [];
      this.processor.port.onmessage = (event) => {
        if (event.data instanceof Float32Array) {
          this.chunks.push(new Float32Array(event.data));
        }
      };

      this.source.connect(this.processor);
      this.processor.connect(this.levelAnalyser);
      this.levelAnalyser.connect(this.audioContext.destination);
      this.processor.channelCountMode = "explicit";
      this.processor.channelCount = 1;

      this.state = "recording";
      this.startedAt = performance.now();
      this._startLevelLoop();

      this.timeoutId = window.setTimeout(() => {
        if (this.state === "recording") {
          this.stop().catch(() => undefined);
        }
      }, this.maxDurationSeconds * 1000);
    } catch (error) {
      this.cancel();
      throw error;
    }
  }

  async stop() {
    if (this.state !== "recording") {
      throw new Error("There is no active recording.");
    }

    this.state = "processing";
    window.clearTimeout(this.timeoutId);
    this.timeoutId = null;
    window.clearInterval(this.levelTimer);
    this.levelTimer = null;
    this.onLevel?.(0);

    const sampleRate = this.audioContext.sampleRate;
    const samples = this._mergeChunks();
    const blob = encodeWav(samples, sampleRate);
    const durationSeconds = samples.length / sampleRate;

    this._cleanup();
    this.state = "idle";

    return {
      blob,
      durationSeconds,
      sampleRate,
    };
  }

  cancel() {
    window.clearTimeout(this.timeoutId);
    this.timeoutId = null;
    window.clearInterval(this.levelTimer);
    this.levelTimer = null;
    this.onLevel?.(0);
    this._cleanup();
    this.state = "idle";
  }

  _mergeChunks() {
    const totalLength = this.chunks.reduce(
      (total, chunk) => total + chunk.length,
      0
    );
    const merged = new Float32Array(totalLength);
    let offset = 0;

    for (const chunk of this.chunks) {
      merged.set(chunk, offset);
      offset += chunk.length;
    }

    return merged;
  }

  _startLevelLoop() {
    const values = new Uint8Array(this.levelAnalyser.fftSize);
    this.levelTimer = window.setInterval(() => {
      if (this.state !== "recording" || !this.levelAnalyser) {
        return;
      }

      this.levelAnalyser.getByteTimeDomainData(values);
      let sum = 0;
      for (const value of values) {
        const centered = (value - 128) / 128;
        sum += centered * centered;
      }
      const rms = Math.sqrt(sum / values.length);
      const normalized = Math.min(1, rms * 4);
      this.onLevel?.(normalized);
    }, 80);
  }

  _cleanup() {
    if (this.source) {
      this.source.disconnect();
    }
    if (this.processor) {
      this.processor.disconnect();
      this.processor.port.onmessage = null;
    }
    if (this.levelAnalyser) {
      this.levelAnalyser.disconnect();
    }
    if (this.stream) {
      for (const track of this.stream.getTracks()) {
        track.stop();
      }
    }
    if (this.audioContext && this.audioContext.state !== "closed") {
      this.audioContext.close().catch(() => undefined);
    }
    if (this.workletUrl) {
      URL.revokeObjectURL(this.workletUrl);
    }

    this.source = null;
    this.processor = null;
    this.levelAnalyser = null;
    this.stream = null;
    this.audioContext = null;
    this.workletUrl = null;
    this.chunks = [];
  }
}
