const Recognition = window.SpeechRecognition || window.webkitSpeechRecognition;
const synth = "speechSynthesis" in window ? window.speechSynthesis : null;

const el = {
  log: document.getElementById("chat-log"),
  empty: document.getElementById("emptyState"),
  form: document.getElementById("composerForm"),
  input: document.getElementById("input"),
  micBtn: document.getElementById("micBtn"),
  micLabel: document.getElementById("micLabel"),
  stopSpeakBtn: document.getElementById("stopSpeakBtn"),
  status: document.getElementById("status"),
  statusText: document.getElementById("statusText"),
  health: document.getElementById("health"),
  healthText: document.getElementById("healthText"),
  banner: document.getElementById("banner"),
  bannerTitle: document.getElementById("bannerTitle"),
  bannerText: document.getElementById("bannerText"),
  bannerClose: document.getElementById("bannerClose"),
  announcer: document.getElementById("announcer"),
  language: document.getElementById("languageSelect"),
  voice: document.getElementById("voiceSelect"),
  voiceNote: document.getElementById("voiceNote"),
  rate: document.getElementById("rateRange"),
  rateOut: document.getElementById("rateOut"),
  handsFree: document.getElementById("handsFree"),
  clearBtn: document.getElementById("clearBtn"),
  clearDialog: document.getElementById("clearDialog"),
};

const PREFS_KEY = "voicechat.prefs.v1";
const MESSAGES_KEY = "voicechat.messages.v1";
const MAX_SAVED_MESSAGES = 200;

const LANGUAGES = [
  { code: "auto", label: "Detect language" },
  { code: "en-US", label: "English (US)" },
  { code: "en-GB", label: "English (UK)" },
  { code: "sw-KE", label: "Kiswahili" },
  { code: "es-ES", label: "Español" },
  { code: "fr-FR", label: "Français" },
  { code: "de-DE", label: "Deutsch" },
  { code: "pt-BR", label: "Português (BR)" },
  { code: "hi-IN", label: "हिन्दी" },
  { code: "ar-SA", label: "العربية" },
  { code: "ja-JP", label: "日本語" },
  { code: "zh-CN", label: "中文" },
];

const RECOGNITION_ERRORS = {
  "not-allowed": "Microphone access was blocked. Allow it from the address bar, or type instead.",
  "service-not-allowed": "This browser blocked speech recognition. You can still type.",
  "audio-capture": "No microphone was found. Check your device, or type instead.",
  network: "Speech recognition needs a network connection in this browser.",
};

const STATE_LABELS = {
  idle: "Ready",
  listening: "Listening…",
  thinking: "Thinking…",
  speaking: "Speaking…",
};

const state = {
  messages: [],
  status: "idle",
  recognition: null,
  controller: null,
  replyComplete: false,
  bannerSource: null,
};

const prefs = Object.assign(
  { voiceURI: "", language: "auto", rate: 1, handsFree: false },
  readJSON(PREFS_KEY) || {},
);

function readJSON(key) {
  try {
    return JSON.parse(localStorage.getItem(key));
  } catch {
    return null;
  }
}

function writeJSON(key, value) {
  try {
    localStorage.setItem(key, JSON.stringify(value));
  } catch {
    /* storage unavailable — the conversation just won't persist */
  }
}

/* ------------------------------------------------------------------ speech */

class Speaker {
  constructor(synthesis) {
    this.synthesis = synthesis;
    this.queue = [];
    this.utterance = null;
    this.watchdog = null;
    this.voice = null;
    this.rate = 1;
    this.onchange = () => {};
  }

  get busy() {
    return Boolean(this.utterance) || this.queue.length > 0;
  }

  configure({ voice, rate }) {
    if (voice !== undefined) this.voice = voice;
    if (rate !== undefined) this.rate = rate;
  }

  say(text) {
    if (!this.synthesis) return;
    const chunks = chunkForSpeech(text);
    if (!chunks.length) return;
    this.queue.push(...chunks);
    this.pump();
    this.onchange();
  }

  stop() {
    this.queue.length = 0;
    this.clearWatchdog();
    if (this.utterance) {
      this.utterance = null;
      this.synthesis.cancel();
    }
    this.onchange();
  }

  clearWatchdog() {
    if (this.watchdog) {
      clearTimeout(this.watchdog);
      this.watchdog = null;
    }
  }

  /** Recover when the engine reports "speaking" but never fires onend. */
  armWatchdog(utterance, text) {
    this.clearWatchdog();
    const budget = Math.max(6000, (text.length / 12) * 1000 / (this.rate || 1)) + 6000;
    this.watchdog = setTimeout(() => {
      if (this.utterance !== utterance) return;
      console.warn("[voice] speech engine stalled; advancing the queue");
      this.utterance = null;
      this.synthesis.cancel();
      this.pump();
    }, budget);
  }

  pump() {
    if (!this.synthesis || this.utterance) return;
    const next = this.queue.shift();
    if (!next) {
      this.onchange();
      return;
    }

    const utterance = new SpeechSynthesisUtterance(next);
    if (this.voice) {
      utterance.voice = this.voice;
      utterance.lang = this.voice.lang;
    }
    utterance.rate = this.rate;
    this.utterance = utterance;

    const advance = () => {
      if (this.utterance === utterance) {
        this.utterance = null;
        this.clearWatchdog();
        this.pump();
      }
    };
    utterance.onend = advance;
    utterance.onerror = advance;
    utterance.onboundary = () => this.armWatchdog(utterance, next);

    this.armWatchdog(utterance, next);
    this.synthesis.speak(utterance);
    this.onchange();
  }
}

const speaker = new Speaker(synth);

function toSpeech(text) {
  return text
    .replace(/```[\s\S]*?```/g, " ")
    .replace(/\[([^\]]+)\]\([^)]*\)/g, "$1")
    .replace(/https?:\/\/\S+/gi, "link")
    .replace(/[*_~`#>]+/g, "")
    .replace(/\s+/g, " ")
    .trim();
}

/** Split into sentences, then hard-wrap anything long so browsers don't stall. */
function chunkForSpeech(text, max = 180) {
  const clean = toSpeech(text);
  if (!clean) return [];
  const sentences = clean.match(/[^.!?…]+[.!?…]+["')\]]*|[^.!?…]+$/g) || [clean];
  const chunks = [];
  for (const sentence of sentences) {
    let rest = sentence.trim();
    while (rest.length > max) {
      let cut = rest.lastIndexOf(",", max);
      if (cut < max * 0.5) cut = rest.lastIndexOf(" ", max);
      if (cut < max * 0.5) cut = max;
      chunks.push(rest.slice(0, cut + 1).trim());
      rest = rest.slice(cut + 1).trim();
    }
    if (rest) chunks.push(rest);
  }
  return chunks;
}

/** Pull finished sentences out of a streaming buffer. */
function takeSentences(buffer) {
  const sentences = [];
  let rest = buffer;
  const re = /^[\s\S]*?[.!?…]["')\]]*(?=\s)/;
  let match;
  while ((match = re.exec(rest))) {
    sentences.push(match[0].trim());
    rest = rest.slice(match[0].length).replace(/^\s+/, "");
  }
  return { sentences, rest };
}

/* ------------------------------------------------------------------ voices */

function currentLanguage() {
  if (prefs.language && prefs.language !== "auto") return prefs.language;
  return navigator.language || "en-US";
}

function voicesForLanguage(language) {
  const all = (synth ? synth.getVoices() : []).filter((voice) => voice.lang);
  const base = language.toLowerCase().split("-")[0];
  const exact = all.filter(
    (voice) => voice.lang.replace("_", "-").toLowerCase() === language.toLowerCase(),
  );
  if (exact.length) return { voices: exact, matched: true };
  const sameBase = all.filter(
    (voice) => voice.lang.replace("_", "-").toLowerCase().split("-")[0] === base,
  );
  if (sameBase.length) return { voices: sameBase, matched: true };
  return { voices: all, matched: false };
}

function populateLanguages() {
  el.language.replaceChildren(...LANGUAGES.map(({ code, label }) => new Option(label, code)));
  el.language.value = prefs.language;
}

function populateVoices() {
  if (!synth) return;
  const { voices, matched } = voicesForLanguage(currentLanguage());
  if (!voices.length) {
    el.voiceNote.hidden = false;
    el.voiceNote.textContent =
      "No voices are available yet. Some browsers load them a moment after the page opens.";
    return;
  }

  const sorted = [...voices].sort(
    (a, b) => Number(b.default) - Number(a.default) || a.name.localeCompare(b.name),
  );
  el.voice.replaceChildren(
    ...sorted.map((voice) => new Option(`${voice.name} (${voice.lang})`, voice.voiceURI)),
  );

  const remembered = sorted.find((voice) => voice.voiceURI === prefs.voiceURI);
  const chosen = remembered || sorted[0];
  el.voice.value = chosen ? chosen.voiceURI : "";
  prefs.voiceURI = el.voice.value;
  speaker.configure({ voice: chosen || null, rate: prefs.rate });

  el.voiceNote.hidden = matched;
  if (!matched) {
    el.voiceNote.textContent =
      "No voice is installed for this language; showing every available voice.";
  }
  savePrefs();
}

/* ------------------------------------------------------------------- status */

function setStatus(name, text) {
  state.status = name;
  el.status.dataset.state = name;
  el.statusText.textContent = text || STATE_LABELS[name] || name;
  el.stopSpeakBtn.hidden = !speaker.busy;
}

/** Surface a problem without leaving the app stuck in an error state. */
function flagProblem(text) {
  el.status.dataset.state = "error";
  el.statusText.textContent = text;
}

function setHealth(kind, text) {
  el.health.dataset.state = kind;
  el.healthText.textContent = text;
}

function showBanner(title, message, source = "general") {
  el.bannerTitle.textContent = title;
  el.bannerText.textContent = message || "";
  el.banner.hidden = false;
  state.bannerSource = source;
}

function closeBanner() {
  el.banner.hidden = true;
  state.bannerSource = null;
}

function announce(message) {
  el.announcer.textContent = message;
}

/* ---------------------------------------------------------------- messages */

function createEmptyState() {
  const wrapper = document.createElement("div");
  wrapper.className = "empty";
  wrapper.id = "emptyState";
  wrapper.innerHTML = "";

  const icon = document.createElementNS("http://www.w3.org/2000/svg", "svg");
  icon.setAttribute("viewBox", "0 0 24 24");
  icon.setAttribute("width", "40");
  icon.setAttribute("height", "40");
  icon.setAttribute("fill", "none");
  icon.setAttribute("stroke", "currentColor");
  icon.setAttribute("stroke-width", "1.4");
  icon.setAttribute("stroke-linecap", "round");
  icon.setAttribute("aria-hidden", "true");
  const path = document.createElementNS("http://www.w3.org/2000/svg", "path");
  path.setAttribute("d", "M4 6.5A2.5 2.5 0 0 1 6.5 4h11A2.5 2.5 0 0 1 20 6.5v8a2.5 2.5 0 0 1-2.5 2.5H9l-5 4z");
  icon.append(path);

  const title = document.createElement("p");
  title.className = "empty-title";
  title.textContent = "Nothing here yet";
  const hint = document.createElement("p");
  hint.textContent = "Tap the microphone and start talking, or type a message below.";

  wrapper.append(icon, title, hint);
  return wrapper;
}

function hideEmpty() {
  if (el.empty && el.empty.isConnected) el.empty.remove();
  el.empty = null;
}

function addMessage(role, content, { streaming = false } = {}) {
  hideEmpty();
  const article = document.createElement("article");
  article.className = `message message--${role}`;
  article.dataset.role = role;

  const avatar = document.createElement("div");
  avatar.className = "avatar";
  avatar.setAttribute("aria-hidden", "true");
  avatar.textContent = role === "user" ? "You" : role === "error" ? "!" : "AI";

  const body = document.createElement("div");
  body.className = "message-body";

  const bubble = document.createElement("div");
  bubble.className = "bubble";

  const text = document.createElement("p");
  text.className = "bubble-text";
  text.textContent = content;
  bubble.append(text);

  const meta = document.createElement("div");
  meta.className = "meta";
  const label = document.createElement("span");
  label.textContent = role === "user" ? "You" : role === "error" ? "Error" : "Assistant";
  const time = document.createElement("time");
  time.textContent = new Date().toLocaleTimeString([], { hour: "2-digit", minute: "2-digit" });
  meta.append(label, time);

  body.append(bubble, meta);
  article.append(avatar, body);
  if (streaming) article.classList.add("is-streaming");
  el.log.append(article);
  scrollIfNearBottom();
  return article;
}

function isNearBottom() {
  return el.log.scrollHeight - el.log.scrollTop - el.log.clientHeight < 80;
}

function scrollIfNearBottom() {
  if (isNearBottom()) el.log.scrollTop = el.log.scrollHeight;
}

function persistMessages() {
  writeJSON(
    MESSAGES_KEY,
    state.messages.slice(-MAX_SAVED_MESSAGES).map(({ role, content, ts }) => ({
      role,
      content,
      ts,
    })),
  );
}

function restoreMessages() {
  const saved = readJSON(MESSAGES_KEY);
  if (!Array.isArray(saved)) return;
  state.messages = saved.filter(
    (entry) =>
      entry &&
      (entry.role === "user" || entry.role === "assistant") &&
      typeof entry.content === "string",
  );
  if (!state.messages.length) return;

  hideEmpty();
  for (const { role, content, ts } of state.messages) {
    const article = addMessage(role, content);
    if (ts) {
      article.querySelector("time").textContent = new Date(ts).toLocaleTimeString([], {
        hour: "2-digit",
        minute: "2-digit",
      });
    }
  }
  el.log.scrollTop = el.log.scrollHeight;
}

/* --------------------------------------------------------------------- chat */

function selectedLanguageLabel() {
  const match = LANGUAGES.find((item) => item.code === prefs.language);
  return match && match.code !== "auto" ? match.label : null;
}

async function apiError(response) {
  let detail = `The server returned ${response.status}.`;
  let hint = "";
  try {
    const data = await response.json();
    detail = data.detail || detail;
    hint = data.hint || "";
  } catch {
    /* non-JSON error body */
  }
  const error = new Error(hint ? `${detail} ${hint}` : detail);
  error.hint = hint;
  return error;
}

function maybeAutoListen() {
  if (!state.replyComplete) return;
  state.replyComplete = false;
  if (prefs.handsFree) window.setTimeout(startListening, 400);
}

async function send(rawText) {
  const content = rawText.trim();
  if (!content || state.status === "thinking") return;

  stopListening({ silent: true });
  speaker.stop();
  closeBanner();
  state.replyComplete = false;

  state.messages.push({ role: "user", content, ts: Date.now() });
  persistMessages();
  addMessage("user", content);

  state.controller?.abort();
  const controller = new AbortController();
  state.controller = controller;

  const bubble = addMessage("assistant", "", { streaming: true });
  const textEl = bubble.querySelector(".bubble-text");
  bubble.querySelector("time").textContent = "";
  setStatus("thinking", "Thinking…");

  let full = "";
  let pending = "";

  try {
    const response = await fetch("/api/chat", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        messages: state.messages.map(({ role, content: text }) => ({ role, content: text })),
        language: selectedLanguageLabel(),
      }),
      signal: controller.signal,
    });

    if (!response.ok) throw await apiError(response);

    const reader = response.body.pipeThrough(new TextDecoderStream()).getReader();
    let buffer = "";

    for (;;) {
      const { value, done } = await reader.read();
      if (done) break;
      buffer += value;

      let newline;
      while ((newline = buffer.indexOf("\n")) >= 0) {
        const line = buffer.slice(0, newline).trim();
        buffer = buffer.slice(newline + 1);
        if (!line) continue;

        let event;
        try {
          event = JSON.parse(line);
        } catch {
          continue;
        }

        if (event.type === "delta" && event.text) {
          full += event.text;
          textEl.textContent = full;
          pending += event.text;
          const { sentences, rest } = takeSentences(pending);
          pending = rest;
          for (const sentence of sentences) speaker.say(sentence);
          if (state.status === "thinking") setStatus("speaking", "Replying…");
          scrollIfNearBottom();
        } else if (event.type === "error") {
          throw new Error(event.message || "The model returned an error.");
        }
      }
    }

    if (pending.trim()) speaker.say(pending);
    if (!full.trim()) throw new Error("The model returned an empty reply.");

    bubble.classList.remove("is-streaming");
    bubble.querySelector("time").textContent = new Date().toLocaleTimeString([], {
      hour: "2-digit",
      minute: "2-digit",
    });
    state.messages.push({ role: "assistant", content: full, ts: Date.now() });
    persistMessages();
    announce(`Assistant said: ${full.slice(0, 300)}`);

    state.replyComplete = true;
    if (!speaker.busy) {
      setStatus("idle", STATE_LABELS.idle);
      maybeAutoListen();
    }
  } catch (error) {
    if (error.name === "AbortError") {
      bubble.remove();
      return;
    }
    speaker.stop();
    bubble.remove();
    addMessage("error", error.message);
    flagProblem("Couldn’t reply");
    showBanner(
      "The reply failed",
      error.hint
        ? error.message
        : `${error.message} Check that Ollama is running, then try again.`,
    );
    announce(`Error: ${error.message}`);
    checkHealth();
  } finally {
    if (state.controller === controller) state.controller = null;
  }
}

/* ---------------------------------------------------------------------- mic */

function setMicPressed(pressed) {
  el.micBtn.setAttribute("aria-pressed", String(pressed));
  el.micLabel.textContent = pressed ? "Listening…" : "Start listening";
  el.micBtn.classList.toggle("is-listening", pressed);
}

function stopListening({ silent = false } = {}) {
  const recognition = state.recognition;
  state.recognition = null;
  if (recognition) {
    try {
      recognition.abort();
    } catch {
      /* already stopped */
    }
  }
  setMicPressed(false);
  if (!silent && state.status === "listening") setStatus("idle", STATE_LABELS.idle);
}

function startListening() {
  if (!Recognition) {
    showBanner(
      "Voice input is unavailable",
      "This browser doesn’t support speech recognition. Chrome or Edge work best — you can still type.",
    );
    return;
  }
  if (state.status === "thinking" || speaker.busy) return;

  stopListening({ silent: true });
  speaker.stop();
  closeBanner();

  const recognition = new Recognition();
  recognition.lang = currentLanguage();
  recognition.interimResults = true;
  recognition.continuous = false;
  recognition.maxAlternatives = 1;

  let finalText = "";
  let failed = false;

  recognition.onstart = () => {
    state.recognition = recognition;
    setMicPressed(true);
    setStatus("listening", "Listening…");
  };

  recognition.onresult = (event) => {
    let interim = "";
    for (let i = event.resultIndex; i < event.results.length; i += 1) {
      const result = event.results[i];
      if (result.isFinal) finalText += result[0].transcript;
      else interim += result[0].transcript;
    }
    if (interim.trim()) el.statusText.textContent = `Hearing: ${interim.trim().slice(-60)}`;
  };

  recognition.onerror = (event) => {
    if (event.error === "aborted" || event.error === "no-speech") return;
    failed = true;
    showBanner(
      "Can’t listen right now",
      RECOGNITION_ERRORS[event.error] || `Speech recognition stopped (${event.error}).`,
    );
    flagProblem("Mic problem");
  };

  recognition.onend = () => {
    setMicPressed(false);
    if (state.recognition === recognition) state.recognition = null;
    if (finalText.trim()) {
      el.input.value = "";
      autosize();
      send(finalText.trim());
    } else if (!failed && state.status === "listening") {
      setStatus("idle", "Didn’t catch that — try again");
    }
  };

  try {
    recognition.start();
    state.recognition = recognition;
    setMicPressed(true);
  } catch (error) {
    stopListening({ silent: true });
    showBanner("Can’t start the microphone", error.message || "Try again in a moment.");
    flagProblem("Mic problem");
  }
}

speaker.onchange = () => {
  el.stopSpeakBtn.hidden = !speaker.busy;
  if (speaker.busy) {
    if (state.status !== "speaking") setStatus("speaking", "Speaking…");
    return;
  }
  if (state.status !== "speaking") return;
  setStatus("idle", STATE_LABELS.idle);
  maybeAutoListen();
};

/* ------------------------------------------------------------------- health */

async function checkHealth() {
  try {
    const response = await fetch("/api/health", { cache: "no-store" });
    const data = await response.json();
    if (!response.ok || !data.ollama) {
      setHealth("offline", "Ollama offline");
      showBanner(
        "Ollama isn’t running",
        data.hint || "Start it with `ollama serve`, then reload this page.",
        "health",
      );
      return;
    }
    if (!data.model_available) {
      setHealth("degraded", `Model “${data.model}” missing`);
      showBanner("Model not pulled", `Run \`ollama pull ${data.model}\` to download it.`, "health");
      return;
    }
    setHealth("ok", `Model ready · ${data.model}`);
    if (state.bannerSource === "health") closeBanner();
  } catch {
    setHealth("offline", "Server offline");
    showBanner(
      "Can’t reach the server",
      "Make sure `python server.py` is running, then reload.",
      "health",
    );
  }
}

/* --------------------------------------------------------------------- init */

function savePrefs() {
  writeJSON(PREFS_KEY, prefs);
}

function autosize() {
  el.input.style.height = "auto";
  el.input.style.height = `${Math.min(el.input.scrollHeight, 160)}px`;
}

function formatRate(value) {
  const rate = Number(value);
  return `${rate % 1 === 0 ? rate.toFixed(0) : rate.toFixed(2).replace(/0$/, "")}×`;
}

function confirmClear() {
  speaker.stop();
  stopListening({ silent: true });
  state.controller?.abort();
  state.replyComplete = false;
  state.messages = [];
  persistMessages();
  el.log.replaceChildren();
  el.empty = createEmptyState();
  el.log.append(el.empty);
  el.input.focus();
  setStatus("idle", "Chat cleared");
}

el.form.addEventListener("submit", (event) => {
  event.preventDefault();
  const text = el.input.value;
  el.input.value = "";
  autosize();
  send(text);
});

el.input.addEventListener("input", autosize);
el.input.addEventListener("keydown", (event) => {
  if (event.key === "Enter" && !event.shiftKey) {
    event.preventDefault();
    el.form.requestSubmit();
  }
});

el.micBtn.addEventListener("click", () => {
  if (state.status === "listening") stopListening();
  else startListening();
});

el.stopSpeakBtn.addEventListener("click", () => {
  speaker.stop();
  setStatus("idle", "Speech stopped");
});

el.bannerClose.addEventListener("click", closeBanner);

el.voice.addEventListener("change", () => {
  const voice = (synth ? synth.getVoices() : []).find((item) => item.voiceURI === el.voice.value);
  if (voice) speaker.configure({ voice });
  prefs.voiceURI = el.voice.value;
  savePrefs();
});

el.language.addEventListener("change", () => {
  prefs.language = el.language.value;
  savePrefs();
  populateVoices();
  if (state.status === "listening") stopListening();
});

el.rate.addEventListener("input", () => {
  prefs.rate = Number(el.rate.value);
  el.rateOut.textContent = formatRate(prefs.rate);
  speaker.configure({ rate: prefs.rate });
  savePrefs();
});

el.handsFree.addEventListener("change", () => {
  prefs.handsFree = el.handsFree.checked;
  savePrefs();
});

el.clearBtn.addEventListener("click", () => {
  if (typeof el.clearDialog.showModal === "function") el.clearDialog.showModal();
  else if (window.confirm("Clear the conversation?")) confirmClear();
});

el.clearDialog.addEventListener("close", () => {
  if (el.clearDialog.returnValue === "confirm") confirmClear();
  el.clearDialog.returnValue = "";
});

document.addEventListener("keydown", (event) => {
  if (event.key === "Escape" && speaker.busy) {
    speaker.stop();
    setStatus("idle", "Speech stopped");
  }
});

document.addEventListener("visibilitychange", () => {
  if (!document.hidden) {
    if (synth && synth.paused && speaker.busy) synth.resume();
    checkHealth();
  }
});

window.addEventListener("focus", checkHealth);

synth?.addEventListener("voiceschanged", populateVoices);

populateLanguages();

if (synth) {
  populateVoices();
  window.setTimeout(populateVoices, 400);
  window.setTimeout(populateVoices, 1500);
} else {
  el.voiceNote.hidden = false;
  el.voiceNote.textContent = "This browser can’t read replies aloud, but the chat still works.";
  el.voice.disabled = true;
  el.rate.disabled = true;
  el.stopSpeakBtn.hidden = true;
}

el.rate.value = String(prefs.rate);
el.rateOut.textContent = formatRate(prefs.rate);
el.handsFree.checked = Boolean(prefs.handsFree);
speaker.configure({ rate: Number(prefs.rate) });

if (!Recognition) {
  el.micBtn.disabled = true;
  el.micLabel.textContent = "Voice input unsupported";
}

restoreMessages();
setStatus("idle", STATE_LABELS.idle);
autosize();
checkHealth();
