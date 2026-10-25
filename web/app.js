const $ = (id) => document.getElementById(id);

const fileInput = $("file");
const guideInput = $("guide");
const drop = $("drop");
const statusEl = $("status");
let recorded = null;
let lastDraw = null;
let busy = false;

function formValues() {
  const body = new FormData();
  body.append("key", $("key").value);
  body.append("scale", $("scale").value);
  body.append("strength", (Number($("strength").value) / 100).toFixed(3));
  body.append("tightness", (Number($("tightness").value) / 100).toFixed(3));
  body.append("retune_ms", $("retune").value);
  body.append("vibrato", (Number($("vibrato").value) / 100).toFixed(3));
  body.append("add_vibrato", (Number($("addVibrato").value) / 100).toFixed(3));
  body.append("presence", (Number($("presence").value) / 100).toFixed(3));
  body.append("reverb", (Number($("reverb").value) / 100).toFixed(3));
  body.append("repair", $("repair").checked ? "true" : "false");
  body.append("harmony", $("harmony").checked ? "true" : "false");
  body.append("backing", $("backing").checked ? "true" : "false");
  body.append("fast", $("fast").checked ? "true" : "false");
  body.append("neural", $("neural").checked ? "true" : "false");
  return body;
}

function bindSlider(input, output, render) {
  const draw = () => { output.textContent = render(input.value); };
  input.addEventListener("input", draw);
  draw();
}

bindSlider($("strength"), $("strengthOut"), (v) => `${v}%`);
bindSlider($("tightness"), $("tightOut"), (v) => `${v}%`);
bindSlider($("retune"), $("retuneOut"), (v) => `${v} ms`);
bindSlider($("vibrato"), $("vibratoOut"), (v) => `${v}%`);
bindSlider($("addVibrato"), $("addOut"), (v) => `${v}%`);
bindSlider($("presence"), $("presenceOut"), (v) => `${v}%`);
bindSlider($("reverb"), $("reverbOut"), (v) => `${v}%`);

fileInput.addEventListener("change", () => {
  recorded = null;
  $("fileLabel").textContent = fileInput.files[0] ? fileInput.files[0].name : "Drop a vocal";
});
guideInput.addEventListener("change", () => {
  $("guideLabel").textContent = guideInput.files[0] ? guideInput.files[0].name : "Follow this tune instead of the nearest note";
});

["dragover", "dragenter"].forEach((name) => drop.addEventListener(name, (event) => {
  event.preventDefault();
  drop.classList.add("hot");
}));
["dragleave", "drop"].forEach((name) => drop.addEventListener(name, (event) => {
  event.preventDefault();
  drop.classList.remove("hot");
}));
drop.addEventListener("drop", (event) => {
  const file = event.dataTransfer.files && event.dataTransfer.files[0];
  if (!file) return;
  const transfer = new DataTransfer();
  transfer.items.add(file);
  fileInput.files = transfer.files;
  recorded = null;
  $("fileLabel").textContent = file.name;
});

let recorder = null;
let chunks = [];
let recordTimer = null;
$("record").addEventListener("click", async () => {
  if (recorder) {
    recorder.stop();
    return;
  }
  try {
    const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
    chunks = [];
    recorder = new MediaRecorder(stream);
    recorder.ondataavailable = (event) => { if (event.data.size) chunks.push(event.data); };
    recorder.onstop = () => {
      stream.getTracks().forEach((track) => track.stop());
      recorded = new Blob(chunks, { type: recorder.mimeType || "audio/webm" });
      fileInput.value = "";
      $("fileLabel").textContent = "Recorded take";
      $("record").setAttribute("aria-pressed", "false");
      $("recordLabel").textContent = "Record";
      recorder = null;
      clearInterval(recordTimer);
      setStatus("Recorded. Enhance when you're ready.");
    };
    recorder.start();
    const started = Date.now();
    $("record").setAttribute("aria-pressed", "true");
    recordTimer = setInterval(() => {
      const seconds = Math.round((Date.now() - started) / 1000);
      $("recordLabel").textContent = `${seconds}s`;
      if (seconds >= 45) recorder.stop();
    }, 250);
  } catch (error) {
    setStatus(error.message || "Microphone permission was blocked.", true);
  }
});

function setStatus(text, bad = false) {
  statusEl.textContent = text;
  statusEl.classList.toggle("bad", bad);
}

function setBusy(on) {
  busy = on;
  ["go", "demoOff", "demoGuide", "record"].forEach((id) => { $(id).disabled = on; });
}

function errorText(data) {
  if (!data) return "The booth could not enhance that take.";
  if (typeof data.detail === "string") return data.detail;
  if (Array.isArray(data.detail)) return data.detail.map((item) => item.msg || String(item)).join(" ");
  return "The booth could not enhance that take.";
}

async function post(url, body) {
  setBusy(true);
  setStatus("Listening through the take…");
  try {
    const response = await fetch(url, { method: "POST", body });
    const data = await response.json();
    if (!response.ok) throw new Error(errorText(data));
    show(data);
  } catch (error) {
    setStatus(error.message, true);
  } finally {
    setBusy(false);
  }
}

$("go").addEventListener("click", () => {
  const body = formValues();
  if (recorded) body.append("file", recorded, "recording.webm");
  else if (fileInput.files[0]) body.append("file", fileInput.files[0]);
  else {
    setStatus("Drop a vocal or record one first.", true);
    return;
  }
  if (guideInput.files[0]) body.append("guide", guideInput.files[0]);
  post("/api/enhance", body);
});

$("demoOff").addEventListener("click", () => {
  const body = new FormData();
  body.append("mode", "offkey");
  body.append("backing", "true");
  post("/api/demo", body);
});

$("demoGuide").addEventListener("click", () => {
  const body = new FormData();
  body.append("mode", "guide");
  body.append("backing", "true");
  post("/api/demo", body);
});

function wavUrl(b64) {
  const binary = atob(b64);
  const bytes = new Uint8Array(binary.length);
  for (let i = 0; i < binary.length; i += 1) bytes[i] = binary.charCodeAt(i);
  return URL.createObjectURL(new Blob([bytes], { type: "audio/wav" }));
}

function setAudio(element, link, b64) {
  if (element.src) URL.revokeObjectURL(element.src);
  const url = wavUrl(b64);
  element.src = url;
  link.href = url;
}

function show(data) {
  $("stage").hidden = false;
  $("statBefore").textContent = Math.round(data.before_cents);
  $("statAfter").textContent = Math.round(data.after_cents);
  $("statKey").textContent = data.key;
  const heard = data.key_source === "detected"
    ? "Heard from the take"
    : (data.detected_key && data.detected_key !== data.key ? `Also sounds like ${data.detected_key}` : "Key set in the booth");
  const repaired = data.repaired_frames ? ` · ${data.repaired_frames} frames repaired` : "";
  $("statMeta").textContent = `${heard}${repaired} · ${data.elapsed_ms} ms`;
  if (data.original_base64) setAudio($("playerBefore"), $("download"), data.original_base64);
  else if (recorded) $("playerBefore").src = URL.createObjectURL(recorded);
  else if (fileInput.files[0]) $("playerBefore").src = URL.createObjectURL(fileInput.files[0]);
  setAudio($("playerAfter"), $("download"), data.audio_base64);
  const mix = Boolean(data.mix_base64);
  $("mixWrap").hidden = !mix;
  $("downloadMix").hidden = !mix;
  if (mix) setAudio($("playerMix"), $("downloadMix"), data.mix_base64);
  const warning = $("warning");
  warning.hidden = !data.warning;
  warning.textContent = data.warning || "";
  fillNotes(data.notes || []);
  lastDraw = data;
  requestAnimationFrame(drawPitch);
  const demoNote = data.demo === "guide"
    ? " Synthesized take, pulled onto the guide melody."
    : data.demo
      ? " Synthesized take."
      : "";
  setStatus(`${Math.round(data.before_cents)}¢ off, then ${Math.round(data.after_cents)}¢. ${data.notes.length} notes.${demoNote}`);
  $("stage").scrollIntoView({ behavior: "smooth", block: "start" });
}

function fillNotes(notes) {
  const body = $("notes");
  body.replaceChildren();
  notes.slice(0, 40).forEach((note) => {
    const row = document.createElement("tr");
    [formatTime(note.t), note.sung, note.target, signed(note.before_cents), signed(note.after_cents)].forEach((text, index) => {
      const cell = document.createElement("td");
      cell.textContent = text;
      if (index === 3) cell.className = Math.abs(note.before_cents) > 25 ? "off" : "";
      if (index === 4) cell.className = Math.abs(note.after_cents) < 20 ? "on" : "";
      row.appendChild(cell);
    });
    body.appendChild(row);
  });
}

function formatTime(seconds) {
  const m = Math.floor(seconds / 60);
  const s = seconds % 60;
  return `${m}:${s.toFixed(2).padStart(5, "0")}`;
}

function signed(cents) {
  const value = Math.round(cents);
  return value > 0 ? `+${value}¢` : `${value}¢`;
}

function drawPitch() {
  if (!lastDraw) return;
  const canvas = $("pitch");
  const rect = canvas.getBoundingClientRect();
  const dpr = window.devicePixelRatio || 1;
  canvas.width = Math.max(320, rect.width) * dpr;
  canvas.height = 280 * dpr;
  const ctx = canvas.getContext("2d");
  ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
  const width = canvas.width / dpr;
  const height = canvas.height / dpr;
  ctx.clearRect(0, 0, width, height);
  const values = [...lastDraw.midi_in, ...lastDraw.midi_out].filter((v) => v != null);
  if (!values.length) return;
  const low = Math.floor(Math.min(...values) - 2);
  const high = Math.ceil(Math.max(...values) + 2);
  const span = Math.max(4, high - low);
  const duration = lastDraw.duration || lastDraw.times[lastDraw.times.length - 1] || 1;
  const left = 36;
  const plotW = width - left - 8;
  const yOf = (midi) => 16 + (high - midi) / span * (height - 32);
  const xOf = (time) => left + (time / duration) * plotW;
  ctx.font = "12px Outfit, sans-serif";
  for (let midi = low; midi <= high; midi += 1) {
    const y = yOf(midi);
    ctx.strokeStyle = midi % 12 === 0 ? "#3a332c" : "#241f1b";
    ctx.beginPath();
    ctx.moveTo(left, y);
    ctx.lineTo(width - 8, y);
    ctx.stroke();
    if (midi % 12 === 0) {
      ctx.fillStyle = "#8c8175";
      ctx.fillText(noteName(midi), 4, y + 4);
    }
  }
  stroke(ctx, lastDraw.times, lastDraw.midi_in, xOf, yOf, "#e07a5f", 1.6);
  stroke(ctx, lastDraw.times, lastDraw.midi_out, xOf, yOf, "#c6dc7a", 2.2);
}

function stroke(ctx, times, series, xOf, yOf, color, width) {
  ctx.strokeStyle = color;
  ctx.lineWidth = width;
  ctx.beginPath();
  let drawing = false;
  series.forEach((midi, index) => {
    if (midi == null) {
      drawing = false;
      return;
    }
    const x = xOf(times[index]);
    const y = yOf(midi);
    if (!drawing) {
      ctx.moveTo(x, y);
      drawing = true;
    } else ctx.lineTo(x, y);
  });
  ctx.stroke();
}

function noteName(midi) {
  const names = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"];
  const rounded = Math.round(midi);
  return `${names[((rounded % 12) + 12) % 12]}${Math.floor(rounded / 12) - 1}`;
}

window.addEventListener("resize", drawPitch);

function recorderFor(button, label, maxSeconds, onDone) {
  let active = null;
  let parts = [];
  let timer = null;
  button.addEventListener("click", async () => {
    if (active) {
      active.stop();
      return;
    }
    try {
      const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
      parts = [];
      active = new MediaRecorder(stream);
      active.ondataavailable = (event) => { if (event.data.size) parts.push(event.data); };
      active.onstop = () => {
        stream.getTracks().forEach((track) => track.stop());
        const blob = new Blob(parts, { type: active.mimeType || "audio/webm" });
        button.setAttribute("aria-pressed", "false");
        label.textContent = "Record";
        active = null;
        clearInterval(timer);
        onDone(blob);
      };
      active.start();
      const started = Date.now();
      button.setAttribute("aria-pressed", "true");
      timer = setInterval(() => {
        const seconds = Math.round((Date.now() - started) / 1000);
        label.textContent = `${seconds}s`;
        if (seconds >= maxSeconds && active) active.stop();
      }, 250);
    } catch (error) {
      label.textContent = "Record";
    }
  });
}

const skinState = { recordings: [], spoken: null, skins: [] };

function skinStatus(id, text, bad = false) {
  $(id).textContent = text;
  $(id).classList.toggle("bad", bad);
}

function describeRecordings() {
  const files = [...$("skinFiles").files].map((file) => file.name);
  const takes = skinState.recordings.length;
  const parts = [];
  if (files.length) parts.push(files.length === 1 ? files[0] : `${files.length} files`);
  if (takes) parts.push(takes === 1 ? "1 recording" : `${takes} recordings`);
  $("skinFilesLabel").textContent = parts.length ? parts.join(" + ") : "Drop recordings of the owner";
}

$("skinFiles").addEventListener("change", describeRecordings);
$("skinDrop").addEventListener("dragover", (event) => event.preventDefault());
$("skinDrop").addEventListener("drop", (event) => {
  event.preventDefault();
  if (!event.dataTransfer.files.length) return;
  const transfer = new DataTransfer();
  [...$("skinFiles").files, ...event.dataTransfer.files].forEach((file) => transfer.items.add(file));
  $("skinFiles").files = transfer.files;
  describeRecordings();
});

recorderFor($("skinRecord"), $("skinRecordLabel"), 300, (blob) => {
  skinState.recordings.push(blob);
  describeRecordings();
  skinStatus("skinBuildStatus", "Recorded. Add more, or make the skin.");
});

$("speakFile").addEventListener("change", () => {
  skinState.spoken = null;
  $("speakFileLabel").textContent = $("speakFile").files[0] ? $("speakFile").files[0].name : "Drop what to say";
});
$("speakDrop").addEventListener("dragover", (event) => event.preventDefault());
$("speakDrop").addEventListener("drop", (event) => {
  event.preventDefault();
  const file = event.dataTransfer.files && event.dataTransfer.files[0];
  if (!file) return;
  const transfer = new DataTransfer();
  transfer.items.add(file);
  $("speakFile").files = transfer.files;
  skinState.spoken = null;
  $("speakFileLabel").textContent = file.name;
});

recorderFor($("speakRecord"), $("speakRecordLabel"), 300, (blob) => {
  skinState.spoken = blob;
  $("speakFile").value = "";
  $("speakFileLabel").textContent = "Recorded";
  skinStatus("skinSpeakStatus", "Recorded. Speak as them when you're ready.");
});

bindSlider($("skinAmount"), $("skinAmountOut"), (v) => `${v}%`);
bindSlider($("skinShift"), $("skinShiftOut"), (v) => `${v > 0 ? "+" : ""}${v} st`);

function showTerms() {
  const skin = skinState.skins.find((item) => item.name === $("skinVoice").value);
  $("skinTermsHint").textContent = skin
    ? `${skin.owner} · ${Math.round(skin.seconds)}s learned · around ${skin.hz} Hz${skin.terms ? ` · Terms: ${skin.terms}` : ""}`
    : "Make a skin first.";
}
$("skinVoice").addEventListener("change", showTerms);

async function loadSkins(select) {
  try {
    const data = await (await fetch("/api/skins")).json();
    skinState.skins = data.skins || [];
  } catch (error) {
    skinState.skins = [];
  }
  const voice = $("skinVoice");
  voice.replaceChildren();
  if (!skinState.skins.length) {
    const option = document.createElement("option");
    option.value = "";
    option.textContent = "No skins yet";
    voice.appendChild(option);
  }
  skinState.skins.forEach((skin) => {
    const option = document.createElement("option");
    option.value = skin.name;
    option.textContent = skin.owner;
    voice.appendChild(option);
  });
  if (select) voice.value = select;
  showTerms();
}

$("skinBuild").addEventListener("click", async () => {
  const body = new FormData();
  [...$("skinFiles").files].forEach((file) => body.append("recordings", file));
  skinState.recordings.forEach((blob, index) => body.append("recordings", blob, `recording-${index + 1}.webm`));
  if (!body.has("recordings")) {
    skinStatus("skinBuildStatus", "Add recordings of the owner first.", true);
    return;
  }
  body.append("owner", $("skinOwner").value);
  body.append("terms", $("skinTerms").value);
  body.append("consent", $("skinConsent").checked ? "true" : "false");
  $("skinBuild").disabled = true;
  skinStatus("skinBuildStatus", "Learning the voice…");
  try {
    const response = await fetch("/api/skins", { method: "POST", body });
    const data = await response.json();
    if (!response.ok) throw new Error(errorText(data));
    skinStatus("skinBuildStatus", `${data.owner}'s skin is ready: ${Math.round(data.seconds)}s of voice, around ${data.hz} Hz.`);
    skinState.recordings = [];
    $("skinFiles").value = "";
    describeRecordings();
    await loadSkins(data.name);
  } catch (error) {
    skinStatus("skinBuildStatus", error.message, true);
  } finally {
    $("skinBuild").disabled = false;
  }
});

$("skinSpeak").addEventListener("click", async () => {
  const voice = $("skinVoice").value;
  if (!voice) {
    skinStatus("skinSpeakStatus", "Make a skin first.", true);
    return;
  }
  const body = new FormData();
  let source = null;
  if (skinState.spoken) {
    source = skinState.spoken;
    body.append("file", source, "spoken.webm");
  } else if ($("speakFile").files[0]) {
    source = $("speakFile").files[0];
    body.append("file", source);
  } else {
    skinStatus("skinSpeakStatus", "Record or drop something to say.", true);
    return;
  }
  body.append("voice", voice);
  body.append("engine", $("skinEngine").value);
  body.append("pitch", $("skinPitch").value);
  body.append("amount", (Number($("skinAmount").value) / 100).toFixed(3));
  body.append("shift", $("skinShift").value);
  $("skinSpeak").disabled = true;
  skinStatus("skinSpeakStatus", $("skinEngine").value === "neural"
    ? "Speaking through the neural engine… the first run downloads its weights."
    : "Speaking through the skin…");
  try {
    const response = await fetch("/api/skins/speak", { method: "POST", body });
    const data = await response.json();
    if (!response.ok) throw new Error(errorText(data));
    $("skinOut").hidden = false;
    $("skinDownloads").hidden = false;
    if ($("skinBefore").src) URL.revokeObjectURL($("skinBefore").src);
    $("skinBefore").src = URL.createObjectURL(source);
    setAudio($("skinAfter"), $("skinDownload"), data.audio_base64);
    $("skinAfterLabel").textContent = `As ${data.owner}`;
    $("skinProvenance").textContent = `The file is tagged: “${data.provenance}”`;
    const pitch = data.source_hz && data.output_hz ? ` · ${Math.round(data.source_hz)} → ${Math.round(data.output_hz)} Hz` : "";
    skinStatus("skinSpeakStatus", `${data.engine} engine${pitch} · ${data.elapsed_ms} ms${data.warning ? ` · ${data.warning}` : ""}`, Boolean(data.warning));
  } catch (error) {
    skinStatus("skinSpeakStatus", error.message, true);
  } finally {
    $("skinSpeak").disabled = false;
  }
});

loadSkins();

fetch("/api/health").then((response) => response.json()).then((data) => {
  if (data.neural) {
    $("neuralHint").textContent = `Checkpoint ${data.checkpoint} is loaded. Neural resynth is available.`;
  }
}).catch(() => {});
