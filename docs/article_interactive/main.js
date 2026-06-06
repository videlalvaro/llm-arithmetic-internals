import * as THREE from "https://cdn.jsdelivr.net/npm/three@0.160.0/build/three.module.js";
import { OrbitControls } from "https://cdn.jsdelivr.net/npm/three@0.160.0/examples/jsm/controls/OrbitControls.js";

const canvas = document.querySelector("#helixCanvas");
const slider = document.querySelector("#integerSlider");
const valueReadout = document.querySelector("#valueReadout");
const phaseReadout = document.querySelector("#phaseReadout");
const cosReadout = document.querySelector("#cosReadout");
const sinReadout = document.querySelector("#sinReadout");
const coarseReadout = document.querySelector("#coarseReadout");

const scene = new THREE.Scene();
scene.background = null;

const camera = new THREE.PerspectiveCamera(42, 1, 0.1, 100);
camera.position.set(6.2, 4.1, 8.6);

const renderer = new THREE.WebGLRenderer({ canvas, antialias: true, alpha: true });
renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
renderer.shadowMap.enabled = true;

const controls = new OrbitControls(camera, canvas);
controls.enableDamping = true;
controls.enablePan = false;
controls.minDistance = 5;
controls.maxDistance = 18;
controls.target.set(0, 0.15, 0);

const keyLight = new THREE.DirectionalLight(0xffffff, 2.2);
keyLight.position.set(4, 8, 5);
scene.add(keyLight);
scene.add(new THREE.AmbientLight(0xffffff, 1.3));

const group = new THREE.Group();
scene.add(group);

const period = 10;
const maxValue = 999;
const helixTurnsShown = 9;
const displayMax = period * helixTurnsShown;
const heightScale = 0.078;
const radius = 1.22;

class IntegerHelixCurve extends THREE.Curve {
  getPoint(t) {
    const n = t * displayMax;
    const theta = (Math.PI * 2 * n) / period;
    return new THREE.Vector3(
      Math.cos(theta) * radius,
      n * heightScale,
      Math.sin(theta) * radius
    );
  }
}

const curve = new IntegerHelixCurve();
const tube = new THREE.TubeGeometry(curve, 1800, 0.024, 24, false);
const tubeMat = new THREE.MeshStandardMaterial({
  color: 0x176d75,
  metalness: 0.12,
  roughness: 0.46,
});
const tubeMesh = new THREE.Mesh(tube, tubeMat);
group.add(tubeMesh);

const pointGeo = new THREE.SphereGeometry(0.075, 18, 18);
const pointMat = new THREE.MeshStandardMaterial({ color: 0xc49125, roughness: 0.34 });
const points = new THREE.InstancedMesh(pointGeo, pointMat, 100);
const dummy = new THREE.Object3D();
for (let i = 0; i < 100; i++) {
  const n = i;
  const p = helixPoint(n);
  dummy.position.copy(p);
  dummy.updateMatrix();
  points.setMatrixAt(i, dummy.matrix);
}
group.add(points);

const activeMat = new THREE.MeshStandardMaterial({
  color: 0xb55f66,
  emissive: 0x7b2330,
  emissiveIntensity: 0.35,
  roughness: 0.25,
});
const active = new THREE.Mesh(new THREE.SphereGeometry(0.22, 32, 32), activeMat);
group.add(active);

const projectionMat = new THREE.LineBasicMaterial({ color: 0x2c5f93, transparent: true, opacity: 0.55 });
const projection = new THREE.Line(new THREE.BufferGeometry(), projectionMat);
group.add(projection);

const planeGeo = new THREE.CircleGeometry(radius, 96);
const planeMat = new THREE.MeshBasicMaterial({
  color: 0xffffff,
  transparent: true,
  opacity: 0.14,
  side: THREE.DoubleSide,
});
const phaseDisc = new THREE.Mesh(planeGeo, planeMat);
phaseDisc.rotation.x = Math.PI / 2;
group.add(phaseDisc);

group.position.set(0.05, -3.15, 0);

function helixPoint(n) {
  const shown = n % displayMax;
  const theta = (Math.PI * 2 * n) / period;
  return new THREE.Vector3(
    Math.cos(theta) * radius,
    shown * heightScale,
    Math.sin(theta) * radius
  );
}

function updateHelixValue(n) {
  const p = helixPoint(n);
  active.position.copy(p);
  phaseDisc.position.y = p.y;

  const base = new THREE.Vector3(p.x, p.y, p.z);
  const axis = new THREE.Vector3(0, p.y, 0);
  projection.geometry.dispose();
  projection.geometry = new THREE.BufferGeometry().setFromPoints([axis, base]);

  const theta = ((Math.PI * 2 * n) / period) % (Math.PI * 2);
  valueReadout.textContent = String(n);
  phaseReadout.textContent = `${(theta * 180 / Math.PI).toFixed(1)}°`;
  cosReadout.textContent = Math.cos(theta).toFixed(2);
  sinReadout.textContent = Math.sin(theta).toFixed(2);
  coarseReadout.textContent = String(Math.floor(n / period));
  updateResidual(n);
}

function resize() {
  const { clientWidth, clientHeight } = canvas;
  renderer.setSize(clientWidth, clientHeight, false);
  camera.aspect = clientWidth / clientHeight;
  camera.updateProjectionMatrix();
}

window.addEventListener("resize", resize);
slider.addEventListener("input", () => updateHelixValue(Number(slider.value)));

document.querySelectorAll(".mode-tab").forEach((button) => {
  button.addEventListener("click", () => {
    document.querySelectorAll(".mode-tab").forEach((b) => b.classList.remove("active"));
    button.classList.add("active");
    setMode(button.dataset.mode);
  });
});

function setMode(mode) {
  if (mode === "helix") {
    camera.position.set(6.2, 4.1, 8.6);
    controls.target.set(0, 0.15, 0);
    tubeMesh.material.color.set(0x176d75);
  } else if (mode === "residual") {
    camera.position.set(5.4, 6.5, 8.8);
    controls.target.set(0, 0.45, 0);
    tubeMesh.material.color.set(0x4d7f3a);
  } else {
    camera.position.set(7.6, 3.2, 6.8);
    controls.target.set(0, 0.15, 0);
    tubeMesh.material.color.set(0x2c5f93);
    animateEmission();
  }
}

const promptTokens = ["What", "is", "327", "x", "48", "?", "Answer"];
const tokenRow = document.querySelector("#promptTokens");
promptTokens.forEach((token, i) => {
  const el = document.createElement("span");
  el.className = "token";
  el.textContent = token;
  el.dataset.index = String(i);
  tokenRow.appendChild(el);
});

const vectorBars = document.querySelector("#vectorBars");
const bars = [];
for (let i = 0; i < 12; i++) {
  const bar = document.createElement("div");
  bar.className = "bar";
  vectorBars.appendChild(bar);
  bars.push(bar);
}

function updateResidual(n) {
  const phase = (Math.PI * 2 * n) / period;
  bars.forEach((bar, i) => {
    const value = 0.5 + 0.42 * Math.sin(phase * (i % 4 + 1) + i * 0.73);
    bar.style.height = `${18 + value * 190}px`;
    bar.style.background = i % 3 === 0
      ? "linear-gradient(180deg, #b55f66, #c49125)"
      : "linear-gradient(180deg, #0f6f78, #4d7f3a)";
  });
  const activeIndex = Math.min(promptTokens.length - 1, Math.floor((n / maxValue) * promptTokens.length));
  document.querySelectorAll(".token").forEach((el, i) => {
    el.classList.toggle("active", i === activeIndex);
  });
  updateMatrices(n);
}

function fillMatrix(id, kind) {
  const root = document.querySelector(id);
  for (let i = 0; i < 25; i++) {
    const cell = document.createElement("div");
    cell.className = "cell";
    cell.dataset.i = String(i);
    root.appendChild(cell);
  }
}

fillMatrix("#matrixA");
fillMatrix("#matrixB");
fillMatrix("#matrixC");

function updateMatrices(n) {
  const roots = ["#matrixA", "#matrixB", "#matrixC"];
  roots.forEach((id, ri) => {
    document.querySelectorAll(`${id} .cell`).forEach((cell, i) => {
      const v = 0.5 + 0.5 * Math.sin(n * 0.017 + i * 0.91 + ri * 1.2);
      const alpha = 0.08 + v * 0.66;
      const color = ri === 0 ? `rgba(15,111,120,${alpha})`
        : ri === 1 ? `rgba(196,145,37,${alpha})`
        : `rgba(77,127,58,${alpha})`;
      cell.style.background = color;
    });
  });
}

const emission = [
  { label: "15", caption: "first chunk" },
  { label: "696", caption: "next chunk" },
  { label: "<eos>", caption: "stop" },
];
const emissionRow = document.querySelector("#emissionRow");
emission.forEach((token) => {
  const el = document.createElement("span");
  el.className = "emission-token";
  el.textContent = token.label;
  emissionRow.appendChild(el);
});

let emissionStep = 0;
function animateEmission() {
  const tokens = document.querySelectorAll(".emission-token");
  tokens.forEach((el, i) => el.classList.toggle("active", i <= emissionStep));
  document.querySelector("#nextTokenFill").style.width = `${((emissionStep + 1) / emission.length) * 100}%`;
  emissionStep = (emissionStep + 1) % emission.length;
}
setInterval(animateEmission, 1500);

const toolResidual = document.querySelector("#toolResidual");
const toolOutput = document.querySelector("#toolOutput");
const toolOutputTitle = document.querySelector("#toolOutputTitle");
const toolArrow = document.querySelector("#toolArrow");
const toolTitle = document.querySelector("#toolTitle");
const toolText = document.querySelector("#toolText");
const toolButtons = document.querySelectorAll(".tool-button");
const toyResidual = [0.72, -0.28, 0.92, 0.18, -0.62, 0.46, 0.82, -0.16];

function drawToolResidual(mode) {
  if (!toolResidual) return;
  toolResidual.innerHTML = "";
  toyResidual.forEach((v, i) => {
    const bar = document.createElement("div");
    bar.className = `tool-dim ${v < 0 ? "negative" : ""}`;
    bar.style.height = `${28 + Math.abs(v) * 170}px`;
    if (mode === "sae" && ![0, 2, 5].includes(i)) bar.style.opacity = "0.22";
    if (mode === "patch" && [1, 4, 7].includes(i)) bar.style.background = "linear-gradient(180deg, #c49125, #b55f66)";
    if (mode === "steer" && [2, 6].includes(i)) bar.style.background = "linear-gradient(180deg, #2c5f93, #c49125)";
    toolResidual.appendChild(bar);
  });
}

function miniVector(values) {
  return `<div class="mini-vector">${values.map((v) => `<span style="height:${18 + Math.abs(v) * 62}px"></span>`).join("")}</div>`;
}

const toolCopy = {
  probe: {
    title: "Probe",
    arrow: "dot",
    outputTitle: "probe score",
    text: "A probe projects the residual vector onto a learned direction. If the score separates classes, the information is readable. It does not prove the direction is causal.",
  },
  sae: {
    title: "Sparse autoencoder",
    arrow: "encode",
    outputTitle: "active features",
    text: "An SAE tries to rebuild the dense vector using a small number of active features. The hope is that some sparse features are more interpretable than raw dimensions.",
  },
  patch: {
    title: "Activation patch",
    arrow: "swap",
    outputTitle: "donor -> recipient",
    text: "Patching copies part of a donor activation into a recipient run. If the behavior follows the donor, the copied state is evidence for a causal variable.",
  },
  steer: {
    title: "Steering",
    arrow: "add",
    outputTitle: "h + direction",
    text: "Steering adds a direction to the residual stream. It can move the model toward a token or feature, but the hard question is whether the rest of the state remains coherent.",
  },
};

function renderTool(mode) {
  if (!toolOutput || !toolTitle || !toolText || !toolArrow || !toolOutputTitle) return;
  if (!toolCopy[mode]) mode = "probe";
  drawToolResidual(mode);
  const copy = toolCopy[mode];
  toolTitle.textContent = copy.title;
  toolText.textContent = copy.text;
  toolArrow.textContent = copy.arrow;
  toolOutputTitle.textContent = copy.outputTitle;

  if (mode === "probe") {
    toolOutput.innerHTML = `
      <div class="score-gauge" style="--score: 78%"><i></i></div>
      <div class="feature-pills">
        <span class="feature-pill">score = h * w</span>
        <span class="feature-pill active">op: lcm</span>
        <span class="feature-pill">margin 0.78</span>
        <span class="feature-pill">readable</span>
      </div>`;
  } else if (mode === "sae") {
    toolOutput.innerHTML = `
      <div class="sae-stack">
        <div class="sparse-row">
          <span class="sparse-feature active">phase</span>
          <span class="sparse-feature">syntax</span>
          <span class="sparse-feature active">scale</span>
          <span class="sparse-feature">noise</span>
          <span class="sparse-feature active">role</span>
          <span class="sparse-feature">style</span>
        </div>
        <div class="reconstruct-row">
          ${miniVector([0.7, 0.2, 0.9, 0.1, 0.6, 0.4])}
          <span class="flow-arrow">~</span>
          ${miniVector([0.65, 0.16, 0.86, 0.18, 0.55, 0.42])}
        </div>
      </div>`;
  } else if (mode === "patch") {
    toolOutput.innerHTML = `
      <div class="patch-pair">
        ${miniVector([0.2, 0.8, 0.5, 0.1, 0.7, 0.3])}
        <span class="flow-arrow">-></span>
        ${miniVector([0.7, 0.2, 0.9, 0.3, 0.2, 0.8])}
      </div>
      <div class="feature-pills"><span class="feature-pill active">decoded tuple follows donor</span></div>`;
  } else {
    toolOutput.innerHTML = `
      <div class="steer-plane">
        <span class="steer-dot" style="left:34%;top:62%"></span>
        <span class="steer-line" style="left:34%;top:62%;width:155px;transform:rotate(-28deg)"></span>
        <span class="steer-dot after" style="left:70%;top:38%"></span>
      </div>`;
  }
}

toolButtons.forEach((button) => {
  button.addEventListener("click", () => {
    toolButtons.forEach((b) => b.classList.remove("active"));
    button.classList.add("active");
    renderTool(button.dataset.tool);
  });
});

const toolSignal = document.querySelector("#toolSignal");
const toolPatch = document.querySelector("#toolPatch");
const toolSteer = document.querySelector("#toolSteer");
const toolSignalValue = document.querySelector("#toolSignalValue");
const toolPatchValue = document.querySelector("#toolPatchValue");
const toolSteerValue = document.querySelector("#toolSteerValue");
const probeGauge = document.querySelector("#probeGauge");
const probeDecode = document.querySelector("#probeDecode");
const patchRecipient = document.querySelector("#patchRecipient");
const steerLine = document.querySelector("#steerLine");
const steerAfter = document.querySelector("#steerAfter");

function updateToolboxSimulator() {
  if (!toolSignal || !toolPatch || !toolSteer) return;

  const signal = Number(toolSignal.value);
  const patch = Number(toolPatch.value);
  const steer = Number(toolSteer.value);

  toolSignalValue.textContent = `${signal}%`;
  toolPatchValue.textContent = `${patch}%`;
  toolSteerValue.textContent = `${steer}%`;

  if (probeGauge && probeDecode) {
    probeGauge.style.setProperty("--score", `${signal}%`);
    if (signal >= 70) {
      probeDecode.innerHTML = "op=gcd<br />a=84 b=36";
    } else if (signal >= 42) {
      probeDecode.innerHTML = "op=gcd<br />a=? b=36";
    } else {
      probeDecode.innerHTML = "weak readout<br />arguments unclear";
    }
  }

  document.querySelectorAll("[data-sae-feature]").forEach((feature) => {
    const key = feature.dataset.saeFeature;
    const threshold = key === "operand-a" ? 35 : key === "operand-b" ? 52 : key === "phase" ? 68 : 84;
    feature.classList.toggle("active", signal >= threshold);
  });

  if (patchRecipient) {
    const donor = [22, 82, 52, 30, 72, 38];
    const original = [74, 24, 90, 36, 28, 84];
    patchRecipient.querySelectorAll("span").forEach((bar, i) => {
      const mixed = original[i] * (1 - patch / 100) + donor[i] * (patch / 100);
      bar.style.height = `${mixed.toFixed(1)}%`;
      bar.style.opacity = String(0.5 + patch / 200);
    });
  }

  if (steerLine && steerAfter) {
    const width = 58 + steer * 1.75;
    const x = 32 + steer * 0.52;
    const y = 64 - steer * 0.34;
    steerLine.style.width = `${width}px`;
    steerAfter.style.left = `${Math.min(84, x)}%`;
    steerAfter.style.top = `${Math.max(24, y)}%`;
  }
}

[toolSignal, toolPatch, toolSteer].forEach((control) => {
  if (control) control.addEventListener("input", updateToolboxSimulator);
});

function tick() {
  resize();
  controls.update();
  group.rotation.y += 0.0009;
  renderer.render(scene, camera);
  requestAnimationFrame(tick);
}

updateHelixValue(Number(slider.value));
animateEmission();
renderTool("probe");
updateToolboxSimulator();
tick();
