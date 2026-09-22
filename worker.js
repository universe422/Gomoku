/* Pinned runtime versions. Single-thread WASM works on ordinary GitHub Pages. */
const PY = 'https://cdn.jsdelivr.net/pyodide/v0.27.7/full/';
const ORT = 'https://cdn.jsdelivr.net/npm/onnxruntime-web@1.22.0/dist/';
let py, session;
const send = (type, payload = {}) => postMessage({type, ...payload});
self.reportProgress = evaluated => send('progress', {evaluated});
self.inferBatch = async raw => {
  const positions = JSON.parse(raw);
  const data = new Float32Array(positions.length * 900);
  positions.forEach(([board, turn, passed], n) => {
    const offset = n * 900;
    for (let i = 0; i < 225; i++) {
      data[offset+i] = +(board[i] === turn);
      data[offset+225+i] = +(board[i] === 3-turn);
      data[offset+450+i] = +(turn === 1);
      data[offset+675+i] = passed;
    }
  });
  const input = new ort.Tensor('float32', data, [positions.length, 4, 15, 15]);
  let output;
  try {
    output = await session.run({board: input});
    return JSON.stringify(positions.map((_, n) => [
      Array.from(output.policy.data.slice(n*226, (n+1)*226)), output.value.data[n]
    ]));
  } finally {
    input.dispose();
    if (output) Object.values(output).forEach(tensor => tensor.dispose());
  }
};
async function init() {
  send('loading', {message: '대국 엔진을 준비하고 있습니다…'});
  importScripts(PY + 'pyodide.js', ORT + 'ort.min.js');
  ort.env.wasm.wasmPaths = ORT;
  ort.env.wasm.numThreads = 1;
  const metadataResponse = await fetch('./model.json', {cache: 'no-cache'});
  if (!metadataResponse.ok) throw new Error(`Model metadata: HTTP ${metadataResponse.status}`);
  const metadata = await metadataResponse.json();
  const modelURL = new URL('./best.onnx', self.location.href);
  modelURL.searchParams.set('v', metadata.onnx_sha256);
  [py, session] = await Promise.all([
    loadPyodide({indexURL: PY}),
    ort.InferenceSession.create(modelURL.href,
      {executionProviders: ['wasm'], graphOptimizationLevel: 'all'})
  ]);
  py.FS.mkdirTree('/engine');
  await Promise.all(['renju', 'patterns', 'threat_search', 'forcing', 'core', 'search', 'browser_support', 'bridge'].map(async name => {
    const response = await fetch(`./engine/${name}.py`);
    if (!response.ok) throw new Error(`${name}: HTTP ${response.status}`);
    py.FS.writeFile(`/engine/${name}.py`, await response.text());
  }));
  py.runPython("import sys; sys.path.insert(0, '/engine')\nfrom bridge import command, ai_turn");
  send('ready', {state: JSON.parse(py.runPython("command('{\"type\":\"new\"}')"))});
}
let busy = false;
self.onmessage = async ({data}) => {
  if (busy) return;
  busy = true;
  try {
    if (data.type === 'init') await init();
    else if (data.type === 'ai') {
      if (!Number.isInteger(data.simulations) || data.simulations < 4 || data.simulations > 16384)
        throw new Error('탐색량은 4~16,384 사이의 정수여야 합니다.');
      py.globals.set('budget', data.simulations);
      const result = JSON.parse(await py.runPythonAsync('await ai_turn(budget)'));
      send('ai-done', result);
    } else {
      py.globals.set('message_json', JSON.stringify(data));
      send('state', {state: JSON.parse(py.runPython('command(message_json)'))});
    }
  } catch (error) {
    send('error', {message: String(error), during: data.type});
  } finally { busy = false; }
};
