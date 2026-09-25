/* Fixed FP32 model and pinned runtimes. Single thread works on ordinary Pages. */
const BUILD = 'speed-20260923-1';
const PY = 'https://cdn.jsdelivr.net/pyodide/v0.27.7/full/';
const ORT = 'https://cdn.jsdelivr.net/npm/onnxruntime-web@1.22.0/dist/';
let py, session, modelBytes, metadata, evaluator, fixtures, device, deviceLost;
let mode = 'optimized', selectedBackend = 'wasm', busy = false, activeId, diagnostics = false;
let info = {}, jsMetrics = {}, lastProgress = 0;
let lastAcceptedId = -1;
const send = (type, payload = {}) => postMessage({type, id: activeId, ...payload});
const now = () => performance.now();
const elapsed = start => now() - start;
const sha256 = async bytes => Array.from(new Uint8Array(await crypto.subtle.digest('SHA-256', bytes)), b => b.toString(16).padStart(2, '0')).join('');
const quantile = (values, q) => [...values].sort((a, b) => a - b)[Math.max(0, Math.ceil(q * values.length) - 1)];
function resetMetrics() {
  jsMetrics = {session_run_ms: 0, output_readback_ms: 0, output_conversion_ms: 0,
    input_encoding_ms: 0, json_parse_ms: 0, json_stringify_ms: 0, packed_copy_ms: 0,
    requested_evaluations: 0, neural_evaluations: 0, inference_calls: 0,
    session_run_calls: 0, attempted_positions: 0,
    batch_histogram: {}, progress_messages: 0, fallback_ms: 0};
  evaluator?.resetMetrics(); lastProgress = 0;
}
self.reportProgress = evaluated => {
  if (mode === 'baseline' || now() - lastProgress >= 150) {
    send('progress', {evaluated}); lastProgress = now(); jsMetrics.progress_messages++;
  }
};
async function fallback(reason) {
  const start = now();
  info.fallback_reasons.push(String(reason));
  const old = session; session = null;
  if (old) { try { await old.release(); } catch (_) {} }
  selectedBackend = 'wasm'; deviceLost = null;
  session = await ort.InferenceSession.create(modelBytes, {executionProviders: ['wasm'], graphOptimizationLevel: 'all'});
  evaluator?.clear(metadata.onnx_sha256 + ':wasm:float32');
  info.backend = 'wasm'; info.provider_evidence = 'WASM-only session successfully created';
  info.partial_cpu_execution = false; jsMetrics.fallback_ms += elapsed(start);
}
async function runTensor(data, count, consume) {
  if (deviceLost && selectedBackend === 'webgpu') await fallback('Device lost: ' + deviceLost);
  const input = new ort.Tensor('float32', data, [count, 4, 15, 15]);
  try {
    for (let attempt = 0; attempt < 2; attempt++) {
      let output, started = now(), stage = 'session_run_ms';
      try {
        jsMetrics.session_run_calls++; jsMetrics.attempted_positions += count;
        output = await session.run({board: input});
        jsMetrics.session_run_ms += elapsed(started);
        started = now(); stage = 'output_readback_ms';
        const reads = await Promise.allSettled([output.policy.getData(), output.value.getData()]);
        const failedRead = reads.find(read => read.status === 'rejected');
        if (failedRead) throw failedRead.reason;
        const [policy, value] = reads.map(read => read.value);
        jsMetrics.output_readback_ms += elapsed(started);
        started = now(); stage = 'output_conversion_ms';
        if (policy.length !== count * 226 || value.length !== count) throw new Error('Model output shape mismatch');
        const result = consume(policy, value);
        jsMetrics.output_conversion_ms += elapsed(started);
        return result;
      } catch (error) {
        jsMetrics[stage] += elapsed(started);
        if (selectedBackend !== 'webgpu' || attempt) throw error;
        // Also recovers a lost device during asynchronous CPU output readback.
        if (output) Object.values(output).forEach(tensor => tensor.dispose());
        output = null;
        await fallback('WebGPU inference/readback failed: ' + String(error));
      } finally { if (output) Object.values(output).forEach(tensor => tensor.dispose()); }
    }
  } finally { input.dispose(); }
}
// Original JSON/encoding path, with measurement only, retained for comparisons.
self.inferBatch = async raw => {
  let started = now(); const positions = JSON.parse(raw);
  jsMetrics.json_parse_ms += elapsed(started);
  started = now(); const data = new Float32Array(positions.length * 900);
  positions.forEach(([board, turn, passed], n) => {
    const offset = n * 900;
    for (let a = 0; a < 225; a++) {
      data[offset + a] = +(board[a] === turn);
      data[offset + 225 + a] = +(board[a] === 3 - turn);
      data[offset + 450 + a] = +(turn === 1);
      data[offset + 675 + a] = passed;
    }
  });
  jsMetrics.input_encoding_ms += elapsed(started);
  jsMetrics.requested_evaluations += positions.length; jsMetrics.neural_evaluations += positions.length;
  jsMetrics.inference_calls++;
  jsMetrics.batch_histogram[positions.length] = (jsMetrics.batch_histogram[positions.length] || 0) + 1;
  const answers = await runTensor(data, positions.length, (policy, value) => positions.map((_, n) =>
    [Array.from(policy.slice(n * 226, (n + 1) * 226)), value[n]]));
  started = now(); const result = JSON.stringify(answers); jsMetrics.json_stringify_ms += elapsed(started);
  return result;
};
self.inferPacked = async (proxy, count) => {
  const started = now(); let buffer, packed;
  try {
    buffer = proxy.getBuffer('u8');
    // One explicit small copy: no Pyodide heap view crosses the asynchronous run.
    packed = buffer.data.slice();
  } finally { buffer?.release(); }
  // Borrowed argument proxy is released by Pyodide when this Promise settles.
  jsMetrics.packed_copy_ms += elapsed(started);
  if (deviceLost && selectedBackend === 'webgpu') await fallback('Device lost: ' + deviceLost);
  return evaluator.evaluate(packed, count, true);
};
function combinedMetrics() {
  return {js: {...jsMetrics, ...(mode === 'baseline' ? {} : evaluator.metrics),
      cache_entries: evaluator.cache.size, cache_capacity: evaluator.capacity},
    python: JSON.parse(py.runPython('metrics_json()')),
    timing_note: 'Python inference_wait contains JS work; never add it to JS subtimings. Loading phases overlap. UI time includes worker time.'};
}
async function init(options) {
  if (py) throw new Error('Reinitialize with a new Worker');
  diagnostics = !!options.diagnostics;
  mode = options.mode === 'baseline' ? 'baseline' : 'optimized';
  const backend = options.backend || 'auto';
  if (!['auto', 'wasm', 'webgpu'].includes(backend)) throw new Error('Unknown backend');
  const tryGPU = backend === 'webgpu' || (backend === 'auto' && !!navigator.gpu);
  resetMetrics();
  info = {build: BUILD, mode, requested_backend: backend, backend: 'wasm', precision: 'float32',
    ort_version: '1.22.0', pyodide_version: '0.27.7', wasm_threads: 1,
    crossOriginIsolated: self.crossOriginIsolated, secureContext: self.isSecureContext,
    userAgent: navigator.userAgent, hardwareConcurrency: navigator.hardwareConcurrency,
    webgpu_available: !!navigator.gpu, fallback_reasons: [], loading_ms: {},
    auto_policy: 'Prefer available WebGPU: target-browser interleaved cold searches improved 44-47%; other devices may differ. Fall back to WASM on failure.',
    partial_cpu_execution: false, graph_capture: false, operator_assignment: 'not measured'};
  const loadStart = now(); send('loading', {message: '대국 엔진을 준비하고 있습니다…'});
  let started = now();
  importScripts(PY + 'pyodide.js', ORT + (tryGPU ? 'ort.webgpu.min.js' : 'ort.min.js'));
  info.loading_ms.runtime_scripts_download_and_evaluate = elapsed(started);
  ort.env.wasm.wasmPaths = ORT; ort.env.wasm.numThreads = 1; ort.env.wasm.proxy = false;
  const manifestResponse = await fetch('./web-assets.json?v=' + BUILD, {cache: 'no-cache'});
  if (!manifestResponse.ok) throw new Error('웹 파일 목록을 읽지 못했습니다.');
  const manifest = await manifestResponse.json();
  if (manifest.build !== BUILD) throw new Error('웹 파일 버전이 섞였습니다. 새로고침해 주세요.');
  info.asset_hashes = manifest.hashes;
  async function asset(path, loadingKey) {
    const t = now(), response = await fetch('./' + path + '?v=' + manifest.hashes[path], {cache: 'no-cache'});
    if (!response.ok) throw new Error(`${path}: HTTP ${response.status}`);
    const bytes = await response.arrayBuffer();
    if (await sha256(bytes) !== manifest.hashes[path]) throw new Error(`${path}: 파일 버전 불일치. 새로고침해 주세요.`);
    if (loadingKey) info.loading_ms[loadingKey] = elapsed(t);
    return bytes;
  }
  const metadataBytes = await asset('model.json', 'metadata_download');
  await asset('inference.js');
  importScripts('./inference.js?v=' + manifest.hashes['inference.js']);
  metadata = JSON.parse(new TextDecoder().decode(metadataBytes));
  info.model = metadata; info.model_json_sha256 = await sha256(metadataBytes);
  const pyStart = now();
  const pyReady = loadPyodide({indexURL: PY}).then(result => {
    info.loading_ms.pyodide_download_and_init = elapsed(pyStart); return result;
  });
  modelBytes = await asset('best.onnx', 'model_download_and_hash');
  if (await sha256(modelBytes) !== metadata.onnx_sha256) throw new Error('모델 해시 불일치');
  started = now();
  if (tryGPU) {
    try {
      if (!navigator.gpu) throw new Error('WebGPU is unavailable in this Worker');
      const adapter = await navigator.gpu.requestAdapter({powerPreference: 'high-performance'});
      if (!adapter) throw new Error('No WebGPU adapter');
      const ai = adapter.info || (adapter.requestAdapterInfo ? await adapter.requestAdapterInfo() : {});
      info.adapter = Object.fromEntries(['vendor', 'architecture', 'device', 'description'].map(k => [k, ai[k] || 'undisclosed']));
      ort.env.webgpu.adapter = adapter;
      // Collect positive execution evidence during warmup only; timestamps disabled for samples.
      info.gpu_probe = {kernel_count: 0, kernel_types: {}, profiling_supported: adapter.features.has('timestamp-query')};
      if (info.gpu_probe.profiling_supported) ort.env.webgpu.profiling = {mode: 'default', ondata: event => {
        info.gpu_probe.kernel_count++;
        const key = event.kernelType || event.kernelName || 'GPU kernel';
        info.gpu_probe.kernel_types[key] = (info.gpu_probe.kernel_types[key] || 0) + 1;
      }};
      session = await ort.InferenceSession.create(modelBytes, {executionProviders: ['webgpu'], graphOptimizationLevel: 'all'});
      selectedBackend = 'webgpu'; info.backend = 'webgpu';
      device = await ort.env.webgpu.device;
      if (!device) throw new Error('ORT did not expose a WebGPU device');
      info.provider_evidence = 'ORT WebGPU device created; inspect gpu_probe for executed kernels';
      info.partial_cpu_execution = 'unknown';
      device.lost.then(loss => { deviceLost = loss.message || loss.reason; });
    } catch (error) { await fallback('WebGPU initialization: ' + String(error)); }
  } else {
    session = await ort.InferenceSession.create(modelBytes, {executionProviders: ['wasm'], graphOptimizationLevel: 'all'});
    info.provider_evidence = 'WASM-only session successfully created';
  }
  info.loading_ms.session_create = elapsed(started);
  evaluator = new NeuralEvaluator(async (data, count) => runTensor(data, count, (policy, value) => {
    const flat = new Float32Array(count * 227);
    for (let n = 0; n < count; n++) { flat.set(policy.subarray(n * 226, (n + 1) * 226), n * 227); flat[n * 227 + 226] = value[n]; }
    return flat;
  }), metadata.onnx_sha256 + ':' + selectedBackend + ':float32');
  started = now(); await evaluator.evaluate(new Uint8Array([...new Uint8Array(225), 1, 0]), 1, false);
  info.loading_ms.first_inference_warmup = elapsed(started);
  if (selectedBackend === 'webgpu') {
    try {
      await device.queue.onSubmittedWorkDone();
      ort.env.webgpu.profiling.mode = 'off';
      if (info.gpu_probe.kernel_count) info.provider_evidence = 'Executed WebGPU kernels observed by ORT profiling during warmup';
    } catch (error) {
      await fallback('Device lost during warmup synchronization: ' + String(error));
      await evaluator.evaluate(new Uint8Array([...new Uint8Array(225),1,0]),1,false);
    }
  }
  // Prepare the common variable batch shapes before the first human move.
  // Cache stays empty; this is loading work, excluded from move measurements.
  const shapeWarmupStart = now(); info.shape_warmups = [];
  for (const count of [2, 4, 8, 16]) {
    if (selectedBackend !== 'webgpu') break;
    const packed = new Uint8Array(count * 227);
    for (let n = 0; n < count; n++) packed[n * 227 + 225] = 1;
    const t = now(), before = selectedBackend;
    await evaluator.evaluate(packed, count, false);
    info.shape_warmups.push({batchSize:count,ms:elapsed(t),backend_before:before,backend_after:selectedBackend});
  }
  info.loading_ms.additional_shape_warmup = elapsed(shapeWarmupStart);
  py = await pyReady; py.FS.mkdirTree('/engine');
  started = now();
  await Promise.all(['renju', 'patterns', 'threat_search', 'forcing', 'core', 'search', 'browser_support', 'web_profile', 'bridge'].map(async name => {
    py.FS.writeFile(`/engine/${name}.py`, new Uint8Array(await asset(`engine/${name}.py`)));
  }));
  fixtures = JSON.parse(new TextDecoder().decode(await asset('engine/fixtures.json')));
  if (!Array.isArray(fixtures)) fixtures = fixtures.positions || fixtures.fixtures;
  info.loading_ms.engine_download = elapsed(started);
  started = now();
  py.runPython("import sys; sys.path.insert(0, '/engine')\nfrom bridge import command, ai_turn, configure, reset_metrics, metrics_json, clear_caches, load_position");
  py.globals.set('bridge_mode', mode); py.globals.set('profiling_enabled', diagnostics);
  py.runPython('configure(bridge_mode, profiling_enabled)');
  info.loading_ms.python_import = elapsed(started);
  info.loading_ms.total = elapsed(loadStart);
  info.model_io = {input: 'board: float32 [batch,4,15,15]', outputs: ['policy: float32 [batch,226]', 'value: float32 [batch]']};
  resetMetrics();
  send('ready', {state: JSON.parse(py.runPython("command('{\"type\":\"new\"}')")), info});
}
function loadFixture(fixture) {
  py.globals.set('fixture_json', JSON.stringify(fixture));
  return JSON.parse(py.runPython('load_position(fixture_json)'));
}
async function turn(simulations) {
  if (!Number.isInteger(simulations) || simulations < 4 || simulations > 16384) throw new Error('탐색량은 4~16,384 사이의 정수여야 합니다.');
  resetMetrics(); py.runPython('reset_metrics()'); py.globals.set('budget', simulations);
  const started = now(), result = JSON.parse(await py.runPythonAsync('await ai_turn(budget)'));
  result.worker_turn_ms = elapsed(started); result.metrics = combinedMetrics(); result.info = info;
  return result;
}
function packedPosition(state, passed = 0) { return new Uint8Array([...state.board.flat(), state.turn, passed]); }
async function microbench(repeats = 5) {
  const state = loadFixture(fixtures.find(f => f.category === 'ordinary') || fixtures[0]);
  const position = packedPosition(state), batches = [];
  for (const count of [1, 4, 8, 16]) {
    const packed = new Uint8Array(count * 227);
    for (let i = 0; i < count; i++) packed.set(position, i * 227);
    await evaluator.evaluate(packed, count, false);
    const samples = [], backends = [];
    for (let i = 0; i < repeats; i++) {
      const started = now(), before = selectedBackend;
      await evaluator.evaluate(packed, count, false); samples.push(elapsed(started));
      backends.push({before, after:selectedBackend});
    }
    batches.push({batchSize: count, samples_ms: samples, sample_backends:backends,
      median_ms: quantile(samples, .5), p95_ms: quantile(samples, .95)});
  }
  return {batches, info, note: 'Encoding + inference + CPU readback + output assembly; cache disabled; one excluded warmup per shape'};
}
async function accuracy(restarted = false) {
  const initialBackend = selectedBackend;
  const positions = fixtures.filter(f => !f.winner).slice(0, 7).map(f => {
    const s = loadFixture(f); return {id: f.id, board: s.board.flat(), turn: s.turn, passed: f.passes ? 1 : 0};
  });
  positions.push({...positions[0], id: 'duplicate'}, {...positions[0], passed: 1, id: 'pass-channel'});
  const cases = [], checks = [], tolerance = selectedBackend === 'webgpu' ? {atol:1e-3,rtol:1e-3} : {atol:2e-5,rtol:2e-4};
  for (const count of [1,4,8,16]) {
    const inputs = Array.from({length:count}, (_,i)=>positions[i%positions.length]);
    const packed = new Uint8Array(count*227);
    inputs.forEach((p,i)=>packed.set([...p.board,p.turn,p.passed],i*227));
    evaluator.clear();
    const noCache = await evaluator.evaluate(packed,count,false);
    const cached = await evaluator.evaluate(packed,count,true);
    const again = await evaluator.evaluate(packed,count,true);
    const jsonOutput = JSON.parse(await self.inferBatch(JSON.stringify(inputs.map(p=>[p.board,p.turn,p.passed]))));
    let cacheError=0, bridgeError=0, mismatches=0;
    for(let i=0;i<noCache.length;i++) {
      const jsonValue=i%227===226?jsonOutput[Math.floor(i/227)][1]:jsonOutput[Math.floor(i/227)][0][i%227];
      const c=Math.max(Math.abs(noCache[i]-cached[i]),Math.abs(cached[i]-again[i]));
      const b=Math.abs(noCache[i]-jsonValue), allowed=tolerance.atol+tolerance.rtol*Math.abs(noCache[i]);
      cacheError=Math.max(cacheError,c);bridgeError=Math.max(bridgeError,b);
      if(!Number.isFinite(c+b) || c>allowed || b>allowed)mismatches++;
    }
    checks.push({batchSize:count,cache_max_absolute_error:cacheError,packed_vs_json_max_absolute_error:bridgeError,mismatches,passed:mismatches===0});
    inputs.forEach((p,i)=>cases.push({...p,id:`batch${count}-${i}-${p.id}`,
      backend:selectedBackend,policy:Array.from(noCache.subarray(i*227,i*227+226)),value:noCache[i*227+226]}));
  }
  if(initialBackend!==selectedBackend && !restarted) return accuracy(true);
  return {cases,checks:{batches:checks,tolerance,passed:checks.every(c=>c.passed)},info};
}
self.onmessage = async ({data}) => {
  if (busy) { postMessage({id: data.id, type: 'error', during: data.type, message: 'Worker is busy'}); return; }
  if (Number.isInteger(data.id)) {
    if (data.id <= lastAcceptedId) { postMessage({id:data.id,type:'error',during:data.type,message:'Stale request ID'}); return; }
    lastAcceptedId = data.id;
  }
  busy = true; activeId = data.id;
  try {
    if (data.type === 'init') await init(data);
    else if (!py) throw new Error('Worker is not initialized');
    else if (data.type === 'ai') send('ai-done', await turn(data.simulations));
    else if (data.type === 'fixtures') send('fixtures-result', {fixtures});
    else if (data.type === 'microbench') send('microbench-result', await microbench(data.repeats));
    else if (data.type === 'accuracy') send('accuracy-result', await accuracy());
    else if (data.type === 'bench') {
      if (data.cache === 'cold') { evaluator.clear(); py.runPython('clear_caches()'); }
      const start = now(); loadFixture(data.fixture); const fixtureLoad = elapsed(start);
      // Replay is fixture setup, not the measured search. Start cold search with empty caches.
      if (data.cache === 'cold') { evaluator.clear(); py.runPython('clear_caches()'); }
      const result = await turn(data.simulations); result.fixture_load_ms = fixtureLoad;
      send('bench-result', result);
    } else if (data.type === 'test-fallback' && diagnostics) {
      await fallback('Diagnostic forced fallback'); send('fallback-result', {info});
    } else if (data.type === 'test-device-loss' && diagnostics) {
      if(selectedBackend==='webgpu') {
        device.destroy();const loss=await device.lost;deviceLost=loss.message||loss.reason||'diagnostic destroy';
        send('device-loss-result',{triggered:true,info});
      } else send('device-loss-result',{triggered:false,info});
    } else {
      if (data.type === 'new') { evaluator.clear(); py.runPython('clear_caches()'); }
      py.globals.set('message_json', JSON.stringify(data));
      const started = now(); const state = JSON.parse(py.runPython('command(message_json)'));
      send('state', {state, command_ms: elapsed(started)});
    }
  } catch (error) { send('error', {message: String(error), during: data.type, info}); }
  finally { busy = false; activeId = undefined; }
};
