'use strict';
const BUILD = 'speed-20260923-1';
const $ = id => document.getElementById(id);
let report, activeClient, cancelled = false;
const assert = (condition, message) => { if (!condition) throw new Error(message); };
const same = (a, b) => JSON.stringify(a) === JSON.stringify(b);
const snapshot = state => JSON.parse(JSON.stringify(state));

class WorkerClient {
  constructor() {
    this.sequence = 0; this.pending = new Map(); this.closed = false;
    this.worker = new Worker('./worker.js?v=' + BUILD);
    this.worker.onmessage = ({data}) => {
      const pending = this.pending.get(data.id);
      if (!pending) return;
      if (data.type === 'loading' || data.type === 'progress') return;
      this.pending.delete(data.id); clearTimeout(pending.timer); pending.resolve(data);
    };
    this.worker.onerror = event => this.close(new Error(event.message || 'Worker error'));
  }
  request(type, fields = {}, replayId) {
    if (this.closed) return Promise.reject(new Error('Worker closed'));
    const id = replayId === undefined ? ++this.sequence : replayId;
    assert(!this.pending.has(id), 'A request with this ID is already pending');
    return new Promise((resolve, reject) => {
      const timer = setTimeout(() => this.close(new Error('Worker request timed out: ' + type)), 180000);
      this.pending.set(id, {resolve, reject, timer});
      this.worker.postMessage({type, ...fields, id});
    });
  }
  close(error = new Error('Worker closed')) {
    if (this.closed) return;
    this.closed = true; this.worker.terminate();
    for (const pending of this.pending.values()) { clearTimeout(pending.timer); pending.reject(error); }
    this.pending.clear();
  }
}

function writeReport() { $('report-json').value = JSON.stringify(report, null, 2); }
function download() {
  if (!report) return;
  if(location.origin==='http://127.0.0.1:8765' && new URLSearchParams(location.search).get('save')==='local') {
    fetch('./__save_diagnostic',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify(report)})
      .then(r=>{if(!r.ok)throw Error(r.status);return r.json();}).then(r=>{$('status').textContent+=' · 로컬 JSON 저장: '+r.saved;})
      .catch(e=>{$('status').textContent+=' · 로컬 저장 오류: '+e.message;});
  }
  const url = URL.createObjectURL(new Blob([JSON.stringify(report, null, 2)], {type:'application/json'}));
  const link = document.createElement('a'); link.href = url;
  link.download = 'renju-browser-acceptance-' + report.started_at.replace(/[:.]/g, '-') + '.json';
  link.click(); setTimeout(() => URL.revokeObjectURL(url), 10000);
}
function reply(message, expected) {
  assert(message.type === expected, `${expected} expected, got ${message.type}: ${message.message || ''}`);
  return message;
}
function empty(state) {
  assert(state.count === 0 && state.turn === 1 && state.winner === null, 'Reset state metadata differs');
  assert(state.board.flat().every(value => value === 0), 'Reset left a stone on the board');
  assert(state.forbidden.length === 0 && !state.canPass, 'Opening restriction was represented as forbidden or pass');
}
function oneMove(before, after) {
  assert(before.winner === null, 'Cannot advance a finished game');
  const changes = [];
  before.board.flat().forEach((value, i) => {
    const next = after.board.flat()[i];
    if (value !== next) {
      assert(value === 0 && next === before.turn, 'AI changed an existing stone or used the wrong color');
      changes.push(i);
    }
  });
  assert(changes.length === 1 && after.count === before.count + 1, 'Expected exactly one new AI stone');
  assert(after.turn === 3 - before.turn || after.winner !== null, 'Unexpected side to move');
  const forbidden = new Set(before.forbidden.map(([r,c]) => r * 15 + c));
  assert(!forbidden.has(changes[0]), 'AI selected a displayed forbidden point');
  return changes[0];
}

async function runBackend(backend) {
  const runtime = {requested_backend:backend, status:'running', checks:[], info:null};
  report.runtimes.push(runtime);
  const client = activeClient = new WorkerClient();
  let state;
  async function check(name, action) {
    if (cancelled) throw new Error('Cancelled');
    $('status').textContent = `${backend} · ${name}`;
    const element = document.createElement('li'); element.textContent = `${backend} · ${name} …`; $('checks').append(element);
    const item = {name, status:'running'}; runtime.checks.push(item);
    try {
      const details = await action(); Object.assign(item, details || {});
      item.status = details?.skipped ? 'skipped' : 'passed';
      element.textContent = `${backend} · ${name}: ${details?.skipped ? '미실행 · ' + details.reason : '통과'}`;
      element.className = details?.skipped ? '' : 'good';
    } catch (error) {
      item.status = cancelled ? 'cancelled' : 'failed'; item.error = String(error);
      element.textContent = `${backend} · ${name}: ${cancelled ? '중단' : '실패'} · ${error.message}`;
      element.className = 'bad'; throw error;
    } finally { writeReport(); }
  }
  async function inspect() { return reply(await client.request('inspect'), 'state').state; }
  async function ai() {
    const before = snapshot(state), result = reply(await client.request('ai', {simulations:32}), 'ai-done');
    const action = oneMove(before, result.state); state = result.state; runtime.info = result.info;
    return {request_id:result.id, action, before, after:snapshot(state), backend:result.info.backend,
      stats:result.stats, metrics:result.metrics};
  }
  try {
    await check('새 Worker 초기화', async () => {
      const result = reply(await client.request('init', {mode:'optimized', backend, diagnostics:true}), 'ready');
      state = result.state; runtime.info = result.info; empty(state);
      return {request_id:result.id, state:snapshot(state), info:result.info};
    });
    await check('중앙 밖 첫 수 거부', async () => {
      const before = snapshot(state), result = await client.request('move', {action:0});
      assert(result.type === 'error' && result.during === 'move', 'Off-center opening was accepted');
      state = await inspect(); assert(same(before, state), 'Rejected move mutated the board');
      return {request_id:result.id, error:result.message, after:snapshot(state)};
    });
    await check('중앙 H8 착수와 AI 32회 탐색', async () => {
      state = reply(await client.request('move', {action:112}), 'state').state;
      assert(state.count === 1 && state.board[7][7] === 1 && state.turn === 2, 'Center move differs');
      const result = await ai(); runtime.first_ai = result; return result;
    });
    await check('완료된 AI 요청 재전송 차단', async () => {
      const before = snapshot(state), id = runtime.first_ai.request_id;
      const result = await client.request('ai', {simulations:32}, id);
      assert(result.type === 'error' && /Stale request ID/.test(result.message), 'Completed request was not rejected');
      state = await inspect(); assert(same(before, state), 'Stale AI request applied another move');
      return {replayed_id:id, after:snapshot(state)};
    });
    await check('새 대국 0수 초기화', async () => {
      state = reply(await client.request('new'), 'state').state; empty(state);
      return {state:snapshot(state)};
    });
    await check('복구 검사 시작판 준비', async () => {
      state = reply(await client.request('move', {action:112}), 'state').state;
      return ai();
    });
    await check('실제 GPU 장치 손실 후 판 보존·AI 재개', async () => {
      if (runtime.info.backend !== 'webgpu') return {skipped:true, reason:'현재 실제 backend가 WebGPU가 아닙니다.', info:runtime.info};
      const before = snapshot(state);
      const loss = reply(await client.request('test-device-loss'), 'device-loss-result');
      assert(loss.triggered === true, 'GPU device loss was not triggered');
      state = await inspect(); assert(same(before, state), 'Device loss mutated the board before inference');
      const result = await ai(); assert(result.backend === 'wasm', 'Device loss did not recover through WASM');
      assert(runtime.info.fallback_reasons.some(reason => /Device lost/.test(reason)), 'Device loss fallback reason missing');
      return {loss_request_id:loss.id, ...result};
    });
    await check('명시적 WASM 복구 후 판 보존·AI 재개', async () => {
      const before = snapshot(state), result = reply(await client.request('test-fallback'), 'fallback-result');
      assert(result.info.backend === 'wasm', 'Forced fallback did not create a WASM session');
      state = await inspect(); assert(same(before, state), 'Forced fallback changed the board');
      return {fallback_request_id:result.id, ...await ai()};
    });
    runtime.status = 'passed';
  } catch (error) {
    runtime.status = cancelled ? 'cancelled' : 'failed'; runtime.error = String(error);
  } finally { runtime.final_state = state; client.close(); if (activeClient === client) activeClient = null; writeReport(); }
}

$('start').addEventListener('click', async () => {
  cancelled = false; $('start').disabled = true; $('cancel').disabled = false; $('download').disabled = true; $('checks').replaceChildren();
  report = {schema:'renju-browser-acceptance-v1', schemaVersion:'renju-speed-acceptance-1', build:BUILD, started_at:new Date().toISOString(), status:'running',
    environment:{userAgent:navigator.userAgent, crossOriginIsolated, isSecureContext, location:location.href},
    note:'Sequential isolated workers, fixed model, 32 simulations. Functional checks only; not performance or playing-strength measurement.', runtimes:[]};
  writeReport();
  try {
    for (const backend of ['wasm', 'webgpu']) { if (cancelled) break; await runBackend(backend); }
    const hashes = report.runtimes.map(runtime => runtime.info?.model?.onnx_sha256).filter(Boolean);
    report.same_model = hashes.length === 2 && hashes.every(value => value === hashes[0]);
    report.status = cancelled ? 'cancelled' : report.runtimes.every(runtime => runtime.status === 'passed') && report.same_model ? 'passed' : 'failed';
    report.gpu_device_loss_verified = report.runtimes.some(runtime => runtime.checks.some(item => item.name === '실제 GPU 장치 손실 후 판 보존·AI 재개' && item.status === 'passed'));
  } catch (error) { report.status = cancelled ? 'cancelled' : 'failed'; report.error = String(error); }
  finally {
    report.finished_at = new Date().toISOString(); writeReport();
    $('status').textContent = report.status === 'passed' ? `검사 통과${report.gpu_device_loss_verified ? ' · 실제 GPU 장치 손실 복구 확인' : ' · GPU 장치 손실은 미실행'}` : report.status === 'cancelled' ? '중단됨 · 수집한 결과를 저장할 수 있습니다.' : '실패한 검사가 있습니다. JSON에서 상세 결과를 확인하세요.';
    $('start').disabled = false; $('cancel').disabled = true; $('download').disabled = false;
    if ($('auto-download').checked && !cancelled) download();
  }
});
$('cancel').addEventListener('click', () => { cancelled = true; activeClient?.close(new Error('Cancelled')); });
$('download').addEventListener('click', download);
