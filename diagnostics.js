'use strict';
// No telemetry. The exported JSON is assembled entirely in this page.
const $ = id => document.getElementById(id);
const VERSION = 'renju-speed-diagnostics-1';
const BUILD = 'speed-20260923-1';
let fixtures = [], active = null, report = null;

const median = values => quantile(values, 0.5);
function quantile(values, q) {
  const sorted = values.filter(Number.isFinite).slice().sort((a, b) => a - b);
  if (!sorted.length) return null;
  const index = (sorted.length - 1) * q, lo = Math.floor(index), hi = Math.ceil(index);
  return sorted[lo] + (sorted[hi] - sorted[lo]) * (index - lo);
}
const milliseconds = value => Number.isFinite(value) ? value.toFixed(2) : '—';
const seconds = value => Number.isFinite(value) ? (value / 1000).toFixed(3) : '—';
const idOf = (fixture, index = 0) => String(fixture.id ?? fixture.name ?? `fixture-${index}`);
const labelOf = fixture => String(fixture.label ?? fixture.name ?? fixture.id ?? '국면');
const aborted = () => new DOMException('사용자가 측정을 중단했습니다.', 'AbortError');
function assertActive(run) { if (run.controller.signal.aborted || active !== run) throw aborted(); }
function log(message) {
  $('log').textContent += `[${new Date().toLocaleTimeString()}] ${message}\n`;
  $('log').scrollTop = $('log').scrollHeight;
}
function setStatus(message) { $('status').textContent = message; log(message); }
function backendOf(info = {}) {
  return String(info.actualBackend ?? info.actual_backend ?? info.backend?.actual ?? info.backend ?? info.provider ?? '미확인');
}
function hashOf(info = {}) {
  return info.onnx_sha256 ?? info.model?.onnx_sha256 ?? info.metadata?.onnx_sha256 ?? null;
}
function runtimeLabel(runtime) {
  const backend = backendOf(runtime.info);
  const fallback = runtime.requestedBackend === 'webgpu' && backend !== 'webgpu';
  return `${runtime.label} · ${backend}${fallback ? ' (GPU 사용 미확인/복구)' : ''}`;
}
const completedOf = row => row.stats?.used ?? row.metrics?.python?.counters?.completed_rollouts;
function statsTolerance(backend) { return backend === 'webgpu' ? {atol:1e-3,rtol:1e-3} : {atol:1e-5,rtol:1e-4}; }
function semanticComparison(expected, actual) {
  const differences = [], tolerance = statsTolerance(actual.actualBackend);
  const floatStats = new Set(['value','q_range','chosen_target_probability','policy_weight','priors','q']);
  function compare(left, right, path, floating = false) {
    if (differences.length >= 30) return;
    if (typeof left === 'number' && typeof right === 'number') {
      const limit = floating ? tolerance.atol+tolerance.rtol*Math.abs(left) : 0;
      if (!Number.isFinite(left) || !Number.isFinite(right) || Math.abs(left-right)>limit) differences.push(path);
    } else if (left && right && typeof left === 'object' && typeof right === 'object') {
      const keys = new Set([...Object.keys(left),...Object.keys(right)]);
      for (const key of keys) {
        // Solver/runtime clocks are measured separately and are not search semantics.
        if (path.startsWith('stats') && /(^|_)(seconds|elapsed|ms)$/.test(key)) continue;
        compare(left[key],right[key],`${path}.${key}`,floating || (path === 'stats' && floatStats.has(key)));
      }
    } else if (left !== right) differences.push(path);
  }
  compare(expected.state,actual.state,'state');
  const stateEqual = differences.length === 0 && !!expected.state && !!actual.state;
  compare(expected.stats,actual.stats,'stats');
  const statsEqual = !differences.some(path=>path.startsWith('stats')) && !!expected.stats && !!actual.stats;
  const baselineCompleted = completedOf(expected), optimizedCompleted = completedOf(actual);
  const completedEqual = Number.isInteger(baselineCompleted) && baselineCompleted === optimizedCompleted;
  const countersConsistent = [expected,actual].every(row=>row.metrics?.python?.counters?.completed_rollouts == null || row.metrics.python.counters.completed_rollouts === completedOf(row));
  return {repeat:actual.repeat,stateEqual,statsEqual,baselineCompleted,optimizedCompleted,completedEqual,countersConsistent,
    numericStatsTolerance:tolerance,differences,eligible:stateEqual && statsEqual && completedEqual && countersConsistent};
}

class WorkerClient {
  constructor(run) {
    const workerURL = new URL('./worker.js', location.href);
    workerURL.searchParams.set('v', BUILD);
    this.worker = new Worker(workerURL);
    this.nextId = 0;
    this.pending = new Map();
    this.stopped = false;
    run.clients.add(this);
    this.worker.onmessage = ({data}) => {
      if (this.stopped) return;
      if (data.type === 'loading') { log(data.message); return; }
      if (data.type === 'progress') return;
      const request = this.pending.get(data.id);
      if (!request) return;
      this.pending.delete(data.id);
      clearTimeout(request.timer);
      if (data.type === 'error') request.reject(new Error(data.message ?? 'Worker 오류'));
      else request.resolve(data);
    };
    this.worker.onerror = event => this.fail(new Error(event.message || 'Worker를 실행하지 못했습니다.'));
    this.worker.onmessageerror = () => this.fail(new Error('Worker 결과를 읽지 못했습니다.'));
  }
  request(type, data = {}, timeout = 600000) {
    if (this.stopped) return Promise.reject(aborted());
    const id = ++this.nextId;
    return new Promise((resolve, reject) => {
      const timer = setTimeout(() => {
        this.pending.delete(id);
        reject(new Error(`${type} 응답 시간 초과 · Worker를 종료했습니다.`));
        this.stop();
      }, timeout);
      this.pending.set(id, {resolve, reject, timer});
      this.worker.postMessage({id, type, ...data});
    });
  }
  fail(error) {
    for (const {reject, timer} of this.pending.values()) { clearTimeout(timer); reject(error); }
    this.pending.clear();
    this.stop();
  }
  stop() {
    if (this.stopped) return;
    this.stopped = true;
    this.worker.terminate();
    for (const {reject, timer} of this.pending.values()) { clearTimeout(timer); reject(aborted()); }
    this.pending.clear();
  }
}

async function loadFixtures() {
  try {
    const response = await fetch(`./engine/fixtures.json?v=${BUILD}`, {cache:'no-cache'});
    if (!response.ok) throw new Error(`HTTP ${response.status}`);
    const data = await response.json();
    fixtures = Array.isArray(data) ? data : (data.fixtures ?? data.positions);
    if (!Array.isArray(fixtures) || !fixtures.length) throw new Error('기보가 비어 있습니다.');
    renderFixtureOptions();
  } catch (error) {
    $('fixture-options').textContent = `기보 목록을 미리 읽지 못했습니다 (${error.message}). 시작하면 엔진에서 전체 기보를 불러옵니다.`;
  }
}
function renderFixtureOptions() {
  $('fixture-options').replaceChildren();
  fixtures.forEach((fixture, index) => {
    const label = document.createElement('label');
    label.className = 'fixture';
    const input = document.createElement('input');
    input.type = 'checkbox'; input.checked = true; input.value = idOf(fixture, index);
    input.name = 'fixture';
    const text = document.createElement('span');
    text.textContent = labelOf(fixture);
    const note = document.createElement('small');
    note.textContent = [fixture.category, fixture.phase, fixture.turn === 1 ? '흑 차례' : fixture.turn === 2 ? '백 차례' : ''].filter(Boolean).join(' · ');
    text.append(note); label.append(input, text); $('fixture-options').append(label);
  });
}

function configFromForm() {
  const budgets = [...new Set($('budgets').value.split(',').map(value => Number(value.trim())))];
  if (!budgets.length || budgets.length > 8 || budgets.some(value => !Number.isInteger(value) || value < 4 || value > 16384))
    throw new Error('탐색량은 4~16,384 정수를 쉼표로 구분해 1~8개 입력하세요.');
  const repeats = Number($('repeats').value);
  if (!Number.isInteger(repeats) || repeats < 1 || repeats > 30) throw new Error('반복 횟수는 1~30 사이의 정수로 입력하세요.');
  const selected = new Set([...document.querySelectorAll('input[name=fixture]:checked')].map(input => input.value));
  const chosen = fixtures.filter((fixture, index) => selected.has(idOf(fixture, index)));
  if (fixtures.length && !chosen.length) throw new Error('측정할 국면을 하나 이상 선택하세요.');
  const requested = $('backend').value;
  const runtimes = [{id:'baseline-wasm', label:'기존', mode:'baseline', requestedBackend:'wasm'}];
  for (const backend of requested === 'compare' ? ['wasm','webgpu'] : [requested])
    runtimes.push({id:`optimized-${backend}`, label:'개선', mode:'optimized', requestedBackend:backend});
  return {budgets, repeats, fixtures:chosen, runtimes, cacheConditions:['cold','warm'], microbenchmarkRepeats:Math.max(10, repeats), autoDownload:$('auto-download').checked};
}
function lockForm(locked) {
  for (const input of $('configuration').querySelectorAll('input,select')) input.disabled = locked;
  $('start').disabled = locked;
  $('cancel').disabled = !locked;
  for (const button of document.querySelectorAll('[data-preset]')) button.disabled = locked;
}

function preset(name) {
  if (active) return;
  const quickIds = new Set(['early-white','early-black','immediate-win-black','immediate-defence-white','foul-0-foul_attack']);
  const selected = fixtures.filter(fixture=>name === 'full' || (name === 'ordinary' ? fixture.category === 'ordinary' : name === 'interleaved' ? fixture.category === 'ordinary' && ['early','mid'].includes(fixture.phase) : quickIds.has(idOf(fixture))));
  const ids = new Set(selected.map(fixture=>idOf(fixture)));
  for (const input of document.querySelectorAll('input[name=fixture]')) input.checked = ids.has(input.value);
  $('budgets').value = name === 'quick' ? '32' : '32, 128, 512';
  $('repeats').value = name === 'quick' ? '1' : '3';
  $('status').textContent = name === 'quick' ? '빠른 확인은 32회·반복 1회입니다. 전체 성능 비교에는 전체 설정을 사용하세요.' : `${selected.length}개 국면을 선택했습니다. 캐시 cold/warm은 따로 측정합니다.`;
}

function drawBoard(state) {
  const canvas = $('result-board'), context = canvas.getContext('2d');
  const board = state?.board ?? Array.from({length:15}, () => Array(15).fill(0));
  const gap = 20, edge = 25;
  context.fillStyle = '#dfc48d'; context.fillRect(0, 0, canvas.width, canvas.height);
  context.strokeStyle = '#8d794f'; context.lineWidth = 0.8;
  for (let index = 0; index < 15; index++) {
    const position = edge + index * gap;
    context.beginPath(); context.moveTo(edge, position); context.lineTo(305, position); context.stroke();
    context.beginPath(); context.moveTo(position, edge); context.lineTo(position, 305); context.stroke();
  }
  for (let row = 0; row < 15; row++) for (let col = 0; col < 15; col++) {
    const stone = Array.isArray(board[row]) ? board[row][col] : board[row * 15 + col];
    if (!stone) continue;
    const x = edge + col * gap, y = edge + row * gap;
    context.beginPath(); context.arc(x, y, 8, 0, 2 * Math.PI);
    context.fillStyle = stone === 1 ? '#23322c' : '#fafbf7'; context.fill();
    context.strokeStyle = '#485444'; context.lineWidth = 0.7; context.stroke();
    if (state?.last?.[0] === row && state?.last?.[1] === col) {
      context.fillStyle = '#c65f3b'; context.beginPath(); context.arc(x, y, 2.5, 0, 2 * Math.PI); context.fill();
    }
  }
  context.strokeStyle = '#a75539'; context.lineWidth = 1.6;
  for (const [row, col] of state?.forbidden ?? []) {
    const x = edge + col * gap, y = edge + row * gap;
    context.beginPath(); context.moveTo(x-3, y-3); context.lineTo(x+3, y+3);
    context.moveTo(x-3, y+3); context.lineTo(x+3, y-3); context.stroke();
  }
  canvas.setAttribute('aria-label', `마지막 측정 결과 · ${state?.count ?? 0}수`);
}
function afterPaint(signal) {
  // A two-RAF boundary is a paint opportunity, not display hardware telemetry.
  // Hidden tabs are never reported as verified UI timing.
  if (document.visibilityState !== 'visible' || !document.hasFocus()) return Promise.resolve(false);
  return new Promise((resolve, reject) => {
    let first, second, done = false;
    const finish = (value, error) => {
      if (done) return;
      done = true; clearTimeout(timeout); cancelAnimationFrame(first); cancelAnimationFrame(second);
      signal.removeEventListener('abort', onAbort);
      if (error) reject(error); else resolve(value);
    };
    const onAbort = () => finish(false, aborted());
    const timeout = setTimeout(() => finish(false), 2000);
    signal.addEventListener('abort', onAbort, {once:true});
    first = requestAnimationFrame(() => { second = requestAnimationFrame(() => finish(document.visibilityState === 'visible')); });
  });
}

async function sample(run, client, runtime, fixture, simulations, cache, repeat) {
  assertActive(run);
  const started = performance.now(), visibilityStart = document.visibilityState;
  const sampleOrder = ++run.sampleOrder, startedAt = new Date().toISOString();
  const response = await client.request('bench', {fixture, simulations, cache});
  const received = performance.now();
  assertActive(run);
  drawBoard(response.state);
  const workerMs = Number.isFinite(response.seconds) ? response.seconds * 1000 : null;
  $('last-sample').textContent = `${runtimeLabel(runtime)} · ${labelOf(fixture)} · ${simulations}회 · ${cache} · 반복 ${repeat + 1} · AI ${seconds(workerMs)}초`;
  const paintObserved = await afterPaint(run.controller.signal), finished = performance.now();
  const result = {
    fixtureId:idOf(fixture), fixtureLabel:labelOf(fixture), requestedSimulations:simulations,
    sampleOrder, startedAt, finishedAt:new Date().toISOString(),
    pageStarted_ms:started,pageReceived_ms:received,pageFinished_ms:finished,
    cache, repeat:repeat + 1, ai_ms:workerMs,
    worker_turn_ms:response.worker_turn_ms ?? null, fixture_load_ms:response.fixture_load_ms ?? null,
    actualBackend:backendOf(response.info ?? runtime.info),modelHash:hashOf(response.info ?? runtime.info),
    ui:{request_to_response_ms:received-started, request_to_paint_opportunity_ms:paintObserved ? finished-started : null,
      response_to_paint_opportunity_ms:paintObserved ? finished-received : null, paintOpportunityObserved:paintObserved,
      visibilityStart, visibilityEnd:document.visibilityState, definition:'Automated diagnostic request to result board after two requestAnimationFrame callbacks; not main game human input latency.'},
    stats:response.stats ?? null, metrics:response.metrics ?? null, state:response.state ?? null, info:response.info ?? null
  };
  runtime.samples.push(result);
  if (response.info) runtime.info = {...runtime.info, ...response.info};
  run.completed++;
  $('progress').value = run.completed;
  $('download').disabled = false;
  summarize(report);
  renderResults(report);
}

function summarize(result) {
  const summaries = [];
  result.semanticComparisons = [];
  for (const runtime of result.runtimes) {
    const groups = new Map();
    for (const row of runtime.samples) {
      const key = `${row.fixtureId}|${row.requestedSimulations}|${row.cache}|${row.actualBackend}|${row.modelHash}`;
      if (!groups.has(key)) groups.set(key, []);
      groups.get(key).push(row);
    }
    for (const rows of groups.values()) {
      const first = rows[0], times = rows.map(row => row.ai_ms).filter(Number.isFinite);
      summaries.push({runtimeId:runtime.id, fixtureId:first.fixtureId, fixtureLabel:first.fixtureLabel,
        requestedSimulations:first.requestedSimulations, cache:first.cache, samples:times.length,
        median_ms:median(times), p95_ms:quantile(times,0.95), quantileMethod:'linear interpolation (n-1)*p',
        request_to_paint_median_ms:median(rows.map(row => row.ui.request_to_paint_opportunity_ms)),
        completedRollouts:rows.map(completedOf),smallSample:times.length < 20, modelHash:first.modelHash, actualBackend:first.actualBackend});
    }
  }
  result.summaries = summaries;
  result.comparisons = [];
  const baselineRuntime = result.runtimes.find(runtime=>runtime.id === 'baseline-wasm');
  for (const item of summaries.filter(item => item.runtimeId !== 'baseline-wasm')) {
    const baseline = summaries.find(row => row.runtimeId === 'baseline-wasm' && row.fixtureId === item.fixtureId && row.requestedSimulations === item.requestedSimulations && row.cache === item.cache);
    const sameModel = baseline?.modelHash && item.modelHash && baseline.modelHash === item.modelHash;
    if (!baseline || !sameModel) continue;
    const runtime = result.runtimes.find(runtime=>runtime.id === item.runtimeId);
    const rows = runtime.samples.filter(row=>row.fixtureId === item.fixtureId && row.requestedSimulations === item.requestedSimulations && row.cache === item.cache && row.actualBackend === item.actualBackend && row.modelHash === item.modelHash);
    const paired = [];
    for (const row of rows) {
      const before = baselineRuntime.samples.find(reference=>reference.fixtureId === row.fixtureId && reference.requestedSimulations === row.requestedSimulations && reference.cache === row.cache && reference.repeat === row.repeat);
      if (!before) continue;
      const check = {...semanticComparison(before,row),runtimeId:item.runtimeId,fixtureId:item.fixtureId,requestedSimulations:item.requestedSimulations,cache:item.cache,actualBackend:item.actualBackend};
      result.semanticComparisons.push(check);
      if (check.eligible && Number.isFinite(before.ai_ms) && Number.isFinite(row.ai_ms) && row.ai_ms>0) paired.push([before,row]);
    }
    const beforeMedian = median(paired.map(pair=>pair[0].ai_ms)), afterMedian = median(paired.map(pair=>pair[1].ai_ms));
    result.comparisons.push({runtimeId:item.runtimeId, fixtureId:item.fixtureId, requestedSimulations:item.requestedSimulations, cache:item.cache,
      baselineSamples:baseline.samples,optimizedSamples:item.samples,eligiblePairs:paired.length,excludedPairs:rows.length-paired.length,
      baseline_median_ms:beforeMedian,optimized_median_ms:afterMedian,
      speedup:paired.length ? beforeMedian/afterMedian : null,reduction_percent:paired.length ? 100*(beforeMedian-afterMedian)/beforeMedian : null,
      sameModel:true,actualBackend:item.actualBackend});
  }
}
function textCell(row, value, className) {
  const cell = document.createElement('td'); cell.textContent = String(value); if (className) cell.className = className;
  row.append(cell); return cell;
}
function renderResults(result) {
  const target = $('results'); target.replaceChildren();
  if (!result.summaries?.length) { const row = document.createElement('tr'); const cell = textCell(row,'측정 후 표시됩니다.'); cell.colSpan = 8; target.append(row); return; }
  for (const item of result.summaries) {
    const runtime = result.runtimes.find(runtime => runtime.id === item.runtimeId);
    const comparison = result.comparisons.find(row => row.runtimeId === item.runtimeId && row.fixtureId === item.fixtureId && row.requestedSimulations === item.requestedSimulations && row.cache === item.cache && row.actualBackend === item.actualBackend);
    const row = document.createElement('tr');
    textCell(row,`${item.fixtureLabel} / ${item.requestedSimulations}`);
    textCell(row,runtimeLabel({...runtime,info:{...runtime.info,backend:item.actualBackend}})); textCell(row,item.cache); textCell(row,item.samples);
    textCell(row,seconds(item.median_ms)); textCell(row,seconds(item.p95_ms));
    const comparable = comparison?.eligiblePairs > 0;
    textCell(row,comparable ? `${Math.abs(comparison.reduction_percent).toFixed(1)}% ${comparison.reduction_percent >= 0 ? '단축' : '증가'} · ${comparison.speedup.toFixed(2)}배 (${comparison.eligiblePairs}쌍)` : comparison ? '탐색 수·결과 불일치' : item.runtimeId === 'baseline-wasm' ? '기준' : '비교 대기', comparable ? comparison.reduction_percent >= 0 ? 'good' : 'bad' : '');
    textCell(row,seconds(item.request_to_paint_median_ms)); target.append(row);
  }
}

function compareAccuracy(result) {
  const reference = result.runtimes.find(runtime => runtime.id === 'baseline-wasm');
  result.accuracyComparisons = [];
  if (!reference?.accuracy?.cases) return;
  for (const runtime of result.runtimes.filter(runtime => runtime !== reference && runtime.accuracy?.cases)) {
    const accuracyBackend = backendOf(runtime.accuracy.info ?? runtime.info);
    const defaults = statsTolerance(accuracyBackend);
    const tolerance = runtime.info.accuracyTolerance ?? defaults;
    const comparison = {runtimeId:runtime.id, actualBackend:accuracyBackend, tolerance,
      criterion:'abs(actual-reference) <= atol + rtol*abs(reference)', maxAbsoluteError:0, maxRelativeError:0,
      relativeDenominatorFloor:1e-8, comparedValues:0, mismatchedValues:0, cases:[], sameModel:!!hashOf(reference.info) && hashOf(reference.info) === hashOf(runtime.info)};
    const actualCases = runtime.accuracy.cases;
    for (let index = 0; index < reference.accuracy.cases.length; index++) {
      const expected = reference.accuracy.cases[index], actual = actualCases[index];
      if (!actual || (expected.id != null && expected.id !== actual.id)) {
        comparison.cases.push({id:expected.id ?? index, passed:false, error:'Case order or identifier mismatch'}); continue;
      }
      const left = [...(expected.policy ?? []), expected.value], right = [...(actual.policy ?? []), actual.value];
      const item = {id:expected.id ?? index, passed:true, maxAbsoluteError:0, maxRelativeError:0, mismatchedValues:0};
      if (expected.policy?.length !== 226 || actual.policy?.length !== 226 || left.length !== right.length) {
        item.passed = false; item.error = 'Expected 226 policy logits and one value';
      } else if (JSON.stringify([expected.board,expected.turn,expected.passed]) !== JSON.stringify([actual.board,actual.turn,actual.passed])) {
        item.passed = false; item.error = 'Input board, turn or pass channel differs';
      } else {
        left.forEach((value, outputIndex) => {
          comparison.comparedValues++;
          const abs = Math.abs(value-right[outputIndex]), relative = abs/Math.max(1e-8,Math.abs(value));
          if (!Number.isFinite(abs) || abs > tolerance.atol+tolerance.rtol*Math.abs(value)) item.mismatchedValues++;
          item.maxAbsoluteError = Math.max(item.maxAbsoluteError, Number.isFinite(abs) ? abs : Infinity);
          item.maxRelativeError = Math.max(item.maxRelativeError, Number.isFinite(relative) ? relative : Infinity);
        });
        item.passed = item.mismatchedValues === 0;
      }
      comparison.maxAbsoluteError = Math.max(comparison.maxAbsoluteError,item.maxAbsoluteError);
      comparison.maxRelativeError = Math.max(comparison.maxRelativeError,item.maxRelativeError);
      comparison.mismatchedValues += item.mismatchedValues;
      comparison.cases.push(item);
    }
    comparison.workerChecksPassed = reference.accuracy.checks?.passed === true && runtime.accuracy.checks?.passed === true;
    comparison.passed = comparison.sameModel && comparison.workerChecksPassed && actualCases.length === reference.accuracy.cases.length && comparison.cases.length > 0 && comparison.cases.every(item => item.passed);
    result.accuracyComparisons.push(comparison);
  }
}
function renderDetails(result) {
  compareAccuracy(result);
  const comparisons = result.accuracyComparisons;
  $('accuracy-status').textContent = comparisons.length ? comparisons.map(item => `${item.runtimeId}: ${item.passed ? '허용 오차 내 일치' : '불일치 또는 비교 불가'} · 최대 절대 오차 ${item.maxAbsoluteError.toExponential(3)} · ${item.comparedValues}개 값`).join(' / ') : '비교를 기다리고 있습니다. 원시 출력과 캐시·배치 순서 검사는 JSON에 함께 저장됩니다.';
  $('environment').textContent = JSON.stringify({environment:result.environment, model:result.model, runtimes:result.runtimes.map(runtime => ({id:runtime.id, requestedBackend:runtime.requestedBackend, info:runtime.info, accuracyChecks:runtime.accuracy?.checks, error:runtime.error})), accuracyComparisons:comparisons}, null, 2);
  const target = $('microbench'); target.replaceChildren();
  for (const runtime of result.runtimes) for (const batch of runtime.microbenchmark?.batches ?? []) {
    const row = document.createElement('tr'); textCell(row,runtimeLabel(runtime)); textCell(row,batch.batchSize);
    textCell(row,batch.samples_ms?.length ?? batch.samples ?? '—'); textCell(row,milliseconds(batch.median_ms)); textCell(row,milliseconds(batch.p95_ms)); target.append(row);
  }
  if (!target.children.length) { const row = document.createElement('tr'); textCell(row,'측정 후 표시됩니다.').colSpan = 5; target.append(row); }
}

async function begin(event) {
  event.preventDefault();
  if (active) return;
  let configuration;
  try { configuration = configFromForm(); } catch (error) { $('status').textContent = error.message; return; }
  const run = {controller:new AbortController(), clients:new Set(), completed:0,sampleOrder:0};
  active = run;
  report = {
    schemaVersion:VERSION, status:'running', startedAt:new Date().toISOString(),
    environment:{url:location.href,userAgent:navigator.userAgent,language:navigator.language,platform:navigator.platform,
      hardwareConcurrency:navigator.hardwareConcurrency ?? null,deviceMemoryGiB:navigator.deviceMemory ?? null,
      secureContext:isSecureContext,crossOriginIsolated:crossOriginIsolated,webgpuExposed:!!navigator.gpu,
      screen:{width:screen.width,height:screen.height,devicePixelRatio},initialVisibility:document.visibilityState,
      timeOrigin:performance.timeOrigin},
    configuration,model:null,runtimes:[],summaries:[],comparisons:[],semanticComparisons:[],accuracyComparisons:[],visibilityEvents:[],executionOrder:[],
    methodology:{schedule:'interleaved-cyclic-v1',runtimeOrder:configuration.runtimes.map(runtime=>runtime.id),
      searchOrder:'fixture -> budget -> repeat -> cyclic runtime order rotated by repeat -> cold immediately followed by warm. All sessions initialized first; exactly one search at a time.',
      warmup:'Unrecorded search for every fixture/budget/runtime before measured repeats, plus inference warmups in microbenchmark.',
      cache:'Each measured cold sample clears neural AND existing Python rule caches; immediate paired warm sample retains them. Baseline has no neural cache.',
      timing:'AI timer is supplied by worker; request-to-response and two-RAF diagnostic board timing measured by page. Nested detail timers must not be summed.',
      quantile:'Linear interpolation at (n-1)*p; p95 with n<20 is exploratory, not a stable tail-latency estimate.',
      comparisonEligibility:'Pair same fixture/budget/cache/repeat with same model hash. Require identical completed rollouts, final state, structural search stats, and numeric stats within recorded tolerance. Durations excluded from semantic equality. Different GPU moves are reported, not declared rule violations.',
      defaultBackend:'The app prefers available WebGPU based on the measured target-browser cold-search benefit, with WASM fallback. Other devices may differ; diagnostics never changes app preferences.'},
    limitations:['GPU request or adapter detection alone does not prove every operator ran on GPU; read runtime provider evidence and fallback reason.',
      'Interleaving and cyclic runtime rotation reduce long-term environment drift, but do not eliminate power, thermals, background work or short-term carry-over effects.',
      'All selected Worker/Pyodide/ORT sessions remain resident during the comparison. Only one search runs at a time; memory usage exceeds the single-session game.',
      'Two RAF callbacks measure a paint opportunity, not physical display latency or main game input.',
      'Output tolerances test numerical agreement, not an assertion of equal playing strength.']
  };
  $('log').textContent=''; $('download').disabled=true; lockForm(true);
  $('report-json').value = '측정 중입니다. 종료 또는 중단 후 전체 JSON이 여기에 표시됩니다.';
  $('progress').value=0;
  renderResults(report); renderDetails(report);
  const visibility = () => { report.visibilityEvents.push({at:new Date().toISOString(),state:document.visibilityState}); if(document.visibilityState !== 'visible') log('탭이 백그라운드로 바뀌었습니다. 해당 표본의 화면 표시 시간은 유효하지 않을 수 있습니다.'); };
  document.addEventListener('visibilitychange', visibility);
  try {
    const metadataResponse = await fetch('./model.json', {cache:'no-cache',signal:run.controller.signal});
    if (!metadataResponse.ok) throw new Error(`모델 정보: HTTP ${metadataResponse.status}`);
    report.model = await metadataResponse.json();
    const sessions = [];
    function failSession(entry, error, phase) {
      entry.runtime.error = {message:String(error.message ?? error),at:new Date().toISOString(),phase};
      entry.failed = true;entry.client.stop();run.clients.delete(entry.client);
      log(`${entry.runtime.id} 중단: ${entry.runtime.error.message} · 다른 실행 경로를 계속합니다.`);
    }
    // Sessions stay resident; this loop and all measurements remain sequential.
    for (const definition of configuration.runtimes) {
      assertActive(run);
      const runtime = {...definition,info:{},samples:[],warmups:[],accuracy:null,microbenchmark:null};
      report.runtimes.push(runtime);
      const client = new WorkerClient(run);
      const entry = {runtime,client,failed:false};sessions.push(entry);
      try {
        setStatus(`${definition.id} · 실행 엔진을 준비합니다.`);
        const initStart = performance.now();
        const initial = await client.request('init',{mode:definition.mode,backend:definition.requestedBackend,diagnostics:true});
        assertActive(run);
        runtime.info = initial.info ?? {};
        runtime.pageInitElapsed_ms = performance.now()-initStart;
        if (!configuration.fixtures.length) {
          const loaded = await client.request('fixtures');
          if (!Array.isArray(loaded.fixtures) || !loaded.fixtures.length) throw new Error('엔진이 고정 기보를 제공하지 않았습니다.');
          configuration.fixtures = loaded.fixtures;
        }
        $('progress').max = configuration.runtimes.length * configuration.fixtures.length * configuration.budgets.length * configuration.repeats * 2;
        setStatus(`${runtimeLabel(runtime)} · 출력 정확성을 검사합니다.`);
        runtime.accuracy = await client.request('accuracy');
        if (runtime.accuracy.info) runtime.info = {...runtime.info,...runtime.accuracy.info};
        renderDetails(report);
        setStatus(`${runtimeLabel(runtime)} · 배치 1·4·8·16 추론 지연을 측정합니다.`);
        runtime.microbenchmark = await client.request('microbench',{batchSizes:[1,4,8,16],repeats:configuration.microbenchmarkRepeats});
        if (runtime.microbenchmark.info) runtime.info = {...runtime.info,...runtime.microbenchmark.info};
        renderDetails(report);
      } catch (error) {
        if (error.name==='AbortError') throw error;
        failSession(entry,error,'initialization');
      } finally { renderDetails(report); }
    }
    let groupIndex = 0;
    for (const fixture of configuration.fixtures) for (const simulations of configuration.budgets) {
      assertActive(run);
      const warmupOffset = groupIndex++ % sessions.length;
      const warmupOrder = [...sessions.slice(warmupOffset),...sessions.slice(0,warmupOffset)];
      for (const entry of warmupOrder) {
        if (entry.failed) continue;
        const {runtime,client} = entry;
        try {
          setStatus(`${runtimeLabel(runtime)} · ${labelOf(fixture)} · ${simulations}회 워밍업 (통계 제외)`);
          const startedAt = new Date().toISOString(), pageStarted = performance.now();
          const warmup = await client.request('bench',{fixture,simulations,cache:'cold'});
          runtime.warmups.push({fixtureId:idOf(fixture),simulations,seconds:warmup.seconds,metrics:warmup.metrics,
            startedAt,finishedAt:new Date().toISOString(),pageStarted_ms:pageStarted,pageFinished_ms:performance.now(),excludedFromStatistics:true});
          if(warmup.info) runtime.info={...runtime.info,...warmup.info};
        } catch(error) { if(error.name==='AbortError') throw error;failSession(entry,error,'warmup'); }
      }
      for (let repeat=0; repeat<configuration.repeats; repeat++) {
        const offset = repeat % sessions.length;
        const order = [...sessions.slice(offset),...sessions.slice(0,offset)].filter(entry=>!entry.failed);
        report.executionOrder.push({fixtureId:idOf(fixture),simulations,repeat:repeat+1,
          runtimeOrder:order.map(entry=>entry.runtime.id),cacheOrder:[...configuration.cacheConditions],at:new Date().toISOString()});
        for (const entry of order) {
          const {runtime,client} = entry;
          try {
            // Keep cold/warm paired without another runtime in between.
            for (const cache of configuration.cacheConditions) {
              assertActive(run);
              setStatus(`${runtimeLabel(runtime)} · ${labelOf(fixture)} · ${simulations}회 · ${cache} · ${repeat+1}/${configuration.repeats} (교차 비교)`);
              await sample(run,client,runtime,fixture,simulations,cache,repeat);
            }
          } catch(error) { if(error.name==='AbortError') throw error;failSession(entry,error,'measurement'); }
        }
      }
    }
    report.status = report.runtimes.some(runtime=>runtime.error) ? 'partial' : 'complete';
    const failed = report.runtimes.filter(runtime=>runtime.error).length;
    setStatus(failed ? `측정 종료 · ${failed}개 경로에서 오류가 있었습니다. 완료한 표본과 오류를 JSON에 저장했습니다.` : '측정 완료 · 결과를 내려받아 같은 조건의 cold와 warm을 각각 비교하세요.');
  } catch (error) {
    report.status = error.name==='AbortError' ? 'cancelled' : 'failed';
    report.error = String(error.message ?? error);
    setStatus(error.name==='AbortError' ? '측정을 중단하고 Worker를 종료했습니다. 수집한 결과는 내려받을 수 있습니다.' : `측정을 완료하지 못했습니다: ${report.error}`);
  } finally {
    for (const client of run.clients) client.stop(); run.clients.clear();
    document.removeEventListener('visibilitychange',visibility);
    report.finishedAt = new Date().toISOString();
    summarize(report); renderResults(report); renderDetails(report);
    $('report-json').value = JSON.stringify(report,null,2);
    active=null; lockForm(false); $('download').disabled=false;
    if (configuration.autoDownload && report.status==='complete') download();
  }
}
function cancel() {
  if (!active) return;
  active.controller.abort();
  for (const client of active.clients) client.stop();
}
function download() {
  if (!report) return;
  summarize(report); compareAccuracy(report);
  const blob = new Blob([JSON.stringify(report,null,2)],{type:'application/json'});
  // Explicit local testing URL only; public Pages never uploads a visitor's report.
  if(location.origin==='http://127.0.0.1:8765' && new URLSearchParams(location.search).get('save')==='local') {
    fetch('./__save_diagnostic',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify(report)})
      .then(r=>{if(!r.ok)throw Error(r.status);return r.json();}).then(r=>log('로컬 검사 결과 저장: '+r.saved)).catch(e=>log('로컬 저장 오류: '+e.message));
  }
  const url = URL.createObjectURL(blob), link = document.createElement('a');
  link.href=url; link.download=`renju-benchmark-${report.startedAt.replace(/[:.]/g,'-')}-${report.status}.json`;
  document.body.append(link); link.click(); link.remove();
  setTimeout(()=>URL.revokeObjectURL(url),10000);
}
$('configuration').addEventListener('submit',begin);
$('cancel').addEventListener('click',cancel);
$('download').addEventListener('click',download);
for (const button of document.querySelectorAll('[data-preset]')) button.addEventListener('click',()=>preset(button.dataset.preset));
addEventListener('pagehide',cancel);
drawBoard();
loadFixtures();
