const $ = id => document.getElementById(id);
let worker, state, ready = false, busy = true, human = 1, thinkingSince = 0, timer;
let pending = '', lastSeconds = null, focused = 112;
const cells = [];
for (let r = 0; r < 15; r++) for (let c = 0; c < 15; c++) {
  const cell = document.createElement('button');
  cell.type = 'button'; cell.setAttribute('role', 'gridcell');
  cell.dataset.position = r * 15 + c;
  cell.tabIndex = r * 15 + c === focused ? 0 : -1;
  cell.innerHTML = `${[3,7,11].includes(r) && [3,7,11].includes(c) && (r===c || r+c===14) ? '<span class="star"></span>' : ''}<span class="stone"></span>`;
  cell.addEventListener('click', () => play(r * 15 + c));
  cell.addEventListener('focus', () => {cells[focused].tabIndex=-1; focused=r*15+c; cell.tabIndex=0;});
  cell.addEventListener('keydown', event => {
    const shifts = {ArrowLeft:[0,-1],ArrowRight:[0,1],ArrowUp:[-1,0],ArrowDown:[1,0]};
    if (!shifts[event.key]) return;
    event.preventDefault(); const [dr,dc]=shifts[event.key];
    const next = Math.max(0,Math.min(14,r+dr))*15+Math.max(0,Math.min(14,c+dc));
    cells[next].focus();
  });
  cells.push(cell); $('board').append(cell);
}
function status(message, detail = '') { $('status').textContent=message; $('detail').textContent=detail; }
function render() {
  const forbidden = new Set((state?.forbidden || []).map(([r,c])=>r*15+c));
  cells.forEach((cell,a)=>{
    const r=Math.floor(a/15),c=a%15,stone=state?.board[r][c] || 0;
    cell.className=['cell',r===0?'top':'',r===14?'bottom':'',c===0?'left':'',c===14?'right':'',stone===1?'black':stone===2?'white':'',forbidden.has(a)?'forbidden':'',state?.last?.[0]===r && state?.last?.[1]===c?'last':''].join(' ');
    // Keep all intersections keyboard reachable; validate actions in play().
    cell.setAttribute('aria-disabled',String(!ready || busy || !!stone || state?.winner!==null || forbidden.has(a)));
    cell.setAttribute('aria-label',`${String.fromCharCode(65+c)}${15-r}, ${stone===1?'흑':stone===2?'백':forbidden.has(a)?'금수':'빈칸'}`);
  });
  $('move-count').textContent=`${state?.count || 0}수`;
  $('turn').textContent=!ready?'대국 준비':state.winner!==null?'대국 종료':`${state.turn===1?'● 흑':'○ 백'} 차례 · ${state.turn===human?'나':'AI'}`;
  $('settings').disabled=!ready || busy;
  $('new-game').disabled=!ready || busy;
  $('pass').hidden=!state?.canPass || state?.turn!==human || state?.winner!==null;
  $('pass').disabled=busy;
}
function settle() {
  busy=false; render();
  if (state.winner!==null) {
    status(state.winner===0?'무승부입니다':state.winner===human?'이겼습니다!':'AI가 이겼습니다', '새 대국으로 다시 도전해 보세요.');
  } else if (state.turn!==human) {
    busy=true;pending='ai';thinkingSince=Date.now();render();
    status('AI가 생각하고 있습니다…','탐색을 준비하는 중');
    worker.postMessage({type:'ai',simulations:Number($('simulations').value)});
    timer=setInterval(()=>{ $('detail').textContent=`${((Date.now()-thinkingSince)/1000).toFixed(1)}초 경과 · 탐색 중`; },1000);
  } else {
    status('내 차례입니다',state.count===0?'중앙에 첫 돌을 놓아 주세요.':lastSeconds===null?'빈 교차점을 선택하세요.':`AI 생각 시간 ${lastSeconds.toFixed(1)}초 · 빈 교차점을 선택하세요.`);
  }
}
function play(action) {
  if (!ready || busy || state.winner!==null || state.turn!==human) return;
  if (action!==225 && state.board[Math.floor(action/15)][action%15]) return;
  busy=true; pending='move';render();worker.postMessage({type:'move',action});
}
function boot() {
  worker?.terminate();clearInterval(timer);ready=false;busy=true;state=null;lastSeconds=null;
  $('retry').hidden=true;status('대국 엔진을 불러오는 중…','처음 접속하면 준비에 잠시 시간이 걸립니다.');render();
  worker = new Worker('./worker.js');
  worker.onmessage=({data})=>{
    if(data.type==='loading') status(data.message,'첫 접속에는 실행 엔진 다운로드가 필요합니다.');
    else if(data.type==='ready'){ready=true;state=data.state;human=Number($('side').value);settle();}
    else if(data.type==='state'){state=data.state;settle();}
    else if(data.type==='ai-done'){clearInterval(timer);state=data.state;lastSeconds=data.seconds;settle();}
    else if(data.type==='progress') $('detail').textContent=`${data.evaluated}개 국면 검토 · ${((Date.now()-thinkingSince)/1000).toFixed(1)}초`;
    else if(data.type==='error') {
      clearInterval(timer);busy=false;
      const recoverable=data.during==='move';
      if (!recoverable) ready=false;
      render();status(recoverable?'이곳에는 둘 수 없습니다':'엔진을 실행하지 못했습니다',recoverable?data.message.split('ValueError: ').pop().trim():'인터넷 연결을 확인하고 엔진을 다시 불러와 주세요.');
      $('retry').hidden=recoverable;console.error(data.message);
    }
  };
  worker.onerror=event=>{clearInterval(timer);ready=false;busy=false;render();status('엔진을 불러오지 못했습니다','연결 상태를 확인한 뒤 다시 시도해 주세요.');$('retry').hidden=false;console.error(event.message);};
  worker.postMessage({type:'init'});
}
$('new-game').addEventListener('click',()=>{if(busy)return;human=Number($('side').value);lastSeconds=null;busy=true;render();worker.postMessage({type:'new'});});
$('pass').addEventListener('click',()=>play(225));
$('retry').addEventListener('click',boot);
fetch('./model.json').then(r=>{if(!r.ok)throw Error(r.status);return r.json();}).then(m=>{$('model-info').textContent=`best · 학습 ${Number(m.games).toLocaleString()}판`;}).catch(()=>{});
boot();
