# 렌주 AI 웹 대국

이 폴더 자체가 GitHub Pages에 배포할 완성된 정적 사이트입니다. Node 빌드나 Python 서버는 배포 후 필요하지 않습니다.

## 내 컴퓨터에서 실행

`index.html`을 더블클릭하면 브라우저 보안 제한 때문에 엔진이 로드되지 않습니다. 프로젝트 루트에서 다음 명령으로 실행한 뒤 http://127.0.0.1:8765/Gomoku/ 를 여세요.

```powershell
.\.venv\Scripts\python.exe renju_next\prepare_web.py
.\.venv\Scripts\python.exe renju_next\serve_web.py
```

## GitHub Pages 배포

1. GitHub에서 공개 저장소를 하나 만듭니다. 예: `renju-play`.
2. `renju_next/package_web.py`를 실행하고 생성된 zip의 **내부 내용 전체**를 저장소 최상위에 넣습니다. `inference.js`, `web-assets.json`, `diagnostics.*`, `engine/web_profile.py`, `engine/fixtures.json`도 필요합니다. 관련 파일을 하나의 커밋으로 함께 갱신하면 배포 중 서로 다른 버전이 섞이지 않습니다.
3. 저장소 **Settings → Pages → Build and deployment**에서 **Deploy from a branch**, **main**, **/(root)**를 선택하고 저장합니다.
4. 배포가 완료되면 Pages 설정에 표시되는 공개 주소를 공유합니다. 일반적인 형태는 `https://사용자명.github.io/저장소명/`입니다.

상대 경로를 사용하므로 저장소 이름이 붙는 GitHub Pages 주소에서도 실행됩니다. 학습 데이터, 가상환경, 원본 `.pt` 체크포인트를 올릴 필요는 없습니다.

## 구성

- `best.onnx`: `models_next/best.pt`에서 내보낸 정책·가치 신경망.
- `engine/renju.py`, `patterns.py`, `threat_search.py`: 기존 소스 그대로 복사한 규칙·전술 엔진.
- `engine/core.py`: 기존 core의 신경망 정의 앞부분을 추출한 게임 어댑터.
- `engine/search.py`: 기존 탐색 코드. 학습용 정책 벡터를 만드는 `torch` 호출만 작은 리스트 어댑터로 대체했습니다. 탐색·후보 제거·전술 판정은 동일합니다.
- `worker.js`: Pyodide 0.27.7 및 ONNX Runtime Web 1.22.0 실행. 첫 접속에는 jsDelivr에서 실행 엔진을 받습니다. 따라서 모델 크기보다 초기 다운로드가 큽니다.
- `inference.js`: 입력 버퍼 재사용, 동일 배치의 중복 제거, 원시 logits/value 2,048개 LRU. 키는 모델 SHA-256·backend·FP32·225칸·차례·이전 패스입니다. 새 대국/새 Worker/장치 복구에서 비웁니다.
- WebGPU는 동일 1.22.0의 `ort.webgpu.min.js` 번들을 사용하며, 실패하면 단일 스레드 WASM으로 복구합니다. 입력/출력 CPU 복사까지 포함해 측정합니다. COOP/COEP 없는 Pages에서 proxy Worker나 다중 WASM 스레드를 강제로 사용하지 않습니다.
- 기본 `auto`는 WebGPU가 노출된 브라우저에서 GPU를 우선 시도합니다. 2026-09-24 대상 브라우저의 같은 탐색량 cold 교차 측정에서 44~47% 단축되어 선택했습니다. 다른 GPU에서도 같은 성능을 보장하지 않으며 미지원·실패·장치 손실 시 WASM으로 복구합니다. 모델 세션은 착수마다 새로 만들지 않습니다.
- 신경망·규칙·탐색 후보·탐색 횟수는 유지합니다. FP16/양자화/Graph Capture/트리 재사용은 적용하지 않았습니다.

AI 계산은 방문자 기기에서 수행합니다. 기기 속도, 탐색량에 따라 생각 시간이 다릅니다. Python 데스크톱 버전과 같은 탐색을 쓰지만 부동소수점 차이로 아주 가까운 후보의 순서가 달라질 수 있고, 기력 동등성은 별도 대국 평가가 필요합니다.

흑은 중앙에서 시작하며, 이후 자유 착수합니다. 흑 금수를 차단하고, 둘 곳이 없을 때만 패스합니다. 대회용 교환 오프닝·시간제는 없습니다. 새로고침하면 현재 대국은 사라집니다. 이 배포 방식에서는 모델과 엔진이 공개됩니다.

## 모델 갱신

프로젝트 루트에서 실행합니다. 원본 모델과 학습 파일은 수정하지 않습니다.

```powershell
.\.venv\Scripts\python.exe -m pip install onnx onnxruntime
.\.venv\Scripts\python.exe renju_next\export_web.py
.\.venv\Scripts\python.exe renju_next\test_web.py
```

변환기는 원본과 ONNX 출력을 배치 1·4·16에서 비교한 후 `model.json`에 오차와 체크포인트 SHA-256을 기록하고 `web-assets.json`도 갱신합니다. 변경된 파일과 자산 목록을 함께 배포하세요. **웹 코드만 고칠 때는 export_web.py를 실행하지 않습니다.** `prepare_web.py`로 해시만 갱신하면 학습 가중치를 그대로 유지합니다.

## 브라우저 속도 진단

`diagnostics.html`에서 **전후 비교 시작**을 누르면 같은 고정 합법 기보와 32·128·512회 예산을 기존 WASM/개선 WASM/개선 WebGPU로 각각 측정하고 JSON을 내려받습니다. 기본은 조건마다 3회이며, p95는 표본이 적은 참고치입니다. 화면을 앞에 두고 다른 무거운 작업을 멈추면 비교하기 좋습니다. 중단 후에도 수집한 결과를 저장할 수 있습니다.

세 경로를 먼저 준비한 뒤 국면·예산마다 순서를 바꾸어 교차 측정합니다. 한 번에 한 경로만 탐색하며, 각 경로의 cold 직후 warm을 실행합니다. **일반 탐색 4개 · 교차 비교** 버튼은 실제 예산을 모두 사용하는 초반·중반 흑백 네 국면만 선택합니다. 전체 설정은 후반·즉시 전술·금수 유도까지 16국면을 포함합니다. 진단 중에는 세 세션을 유지하므로 대국 화면보다 메모리를 더 사용합니다.

- cold: 기보 준비 후 기존 규칙 캐시와 신경망 캐시를 모두 비운 탐색. warm: 그 직후 같은 국면을 캐시를 유지해 다시 탐색. 두 값을 합쳐 가속 배수를 만들지 않습니다.
- 실제 완료 rollout, 신경망 평가 수, 배치 분포, 규칙 캐시, 깊이, 전술 증명, 진행 메시지를 기록합니다. 같은 모델·완료 탐색 수·결과판·탐색 통계인 표본끼리 비교합니다. 전술로 즉시 끝나 실제 탐색이 0회인 경우는 일반 탐색과 분리합니다.
- Python 세부 시간은 중첩을 뺀 값입니다. `inference_wait` 안에 JavaScript/ORT 시간이 들어 있으므로 둘을 다시 더하지 않습니다. 로딩도 겹치는 구간은 합계 대신 `loading_ms.total`을 사용합니다.
- GPU 요청명만으로 실행을 단정하지 않습니다. 실제 어댑터 정보, 워밍업 GPU kernel 실행 증거와 fallback 사유를 남깁니다. 전체 연산자 배치의 CPU/GPU 분할은 확정하지 않으며 일부 Shape 연산은 GPU kernel이 없습니다.
- `/?diagnostics=1&mode=baseline&backend=wasm`와 `/?diagnostics=1&backend=webgpu`에서 실제 사람 클릭→AI 보드 갱신/두 RAF 시점도 기록하고 JSON으로 저장할 수 있습니다. 숨겨진 탭은 유효한 화면 표시 시간으로 취급하지 않습니다. 기보 자동 측정의 표시 시간과 별개입니다.
- 브리지는 Python bytearray의 작은 입력을 JS 소유 버퍼로 복사하고, 반환 Float32Array도 Python으로 복사합니다. zero-copy라고 주장하지 않습니다. borrowed 버퍼는 finally에서 해제하며 비동기 실행 중 입력 저장소를 덮어쓰지 않습니다.
- `acceptance.html`에서 중앙 첫 수 차단, 새 대국, 오래된 요청 차단, 실제 GPU 장치 손실 및 WASM 복구 후 판 보존·단일 착수를 검사하고 JSON을 저장할 수 있습니다. 속도 진단과 동시에 실행하지 마세요.

### 2026-09-24 실측

Windows / Chrome 153 계열 브라우저, NVIDIA Lovelace 어댑터, ORT Web 1.22.0, FP32 동일 모델. 정확한 GPU 제품명은 브라우저에서 공개하지 않았습니다. 초반·중반 흑백 4국면, 조건당 3회, 캐시를 비운 탐색이며 **아래 시간은 국면별 중앙값의 평균**입니다. 기준은 기존 JSON/WASM 경로에 동일한 계측을 추가한 버전입니다.

| 실제 완료 탐색 | 기존 WASM | 개선 WebGPU | 단축 |
|---:|---:|---:|---:|
| 32회 | 0.219초 | 0.117초 | 46.7% |
| 128회 | 0.811초 | 0.453초 | 44.2% |
| 512회 | 3.287초 | 1.835초 | 44.2% |

216회 교차 측정에서 전후 144쌍의 결과판·완료 탐색 수·탐색 통계가 일치했습니다. WebGPU 정책/가치 6,583개 값의 최대 절대 오차는 1.824e-5로 허용 기준 이내였습니다. GPU 커널 실행도 관측했지만 전체 연산의 CPU/GPU 배치를 확정한 것은 아닙니다.

주 병목은 신경망 추론이었습니다. 128회 cold의 평균 `session.run`은 687.7→319.0ms, Python 변환은 9.42→2.17ms였습니다(중첩 시간은 더하지 않음). WASM만의 cold 종합 변화는 -0.5%, -0.3%, +2.6%로 거의 같았습니다. 동일 국면의 warm 반복은 84~88% 빨랐지만 실제 대국 전체의 가속으로 해석하지 않습니다.

먼저 수행한 16국면 순차 측정 864회는 전후 576쌍 모두 일치했으나 시간대별 추론 지연이 크게 변해 성능 결론에서 분리했습니다. 위 교차 측정도 작은 표본이며 다른 기기·배경 부하에서는 달라집니다. 전체 대국을 통한 기력 동등성 평가는 하지 않았습니다. 초기 GPU 배치 준비는 로딩 시간으로 기록하고 착수 통계에서 제외합니다.

결과와 검증 스크립트는 로컬 프로젝트 `renju_next/verification/web-speed/`에 있습니다. `serve_web.py`의 선택적인 로컬 결과 저장 기능은 개발용이며, GitHub Pages에서는 방문자 결과를 서버에 전송하지 않습니다.

공식 문서: [GitHub Pages 게시 소스](https://docs.github.com/en/pages/getting-started-with-github-pages/configuring-a-publishing-source-for-your-github-pages-site), [ONNX Runtime Web](https://onnxruntime.ai/docs/get-started/with-javascript/web.html), [Pyodide](https://pyodide.org/en/stable/).

구현 참고: [ORT WebGPU](https://onnxruntime.ai/docs/tutorials/web/ep-webgpu.html), [ORT Worker·스레드·자산 버전](https://onnxruntime.ai/docs/tutorials/web/env-flags-and-session-options.html), [Pyodide 0.27.7 버퍼 수명과 변환](https://pyodide.org/en/0.27.7/usage/type-conversions.html).
