# 렌주 AI 웹 대국

이 폴더 자체가 GitHub Pages에 배포할 완성된 정적 사이트입니다. Node 빌드나 Python 서버는 배포 후 필요하지 않습니다.

## 내 컴퓨터에서 실행

`index.html`을 더블클릭하면 브라우저 보안 제한 때문에 엔진이 로드되지 않습니다. 프로젝트 루트에서 다음 명령으로 실행한 뒤 http://127.0.0.1:8765 를 여세요.

```powershell
.\.venv\Scripts\python.exe -m http.server 8765 --bind 127.0.0.1 --directory renju_next\github-pages
```

## GitHub Pages 배포

1. GitHub에서 공개 저장소를 하나 만듭니다. 예: `renju-play`.
2. **이 폴더 안의 내용**을 저장소 최상위에 업로드합니다. `index.html`, `app.js`, `style.css`, `worker.js`, `best.onnx`, `model.json`, `engine/`를 모두 포함하세요. 가능하면 `.nojekyll`도 포함하세요.
3. 저장소 **Settings → Pages → Build and deployment**에서 **Deploy from a branch**, **main**, **/(root)**를 선택하고 저장합니다.
4. 배포가 완료되면 Pages 설정에 표시되는 공개 주소를 공유합니다. 일반적인 형태는 `https://사용자명.github.io/저장소명/`입니다.

상대 경로를 사용하므로 저장소 이름이 붙는 GitHub Pages 주소에서도 실행됩니다. 학습 데이터, 가상환경, 원본 `.pt` 체크포인트를 올릴 필요는 없습니다.

## 구성

- `best.onnx`: `models_next/best.pt`에서 내보낸 정책·가치 신경망.
- `engine/renju.py`, `patterns.py`, `threat_search.py`: 기존 소스 그대로 복사한 규칙·전술 엔진.
- `engine/core.py`: 기존 core의 신경망 정의 앞부분을 추출한 게임 어댑터.
- `engine/search.py`: 기존 탐색 코드. 학습용 정책 벡터를 만드는 `torch` 호출만 작은 리스트 어댑터로 대체했습니다. 탐색·후보 제거·전술 판정은 동일합니다.
- `worker.js`: Pyodide 0.27.7 및 ONNX Runtime Web 1.22.0 실행. 첫 접속에는 jsDelivr에서 실행 엔진을 받습니다. 따라서 모델 크기보다 초기 다운로드가 큽니다.
- ONNX Runtime은 일반 Pages에서 지원되는 단일 스레드 WASM CPU 모드를 사용합니다. WebGPU 가속은 이번 버전에 포함하지 않습니다.

AI 계산은 방문자 기기에서 수행합니다. 기기 속도, 탐색량에 따라 생각 시간이 다릅니다. Python 데스크톱 버전과 같은 탐색을 쓰지만 부동소수점 차이로 아주 가까운 후보의 순서가 달라질 수 있고, 기력 동등성은 별도 대국 평가가 필요합니다.

흑은 중앙에서 시작하며, 이후 자유 착수합니다. 흑 금수를 차단하고, 둘 곳이 없을 때만 패스합니다. 대회용 교환 오프닝·시간제는 없습니다. 새로고침하면 현재 대국은 사라집니다. 이 배포 방식에서는 모델과 엔진이 공개됩니다.

## 모델 갱신

프로젝트 루트에서 실행합니다. 원본 모델과 학습 파일은 수정하지 않습니다.

```powershell
.\.venv\Scripts\python.exe -m pip install onnx onnxruntime
.\.venv\Scripts\python.exe renju_next\export_web.py
.\.venv\Scripts\python.exe renju_next\test_web.py
```

변환기는 원본과 ONNX 출력을 배치 1·4·16에서 비교한 후 `model.json`에 오차와 체크포인트 SHA-256을 기록합니다. 갱신된 `best.onnx`, `model.json`, `engine/`를 함께 다시 업로드하세요.

공식 문서: [GitHub Pages 게시 소스](https://docs.github.com/en/pages/getting-started-with-github-pages/configuring-a-publishing-source-for-your-github-pages-site), [ONNX Runtime Web](https://onnxruntime.ai/docs/get-started/with-javascript/web.html), [Pyodide](https://pyodide.org/en/stable/).
