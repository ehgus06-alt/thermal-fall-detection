 Thermal Fall Detection (열화상 낙상감지)

FLIR **Lepton** 열화상 카메라로 **실시간 낙상**을 감지하는 딥러닝 프로젝트.
짧은 시간창의 움직임을 **MHI(Motion History Image)** 로 요약해, ImageNet 프리트레인
**MobileNetV3-small** 을 파인튜닝하고, **MIL(multiple-instance)** 로 클립 라벨만으로
낙상 구간을 학습한다. 추론은 슬라이딩 윈도우 + EMA + 지속성 필터로 실시간 알람을 낸다.

> 열화상은 **어둠·프라이버시**에 강해 노인 돌봄/화장실·침실 낙상감지에 적합하다.

 특징
**실시간** — RTX GPU에서 100+ win/s, Lepton 9fps 대비 여유 (CPU도 가능)
**데이터 효율** — MHI가 움직임을 미리 주입 → 적은 데이터로 학습(클립 348개)
 **라벨 견고** — 프레임 라벨 불필요, 클립 라벨 + MIL로 낙상 순간 자동 탐지
**Lepton 도메인 맞춤** — 학습 시 160×120 그레이스케일로 다운스케일(도메인 갭 완화)
 **바로 배포** — 오탐 억제(persistence)·쿨다운·낙상 스냅샷 저장 내장

## 결과 (검증 70클립, 배포와 동일한 스트리밍 방식)
| AP | F1 | 정밀도 | 재현율 |
|----|----|-------|-------|
| 0.985 | 0.952 | 0.943 | 0.962 |

>  현재 분할은 participant-safe가 아니라 **낙관적일 수 있음**(아래 한계 참고).

## 🧩 동작 원리
```
프레임 시퀀스 ─► [Lepton 화질화: 그레이·퍼센타일정규화·128²]
              ─► [짧은 윈도우 → 3채널 요약: 현재프레임 / MHI / 모션에너지]
              ─► [MobileNetV3-small] ─► 윈도우 낙상확률
              ─► MIL max-pool(학습) / 슬라이딩+EMA+persistence(추론) ─► 낙상 알람
```

##  구조
```
cache_frames.py   1) 원본 프레임 → Lepton 화질 캐시(.npy)
dataset.py        MIL 백 데이터셋 + MHI 윈도우 생성
train.py          2) MobileNetV3 + MIL 학습 → runs/best.pt
eval_stream.py    3) 스트리밍(배포) 방식 검증
infer.py          오프라인 추론(캐시 클립, 카메라 불필요)
list_cameras.py   Lepton UVC 인덱스 찾기
capture_test.py   카메라 프레임 수신 헤드리스 검증
lepton_live.py    4) 실시간 Lepton 낙상감지 (메인 배포 스크립트)
runs/best.pt      학습된 모델 (동봉)
```

##  데이터셋은 포함되지 않음
원본 열화상 클립(147k 프레임)과 2.3GB 캐시는 용량/라이선스 문제로 **저장소에 없다**(`.gitignore`).
자신의 데이터를 `Fall/FallN/frame*.jpg`, `NonFall/NonFallN/frame*.jpg` 구조로 두고
`python cache_frames.py` 로 캐시를 생성하면 된다.

##  설치 & 사용
```bash
pip install -r requirements.txt
# GPU torch는 https://pytorch.org 에서 CUDA에 맞춰 설치 권장

python cache_frames.py      # 1) 전처리 캐시
python train.py --epochs 25 # 2) 학습
python eval_stream.py       # 3) 검증
python infer.py --clip Fall/Fall29   # 오프라인 데모(카메라 불필요)
```

### 실시간 (FLIR Lepton + PureThermal)
```bash
python list_cameras.py                     # Lepton 인덱스 확인
python lepton_live.py --camera 0 --display # 실시간 낙상감지
#  튜닝: --thr 0.47 --persist 3 --ema 0.4
```
낙상 시 콘솔 경보 + `alarms/` 에 스냅샷 저장. 자세한 하드웨어 가이드는
[`DEPLOY_LEPTON.md`](DEPLOY_LEPTON.md).

##  알려진 한계
1. **데이터 누수** — participant-safe 분할이 아님 → 지표가 낙관적일 수 있음.
2. **도메인 갭** — 학습 데이터는 고해상도 FLIR One 계열. 실제 Lepton에서 임계 재보정 권장.
3. **하드 네거티브** — 눕기/앉기 등 일부 정상활동을 오탐할 수 있음(재학습으로 개선).

##  카메라 설치 팁
높이 2~2.5m, 아래로 30~45° 기울여 **바닥이 보이게**, 전신이 프레임에, 열원(히터·햇빛)은 화면 밖.

## License
MIT — [LICENSE](LICENSE). 모델 가중치는 원본 데이터셋 라이선스의 영향을 받을 수 있으니
공개 전 데이터 출처의 재배포 조건을 확인할 것.
