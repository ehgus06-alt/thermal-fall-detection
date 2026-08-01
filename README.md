# Thermal Fall Detection (열화상 낙상감지)

FLIR **Lepton** 열화상 카메라로 **실시간 낙상**을 감지하는 딥러닝 프로젝트.
짧은 시간창의 움직임을 **MHI(Motion History Image)** 로 요약해, ImageNet 프리트레인
**MobileNetV3-small** 을 파인튜닝하고, **MIL(multiple-instance)** 로 클립 라벨만으로
낙상 구간을 학습한다. 추론은 슬라이딩 윈도우 + EMA + 지속성 필터로 실시간 알람을 낸다.

> 열화상은 **어둠·프라이버시**에 강해 노인 돌봄/화장실·침실 낙상감지에 적합하다.

## 특징
- **실시간** — RTX GPU에서 100+ win/s, Lepton 9fps 대비 여유 (CPU도 가능)
- **데이터 효율** — MHI가 움직임을 미리 주입 → 적은 데이터로 학습
- **라벨 견고** — 프레임 라벨 불필요, 클립 라벨 + MIL로 낙상 순간 자동 탐지
- **멀티 데이터셋** — 서로 다른 열화상 데이터셋(다른 방·카메라·팔레트)을 소스 태그로 통합, 교차검증 지원
- **Lepton 도메인 맞춤** — 학습 시 그레이스케일·정규화·다운스케일(도메인 갭 완화)
- **바로 배포** — 오탐 억제(persistence)·쿨다운·낙상 스냅샷 저장 내장

## 결과 (2개 열화상 데이터셋, 391 클립)
| 평가 | AP | F1 | 비고 |
|---|----|----|----|
| in-domain (같은 데이터셋) | 0.985 | 0.95 | 낙관적 (같은 환경) |
| **교차검증** (A 학습→B 테스트) | **~0.92** | 0.90 | **진짜 일반화 실력** |
| 합본 (2개 섞어 학습) | 0.974 | 0.93 | 다양성 모델 |
| **최종 배포** (`--full`) | — | — | **전량 100% 학습** |

> 교차검증으로 확인: 전혀 다른 데이터셋에서도 **AP ~0.92** → 배경이 아닌 실제 "낙상 움직임"을 학습.
> 새 환경 현실 기대치는 **AP 0.85~0.92**.

## 동작 원리
```
프레임 시퀀스 ─► [Lepton 화질화: 그레이·퍼센타일정규화·128²]
              ─► [짧은 윈도우 → 3채널 요약: 현재프레임 / MHI / 모션에너지]
              ─► [MobileNetV3-small] ─► 윈도우 낙상확률
              ─► MIL max-pool(학습) / 슬라이딩+EMA+persistence(추론) ─► 낙상 알람
```

## 구조
```
cache_frames.py   1) 원본 프레임 → Lepton 화질 캐시(.npy)
ingest_khan.py    1') zip으로 압축된 다른 데이터셋 통합 (손상·중복·잡파일 자동 제외)
dataset.py        MIL 백 데이터셋 + MHI 윈도우 생성 (멀티소스 매니페스트 병합)
train.py          2) MobileNetV3 + MIL 학습 → runs/best.pt
                     --train_source/--eval_source (교차검증), --full (전량 배포모델)
eval_stream.py    3) 스트리밍(배포) 방식 검증
infer.py          오프라인 추론(캐시 클립, 카메라 불필요)
list_cameras.py   Lepton UVC 인덱스 찾기
capture_test.py   카메라 프레임 수신 헤드리스 검증
lepton_live.py    4) 실시간 Lepton 낙상감지 (메인 배포 스크립트)
runs/best.pt      학습된 최종 배포 모델 (동봉)
```

## 데이터셋은 포함되지 않음
원본 열화상 클립과 캐시는 용량/라이선스 문제로 **저장소에 없다**(`.gitignore`).
자신의 데이터를 `Fall/FallN/frame*.jpg`, `NonFall/NonFallN/frame*.jpg` 구조로 두고
`python cache_frames.py` 로 캐시를 생성하면 된다. zip 형태의 다른 데이터셋은
`ingest_khan.py` 방식으로 흡수할 수 있다.

## 설치 & 사용
```bash
pip install -r requirements.txt
# GPU torch는 https://pytorch.org 에서 CUDA에 맞춰 설치 권장

python cache_frames.py      # 1) 전처리 캐시 (기본 데이터셋)
python ingest_khan.py       # 1') 추가 zip 데이터셋 통합 (선택)
python train.py --full --tag _final   # 2) 전량 학습 → runs/best_final.pt
python eval_stream.py       # 3) 검증
python infer.py --clip Fall/Fall29    # 오프라인 데모(카메라 불필요)

# 교차검증 (진짜 일반화 실력 측정)
python train.py --train_source flir --eval_source khan --tag _cross
```

### 실시간 (FLIR Lepton + PureThermal)
```bash
python list_cameras.py                     # Lepton 인덱스 확인
python lepton_live.py --camera 0 --display # 실시간 낙상감지
#  튜닝: --thr 0.47 --persist 3 --ema 0.4
```
낙상 시 콘솔 경보 + `alarms/` 에 스냅샷 저장. 자세한 하드웨어 가이드는
[`DEPLOY_LEPTON.md`](DEPLOY_LEPTON.md).

## 알려진 한계
1. **데이터 누수** — participant-safe 분할이 아님 → in-domain 지표는 낙관적일 수 있음(교차검증으로 보정).
2. **도메인 갭** — 실제 Lepton에서 임계 재보정 권장.
3. **하드 네거티브** — 눕기/앉기 등 일부 정상활동을 오탐할 수 있음(데이터 추가·재학습으로 개선).

## 카메라 설치 팁
높이 2~2.5m, 아래로 30~45° 기울여 **바닥이 보이게**, 전신이 프레임에, 열원(히터·햇빛)은 화면 밖.

## License
MIT — [LICENSE](LICENSE). 모델 가중치는 원본 데이터셋 라이선스의 영향을 받을 수 있으니
공개 전 데이터 출처의 재배포 조건을 확인할 것.
