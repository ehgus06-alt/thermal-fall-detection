# FLIR Lepton 실시간 낙상감지 배포 가이드

카메라를 꽂는 순간 바로 쓸 수 있게 준비 완료됨. `runs/best.pt` 모델 사용.

## 1. 하드웨어
- **FLIR Lepton 3.5** (160×120) + **PureThermal 2/3** 브레이크아웃 보드 (USB-UVC).
- USB로 PC에 연결하면 Windows가 표준 UVC 웹캠으로 인식 (별도 드라이버 대개 불필요).

## 2. 카메라 인덱스 찾기
```
python list_cameras.py
```
`160x120` 또는 `120x160`으로 뜨는 인덱스가 Lepton. (예: index 1)

## 3. 실시간 실행
```
# 컬러화(RGB) 출력 Lepton (기본)
python lepton_live.py --camera 1 --display

# 라디오메트릭 16bit 출력 Lepton
python lepton_live.py --camera 1 --y16 --display
```
- 낙상 감지 시 콘솔에 `>>> FALL ALARM <<<` + `alarms/` 폴더에 스냅샷 저장.
- 처리속도 100+ 윈도우/초 (RTX 5080) → Lepton 9fps 대비 여유. CPU만으로도 실시간 가능.

## 4. 튜닝 노브
| 옵션 | 기본 | 설명 |
|---|---|---|
| `--thr` | 0.47 | 알람 EMA 임계 (↑ 오탐↓/놓침↑) |
| `--ema` | 0.4 | 평활 계수 (↑ 반응빠름/노이즈↑) |
| `--persist` | 3 | 연속 N윈도우 초과해야 알람 (순간 오탐 억제) |
| `--cooldown` | 5.0 | 알람 간 최소 간격(초) |

## 5. 카메라 없이 로직 검증
```
python lepton_live.py --simclip Fall/Fall29      # 낙상 → 알람
python lepton_live.py --simclip NonFall/NonFall64 # 정상 → 무알람
```

## ⚠️ 실배포 전 유의
1. **도메인 갭**: 이 모델은 고해상도 FLIR One 계열로 학습됨. 실제 Lepton에서 첫 구동 시
   `--simclip` 대신 실제 카메라로 몇 번 넘어져보며 `--thr`/`--persist` 재보정 권장.
2. **하드 네거티브**: 바닥에 눕기/앉기 등 일부 정상활동을 낙상으로 오탐할 수 있음
   (예: NonFall1). 실환경 정상영상을 모아 재학습(하드네거티브 마이닝)하면 개선됨.
3. **누수 미검증**: 현재 지표는 participant-safe 분할이 아니라 낙관적일 수 있음.
