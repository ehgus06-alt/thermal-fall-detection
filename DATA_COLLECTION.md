# 내 환경 데이터 수집 프로토콜 (하드 네거티브 중심)

목표: **눕기·앉기를 낙상으로 오탐하는 문제**를 데이터로 해결한다.
`record_lepton.py`로 아래 시나리오를 녹화하면 `cache/`에 바로 들어가고, 재학습하면 모델이
"느린 눕기 ≠ 낙상"을 직접 배운다. (hand-crafted 속도 게이트보다 훨씬 확실함이 검증됨.)

## 준비
- 카메라 각도: 높이 1.5~2m, 아래로 ~30°, **바닥·착지지점이 프레임에** ([DEPLOY_LEPTON.md] 참고)
- 안전: 매트리스/이불 깔고. FLIR 앱은 닫기.

## 1) 하드 네거티브 (NonFall) — 이게 핵심, 많이!
헷갈리는 "천천히/통제된 하강"을 라벨 `nonfall`로:
```bash
python record_lepton.py --camera 0 --label nonfall --name lie_slow1     # 바닥에 천천히 눕기
python record_lepton.py --camera 0 --label nonfall --name sit_floor1    # 바닥에 앉았다 일어나기
python record_lepton.py --camera 0 --label nonfall --name bed_lie1      # 침대/소파에 눕기
python record_lepton.py --camera 0 --label nonfall --name pickup1       # 물건 주우려 숙이기
python record_lepton.py --camera 0 --label nonfall --name kneel1        # 무릎 꿇기
python record_lepton.py --camera 0 --label nonfall --name stretch1      # 바닥 스트레칭/요가
python record_lepton.py --camera 0 --label nonfall --name walk1         # 그냥 걷기·서성이기
```
> 각 시나리오 2~3번씩, 위치·방향 바꿔가며. **목표 15~20클립+**

## 2) 낙상 (Fall) — 다양하게
```bash
python record_lepton.py --camera 0 --label fall --name fall_fwd1    # 앞으로
python record_lepton.py --camera 0 --label fall --name fall_back1   # 뒤로
python record_lepton.py --camera 0 --label fall --name fall_side1   # 옆으로
python record_lepton.py --camera 0 --label fall --name fall_slip1   # 미끄러지듯
```
> 방향·속도·위치 바꿔 **15~20클립+**. 넘어진 뒤 몇 초 그대로 있기(실제 상황처럼).

## 3) 재학습 & 교체
```bash
python train.py --full --tag _final     # 내 클립 포함 전량 재학습
# runs/best_final.pt -> runs/best.pt 로 교체 후 테스트
python lepton_live.py --camera 0 --display
```

## 팁
- 오탐이 특정 동작(예: 소파에 눕기)에서 나면 → 그 동작을 **더 많이** 녹화해 재학습.
- 한 번에 완벽 안 됨. **녹화→재학습→테스트** 몇 바퀴 돌리면 내 방에 맞게 좋아진다.
- 클립은 최소 30프레임(~3초) 이상이어야 저장됨.
