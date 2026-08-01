# 라즈베리파이(Raspberry Pi) 배포 가이드

FLIR Lepton + PureThermal 을 라즈베리파이에 물려 실시간 낙상감지를 돌리는 방법.

## 0. 권장 사양 (중요)
- **Raspberry Pi 5 권장** (Pi 4도 가능하나 느림). Pi 3/Zero는 torch가 무거워 비추천.
- **64-bit Raspberry Pi OS (Bookworm)** — 32-bit는 torch 공식 휠이 없어 사실상 불가.
- 여유 저장공간 4GB+ (torch가 큼), 가능하면 방열/전원 넉넉히.

## 1. 시스템 패키지
```bash
sudo apt update
sudo apt install -y python3-pip python3-venv python3-opencv libatlas-base-dev
sudo usermod -aG video $USER      # 카메라 접근 권한 (재로그인 필요)
```

## 2. 저장소 받기 + 파이썬 환경
```bash
git clone https://github.com/ehgus06-alt/thermal-fall-detection
cd thermal-fall-detection
python3 -m venv .venv --system-site-packages   # apt의 opencv 재사용
source .venv/bin/activate
pip install torch torchvision numpy pillow scikit-learn   # aarch64 CPU 휠 (다운로드 큼)
```
> torch 설치가 오래 걸립니다. 실패하면 `pip install torch --index-url https://download.pytorch.org/whl/cpu` 시도.

## 3. Lepton 연결 확인
PureThermal을 USB에 꽂고:
```bash
ls /dev/video*          # video0 등 생기는지
python list_cameras.py  # 160x120(또는 160x122)로 뜨는 인덱스 = Lepton
v4l2-ctl --list-formats-ext -d /dev/video0   # (선택) 지원 포맷 확인
```
> Linux에선 Windows 때와 달리 V4L2로 대체로 잘 잡힙니다.

## 4. 실행
```bash
# 화면 없는(헤드리스) SSH 환경 → --display 빼고 콘솔 경보만
python lepton_live.py --camera 0

# 데스크톱/모니터 연결 시
python lepton_live.py --camera 0 --display
```
- 낙상 시 콘솔 `>>> FALL ALARM <<<` + `alarms/` 폴더에 스냅샷 저장.
- Lepton이 RGB가 아니라 16bit 원시로 잡히면 `--y16` 추가.

## 5. 실시간 성능 튜닝 (Pi는 CPU라 느림)
| 옵션 | 효과 |
|---|---|
| `--stride 2` (또는 3) | 프레임 N개마다 1번만 추론 → CPU 부담↓ (권장) |
| `--thr 0.5 --persist 3` | 오탐/민감도 보정 |

Pi 5에선 stride 1~2로 실시간 가능, Pi 4는 stride 2~3 권장.

## 6. 부팅 시 자동 실행 (선택, systemd)
`/etc/systemd/system/fall.service`:
```ini
[Unit]
Description=Thermal Fall Detection
After=multi-user.target
[Service]
User=pi
WorkingDirectory=/home/pi/thermal-fall-detection
ExecStart=/home/pi/thermal-fall-detection/.venv/bin/python lepton_live.py --camera 0 --stride 2
Restart=always
[Install]
WantedBy=multi-user.target
```
```bash
sudo systemctl enable --now fall.service
```

## 참고
- 카메라 없이 코드만 테스트하려면 데이터(캐시)가 필요 → Pi에선 보통 라이브 카메라로 바로 테스트.
- 모델(`runs/best.pt`)은 저장소에 포함돼 있어 학습 없이 바로 추론 가능.
