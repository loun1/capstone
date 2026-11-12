# 캡스톤

LG M50T 기반 배터리 디지털 트윈으로 팩 데이터를 재현하고, 운전 습관과 충전 전류가 배터리에 주는 영향을 비교한다.

| 작업 | 코드 | 결과 |
| --- | --- | --- |
| SPMeT 디지털 트윈과 GP | [팩 재현](bayesian/pack_replay.py) · [전압 선별 전송](bayesian/gp_telemetry.py) · [펄스 보정](simulation/gp_pulse_calibration.py) · [리튬 손실 보정](simulation/gp_internal_state.py) | 전압·온도 재현, 전송량·복원 오차, 보정 후보의 별도 데이터 평가 |
| 운전자 습관 이차 회귀 | [전류 부담 예측](bayesian/habit_predictor.py) · [전류 파형 예측](bayesian/habit_waveform.py) · [수명 비교](bayesian/habit_life_benchmark.py) | 같은 가상 도로·혼잡도에서 습관별 전류 지표와 장기 모의 수명 차이 |
| 충전 전류 베이지안 최적화 | [CC–CV 평가](bayesian/charging.py) · [탐색 엔진](optimization/bayesian_charge.py) · [반복 검증](bayesian/charging_stability.py) | 시간·전압·전류 제약, 후보별 계산 용량 손실, 반복 실행 편차 |

## 설치

저장소 루트에서 Python 3.11 가상환경을 활성화한 후 설치한다. 각 예제의 `--output`에는 새 파일이나 폴더 경로를 지정한다.

```sh
python -m pip install -r requirements.txt
```

## 1. SPMeT 디지털 트윈과 GP

팀 ROM의 `raw_input.csv`와 `metadata.json`을 사용한다. CSV 열은 `vehicle_id,cycle_id,timestamp_s,voltage_v,current_a,module_01_temperature_c`이며 방전 전류가 양수다. [108S32P 설정](configs/team_pack_108s32p.json)은 팩 전압을 108로, 전류를 32로 나누어 대표 셀에 입력한다. `--ambient-temperature-c`에는 재현에 사용할 주변 온도를 지정한다.

```sh
python -m bayesian replay-input --csv data/raw_input.csv --metadata data/metadata.json --pack-config configs/team_pack_108s32p.json --cycle-id 1 --duration-s 300 --thermal-mode lumped --ambient-temperature-c 20 --cell-calibration m50-c-joint-v3 --output outputs/replay
python -m bayesian.gp_telemetry --replay outputs/replay/result.json --output outputs/telemetry
```

두 번째 명령은 전체·균등·GP 선별 전송을 비교한다. 시간·전류·온도·모델 예측 전압은 공통으로 보내고 관측 전압을 선별한다. 저장된 JSONL만으로 수신 측 전압을 복원한다. 검증 범위는 대표 전압 채널 하나의 오프라인 ROM 재생이다.

펄스 보정은 시간 `t`(s), 전류 `i`(A, 방전 양수), 전압 `v`(V), `temperature_C`, 부하 여부 `loaded`가 포함된 실측 GITT NPZ가 필요하다. `--parameter`에는 `contact-resistance` 또는 `exchange-current`를 지정한다. 리튬 손실 보정에는 기존 A–D셀 기준 성능시험(RPT) CSV와 동결 SEI 계산 JSON이 필요하다. `--capstone-root`는 `work/validation_proposal_20261103`와 `work/external_m50_benchmark`가 있는 외부 작업 폴더다. 실측 자료 출처는 [Kirkaldy LG M50T 데이터](https://doi.org/10.5281/zenodo.10637534)다.

```sh
python -m simulation.gp_pulse_calibration --observed data/gitt_observed.npz --parameter contact-resistance --output outputs/pulse
python -m simulation.gp_internal_state --capstone-root ../capstone-data --output outputs/internal-state
```

## 2. 운전자 습관 이차 회귀

가상 도로·혼잡도·거리·신호 대기 조건을 고정한다. 가속·제동 패턴을 바꾸어 최대 방전·회생 전류, 전류 제곱 누적값, 절대 전하량이 기준 운전과 얼마나 달라지는지 예측한다. 장기 비교는 생성 전류를 기준으로 GP와 이차 회귀가 모의 수명 차이를 얼마나 재현하는지 평가한다.

```sh
python -m bayesian predict-habit-burden --launch-ramp-s 10 --brake-ramp-s 11 --output outputs/habit_prediction.json
python -m bayesian compare-habit-stress --metrics-only --output outputs/habit_stress
```

장기 수명 비교는 별도 환경에서 NumPy 2.3.5와 [BLAST-Lite 1.1.0](https://github.com/NatLabRockies/BLAST-Lite/releases/tag/v1.1.0)을 사용한다.

```sh
python -m bayesian compare-habit-life-models --output outputs/habit_life_models
```

## 3. 충전 전류 베이지안 최적화

동일한 초기 조건과 목표 SOC에서 셀 단위 CC–CV 전류 후보를 평가한다. GP 기반 탐색은 충전 시간과 물리 제약을 만족하는 후보를 찾고 계산된 용량 손실을 비교한다. 탐색 결과와 상태 저널로 이어서 평가하거나 최선 후보를 반복 검증할 수 있다.

```sh
python -m bayesian charging --current-a 3.2 --output outputs/charging_baseline.json
python -m bayesian charging --budget 8 --output outputs/charging_search
python -m bayesian charging-stability --prior-state outputs/charging_search/charge_bo_state.json --prior-report outputs/charging_search/charging_search.json --budget 12 --output outputs/charging_stability
```

## 테스트

```sh
python -m unittest discover -s bayesian -t . -q
python -m unittest optimization.test_bayesian_charge simulation.test_gp_internal_state -q
```
