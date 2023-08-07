# lv2-RecSys-5 '스나이퍼'

## Movie Recommendation

**프로젝트 개요**  

```
사용자의 영화 시청 이력 데이터를 바탕으로 사용자가 다음에 시청할 영화 및 좋아할 영화를 예측합니다.
```
<br>

**고려할 것**

1. 시간 정보의 활용 - 순차적인 사용자 행동
2. 누락된 아이템 예측
3. 부가 정보 활용

<br>

**프로젝트 수행방법**

1. EDA & Feature Engineering
2. 모델 탐구
3. 모델 선정
4. 결과 개선 및 분석

<br>

**데이터**

```
train
├── Ml_item2attributes.json
├── directors.tsv
├── genres.tsv
├── titles.tsv
├── train_ratings.csv
├── writers.tsv
└── years.tsv
```

| data | 소개 | 행 | 열 |
| :-------: | :-----------: | :-----------: | :-----------: |
| `train_ratings.csv` | 주 학습 데이터 | 5,154,471 | `userid`, `itemid`, `timestamp` |
| `Ml_item2attributes.json` | item과 genre의 mapping 데이터 | | |
| `titles.tsv` | 영화 제목 | 6,807 | |
| `years.tsv` | 영화 개봉년도 | 6,799 | |
| `directors.tsv` | 영화별 감독 | 5,905 | |
| `genres.tsv` | 영화 장르 (한 영화에 여러 장르 가능) | 15,934 | |
| `writers.tsv` | 영화 작가 | 11,307 | |

<br>

## EDA & Feature Engineering

강의에서 설명한 정도로만 진행하였다.

- interaction이 많은 상위 4명 유저를 봤을 때, 딱히 연도와의 관련성을 찾을 수 없음
- 2008년 데이터가 제일 많고, 2009년 이후 감소하는 추세

<br>


## 모델
RecBole을 적극 활용하여 모델들을 구현하고 선정하였다.

### 1. 모델 탐구
데이터셋의 전체 또는 일부(1%)만 이용하여 모델들을 테스트하고, 성능 위주로 더 탐구해 볼만한 후보 모델들을 정했다.

- General Recommendation : BPR_MF, LightGCN, EASE, FISM, VAE, MultiVAE, MultiDAE, RecVAE, VASP
- Sequential Recommendation : GRU4Rec, GRU4RecF, RepeatNet, BERT4REC

비교적 성능이 좋았던 모델은 다음과 같다.

- MultiDAE, VAE(0.032, 0.033), FISM(0.034), GRU4Rec(0.0133), RepeatNet(0.1571)

<br>

### 2. 모델 선정 및 결과

- 가장 성능이 좋은 모델들을 골랐다.
  ```
  RecBole을 이용한 GRU4Rec, GRU4RecF, EASE, RecVAE
  ```
- 결과 앙상블

<br>

## 결론

- Cross Validation, 하이퍼파라미터 조절, 앙상블
  - 모델마다 seed를 수정하며 train, valid를 다르게 구성해 k-fold 검증을 수행
  - MultiVAE, MultiDAE의 결과값을 기존에 봤던 영화 제외하고 상위 30개씩 추출해서 영화별로 점수들을 더해 평균을 구하기
  - Recall@10 값이 단일 모델로 돌렸을 때(0.1358~0.1305)보다 0.1485까지 약 10%의 성능 향상을 보임

- Sequential 모델의 앙상블
  - MF, RecVAE, EASE, BERT4Rec의 결과를 시청 이력 top 100개의 영화로 비교
    - MF, RecVAE, EASE의 경우 평균 40%~50%, BERT4Rec은 평균 10% 정도가 인기 영화로 채워짐
    - Sequential한 모델들을 앙상블하는 것이 필요하다 판단함
  - 성능이 잘 나온 순서대로 RecVAE, EASE, BERT4Rec을 앙상블하니, Recall@10 값에서 0.04 정도의 성능 향상을 보임


<br>

## 최종 제출
```
RecBole을 이용한 GRU4Rec, GRU4RecF, EASE, RecVAE
```
- Recall@10 : 0.1714

<br>
