# Words: MLP report

Model: MLP on the variant-C word sequence (hand shape + shoulder/nose-relative position, both hands), classes = 8 MVP words + other (LSC50 impostors). Config: `{"T": 15, "hidden": [256, 128], "alpha": 0.001, "copies": 8, "rot": 15, "loc_shift": 0.2, "warp": true, "shape2": false, "seeds": 3, "swap": true}`.
All numbers are on held-out signers (same 5 signer folds as the DTW eval). Target mode: pass = p(word) >= its threshold; threshold = lowest value in [0.30, 0.95] where impostors pass as the word <= 10% and every other single MVP word passes as it <= 10%. Thresholds are fit on the same out-of-fold scores, so recall is slightly optimistic.

## Variant search

| trial | mean recall | words >= 80% | mean top-1 | kept |
|---|---|---|---|---|
| base | 0.823 | 6/8 | 0.832 | yes |

## Per word (chosen config)

| word | n | top-1 | recall @thr | impostor FA | worst other word (rate) | threshold | Gate 5 |
|---|---|---|---|---|---|---|---|
| GRACIAS | 46 | 0.70 | 0.67 | 0.048 | BUENAS_TARDES (0.08) | 0.33 | fail |
| BUENOS_DIAS | 74 | 0.86 | 0.86 | 0.038 | BUENAS_TARDES (0.05) | 0.3 | pass |
| BUENAS_TARDES | 75 | 0.76 | 0.81 | 0.053 | BUENAS_NOCHES (0.09) | 0.35 | pass |
| BUENAS_NOCHES | 77 | 0.79 | 0.68 | 0.000 | GRACIAS (0.09) | 0.48 | fail |
| BIEN | 74 | 0.89 | 0.89 | 0.082 | ADIOS (0.09) | 0.3 | pass |
| HOLA | 70 | 0.81 | 0.83 | 0.019 | ADIOS (0.09) | 0.36 | pass |
| ADIOS | 74 | 0.85 | 0.86 | 0.019 | HOLA (0.10) | 0.37 | pass |
| POR_FAVOR | 70 | 0.99 | 0.97 | 0.067 | BIEN (0.08) | 0.3 | pass |

## Cross-dataset (top-1, no reject)

- lsc50->lsc54: 0.50  GRACIAS 0.42, BUENOS_DIAS 0.81, BUENAS_TARDES 0.56, BUENAS_NOCHES 0.49, BIEN 0.81, HOLA 0.48, ADIOS 0.11, POR_FAVOR 0.24
- lsc54->lsc50: 0.58  GRACIAS 0.80, BUENOS_DIAS 0.65, BUENAS_TARDES 0.75, BUENAS_NOCHES 0.50, BIEN 0.55, HOLA 0.25, ADIOS 0.40, POR_FAVOR 0.75

DTW comparison (variant C, `ml/NOTES.md` phase 5): top-1 0.81, 0/8 words at 80% with impostor FA <= 10%, cross-dataset 0.56 / 0.58.

Gate 5 (6 of 8 needed): 6/8 pass -> PASS. Camera-graded: BUENOS_DIAS, BUENAS_TARDES, BIEN, HOLA, ADIOS, POR_FAVOR. The others stay video-only ("Mira y elige").

## Caveats
- Impostors are LSC50 only (5 people, 42 words); LSC-54 non-MVP signs have no pose.
- No team clips yet: the cross-dataset number is the closest to "new user, new phone".
- The app pipeline is the DTW one: 2.5 s window, idle trim, resample to 30, PoseLandmarker lite.
