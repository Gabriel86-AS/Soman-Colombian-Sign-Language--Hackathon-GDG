# Words: MLP report

Model: MLP on the variant-C word sequence (hand shape + shoulder/nose-relative position of both hands, no second-hand shape), classes = 8 MVP words + other (LSC50 impostors). Final config: `{"T": 15, "hidden": [256, 128], "alpha": 0.001, "copies": 8, "rot": 15, "loc_shift": 0.2, "warp": true, "shape2": false, "seeds": 3}`. Layout per step: shape 40 | locB 2 | locN 2 | locB2 2 = 46 dims x 15 = 690 inputs, 3 members, 6.0 MB JSON.
Data: 768 clips (560 MVP, 208 LSC50 impostors), 23 signers.
All numbers are on held-out signers (same 5 signer folds as the DTW eval). Target mode: pass = p(word) >= its threshold; threshold = lowest value in [0.50, 0.95] where impostors pass as the word <= 10% and every other single MVP word passes as it <= 10%. Thresholds are fit on the same out-of-fold scores, so recall is slightly optimistic.

## Variant search, round 1 (from base; kept on n_pass, then mean recall)

| trial | mean recall | words >= 80% | mean top-1 | kept |
|---|---|---|---|---|
| base (T30, aug x4, warp, shape2) | 0.716 | 2/8 | 0.799 | yes |
| no_aug | 0.664 | 1/8 | 0.774 | |
| no_warp | 0.706 | 3/8 | 0.800 | yes (noise: HOLA 0.79 -> 0.80) |
| T15 | 0.718 | 2/8 | 0.803 | |
| no_shape2 | 0.736 | 3/8 | 0.789 | yes |
| copies8 | 0.742 | 3/8 | 0.796 | yes |
| loc_shift0.2 | 0.752 | 4/8 | 0.793 | yes |
| small (128,64) | 0.727 | 4/8 | 0.783 | |
| alpha1e-2 | 0.738 | 3/8 | 0.794 | |
| ensemble5 | 0.739 | 4/8 | 0.795 | |

## Variant search, round 2 (from round 1's choice; kept on mean recall +0.005, T15 if no loss)

| trial | mean recall | words >= 80% | mean top-1 | floor 0.3: mean / pass | kept |
|---|---|---|---|---|---|
| base | 0.752 | 4/8 | 0.793 | 0.779 / 4 | yes |
| vel (location deltas) | 0.747 | 3/8 | 0.796 | 0.763 / 3 | |
| shape2_tips | 0.747 | 4/8 | 0.808 | 0.780 / 4 | |
| balance (oversample) | 0.753 | 4/8 | 0.806 | 0.769 / 4 | |
| loc_shift0.3 | 0.731 | 3/8 | 0.779 | 0.770 / 4 | |
| warp | 0.762 | 4/8 | 0.812 | 0.779 / 4 | yes |
| T15 | 0.770 | 4/8 | 0.812 | 0.791 / 4 | yes |

Floor 0.3 = lowering the minimum p(word) from 0.5 to 0.3 (same 10% limits). It adds about 2-3 points of mean recall but no extra word passes, so the export keeps 0.5.

## Per word (final)

| word | n | top-1 | recall @thr | impostor FA | worst other word (rate) | threshold | Gate 5 |
|---|---|---|---|---|---|---|---|
| GRACIAS | 46 | 0.67 | 0.54 | 0.034 | BUENAS_TARDES (0.04) | 0.5 | fail |
| BUENOS_DIAS | 74 | 0.80 | 0.77 | 0.029 | BUENAS_NOCHES (0.08) | 0.5 | fail |
| BUENAS_TARDES | 75 | 0.73 | 0.69 | 0.019 | BUENAS_NOCHES (0.05) | 0.5 | fail |
| BUENAS_NOCHES | 77 | 0.78 | 0.77 | 0.005 | BUENOS_DIAS (0.08) | 0.5 | fail |
| BIEN | 74 | 0.85 | 0.84 | 0.043 | HOLA (0.03) | 0.5 | pass |
| HOLA | 70 | 0.87 | 0.83 | 0.024 | ADIOS (0.08) | 0.5 | pass |
| ADIOS | 74 | 0.88 | 0.84 | 0.024 | BUENOS_DIAS (0.04) | 0.5 | pass |
| POR_FAVOR | 70 | 0.91 | 0.89 | 0.024 | BIEN (0.07) | 0.5 | pass |

## Cross-dataset (top-1, no reject)

- lsc50->lsc54: 0.61  GRACIAS 0.46, BUENOS_DIAS 0.76, BUENAS_TARDES 0.44, BUENAS_NOCHES 0.30, BIEN 0.85, HOLA 0.74, ADIOS 0.48, POR_FAVOR 0.80
- lsc54->lsc50: 0.59  GRACIAS 0.80, BUENOS_DIAS 0.40, BUENAS_TARDES 0.60, BUENAS_NOCHES 0.40, BIEN 0.90, HOLA 0.40, ADIOS 0.45, POR_FAVOR 0.75

## Versus DTW (variant C, `ml/NOTES.md` phase 5)

| | DTW | MLP |
|---|---|---|
| words at >= 80% | 0/8 | 4/8 |
| mean recall at the reject threshold | 0.63 | 0.77 |
| top-1 (no reject) | 0.81 | 0.81 |
| impostor false-accept | 9.6% | 0.5-4.3% per word |
| cross-dataset top-1 | 0.56 / 0.58 | 0.61 / 0.59 |

Gate 5 (6 of 8 needed): 4/8 pass -> FAIL. Camera-graded: BIEN, HOLA, ADIOS, POR_FAVOR. The others stay video-only ("Mira y elige") until team clips are added.

## Caveats
- The DTW recall used a global impostor rule; the MLP uses per-word target mode with an extra "other words <= 10%" limit. Top-1 is the like-for-like number (equal).
- Impostors are LSC50 only (5 people, 42 words); LSC-54 non-MVP signs have no pose.
- No team clips yet: the cross-dataset number is the closest to "new user, new phone".
- App pipeline = the DTW one (2.5 s window, idle trim, resample to 30, PoseLandmarker lite), then each block resampled linearly to 15 steps.
