# Moving letters: MLP report

Model: MLP on the hand-crop shape sequence (12 x 40 = 480 dims), classes = G H J N Ñ S Z + static. Config: `{"hidden": [256, 128], "alpha": 0.001, "copies": 4, "rot": 15, "aug3d": true, "warp": true, "seeds": 3}`.
Data: LSC70ANH crops, 6 frames per clip (first second, 6 fps), 70 signers; static = the 20 static letters' clips.
All numbers are on held-out signers (GroupKFold 5). Target mode: pass = p(target) >= its threshold; threshold = lowest value (>= 0.5) at which every other single letter passes as the target <= 5%.
Gate 3: recall >= 80% and the contact sheet (`qc/sheets/moving_<X>.png`) shows the motion in the first second.

## Variant search (mean recall over the 7 letters)

| trial | mean recall | kept |
|---|---|---|
| base | 0.797 | yes |
| aug3d | 0.814 | yes |
| warp | 0.838 | yes |
| copies8 | 0.833 |  |
| rot25 | 0.838 |  |
| small | 0.819 |  |
| alpha1e-2 | 0.832 |  |
| ensemble5 | 0.820 |  |

## Per letter (chosen config)

| letter | n | recall | worst other letter passing (rate) | FA static (avg) | threshold p | Gate 3 |
|---|---|---|---|---|---|---|
| G | 70 | 0.89 | X (0.03) | 0.002 | 0.5 | pass |
| H | 61 | 0.67 | Q (0.02) | 0.001 | 0.5 | fail |
| J | 67 | 0.88 | A (0.00) | 0.000 | 0.5 | pass |
| N | 69 | 0.96 | P (0.01) | 0.001 | 0.5 | pass |
| Ñ | 63 | 0.90 | P (0.01) | 0.001 | 0.5 | pass |
| S | 68 | 0.76 | X (0.01) | 0.001 | 0.6059 | fail |
| Z | 60 | 0.80 | E (0.02) | 0.001 | 0.5201 | pass |

DTW comparison (same rule): crop shape DTW 0.36 mean recall, with static reject templates 0.65, 0/7 pass (see `ml/NOTES.md`, phase 6).

Ship (camera-graded): G, J, N, Ñ, Z. The others stay video-only.

## Caveats
- LSC70 covers only the first second at 6 fps: the app must use the same window (about 1 s after the countdown, hand crop, resample to 12 by time).
- No team clips yet: this is the dataset number, not the new-user/new-camera number.
- `worst other letter` covers static letters only; moving-vs-moving confusions are inside the worst5 rule too.
