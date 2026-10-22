# Numbers model report (v2)

Frames: 4737 numbers frames (12 numbers (1-10, MIL, MILLON; 2 and 3 only from our captures), 73 signers, LSC70ANH crops + 1250 frames of our own captures) + 2000 none frames.
All numbers are on held-out signers (GroupKFold 5 by signer).

## Variants tried (macro accuracy over 12 numbers + none, frame level / sample level)

| variant | frame | sample |
|---|---|---|
| feats=xy | 0.809 | 0.852 |
| feats=xy+z | 0.799 | 0.827 |
| feats=xy+geo | 0.808 | 0.836 |
| feats=xy+z+geo3 | 0.791 | 0.824 |
| wider (256,128) a=1e-3 | 0.803 | 0.833 |
| aug x4 rot 25 | 0.794 | 0.828 |
| aug 3D viewpoint | 0.805 | 0.844 |
| +full-frame LSC70 | 0.798 | 0.827 |
| ensemble x5 | 0.810 | 0.843 |
| quick: chosen | 0.809 | 0.852 |

Chosen: `{"feats": "xy", "hidden": [128, 64], "alpha": 0.0001, "copies": 2, "rot": 15, "aug3d": false, "extra": false, "seeds": 1}`

## Per sign

Free = argmax over all classes. Target = the app knows the sign: frame passes if p(sign) >= its own threshold.
Shape spread = how much the handshape changes within one sample (high = the sign moves; a static model may not fit).

| sign | frame acc | sample acc | target threshold | target recall | worst false accept | shape spread |
|---|---|---|---|---|---|---|
| 1 | 0.863 | 0.870 | 0.94 | 0.352 | 6 0.036 | 0.140 |
| 10 | 0.888 | 0.957 | 0.50 | 0.888 | none 0.032 | 1.360 |
| 2 | 0.714 | 0.759 | 0.50 | 0.710 | 3 0.038 | 0.183 |
| 3 | 0.433 | 0.444 | 0.60 | 0.408 | 8 0.047 | 0.186 |
| 4 | 0.868 | 0.917 | 0.82 | 0.616 | 9 0.048 | 0.134 |
| 5 | 0.855 | 0.904 | 0.50 | 0.855 | none 0.021 | 0.102 |
| 6 | 0.765 | 0.843 | 0.83 | 0.601 | 1 0.049 | 0.253 |
| 7 | 0.935 | 0.957 | 0.77 | 0.916 | 2 0.048 | 0.297 |
| 8 | 0.892 | 1.000 | 0.95 | 0.662 | 3 0.126 | 0.350 |
| 9 | 0.783 | 0.886 | 0.61 | 0.732 | 4 0.049 | 0.397 |
| MIL | 0.851 | 0.875 | 0.50 | 0.851 | none 0.017 | 0.759 |
| MILLON | 0.800 | 0.810 | 0.50 | 0.800 | none 0.030 | 0.629 |
| none | 0.873 | | | | |

Macro: frame 0.809, sample 0.852.

## Most confused (true -> predicted, frame level)

- 3 -> 8: 0.353
- 6 -> 1: 0.213
- MILLON -> none: 0.200
- 5 -> none: 0.145
- 2 -> 7: 0.141
- MIL -> none: 0.131
- 1 -> 6: 0.123
- 9 -> 4: 0.121
- 10 -> none: 0.109
- 3 -> none: 0.105
- 2 -> none: 0.097
- 4 -> 9: 0.068

## Our captures (each person held out)

| sign | aris_p3 | gabriel_p1 | santiago_p2 | all | frames |
|---|---|---|---|---|---|
| 1 | 1.00 | 0.75 | 1.00 | 0.899 | 276 |
| 2 | 0.48 | 0.83 | 0.78 | 0.714 | 269 |
| 3 | 0.32 | 0.37 | 0.57 | 0.433 | 238 |
| 4 | 0.59 | 0.79 | 1.00 | 0.793 | 241 |
| 5 | 0.48 | 0.57 | 1.00 | 0.681 | 226 |

Our captures macro 0.704; LSC70 + none frames only, macro 0.865.

Free-mode threshold tau = 0.873 (<= 5% of none frames above it, floor 0.70). holdFrames = 8.
Target thresholds: for every other class, <= 5% of its frames reach the threshold (per frame, before the 8-frame hold).

Golden: 16 cases (4 mirrored); JSON forward pass vs sklearn max error 5.78e-07
