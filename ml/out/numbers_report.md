# Numbers model report (v2)

Frames: 3487 numbers frames (10 numbers (1, 4-10, MIL, MILLON), 70 signers, LSC70ANH crops) + 2000 none frames.
All numbers are on held-out signers (GroupKFold 5 by signer).

## Variants tried (macro accuracy over 10 numbers + none, frame level / sample level)

| variant | frame | sample |
|---|---|---|
| feats=xy | 0.861 | 0.886 |
| feats=xy+z | 0.854 | 0.888 |
| feats=xy+geo | 0.847 | 0.870 |
| feats=xy+z+geo3 | 0.844 | 0.876 |
| wider (256,128) a=1e-3 | 0.854 | 0.881 |
| aug x4 rot 25 | 0.858 | 0.896 |
| aug 3D viewpoint | 0.868 | 0.908 |
| +full-frame LSC70 | 0.869 | 0.910 |
| ensemble x5 | 0.872 | 0.911 |

Chosen: `{"feats": "xy", "hidden": [128, 64], "alpha": 0.0001, "copies": 2, "rot": 15, "aug3d": true, "extra": false, "seeds": 1}`

## Per sign

Free = argmax over all classes. Target = the app knows the sign: frame passes if p(sign) >= its own threshold.
Shape spread = how much the handshape changes within one sample (high = the sign moves; a static model may not fit).

| sign | frame acc | sample acc | target threshold | target recall | worst false accept | shape spread |
|---|---|---|---|---|---|---|
| 1 | 0.768 | 0.812 | 0.85 | 0.352 | 6 0.049 | 0.178 |
| 10 | 0.871 | 0.943 | 0.50 | 0.871 | none 0.029 | 1.360 |
| 4 | 0.943 | 0.971 | 0.83 | 0.663 | 9 0.048 | 0.147 |
| 5 | 0.949 | 0.957 | 0.50 | 0.949 | none 0.019 | 0.111 |
| 6 | 0.801 | 0.871 | 0.81 | 0.645 | 1 0.048 | 0.253 |
| 7 | 0.962 | 0.971 | 0.50 | 0.962 | 6 0.014 | 0.297 |
| 8 | 0.929 | 0.971 | 0.50 | 0.926 | 9 0.040 | 0.350 |
| 9 | 0.783 | 0.886 | 0.50 | 0.777 | 4 0.045 | 0.397 |
| MIL | 0.845 | 0.875 | 0.50 | 0.845 | none 0.019 | 0.759 |
| MILLON | 0.810 | 0.825 | 0.50 | 0.807 | none 0.020 | 0.629 |
| none | 0.884 | | | | |

Macro: frame 0.868, sample 0.908.

## Most confused (true -> predicted, frame level)

- MILLON -> none: 0.190
- 6 -> 1: 0.172
- 1 -> 6: 0.154
- MIL -> none: 0.143
- 9 -> 4: 0.137
- 10 -> none: 0.123
- 1 -> none: 0.075
- 5 -> none: 0.051
- 4 -> 9: 0.048
- 8 -> 9: 0.045
- 9 -> 8: 0.043
- 9 -> none: 0.024

Free-mode threshold tau = 0.860 (<= 5% of none frames above it, floor 0.70). holdFrames = 8.
Target thresholds: for every other class, <= 5% of its frames reach the threshold (per frame, before the 8-frame hold).

Golden: 14 cases (4 mirrored); JSON forward pass vs sklearn max error 3.20e-06
