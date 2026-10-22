# Numbers model report (v2)

Frames: 2249 numbers frames (5 numbers (1, 2, 3, 4, 5), 73 signers, LSC70ANH crops + 1250 frames of our own captures) + 2000 none frames.
All numbers are on held-out signers (GroupKFold 5 by signer).

## Variants tried (macro accuracy over 5 numbers + none, frame level / sample level)

| variant | frame | sample |
|---|---|---|
| feats=xy | 0.883 | 0.880 |
| feats=xy+z | 0.876 | 0.874 |
| feats=xy+geo | 0.880 | 0.877 |
| feats=xy+z+geo3 | 0.876 | 0.879 |
| wider (256,128) a=1e-3 | 0.875 | 0.869 |
| aug x4 rot 25 | 0.904 | 0.910 |
| aug 3D viewpoint | 0.887 | 0.879 |
| +full-frame LSC70 | 0.901 | 0.907 |
| ensemble x5 | 0.902 | 0.905 |

Chosen: `{"feats": "xy", "hidden": [128, 64], "alpha": 0.0001, "copies": 4, "rot": 25, "aug3d": false, "extra": false, "seeds": 1}`

## Per sign

Free = argmax over all classes. Target = the app knows the sign: frame passes if p(sign) >= its own threshold.
Shape spread = how much the handshape changes within one sample (high = the sign moves; a static model may not fit).

| sign | frame acc | sample acc | target threshold | target recall | worst false accept | shape spread |
|---|---|---|---|---|---|---|
| 1 | 0.979 | 0.980 | 0.50 | 0.979 | none 0.013 | 0.140 |
| 2 | 0.892 | 0.897 | 0.50 | 0.892 | 3 0.038 | 0.183 |
| 3 | 0.777 | 0.778 | 0.50 | 0.773 | none 0.004 | 0.186 |
| 4 | 0.957 | 0.979 | 0.71 | 0.929 | 3 0.046 | 0.134 |
| 5 | 0.867 | 0.915 | 0.50 | 0.867 | none 0.022 | 0.102 |
| none | 0.952 | | | | |

Macro: frame 0.904, sample 0.910.

## Most confused (true -> predicted, frame level)

- 5 -> none: 0.133
- 3 -> none: 0.126
- 2 -> none: 0.097
- 3 -> 4: 0.059
- 4 -> none: 0.042
- 3 -> 2: 0.038
- 1 -> none: 0.021
- 2 -> 1: 0.011
- 4 -> 3: 0.002
- 5 -> 4: 0.000
- 5 -> 3: 0.000
- 5 -> 2: 0.000

## Our captures (each person held out)

| sign | aris_p3 | gabriel_p1 | santiago_p2 | all | frames |
|---|---|---|---|---|---|
| 1 | 1.00 | 1.00 | 1.00 | 1.000 | 276 |
| 2 | 0.77 | 0.98 | 0.91 | 0.892 | 269 |
| 3 | 0.63 | 0.63 | 1.00 | 0.777 | 238 |
| 4 | 0.88 | 0.89 | 1.00 | 0.921 | 241 |
| 5 | 0.71 | 0.49 | 0.96 | 0.721 | 226 |

Our captures macro 0.862; LSC70 + none frames only, macro 0.965.

Free-mode threshold tau = 0.700 (<= 5% of none frames above it, floor 0.70). holdFrames = 8.
Target thresholds: for every other class, <= 5% of its frames reach the threshold (per frame, before the 8-frame hold).

Golden: 9 cases (4 mirrored); JSON forward pass vs sklearn max error 2.35e-07
