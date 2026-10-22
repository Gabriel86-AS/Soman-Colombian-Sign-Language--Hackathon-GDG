# Letter model report (v2)

Frames: 6959 letter frames (20 static letters, 70 signers, LSC70ANH crops) + 2000 none frames.
All numbers are on held-out signers (GroupKFold 5 by signer).

## Variants tried (macro accuracy over 20 letters + none, frame level / sample level)

| variant | frame | sample |
|---|---|---|
| feats=xy | 0.918 | 0.928 |
| feats=xy+z | 0.921 | 0.933 |
| feats=xy+geo | 0.922 | 0.930 |
| feats=xy+z+geo3 | 0.922 | 0.933 |
| wider (256,128) a=1e-3 | 0.923 | 0.931 |
| aug x4 rot 25 | 0.923 | 0.931 |
| aug 3D viewpoint | 0.926 | 0.938 |
| +full-frame LSC70 | 0.912 | 0.922 |
| ensemble x5 | 0.930 | 0.939 |
| quick: chosen | 0.923 | 0.931 |
| quick: chosen + {"aug3d": true, "seeds": 5} | 0.930 | 0.938 |

Chosen: `{"feats": "xy", "hidden": [256, 128], "alpha": 0.001, "copies": 2, "rot": 15, "aug3d": true, "extra": false, "seeds": 5}`

## Per letter

Free = argmax over all classes. Target = the app knows the letter: frame passes if p(letter) >= its own threshold.

| letter | frame acc | sample acc | target threshold | target recall | worst false accept |
|---|---|---|---|---|---|
| A | 0.972 | 0.986 | 0.50 | 0.969 | Y 0.014 |
| B | 0.930 | 0.913 | 0.50 | 0.930 | none 0.018 |
| C | 0.869 | 0.882 | 0.58 | 0.857 | E 0.049 |
| D | 0.917 | 0.929 | 0.50 | 0.917 | X 0.008 |
| E | 0.832 | 0.838 | 0.78 | 0.737 | C 0.048 |
| F | 0.918 | 0.929 | 0.50 | 0.915 | L 0.043 |
| I | 0.941 | 0.957 | 0.50 | 0.938 | D 0.014 |
| K | 0.928 | 0.928 | 0.50 | 0.928 | none 0.003 |
| L | 0.939 | 0.957 | 0.77 | 0.872 | F 0.048 |
| M | 0.954 | 0.957 | 0.50 | 0.954 | none 0.003 |
| O | 0.927 | 0.929 | 0.50 | 0.916 | E 0.037 |
| P | 0.916 | 0.913 | 0.50 | 0.916 | M 0.014 |
| Q | 0.962 | 0.984 | 0.50 | 0.958 | none 0.003 |
| R | 0.953 | 0.971 | 0.50 | 0.951 | K 0.028 |
| T | 0.917 | 0.943 | 0.50 | 0.903 | W 0.014 |
| U | 0.966 | 0.971 | 0.95 | 0.868 | Y 0.045 |
| V | 0.963 | 0.971 | 0.50 | 0.963 | T 0.006 |
| W | 0.963 | 0.957 | 0.50 | 0.957 | V 0.014 |
| X | 0.935 | 0.943 | 0.50 | 0.935 | D 0.011 |
| Y | 0.898 | 0.900 | 0.50 | 0.895 | I 0.044 |
| none | 0.933 | | | | |

Macro: frame 0.930, sample 0.938.

## Most confused (true -> predicted, frame level)

- P -> none: 0.084
- C -> E: 0.078
- E -> none: 0.064
- E -> C: 0.061
- T -> none: 0.060
- F -> L: 0.056
- Y -> U: 0.054
- B -> none: 0.053
- X -> none: 0.049
- I -> Y: 0.044
- L -> F: 0.043
- E -> O: 0.043

Free-mode threshold tau = 0.700 (<= 5% of none frames above it, floor 0.70). holdFrames = 8.
Target thresholds: for every other class, <= 5% of its frames reach the threshold (per frame, before the 8-frame hold).

Golden: 24 cases (4 mirrored); JSON forward pass vs sklearn max error < 1e-5 (weights rounded to 6 decimals)
