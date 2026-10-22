# Numbers demo (team model)

Live webcam demo of the numbers model retrained on 2026-10-22 with our own captures (aris, gabriel, santiago). It has 12 numbers: 1-10, MIL, MILLON. 2 and 3 come only from our captures.

The model is a frozen copy in `model/`, so retraining `ml/out/` does not change this demo. The original demo (`ml/demo_numbers/`) always uses the latest `ml/out/`.

## Run

Double-click `run.bat`, or from the `ml/` folder:

```
.venv-demo\Scripts\python demo_numbers_team\demo.py            # webcam 0
.venv-demo\Scripts\python demo_numbers_team\demo.py 1          # another camera
.venv-demo\Scripts\python demo_numbers_team\demo.py --check    # no camera: golden test
```

## Keys

`1`-`9` = check that number, `0` = 10, `M` = MIL, `N` = MILLON, Space = free mode (guess any), Esc = quit.

## Screen

- Left: camera with the hand skeleton, the current guess, and a green bar that fills while the sign is held (8 frames).
- Right: the hand crop the model sees, then one bar per class. The red tick is the threshold the bar must pass.

## Tips

- Hold the hand close to the camera (the model was trained on hand crops).
- Weak spots: 3 is often read as 8, 6 and 1 get mixed up, 2 sometimes reads as 7.
- 10, MIL and MILLON move: hold the final shape.

Held-out scores (people not seen in training): all numbers 0.81 per frame; our captures 0.70 (1 0.90, 2 0.71, 3 0.43, 4 0.79, 5 0.68).
