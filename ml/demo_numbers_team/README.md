# Numbers demo (team model, 1-5)

Live webcam demo of the numbers 1-5 model, trained on 2026-10-22 with our own captures (aris, gabriel, santiago) plus the LSC70 signers for 1, 4 and 5. 2 and 3 come only from our captures.

The model is a frozen copy in `model/`, so retraining `ml/out/` does not change this demo. The original demo (`ml/demo_numbers/`) always uses the latest `ml/out/`.

## Run

Double-click `run.bat`, or from the `ml/` folder:

```
.venv-demo\Scripts\python demo_numbers_team\demo.py            # webcam 0
.venv-demo\Scripts\python demo_numbers_team\demo.py 1          # another camera
.venv-demo\Scripts\python demo_numbers_team\demo.py --check    # no camera: golden test
```

## Keys

`1`-`5` = check that number, Space = free mode (guess any), Esc = quit.

## Screen

- Left: camera with the hand skeleton, the current guess, and a green bar that fills while the sign is held (8 frames).
- Right: the hand crop the model sees, then one bar per class. The red tick is the threshold the bar must pass.

## Tips

- Hold the hand close to the camera (the model was trained on hand crops).
- Weak spots: 3 and 5 sometimes read as "no sign"; hold them steady and clear.

Held-out scores (people not seen in training): 0.90 per frame overall; our captures 0.86 (1 1.00, 2 0.89, 3 0.78, 4 0.92, 5 0.72).
