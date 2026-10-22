# Numbers demo

Live webcam demo of the numbers model (`ml/out/numbers_mlp.json`). Separate from the letters demo (`ml/demo_letters.py`).

From the `ml/` folder (same environment as the letters demo, `.venv-demo`):

```
.venv-demo\Scripts\python demo_numbers\demo_numbers.py            # webcam 0
.venv-demo\Scripts\python demo_numbers\demo_numbers.py 1          # another camera
.venv-demo\Scripts\python demo_numbers\demo_numbers.py --check    # no camera: golden test
```

Keys: `1` `4`-`9` = check that number, `0` = 10, `M` = MIL, `N` = MILLON, Space = free mode (guess any), Esc = quit.

Tips: hold the hand close to the camera (the model was trained on hand crops). 1 and 6 are often confused. 10, MIL and MILLON move: hold the final shape.
