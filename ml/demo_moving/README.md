# Moving letters demo (G H J N Ñ S Z)

Live webcam test of the moving-letter model (`ml/out/moving_mlp.json`, always the latest export). It works like the app's "Haz la seña": you pick a letter, get a countdown, and sign during a 1 s window.

## Run

Double-click `run.bat`, or from the `ml/` folder:

```
.venv-demo\Scripts\python demo_moving\demo.py            # webcam 0
.venv-demo\Scripts\python demo_moving\demo.py 1          # another camera
.venv-demo\Scripts\python demo_moving\demo.py --check    # no camera: golden test
```

## Keys

`G` `H` `J` `N` `S` `Z` = try that letter, `1` = Ñ, Space = try the same letter again, Esc = quit.

## How it works

1. Countdown (3 s), then a 1 s recording window. **Start the movement right at "GO!"**: the training data (LSC70) is the first second of each sign.
2. Each frame: hand found on the full frame, cropped (LSC70ANH look), landmarks again on the crop.
3. The 6 frames closest to 0, 1/6 ... 5/6 s are kept (LSC70 is 6 fps), turned into 12 x 40 shape features, then the MLP.
4. Pass = p(letter) at or above its red tick. The right panel shows the 6 crops used and one bar per class ("static" = a static letter, not a moving one).

## Tips

- Hold the hand close to the camera (trained on hand crops).
- Letters that failed Gate 3 (see `ml/out/moving_report.md`) can still be tried here; they are just not reliable.
