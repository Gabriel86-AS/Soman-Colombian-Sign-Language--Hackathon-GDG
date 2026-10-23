import 'package:flutter/material.dart';

import '../app/theme/soman_colors.dart';
import '../app/theme/soman_type.dart';
import '../app/theme/spacing.dart';

/// MediaPipe hand bones (pairs of landmark indices).
const handBones = [
  (0, 1), (1, 2), (2, 3), (3, 4), // thumb
  (0, 5), (5, 6), (6, 7), (7, 8), // index
  (5, 9), (9, 10), (10, 11), (11, 12), // middle
  (9, 13), (13, 14), (14, 15), (15, 16), // ring
  (13, 17), (17, 18), (18, 19), (19, 20), // pinky
  (0, 17),
];

/// The open hand drawn on the canvas, as 21 points normalized to a panel.
/// For the gallery and screens without a camera.
final List<Offset> demoHandLandmarks = [
  for (final (x, y) in const [
    (100, 225),
    (70, 200),
    (50, 170),
    (38, 145),
    (30, 120),
    (75, 130),
    (70, 95),
    (67, 70),
    (65, 48),
    (100, 125),
    (100, 85),
    (100, 58),
    (100, 35),
    (123, 130),
    (128, 95),
    (131, 70),
    (134, 50),
    (143, 140),
    (152, 112),
    (158, 93),
    (163, 75),
  ])
    Offset((65 + x * 1.1) / 350, (40 + y * 1.1) / 380),
];

/// Rounded panel (radius 34) for the camera: four rounded bracket corners,
/// the 21-point hand overlay (ochre points, cream bones) and a cream
/// precision pill at the bottom.
class CameraPanel extends StatelessWidget {
  const CameraPanel({
    super.key,
    this.child,
    this.landmarks,
    this.precision,
    this.height = 380,
  });

  /// Camera preview (Phase E). Null shows the plain panel.
  final Widget? child;

  /// 21 points, 0..1 inside the panel. Null hides the overlay.
  final List<Offset>? landmarks;

  /// 0..1. Null hides the pill.
  final double? precision;
  final double height;

  @override
  Widget build(BuildContext context) {
    final c = context.colors;
    final hasPill = precision != null;
    return Semantics(
      label: 'Vista de cámara',
      container: true,
      child: ClipRRect(
        borderRadius: BorderRadius.circular(Radii.media),
        child: Container(
          height: height,
          color: c.pill,
          child: Stack(
            fit: StackFit.expand,
            children: [
              ?child,
              Positioned.fill(
                left: 44,
                right: 44,
                top: 36,
                bottom: hasPill ? 72 : 36,
                child: CustomPaint(painter: _BracketPainter(c.primary)),
              ),
              if (landmarks != null)
                Positioned.fill(
                  child: Semantics(
                    label: 'Mano detectada',
                    child: CustomPaint(
                      painter: LandmarkPainter(
                        landmarks!,
                        point: c.primary,
                        bone: SomanPalette.cream.withValues(alpha: 0.75),
                      ),
                    ),
                  ),
                ),
              if (hasPill)
                Positioned(
                  left: 14,
                  right: 14,
                  bottom: 14,
                  child: _PrecisionPill(value: precision!),
                ),
            ],
          ),
        ),
      ),
    );
  }
}

class _PrecisionPill extends StatelessWidget {
  const _PrecisionPill({required this.value});
  final double value;

  @override
  Widget build(BuildContext context) {
    final c = context.colors;
    final v = value.clamp(0.0, 1.0);
    final pct = '${(v * 100).round()}%';
    return Semantics(
      label: 'Precisión de la seña',
      value: pct,
      excludeSemantics: true,
      child: Container(
        height: 50,
        padding: const EdgeInsets.symmetric(horizontal: 18),
        decoration: BoxDecoration(
          color: c.sheet,
          borderRadius: BorderRadius.circular(Radii.pill),
        ),
        child: Row(
          children: [
            Text(
              'Precisión',
              style: SomanType.label.copyWith(color: c.onSheet, fontSize: 14),
            ),
            const SizedBox(width: 12),
            Expanded(
              child: ClipRRect(
                borderRadius: BorderRadius.circular(Radii.pill),
                child: Container(
                  height: 10,
                  color: c.onSheet.withValues(alpha: 0.15),
                  alignment: Alignment.centerLeft,
                  child: TweenAnimationBuilder<double>(
                    tween: Tween(end: v),
                    duration: Motion.fast,
                    builder: (_, t, _) => FractionallySizedBox(
                      widthFactor: t,
                      child: Container(
                        decoration: BoxDecoration(
                          color: c.accent,
                          borderRadius: BorderRadius.circular(Radii.pill),
                        ),
                      ),
                    ),
                  ),
                ),
              ),
            ),
            const SizedBox(width: 12),
            SizedBox(
              width: 50,
              child: Text(
                pct,
                textAlign: TextAlign.right,
                style: SomanType.title.copyWith(color: c.onSheet, fontSize: 20),
              ),
            ),
          ],
        ),
      ),
    );
  }
}

/// Four rounded L-shaped corners.
class _BracketPainter extends CustomPainter {
  _BracketPainter(this.color);
  final Color color;

  static const _len = 38.0;
  static const _r = 18.0;

  @override
  void paint(Canvas canvas, Size s) {
    final p = Paint()
      ..color = color
      ..style = PaintingStyle.stroke
      ..strokeWidth = 4
      ..strokeCap = StrokeCap.round;
    // Each corner: from the end of one arm, around the rounded corner, to
    // the end of the other arm. (dx, dy) point inward.
    for (final (x, y, dx, dy) in [
      (0.0, 0.0, 1.0, 1.0),
      (s.width, 0.0, -1.0, 1.0),
      (0.0, s.height, 1.0, -1.0),
      (s.width, s.height, -1.0, -1.0),
    ]) {
      final path = Path()
        ..moveTo(x, y + dy * _len)
        ..lineTo(x, y + dy * _r)
        ..quadraticBezierTo(x, y, x + dx * _r, y)
        ..lineTo(x + dx * _len, y);
      canvas.drawPath(path, p);
    }
  }

  @override
  bool shouldRepaint(_BracketPainter old) => old.color != color;
}

/// Hand skeleton: cream bones, ochre points (wrist a bit bigger).
class LandmarkPainter extends CustomPainter {
  LandmarkPainter(this.points, {required this.point, required this.bone});

  final List<Offset> points;
  final Color point;
  final Color bone;

  @override
  void paint(Canvas canvas, Size size) {
    if (points.length < 21) return;
    Offset at(int i) =>
        Offset(points[i].dx * size.width, points[i].dy * size.height);
    final b = Paint()
      ..color = bone
      ..strokeWidth = 2.5
      ..strokeCap = StrokeCap.round;
    for (final (i, j) in handBones) {
      canvas.drawLine(at(i), at(j), b);
    }
    final p = Paint()..color = point;
    for (var i = 0; i < 21; i++) {
      canvas.drawCircle(at(i), i == 0 ? 5 : 4, p);
    }
  }

  @override
  bool shouldRepaint(LandmarkPainter old) =>
      !identical(old.points, points) || old.point != point || old.bone != bone;
}
