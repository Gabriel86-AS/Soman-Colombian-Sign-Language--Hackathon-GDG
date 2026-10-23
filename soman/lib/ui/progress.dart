import 'dart:math' as math;

import 'package:flutter/material.dart';

import '../app/theme/soman_colors.dart';
import '../app/theme/spacing.dart';
import 'soman_icons.dart';

/// Lesson progress as a row of pill segments.
class SegmentProgress extends StatelessWidget {
  const SegmentProgress({super.key, required this.total, required this.done});

  final int total;
  final int done;

  @override
  Widget build(BuildContext context) {
    final c = context.colors;
    return Semantics(
      label: 'Progreso de la lección',
      value: '$done de $total',
      excludeSemantics: true,
      child: Row(
        children: [
          for (var i = 0; i < total; i++) ...[
            if (i > 0) const SizedBox(width: 5),
            Expanded(
              child: AnimatedContainer(
                duration: Motion.normal,
                curve: Motion.curve,
                height: 10,
                decoration: BoxDecoration(
                  color: i < done ? c.primary : c.text.withValues(alpha: 0.15),
                  borderRadius: BorderRadius.circular(Radii.pill),
                ),
              ),
            ),
          ],
        ],
      ),
    );
  }
}

/// Ring around [child] that fills while the target sign is held.
/// [progress] runs 0..1; changes animate smoothly.
class HoldRing extends StatelessWidget {
  const HoldRing({
    super.key,
    required this.progress,
    required this.child,
    this.diameter = 76,
    this.gap = 6,
    this.thickness = 5,
  });

  final double progress;
  final Widget child;

  /// Size of [child].
  final double diameter;
  final double gap;
  final double thickness;

  @override
  Widget build(BuildContext context) {
    final c = context.colors;
    final outer = diameter + 2 * (gap + thickness);
    return SizedBox.square(
      dimension: outer,
      child: TweenAnimationBuilder<double>(
        tween: Tween(end: progress.clamp(0, 1)),
        duration: Motion.fast,
        curve: Curves.linear,
        builder: (context, p, child) => CustomPaint(
          painter: _RingPainter(
            p,
            c.primary,
            c.primary.withValues(alpha: 0.25),
            thickness,
          ),
          child: child,
        ),
        child: Center(child: child),
      ),
    );
  }
}

class _RingPainter extends CustomPainter {
  _RingPainter(this.p, this.color, this.track, this.width);

  final double p;
  final Color color;
  final Color track;
  final double width;

  @override
  void paint(Canvas canvas, Size size) {
    final rect = (Offset.zero & size).deflate(width / 2);
    final paint = Paint()
      ..style = PaintingStyle.stroke
      ..strokeWidth = width
      ..strokeCap = StrokeCap.round;
    canvas.drawArc(rect, 0, math.pi * 2, false, paint..color = track);
    if (p > 0) {
      canvas.drawArc(
        rect,
        -math.pi / 2,
        math.pi * 2 * p,
        false,
        paint..color = color,
      );
    }
  }

  @override
  bool shouldRepaint(_RingPainter old) =>
      old.p != p || old.color != color || old.track != track;
}

/// The big ochre check button of the practice screen, inside a [HoldRing].
class HoldCheckButton extends StatelessWidget {
  const HoldCheckButton({
    super.key,
    required this.progress,
    required this.onPressed,
    this.semanticLabel = 'Comprobar seña',
  });

  final double progress;
  final VoidCallback? onPressed;
  final String semanticLabel;

  @override
  Widget build(BuildContext context) {
    final c = context.colors;
    return Semantics(
      button: true,
      enabled: onPressed != null,
      label: semanticLabel,
      value: '${(progress * 100).round()} %',
      excludeSemantics: true,
      child: HoldRing(
        progress: progress,
        child: Material(
          color: c.primary,
          shape: const CircleBorder(),
          clipBehavior: Clip.antiAlias,
          child: InkWell(
            onTap: onPressed,
            child: SizedBox.square(
              dimension: 76,
              child: Center(
                child: SomanIcon(
                  SomanIcons.check,
                  size: 30,
                  color: c.onPrimary,
                ),
              ),
            ),
          ),
        ),
      ),
    );
  }
}
