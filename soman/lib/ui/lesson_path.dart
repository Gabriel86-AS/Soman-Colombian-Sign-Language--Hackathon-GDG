import 'dart:ui' show PathMetric;

import 'package:flutter/material.dart';

import '../app/theme/soman_colors.dart';
import '../app/theme/soman_type.dart';
import '../app/theme/spacing.dart';
import 'illustrations.dart';
import 'soman_icons.dart';

enum NodeState { done, current, locked }

/// 72 px lesson circle with a solid `0 7px 0` shadow. Sinks into its
/// shadow while pressed.
class LessonNode extends StatefulWidget {
  const LessonNode({
    super.key,
    required this.state,
    required this.label,
    this.onTap,
  });

  final NodeState state;

  /// Sign name, read by screen readers with the state.
  final String label;
  final VoidCallback? onTap;

  static const size = 72.0;
  static const _depth = 7.0;

  @override
  State<LessonNode> createState() => _LessonNodeState();
}

class _LessonNodeState extends State<LessonNode> {
  bool _down = false;

  @override
  Widget build(BuildContext context) {
    final c = context.colors;
    final (bg, shadow, Widget icon, String spoken) = switch (widget.state) {
      NodeState.done => (
        c.primary,
        c.nodeShadowDone,
        SomanIcon(SomanIcons.check, size: 30, color: c.onPrimary),
        'completada',
      ),
      NodeState.current => (
        c.inverse,
        c.nodeShadowCurrent,
        HandIllustration(size: 40, color: c.onInverse, cuffColor: c.primary),
        'empezar',
      ),
      NodeState.locked => (
        SomanPalette.red,
        c.nodeShadowLocked,
        SomanIcon(
          SomanIcons.lock,
          size: 26,
          color: SomanPalette.cream.withValues(alpha: 0.55),
        ),
        'bloqueada',
      ),
    };
    final tappable = widget.onTap != null && widget.state != NodeState.locked;
    final sink = _down ? LessonNode._depth - 2 : 0.0;

    Widget node = AnimatedContainer(
      duration: const Duration(milliseconds: 90),
      width: LessonNode.size,
      height: LessonNode.size,
      transform: Matrix4.translationValues(0, sink, 0),
      alignment: Alignment.center,
      decoration: BoxDecoration(
        color: bg,
        shape: BoxShape.circle,
        boxShadow: [
          if (widget.state == NodeState.current)
            BoxShadow(
              color: c.inverse.withValues(alpha: 0.14),
              spreadRadius: 8,
            ),
          BoxShadow(color: shadow, offset: Offset(0, LessonNode._depth - sink)),
        ],
      ),
      child: icon,
    );

    if (tappable) {
      node = GestureDetector(
        onTapDown: (_) => setState(() => _down = true),
        onTapCancel: () => setState(() => _down = false),
        onTapUp: (_) => setState(() => _down = false),
        onTap: widget.onTap,
        child: node,
      );
    }
    return Semantics(
      button: tappable,
      enabled: tappable,
      label: '${widget.label}, $spoken',
      excludeSemantics: true,
      child: node,
    );
  }
}

/// One stop on the path.
class PathStop {
  const PathStop({required this.label, required this.state, this.onTap});
  final String label;
  final NodeState state;
  final VoidCallback? onTap;
}

/// Winding dotted path through lesson nodes. Fixed height (put it in a
/// scroll view). Node i sits at `xs[i % xs.length]` of the width.
class LessonPath extends StatelessWidget {
  const LessonPath({
    super.key,
    required this.stops,
    this.xs = defaultXs,
    this.spacing = 90,
    this.startLabel = 'Empezar →',
  });

  final List<PathStop> stops;

  /// Horizontal position of each node's center, 0..1 of the width.
  final List<double> xs;

  /// Vertical distance between node tops.
  final double spacing;
  final String startLabel;

  /// Same curve as the canvas home screen.
  static const defaultXs = [0.5, 0.31, 0.24, 0.41, 0.62, 0.70, 0.56];

  static const _pad = 8.0;

  double get height =>
      _pad * 2 + (stops.length - 1) * spacing + LessonNode.size + 7;

  /// Center of node [i] for a path [width] wide. Lets a screen place Nua or
  /// other decor next to a node.
  Offset centerOf(int i, double width) => Offset(
    xs[i % xs.length] * width,
    _pad + i * spacing + LessonNode.size / 2,
  );

  @override
  Widget build(BuildContext context) {
    final c = context.colors;
    return LayoutBuilder(
      builder: (context, box) {
        final w = box.maxWidth;
        final centers = [for (var i = 0; i < stops.length; i++) centerOf(i, w)];
        return SizedBox(
          height: height,
          child: Stack(
            clipBehavior: Clip.none,
            children: [
              Positioned.fill(
                child: CustomPaint(
                  painter: _DottedPathPainter(
                    centers,
                    c.text.withValues(alpha: 0.25),
                    spacing / 2,
                  ),
                ),
              ),
              for (var i = 0; i < stops.length; i++)
                ..._stop(context, stops[i], centers[i], w),
            ],
          ),
        );
      },
    );
  }

  List<Widget> _stop(
    BuildContext context,
    PathStop s,
    Offset center,
    double w,
  ) {
    final c = context.colors;
    const r = LessonNode.size / 2;
    final onRight = center.dx < w * 0.55;
    final top = center.dy - r;

    final Widget side;
    if (s.state == NodeState.current) {
      side = _StartBubble(label: s.label, action: startLabel, onTap: s.onTap);
    } else {
      final muted = s.state == NodeState.locked;
      side = ExcludeSemantics(
        child: Text(
          s.label,
          style: SomanType.label.copyWith(
            fontSize: 15,
            fontWeight: muted ? FontWeight.w500 : FontWeight.w600,
            color: muted ? c.textMuted : c.text,
          ),
        ),
      );
    }
    final isBubble = s.state == NodeState.current;
    final gap = isBubble ? 18.0 : 14.0;
    final sideTop = isBubble ? top + 7 : top + 26;

    return [
      Positioned(
        left: center.dx - r,
        top: top,
        child: LessonNode(state: s.state, label: s.label, onTap: s.onTap),
      ),
      if (onRight)
        Positioned(left: center.dx + r + gap, top: sideTop, child: side)
      else
        Positioned(right: w - (center.dx - r - gap), top: sideTop, child: side),
    ];
  }
}

class _StartBubble extends StatelessWidget {
  const _StartBubble({required this.label, required this.action, this.onTap});

  final String label;
  final String action;
  final VoidCallback? onTap;

  @override
  Widget build(BuildContext context) {
    final c = context.colors;
    // Light: wine bubble, ochre action. Dark: cream bubble, wine action
    // (ochre on cream is too faint).
    final actionColor = Theme.of(context).brightness == Brightness.dark
        ? c.onInverse
        : c.primary;
    return ExcludeSemantics(
      child: GestureDetector(
        onTap: onTap,
        child: Container(
          height: 58,
          padding: const EdgeInsets.symmetric(horizontal: 22),
          decoration: BoxDecoration(
            color: c.inverse,
            borderRadius: BorderRadius.circular(Radii.pill),
            boxShadow: [
              BoxShadow(
                color: Colors.black.withValues(alpha: 0.2),
                blurRadius: 24,
                offset: const Offset(0, 10),
              ),
            ],
          ),
          child: Column(
            mainAxisAlignment: MainAxisAlignment.center,
            crossAxisAlignment: CrossAxisAlignment.start,
            children: [
              Text(
                label,
                style: SomanType.title.copyWith(
                  color: c.onInverse,
                  fontSize: 17,
                  height: 1.1,
                ),
              ),
              Text(
                action,
                style: SomanType.label.copyWith(
                  color: actionColor,
                  fontWeight: FontWeight.w700,
                ),
              ),
            ],
          ),
        ),
      ),
    );
  }
}

/// Round dots every 15 px along smooth curves between node centers.
class _DottedPathPainter extends CustomPainter {
  _DottedPathPainter(this.centers, this.color, this.bend);

  final List<Offset> centers;
  final Color color;
  final double bend;

  @override
  void paint(Canvas canvas, Size size) {
    if (centers.length < 2) return;
    final path = Path()..moveTo(centers.first.dx, centers.first.dy);
    for (var i = 1; i < centers.length; i++) {
      final a = centers[i - 1];
      final b = centers[i];
      path.cubicTo(a.dx, a.dy + bend, b.dx, b.dy - bend, b.dx, b.dy);
    }
    final paint = Paint()..color = color;
    for (final PathMetric m in path.computeMetrics()) {
      for (double d = 0; d < m.length; d += 15) {
        final t = m.getTangentForOffset(d);
        if (t != null) canvas.drawCircle(t.position, 3, paint);
      }
    }
  }

  @override
  bool shouldRepaint(_DottedPathPainter old) =>
      old.color != color ||
      old.centers.length != centers.length ||
      !_same(old.centers, centers);

  static bool _same(List<Offset> a, List<Offset> b) {
    for (var i = 0; i < a.length; i++) {
      if (a[i] != b[i]) return false;
    }
    return true;
  }
}
