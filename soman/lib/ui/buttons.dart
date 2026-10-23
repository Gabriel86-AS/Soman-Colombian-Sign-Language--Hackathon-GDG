import 'package:flutter/material.dart';

import '../app/theme/soman_colors.dart';
import '../app/theme/soman_type.dart';
import '../app/theme/spacing.dart';
import 'soman_icons.dart';

/// Soft, low drop shadow for floating elements only (DESIGN.md).
List<BoxShadow> floatShadow([double alpha = 0.25]) => [
  BoxShadow(
    color: Colors.black.withValues(alpha: alpha),
    blurRadius: 26,
    offset: const Offset(0, 12),
  ),
];

/// Shared tap target: pill or circle, ink ripple, button semantics,
/// faded when disabled.
class _Tappable extends StatelessWidget {
  const _Tappable({
    required this.color,
    required this.onPressed,
    required this.child,
    this.circle = false,
    this.shadow,
    this.semanticLabel,
  });

  final Color color;
  final VoidCallback? onPressed;
  final Widget child;
  final bool circle;
  final List<BoxShadow>? shadow;
  final String? semanticLabel;

  @override
  Widget build(BuildContext context) {
    final enabled = onPressed != null;
    final ShapeBorder shape = circle
        ? const CircleBorder()
        : const StadiumBorder();
    return Semantics(
      button: true,
      enabled: enabled,
      label: semanticLabel,
      excludeSemantics: semanticLabel != null,
      child: AnimatedOpacity(
        duration: Motion.fast,
        opacity: enabled ? 1 : 0.45,
        child: DecoratedBox(
          decoration: ShapeDecoration(
            shape: shape,
            shadows: enabled ? shadow : null,
          ),
          child: Material(
            color: color,
            shape: shape,
            clipBehavior: Clip.antiAlias,
            child: InkWell(onTap: onPressed, child: child),
          ),
        ),
      ),
    );
  }
}

/// Ochre pill, 62 high: label on the left, wine circle with an ochre arrow
/// on the right. The one main action of a screen.
class PrimaryPillButton extends StatelessWidget {
  const PrimaryPillButton({
    super.key,
    required this.label,
    required this.onPressed,
    this.icon = SomanIcons.arrowRight,
  });

  final String label;
  final VoidCallback? onPressed;
  final SomanIconData icon;

  @override
  Widget build(BuildContext context) {
    final c = context.colors;
    return _Tappable(
      color: c.primary,
      onPressed: onPressed,
      shadow: floatShadow(),
      child: SizedBox(
        height: 62,
        child: Padding(
          padding: const EdgeInsets.only(left: 26, right: 8),
          child: Row(
            children: [
              Expanded(
                child: Text(
                  label,
                  overflow: TextOverflow.ellipsis,
                  style: SomanType.button.copyWith(color: c.onPrimary),
                ),
              ),
              Container(
                width: 46,
                height: 46,
                alignment: Alignment.center,
                decoration: BoxDecoration(
                  color: c.onPrimary,
                  shape: BoxShape.circle,
                ),
                child: SomanIcon(icon, color: c.primary),
              ),
            ],
          ),
        ),
      ),
    );
  }
}

/// Wine pill (red in dark mode) with cream text. 46 high by default; use 56
/// for the coach sheet's "Intentar de nuevo".
class SecondaryPillButton extends StatelessWidget {
  const SecondaryPillButton({
    super.key,
    required this.label,
    required this.onPressed,
    this.icon,
    this.height = 46,
    this.expand = false,
    this.background,
    this.foreground,
  });

  final String label;
  final VoidCallback? onPressed;
  final SomanIconData? icon;
  final double height;
  final bool expand;
  final Color? background;
  final Color? foreground;

  @override
  Widget build(BuildContext context) {
    final c = context.colors;
    final fg = foreground ?? c.onSurface;
    final style = SomanType.button.copyWith(
      color: fg,
      fontSize: height >= 56 ? 17 : 15,
    );
    return _Tappable(
      color: background ?? c.surface,
      onPressed: onPressed,
      child: SizedBox(
        height: height,
        child: Padding(
          padding: EdgeInsets.only(left: icon == null ? 22 : 16, right: 22),
          child: Row(
            mainAxisSize: expand ? MainAxisSize.max : MainAxisSize.min,
            mainAxisAlignment: MainAxisAlignment.center,
            children: [
              if (icon != null) ...[
                SomanIcon(icon!, size: 18, color: fg),
                const SizedBox(width: 6),
              ],
              Flexible(
                child: Text(
                  label,
                  overflow: TextOverflow.ellipsis,
                  style: style,
                ),
              ),
            ],
          ),
        ),
      ),
    );
  }
}

/// 44 px round icon button (wine in light, red in dark). Icon-only, so a
/// spoken label is required.
class CircleIconButton extends StatelessWidget {
  const CircleIconButton({
    super.key,
    required this.icon,
    required this.semanticLabel,
    required this.onPressed,
    this.size = 44,
    this.background,
    this.foreground,
  });

  final SomanIconData icon;
  final String semanticLabel;
  final VoidCallback? onPressed;
  final double size;
  final Color? background;
  final Color? foreground;

  @override
  Widget build(BuildContext context) {
    final c = context.colors;
    return Tooltip(
      message: semanticLabel,
      excludeFromSemantics: true,
      child: _Tappable(
        circle: true,
        color: background ?? c.surface,
        onPressed: onPressed,
        semanticLabel: semanticLabel,
        child: SizedBox.square(
          dimension: size,
          child: Center(
            child: SomanIcon(
              icon,
              size: size * 0.45,
              color: foreground ?? c.onSurface,
            ),
          ),
        ),
      ),
    );
  }
}
