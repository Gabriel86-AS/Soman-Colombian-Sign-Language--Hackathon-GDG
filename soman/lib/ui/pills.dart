import 'package:flutter/material.dart';

import '../app/theme/soman_colors.dart';
import '../app/theme/soman_type.dart';
import '../app/theme/spacing.dart';
import 'soman_icons.dart';

/// "SOMAN" in Regular Aesthetic. The only place that font is used.
class SomanLogo extends StatelessWidget {
  const SomanLogo({super.key, this.small = false, this.color});

  final bool small;
  final Color? color;

  @override
  Widget build(BuildContext context) => Semantics(
    label: 'Soman',
    header: true,
    excludeSemantics: true,
    child: Text(
      'SOMAN',
      style: (small ? SomanType.logoSmall : SomanType.logo).copyWith(
        color: color ?? context.colors.text,
      ),
    ),
  );
}

/// 38 px pill. Toggles when [onSelected] is set (Lento / Normal / Espejo),
/// otherwise just shows a word (new signs on the completion screen).
class SomanChip extends StatelessWidget {
  const SomanChip({
    super.key,
    required this.label,
    this.selected = false,
    this.onSelected,
    this.icon,
  });

  final String label;
  final bool selected;
  final ValueChanged<bool>? onSelected;
  final SomanIconData? icon;

  @override
  Widget build(BuildContext context) {
    final c = context.colors;
    final bg = selected ? c.inverse : c.pill;
    final fg = selected ? c.onInverse : c.onPill;
    final body = AnimatedContainer(
      duration: Motion.fast,
      curve: Motion.curve,
      height: 38,
      padding: const EdgeInsets.symmetric(horizontal: 16),
      decoration: BoxDecoration(
        color: bg,
        borderRadius: BorderRadius.circular(Radii.pill),
      ),
      child: Row(
        mainAxisSize: MainAxisSize.min,
        children: [
          if (icon != null) ...[
            SomanIcon(icon!, size: 16, color: fg),
            const SizedBox(width: 6),
          ],
          Text(
            label,
            style: SomanType.label.copyWith(
              color: fg,
              fontSize: 14,
              fontWeight: selected ? FontWeight.w600 : FontWeight.w500,
            ),
          ),
        ],
      ),
    );
    if (onSelected == null) return body;
    return Semantics(
      button: true,
      selected: selected,
      child: GestureDetector(
        behavior: HitTestBehavior.opaque,
        onTap: () => onSelected!(!selected),
        child: body,
      ),
    );
  }
}

/// Small ochre tag next to a sign name (e.g. "Saludo").
class SomanTag extends StatelessWidget {
  const SomanTag(this.label, {super.key});

  final String label;

  @override
  Widget build(BuildContext context) {
    final c = context.colors;
    return Container(
      height: 30,
      padding: const EdgeInsets.symmetric(horizontal: 14),
      alignment: Alignment.center,
      decoration: BoxDecoration(
        color: c.primary,
        borderRadius: BorderRadius.circular(Radii.pill),
      ),
      child: Text(label, style: SomanType.label.copyWith(color: c.onPrimary)),
    );
  }
}

/// Header pill with an optional icon: streak, XP, lives.
class StatPill extends StatelessWidget {
  const StatPill({
    super.key,
    required this.label,
    required this.semanticLabel,
    this.icon,
    this.background,
    this.foreground,
    this.iconColor,
    this.height = 34,
    this.ochre = false,
  });

  /// Racha: red pill, ochre flame.
  factory StatPill.streak(int days, {Key? key}) => StatPill(
    key: key,
    icon: SomanIcons.flame,
    label: days == 1 ? '1 día' : '$days días',
    semanticLabel: 'Racha de ${days == 1 ? '1 día' : '$days días'}',
  );

  /// XP: ochre pill, wine text.
  factory StatPill.xp(int xp, {Key? key}) => StatPill(
    key: key,
    label: '$xp XP',
    semanticLabel: '$xp puntos de experiencia',
    ochre: true,
  );

  final String label;
  final String semanticLabel;
  final SomanIconData? icon;
  final Color? background;
  final Color? foreground;
  final Color? iconColor;
  final double height;

  /// Ochre pill with wine text instead of red with cream.
  final bool ochre;

  @override
  Widget build(BuildContext context) {
    final c = context.colors;
    final fg = foreground ?? (ochre ? c.onPrimary : c.onPill);
    return Semantics(
      label: semanticLabel,
      excludeSemantics: true,
      child: Container(
        height: height,
        padding: const EdgeInsets.symmetric(horizontal: 13),
        decoration: BoxDecoration(
          color: background ?? (ochre ? c.primary : c.pill),
          borderRadius: BorderRadius.circular(Radii.pill),
        ),
        child: Row(
          mainAxisSize: MainAxisSize.min,
          children: [
            if (icon != null) ...[
              SomanIcon(icon!, size: 17, color: iconColor ?? c.primary),
              const SizedBox(width: 5),
            ],
            AnimatedSwitcher(
              duration: Motion.normal,
              transitionBuilder: (child, a) => FadeTransition(
                opacity: a,
                child: SlideTransition(
                  position: Tween(
                    begin: const Offset(0, -0.4),
                    end: Offset.zero,
                  ).animate(a),
                  child: child,
                ),
              ),
              child: Text(
                label,
                key: ValueKey(label),
                style: SomanType.label.copyWith(color: fg, fontSize: 14),
              ),
            ),
          ],
        ),
      ),
    );
  }
}

/// Lives are cups of tinto (3 per day). Red pill, ochre cup, the count.
class LivesPill extends StatelessWidget {
  const LivesPill({super.key, required this.lives});

  final int lives;

  @override
  Widget build(BuildContext context) => StatPill(
    icon: SomanIcons.cup,
    label: '$lives',
    height: 36,
    semanticLabel: lives == 1 ? '1 taza de tinto' : '$lives tazas de tinto',
  );
}
