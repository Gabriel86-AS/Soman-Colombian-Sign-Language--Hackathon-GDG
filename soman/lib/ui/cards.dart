import 'package:flutter/material.dart';

import '../app/theme/soman_colors.dart';
import '../app/theme/soman_type.dart';
import '../app/theme/spacing.dart';
import 'illustrations.dart';
import 'soman_icons.dart';

/// Solid fills for tiles and stat cards. Each one picks a text color that
/// passes contrast in both modes.
enum SomanTone { primary, surface, surfaceAlt, accent }

extension SomanToneColors on SomanTone {
  (Color bg, Color fg) of(SomanColors c) => switch (this) {
    SomanTone.primary => (c.primary, c.onPrimary),
    SomanTone.surface => (c.surface, c.onSurface),
    SomanTone.surfaceAlt => (c.surfaceAlt, c.onSurfaceAlt),
    SomanTone.accent => (c.accent, c.onAccent),
  };
}

/// Tap wrapper that keeps the ripple inside the card's rounded shape.
Widget _tapCard({
  required double radius,
  required Color color,
  required VoidCallback? onTap,
  required Widget child,
  String? semanticLabel,
}) {
  final card = Material(
    color: color,
    borderRadius: BorderRadius.circular(radius),
    clipBehavior: Clip.antiAlias,
    child: onTap == null ? child : InkWell(onTap: onTap, child: child),
  );
  if (onTap == null) return card;
  return Semantics(button: true, label: semanticLabel, child: card);
}

/// Home header card for the current unit: ochre, overline, Fraunces title,
/// one line, "Guía" pill, teal circle bleeding off the corner, cream hand.
class UnitCard extends StatelessWidget {
  const UnitCard({
    super.key,
    required this.overline,
    required this.title,
    required this.subtitle,
    this.onGuide,
  });

  final String overline;
  final String title;
  final String subtitle;
  final VoidCallback? onGuide;

  @override
  Widget build(BuildContext context) {
    final c = context.colors;
    return ClipRRect(
      borderRadius: BorderRadius.circular(Radii.card),
      child: Container(
        constraints: const BoxConstraints(minHeight: 156),
        color: c.primary,
        child: Stack(
          children: [
            Positioned(
              right: -36,
              bottom: -46,
              child: DecorCircle(size: 150, color: c.accent),
            ),
            const Positioned(
              right: 14,
              bottom: 6,
              child: HandIllustration(
                size: 84,
                color: SomanPalette.cream,
                cuffColor: SomanPalette.red,
              ),
            ),
            Padding(
              padding: const EdgeInsets.fromLTRB(20, 20, 110, 20),
              child: Column(
                crossAxisAlignment: CrossAxisAlignment.start,
                children: [
                  Text(
                    overline.toUpperCase(),
                    style: SomanType.overline.copyWith(color: c.onPrimary),
                  ),
                  const SizedBox(height: 6),
                  Text(
                    title,
                    style: SomanType.title.copyWith(
                      color: c.onPrimary,
                      fontSize: 27,
                      height: 1.05,
                    ),
                  ),
                  const SizedBox(height: 4),
                  Text(
                    subtitle,
                    style: SomanType.body.copyWith(
                      color: c.onPrimary,
                      fontSize: 15,
                      height: 1.3,
                    ),
                  ),
                  if (onGuide != null) ...[
                    const SizedBox(height: 12),
                    _GuidePill(onTap: onGuide!),
                  ],
                ],
              ),
            ),
          ],
        ),
      ),
    );
  }
}

class _GuidePill extends StatelessWidget {
  const _GuidePill({required this.onTap});
  final VoidCallback onTap;

  @override
  Widget build(BuildContext context) {
    final c = context.colors;
    return Semantics(
      button: true,
      label: 'Guía',
      excludeSemantics: true,
      child: Material(
        color: c.pill,
        shape: const StadiumBorder(),
        clipBehavior: Clip.antiAlias,
        child: InkWell(
          onTap: onTap,
          child: SizedBox(
            height: 36,
            child: Padding(
              padding: const EdgeInsets.symmetric(horizontal: 16),
              child: Row(
                mainAxisSize: MainAxisSize.min,
                children: [
                  SomanIcon(SomanIcons.book, size: 16, color: c.onPill),
                  const SizedBox(width: 6),
                  Text(
                    'Guía',
                    style: SomanType.label.copyWith(
                      color: c.onPill,
                      fontSize: 14,
                    ),
                  ),
                ],
              ),
            ),
          ),
        ),
      ),
    );
  }
}

/// 100-high tile: solid fill, illustration top left, title at the bottom.
/// No numbers.
class CategoryTile extends StatelessWidget {
  const CategoryTile({
    super.key,
    required this.title,
    required this.illustration,
    this.tone = SomanTone.surface,
    this.onTap,
  });

  final String title;
  final Widget illustration;
  final SomanTone tone;
  final VoidCallback? onTap;

  @override
  Widget build(BuildContext context) {
    final (bg, fg) = tone.of(context.colors);
    return _tapCard(
      radius: Radii.tile,
      color: bg,
      onTap: onTap,
      semanticLabel: title,
      child: SizedBox(
        height: 100,
        child: Padding(
          padding: const EdgeInsets.fromLTRB(14, 12, 14, 12),
          child: Column(
            crossAxisAlignment: CrossAxisAlignment.start,
            mainAxisAlignment: MainAxisAlignment.spaceBetween,
            children: [
              SizedBox(height: 40, child: illustration),
              Text(
                title,
                maxLines: 1,
                overflow: TextOverflow.ellipsis,
                style: SomanType.title.copyWith(color: fg, fontSize: 20),
              ),
            ],
          ),
        ),
      ),
    );
  }
}

/// Nua's face, "Consejo" in Fraunces italic, one sentence of advice.
class TipCard extends StatelessWidget {
  const TipCard({super.key, required this.text, this.title = 'Consejo'});

  final String text;
  final String title;

  @override
  Widget build(BuildContext context) {
    final c = context.colors;
    return Semantics(
      container: true,
      child: Container(
        padding: const EdgeInsets.all(Space.l),
        decoration: BoxDecoration(
          color: c.surface,
          borderRadius: BorderRadius.circular(26),
        ),
        child: Row(
          children: [
            const NuaAvatar(size: 52),
            const SizedBox(width: 14),
            Expanded(
              child: Column(
                crossAxisAlignment: CrossAxisAlignment.start,
                children: [
                  Text(
                    title,
                    style: SomanType.title.copyWith(
                      color: c.onSurface,
                      fontSize: 18,
                    ),
                  ),
                  const SizedBox(height: 2),
                  Text(
                    text,
                    style: SomanType.body.copyWith(
                      color: c.onSurface,
                      fontSize: 15,
                      height: 1.4,
                    ),
                  ),
                ],
              ),
            ),
          ],
        ),
      ),
    );
  }
}

/// Big Fraunces italic number over a small Jost label.
class StatCard extends StatelessWidget {
  const StatCard({
    super.key,
    required this.value,
    required this.label,
    this.tone = SomanTone.primary,
  });

  final String value;
  final String label;
  final SomanTone tone;

  @override
  Widget build(BuildContext context) {
    final (bg, fg) = tone.of(context.colors);
    return Semantics(
      label: '$value $label',
      excludeSemantics: true,
      child: Container(
        height: 104,
        padding: const EdgeInsets.symmetric(horizontal: Space.s),
        decoration: BoxDecoration(
          color: bg,
          borderRadius: BorderRadius.circular(28),
        ),
        child: Column(
          mainAxisAlignment: MainAxisAlignment.center,
          children: [
            FittedBox(
              child: Text(
                value,
                style: SomanType.title.copyWith(
                  color: fg,
                  fontSize: 32,
                  height: 1,
                ),
              ),
            ),
            const SizedBox(height: 6),
            Text(
              label,
              textAlign: TextAlign.center,
              maxLines: 2,
              style: SomanType.label.copyWith(color: fg),
            ),
          ],
        ),
      ),
    );
  }
}
