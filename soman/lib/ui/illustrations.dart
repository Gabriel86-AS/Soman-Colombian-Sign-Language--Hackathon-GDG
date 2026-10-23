import 'package:flutter/material.dart';
import 'package:flutter_svg/flutter_svg.dart';

import '../app/theme/soman_colors.dart';
import '../app/theme/spacing.dart';

String _hex(Color c) =>
    '#${(c.toARGB32() & 0xFFFFFF).toRadixString(16).padLeft(6, '0')}';

Widget _svg(String svg, double w, double h, String? label) {
  final pic = SvgPicture.string(svg, width: w, height: h);
  if (label == null) return ExcludeSemantics(child: pic);
  return Semantics(label: label, image: true, child: pic);
}

/// Flat open hand from the canvas (fingers + cuff).
class HandIllustration extends StatelessWidget {
  const HandIllustration({
    super.key,
    this.size = 64,
    this.color,
    this.cuffColor,
    this.semanticLabel,
  });

  final double size;
  final Color? color;
  final Color? cuffColor;
  final String? semanticLabel;

  @override
  Widget build(BuildContext context) {
    final c = context.colors;
    final f = _hex(color ?? c.onSurface);
    final k = _hex(cuffColor ?? c.primary);
    return _svg(
      '<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 64 64">'
      '<rect x="17" y="30" width="33" height="25" rx="12" fill="$f"/>'
      '<rect x="18" y="12" width="7.5" height="28" rx="3.75" fill="$f"/>'
      '<rect x="26.5" y="7" width="7.5" height="32" rx="3.75" fill="$f"/>'
      '<rect x="35" y="9" width="7.5" height="30" rx="3.75" fill="$f"/>'
      '<rect x="43.5" y="16" width="7" height="24" rx="3.5" fill="$f"/>'
      '<rect x="9" y="31" width="7.5" height="20" rx="3.75" fill="$f" transform="rotate(-32 12.75 41)"/>'
      '<rect x="22" y="51" width="23" height="11" rx="4" fill="$k"/>'
      '</svg>',
      size,
      size,
      semanticLabel,
    );
  }
}

/// Coffee mug with steam (teal mug, ochre saucer).
class CoffeeMug extends StatelessWidget {
  const CoffeeMug({super.key, this.size = 58, this.semanticLabel});

  final double size;
  final String? semanticLabel;

  @override
  Widget build(BuildContext context) {
    final steam = _hex(context.colors.text);
    final mug = _hex(SomanPalette.teal);
    final saucer = _hex(SomanPalette.ochre);
    return _svg(
      '<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 64 64">'
      '<path d="M24 17c-3-3 3-5 0-9M33 17c-3-3 3-5 0-9" stroke="$steam" stroke-width="3" fill="none" stroke-linecap="round"/>'
      '<rect x="14" y="21" width="28" height="29" rx="9" fill="$mug"/>'
      '<circle cx="44" cy="34" r="7" fill="none" stroke="$mug" stroke-width="4.5"/>'
      '<rect x="9" y="52" width="40" height="5" rx="2.5" fill="$saucer"/>'
      '</svg>',
      size,
      size,
      semanticLabel,
    );
  }
}

/// Coffee branch with leaves and teal beans.
class CoffeeBranch extends StatelessWidget {
  const CoffeeBranch({super.key, this.height = 90, this.semanticLabel});

  final double height;
  final String? semanticLabel;

  @override
  Widget build(BuildContext context) {
    final r = _hex(SomanPalette.red);
    final o = _hex(SomanPalette.ochre);
    final t = _hex(SomanPalette.teal);
    return _svg(
      '<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 70 90">'
      '<path d="M35 88C35 60 30 35 38 6" stroke="$r" stroke-width="4" stroke-linecap="round" fill="none"/>'
      '<ellipse cx="22" cy="30" rx="15" ry="7" transform="rotate(-30 22 30)" fill="$o"/>'
      '<ellipse cx="52" cy="46" rx="15" ry="7" transform="rotate(30 52 46)" fill="$o"/>'
      '<ellipse cx="20" cy="62" rx="14" ry="6.5" transform="rotate(-25 20 62)" fill="$r"/>'
      '<circle cx="44" cy="20" r="5.5" fill="$t"/>'
      '<circle cx="48" cy="29" r="5.5" fill="$t"/>'
      '<circle cx="42" cy="70" r="5" fill="$t"/>'
      '</svg>',
      height * 70 / 90,
      height,
      semanticLabel,
    );
  }
}

/// Big flat circle meant to bleed off a screen or card corner. Place it with
/// a [Positioned] using negative offsets.
class DecorCircle extends StatelessWidget {
  const DecorCircle({super.key, required this.size, this.color});

  final double size;
  final Color? color;

  @override
  Widget build(BuildContext context) => ExcludeSemantics(
    child: Container(
      width: size,
      height: size,
      decoration: BoxDecoration(
        color: color ?? context.colors.accent,
        shape: BoxShape.circle,
      ),
    ),
  );
}

/// Scattered confetti dots and dashes (completion screen). Fills its parent.
class ConfettiDots extends StatelessWidget {
  const ConfettiDots({super.key});

  // x, y as fractions of the box; w, h in px; rotation in degrees.
  static const _bits = [
    (0.09, 0.10, 14.0, 14.0, 0.0, 0),
    (0.85, 0.18, 26.0, 9.0, 35.0, 1),
    (0.15, 0.85, 22.0, 8.0, -30.0, 2),
    (0.82, 0.82, 12.0, 12.0, 0.0, 2),
    (0.77, 0.02, 10.0, 10.0, 0.0, 0),
    (0.06, 0.52, 24.0, 8.0, 60.0, 1),
    (0.88, 0.55, 16.0, 16.0, 0.0, 3),
  ];

  @override
  Widget build(BuildContext context) {
    final c = context.colors;
    final palette = [c.primary, c.inverse, c.accent, SomanPalette.red];
    return ExcludeSemantics(
      child: LayoutBuilder(
        builder: (context, box) {
          return Stack(
            children: [
              for (final (x, y, w, h, deg, color) in _bits)
                Positioned(
                  left: x * box.maxWidth,
                  top: y * box.maxHeight,
                  child: Transform.rotate(
                    angle: deg * 3.1415926 / 180,
                    child: Container(
                      width: w,
                      height: h,
                      decoration: BoxDecoration(
                        color: palette[color],
                        borderRadius: BorderRadius.circular(Radii.pill),
                      ),
                    ),
                  ),
                ),
            ],
          );
        },
      ),
    );
  }
}

/// Nua's face in a circle (tip card, coach sheet). Crops the mascot image
/// the same way the canvas does.
class NuaAvatar extends StatelessWidget {
  const NuaAvatar({super.key, this.size = 52, this.background});

  final double size;
  final Color? background;

  @override
  Widget build(BuildContext context) {
    final c = context.colors;
    return ExcludeSemantics(
      child: Container(
        width: size,
        height: size,
        clipBehavior: Clip.antiAlias,
        decoration: BoxDecoration(
          color: background ?? c.primary,
          shape: BoxShape.circle,
        ),
        child: OverflowBox(
          alignment: Alignment.topLeft,
          maxWidth: double.infinity,
          maxHeight: double.infinity,
          child: Transform.translate(
            offset: Offset(-size * 0.73, -size * 0.17),
            child: Image.asset(c.nuaImage, width: size * 2.38),
          ),
        ),
      ),
    );
  }
}
