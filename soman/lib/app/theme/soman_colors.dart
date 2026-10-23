import 'package:flutter/material.dart';

/// The five DESIGN.md colors. Widgets should read [SomanColors] from the
/// theme instead of using these directly, so light and dark both work.
abstract final class SomanPalette {
  static const cream = Color(0xFFFFF3B0);
  static const wine = Color(0xFF540B0E);
  static const red = Color(0xFF9E2A2B);
  static const ochre = Color(0xFFE09F3E);
  static const teal = Color(0xFF335C67); // accent only, never a background
}

/// Role-based colors for one theme mode (DESIGN.md front matter).
@immutable
class SomanColors extends ThemeExtension<SomanColors> {
  const SomanColors({
    required this.background,
    required this.surface,
    required this.onSurface,
    required this.surfaceAlt,
    required this.onSurfaceAlt,
    required this.primary,
    required this.onPrimary,
    required this.text,
    required this.textMuted,
    required this.navBar,
    required this.accent,
    required this.onAccent,
    required this.inverse,
    required this.onInverse,
    required this.pill,
    required this.onPill,
    required this.sheet,
    required this.onSheet,
    required this.nodeShadowDone,
    required this.nodeShadowCurrent,
    required this.nodeShadowLocked,
    required this.nuaImage,
  });

  final Color background;
  final Color surface;
  final Color onSurface;
  final Color surfaceAlt;
  final Color onSurfaceAlt;
  final Color primary;
  final Color onPrimary;
  final Color text;
  final Color textMuted;
  final Color navBar;
  final Color accent;
  final Color onAccent;

  /// Opposite of the background: wine in light, cream in dark. Current
  /// lesson node, "Empezar" bubble, selected chip.
  final Color inverse;
  final Color onInverse;

  /// Red in both modes: chips, lives and streak pills.
  final Color pill;
  final Color onPill;

  /// Bottom sheets (coach sheet) are cream in both modes.
  final Color sheet;
  final Color onSheet;

  /// Solid `0 7px 0` offset shadow under lesson nodes.
  final Color nodeShadowDone;
  final Color nodeShadowCurrent;
  final Color nodeShadowLocked;

  /// Nua: green iguana in light mode, yellow in dark mode.
  final String nuaImage;

  static const light = SomanColors(
    background: SomanPalette.cream,
    surface: SomanPalette.wine,
    onSurface: SomanPalette.cream,
    surfaceAlt: SomanPalette.red,
    onSurfaceAlt: SomanPalette.cream,
    primary: SomanPalette.ochre,
    onPrimary: SomanPalette.wine,
    text: SomanPalette.wine,
    textMuted: Color(0xC7540B0E), // wine 78%
    navBar: SomanPalette.red,
    accent: SomanPalette.teal,
    onAccent: SomanPalette.cream,
    inverse: SomanPalette.wine,
    onInverse: SomanPalette.cream,
    pill: SomanPalette.red,
    onPill: SomanPalette.cream,
    sheet: SomanPalette.cream,
    onSheet: SomanPalette.wine,
    nodeShadowDone: SomanPalette.teal,
    nodeShadowCurrent: SomanPalette.ochre,
    nodeShadowLocked: SomanPalette.wine,
    nuaImage: 'assets/images/nua_green.png',
  );

  static const dark = SomanColors(
    background: SomanPalette.wine,
    surface: SomanPalette.red,
    onSurface: SomanPalette.cream,
    surfaceAlt: SomanPalette.cream,
    onSurfaceAlt: SomanPalette.wine, // cream card needs wine text
    primary: SomanPalette.ochre,
    onPrimary: SomanPalette.wine,
    text: SomanPalette.cream,
    textMuted: Color(0xD1FFF3B0), // cream 82%
    navBar: SomanPalette.red,
    accent: SomanPalette.teal,
    onAccent: SomanPalette.cream,
    inverse: SomanPalette.cream,
    onInverse: SomanPalette.wine,
    pill: SomanPalette.red,
    onPill: SomanPalette.cream,
    sheet: SomanPalette.cream,
    onSheet: SomanPalette.wine,
    nodeShadowDone: SomanPalette.teal,
    nodeShadowCurrent: SomanPalette.ochre,
    nodeShadowLocked: Color(0xFF3A0709), // wine, darker so it shows on wine
    nuaImage: 'assets/images/nua_yellow.png',
  );

  /// Every token by name, for the gallery and tests.
  Map<String, Color> get asMap => {
    'background': background,
    'surface': surface,
    'onSurface': onSurface,
    'surfaceAlt': surfaceAlt,
    'onSurfaceAlt': onSurfaceAlt,
    'primary': primary,
    'onPrimary': onPrimary,
    'text': text,
    'textMuted': textMuted,
    'navBar': navBar,
    'accent': accent,
    'onAccent': onAccent,
    'inverse': inverse,
    'onInverse': onInverse,
    'pill': pill,
    'onPill': onPill,
    'sheet': sheet,
    'onSheet': onSheet,
    'nodeShadowDone': nodeShadowDone,
    'nodeShadowCurrent': nodeShadowCurrent,
    'nodeShadowLocked': nodeShadowLocked,
  };

  /// Text/background pairs that must reach 4.5:1 contrast.
  List<(String, Color, Color)> get textPairs => [
    ('text / background', text, background),
    ('textMuted / background', textMuted, background),
    ('onSurface / surface', onSurface, surface),
    ('onSurfaceAlt / surfaceAlt', onSurfaceAlt, surfaceAlt),
    ('onPrimary / primary', onPrimary, primary),
    ('onAccent / accent', onAccent, accent),
    ('onInverse / inverse', onInverse, inverse),
    ('onPill / pill', onPill, pill),
    ('onSheet / sheet', onSheet, sheet),
    ('onSurface / navBar', onSurface, navBar),
  ];

  @override
  SomanColors copyWith({
    Color? background,
    Color? surface,
    Color? onSurface,
    Color? surfaceAlt,
    Color? onSurfaceAlt,
    Color? primary,
    Color? onPrimary,
    Color? text,
    Color? textMuted,
    Color? navBar,
    Color? accent,
    Color? onAccent,
    Color? inverse,
    Color? onInverse,
    Color? pill,
    Color? onPill,
    Color? sheet,
    Color? onSheet,
    Color? nodeShadowDone,
    Color? nodeShadowCurrent,
    Color? nodeShadowLocked,
    String? nuaImage,
  }) {
    return SomanColors(
      background: background ?? this.background,
      surface: surface ?? this.surface,
      onSurface: onSurface ?? this.onSurface,
      surfaceAlt: surfaceAlt ?? this.surfaceAlt,
      onSurfaceAlt: onSurfaceAlt ?? this.onSurfaceAlt,
      primary: primary ?? this.primary,
      onPrimary: onPrimary ?? this.onPrimary,
      text: text ?? this.text,
      textMuted: textMuted ?? this.textMuted,
      navBar: navBar ?? this.navBar,
      accent: accent ?? this.accent,
      onAccent: onAccent ?? this.onAccent,
      inverse: inverse ?? this.inverse,
      onInverse: onInverse ?? this.onInverse,
      pill: pill ?? this.pill,
      onPill: onPill ?? this.onPill,
      sheet: sheet ?? this.sheet,
      onSheet: onSheet ?? this.onSheet,
      nodeShadowDone: nodeShadowDone ?? this.nodeShadowDone,
      nodeShadowCurrent: nodeShadowCurrent ?? this.nodeShadowCurrent,
      nodeShadowLocked: nodeShadowLocked ?? this.nodeShadowLocked,
      nuaImage: nuaImage ?? this.nuaImage,
    );
  }

  @override
  SomanColors lerp(SomanColors? other, double t) {
    if (other == null) return this;
    Color l(Color a, Color b) => Color.lerp(a, b, t)!;
    return SomanColors(
      background: l(background, other.background),
      surface: l(surface, other.surface),
      onSurface: l(onSurface, other.onSurface),
      surfaceAlt: l(surfaceAlt, other.surfaceAlt),
      onSurfaceAlt: l(onSurfaceAlt, other.onSurfaceAlt),
      primary: l(primary, other.primary),
      onPrimary: l(onPrimary, other.onPrimary),
      text: l(text, other.text),
      textMuted: l(textMuted, other.textMuted),
      navBar: l(navBar, other.navBar),
      accent: l(accent, other.accent),
      onAccent: l(onAccent, other.onAccent),
      inverse: l(inverse, other.inverse),
      onInverse: l(onInverse, other.onInverse),
      pill: l(pill, other.pill),
      onPill: l(onPill, other.onPill),
      sheet: l(sheet, other.sheet),
      onSheet: l(onSheet, other.onSheet),
      nodeShadowDone: l(nodeShadowDone, other.nodeShadowDone),
      nodeShadowCurrent: l(nodeShadowCurrent, other.nodeShadowCurrent),
      nodeShadowLocked: l(nodeShadowLocked, other.nodeShadowLocked),
      nuaImage: t < 0.5 ? nuaImage : other.nuaImage,
    );
  }
}

/// WCAG contrast ratio. A see-through foreground is blended over [bg] first.
double contrastRatio(Color fg, Color bg) {
  final solid = Color.alphaBlend(fg, bg);
  final a = solid.computeLuminance();
  final b = bg.computeLuminance();
  final hi = a > b ? a : b;
  final lo = a > b ? b : a;
  return (hi + 0.05) / (lo + 0.05);
}

extension SomanThemeContext on BuildContext {
  SomanColors get colors => Theme.of(this).extension<SomanColors>()!;
}
