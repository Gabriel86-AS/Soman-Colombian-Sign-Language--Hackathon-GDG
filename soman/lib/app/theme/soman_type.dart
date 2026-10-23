import 'package:flutter/material.dart';

/// Text styles from DESIGN.md. Colors come from the theme, not from here.
///
/// Fraunces and Jost are variable fonts: weight and optical size are set with
/// [FontVariation]. Uppercase styles (logo, overline) need the caller to pass
/// uppercase text; [SomanLogo] and the overline helpers do that.
abstract final class SomanType {
  static const _fraunces = 'Fraunces';
  static const _jost = 'Jost';

  /// Only the SomanLogo widget may use this family.
  static const logoFamily = 'RegularAesthetic';

  static TextStyle _serif(double size, {double? height}) => TextStyle(
    fontFamily: _fraunces,
    fontFamilyFallback: const ['serif'],
    fontSize: size,
    fontStyle: FontStyle.italic,
    fontWeight: FontWeight.w400,
    height: height,
    fontVariations: [
      const FontVariation('wght', 400),
      FontVariation('opsz', size.clamp(9, 144).toDouble()),
    ],
  );

  static TextStyle _sans(
    double size,
    FontWeight weight, {
    double? height,
    double? letterSpacing,
  }) => TextStyle(
    fontFamily: _jost,
    fontSize: size,
    fontWeight: weight,
    height: height,
    letterSpacing: letterSpacing,
    fontVariations: [FontVariation('wght', weight.value.toDouble())],
  );

  static const logo = TextStyle(
    fontFamily: logoFamily,
    fontFamilyFallback: [_fraunces, 'serif'],
    fontSize: 96,
    fontWeight: FontWeight.w400,
    letterSpacing: 96 * 0.02,
    height: 1,
  );

  static const logoSmall = TextStyle(
    fontFamily: logoFamily,
    fontFamilyFallback: [_fraunces, 'serif'],
    fontSize: 28,
    fontWeight: FontWeight.w400,
    letterSpacing: 28 * 0.02,
    height: 1,
  );

  static final display = _serif(56, height: 1);
  static final headline = _serif(38, height: 1.05);
  static final title = _serif(24, height: 1.15);
  static final body = _sans(16, FontWeight.w400, height: 1.45);
  static final button = _sans(18, FontWeight.w600);
  static final label = _sans(13, FontWeight.w600);
  static final overline = _sans(11, FontWeight.w600, letterSpacing: 11 * 0.22);

  /// Every style by name, for the gallery.
  static Map<String, TextStyle> get all => {
    'logo': logo,
    'logoSmall': logoSmall,
    'display': display,
    'headline': headline,
    'title': title,
    'body': body,
    'button': button,
    'label': label,
    'overline': overline,
  };

  /// Material slots, so stock widgets pick up our fonts.
  static TextTheme textTheme(Color color) => TextTheme(
    displayLarge: display,
    displayMedium: headline,
    headlineLarge: headline,
    headlineMedium: headline,
    headlineSmall: title,
    titleLarge: title,
    titleMedium: button,
    titleSmall: label,
    bodyLarge: body,
    bodyMedium: body,
    bodySmall: label.copyWith(fontWeight: FontWeight.w400),
    labelLarge: button,
    labelMedium: label,
    labelSmall: overline,
  ).apply(bodyColor: color, displayColor: color);
}
