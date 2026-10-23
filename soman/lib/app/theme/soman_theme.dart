import 'package:flutter/material.dart';
import 'package:flutter_riverpod/flutter_riverpod.dart';

import 'soman_colors.dart';
import 'soman_type.dart';
import 'spacing.dart';

/// Builds the Material theme from [SomanColors], so stock widgets use the
/// Soman palette (no default purple) and nothing gets a border.
abstract final class SomanTheme {
  static final light = _build(Brightness.light, SomanColors.light);
  static final dark = _build(Brightness.dark, SomanColors.dark);

  static ThemeData _build(Brightness brightness, SomanColors c) {
    final scheme = ColorScheme(
      brightness: brightness,
      primary: c.primary,
      onPrimary: c.onPrimary,
      secondary: c.surfaceAlt,
      onSecondary: c.onSurfaceAlt,
      tertiary: c.accent,
      onTertiary: c.onAccent,
      error: SomanPalette.red,
      onError: SomanPalette.cream,
      surface: c.background,
      onSurface: c.text,
      onSurfaceVariant: c.textMuted,
      surfaceContainerLowest: c.background,
      surfaceContainerLow: c.background,
      surfaceContainer: c.background,
      surfaceContainerHigh: c.background,
      surfaceContainerHighest: c.background,
      primaryContainer: c.primary,
      onPrimaryContainer: c.onPrimary,
      secondaryContainer: c.surface,
      onSecondaryContainer: c.onSurface,
      outline: Colors.transparent,
      outlineVariant: Colors.transparent,
      surfaceTint: Colors.transparent,
      shadow: SomanPalette.wine,
    );

    const stadium = StadiumBorder();
    final buttonText = WidgetStatePropertyAll(SomanType.button);

    return ThemeData(
      useMaterial3: true,
      brightness: brightness,
      colorScheme: scheme,
      scaffoldBackgroundColor: c.background,
      canvasColor: c.background,
      fontFamily: 'Jost',
      textTheme: SomanType.textTheme(c.text),
      extensions: [c],
      splashFactory: InkSparkle.splashFactory,
      dividerTheme: const DividerThemeData(color: Colors.transparent, space: 0),
      appBarTheme: AppBarTheme(
        backgroundColor: c.background,
        foregroundColor: c.text,
        elevation: 0,
        scrolledUnderElevation: 0,
        centerTitle: false,
        titleTextStyle: SomanType.title.copyWith(color: c.text),
      ),
      cardTheme: CardThemeData(
        color: c.surface,
        elevation: 0,
        margin: EdgeInsets.zero,
        shape: RoundedRectangleBorder(
          borderRadius: BorderRadius.circular(Radii.card),
        ),
      ),
      filledButtonTheme: FilledButtonThemeData(
        style: ButtonStyle(
          backgroundColor: WidgetStatePropertyAll(c.primary),
          foregroundColor: WidgetStatePropertyAll(c.onPrimary),
          textStyle: buttonText,
          shape: const WidgetStatePropertyAll(stadium),
          minimumSize: const WidgetStatePropertyAll(Size(64, 46)),
          elevation: const WidgetStatePropertyAll(0),
        ),
      ),
      elevatedButtonTheme: ElevatedButtonThemeData(
        style: ButtonStyle(
          backgroundColor: WidgetStatePropertyAll(c.surface),
          foregroundColor: WidgetStatePropertyAll(c.onSurface),
          textStyle: buttonText,
          shape: const WidgetStatePropertyAll(stadium),
          minimumSize: const WidgetStatePropertyAll(Size(64, 46)),
          elevation: const WidgetStatePropertyAll(0),
        ),
      ),
      textButtonTheme: TextButtonThemeData(
        style: ButtonStyle(
          foregroundColor: WidgetStatePropertyAll(c.text),
          textStyle: buttonText,
          shape: const WidgetStatePropertyAll(stadium),
        ),
      ),
      outlinedButtonTheme: OutlinedButtonThemeData(
        style: ButtonStyle(
          foregroundColor: WidgetStatePropertyAll(c.text),
          textStyle: buttonText,
          shape: const WidgetStatePropertyAll(stadium),
          side: const WidgetStatePropertyAll(BorderSide.none),
        ),
      ),
      chipTheme: ChipThemeData(
        backgroundColor: c.surface,
        selectedColor: c.primary,
        labelStyle: SomanType.label.copyWith(color: c.onSurface),
        side: BorderSide.none,
        shape: stadium,
      ),
      inputDecorationTheme: InputDecorationTheme(
        filled: true,
        fillColor: c.surface,
        border: OutlineInputBorder(
          borderRadius: BorderRadius.circular(Radii.pill),
          borderSide: BorderSide.none,
        ),
      ),
      bottomSheetTheme: BottomSheetThemeData(
        backgroundColor: c.sheet,
        modalBackgroundColor: c.sheet,
        surfaceTintColor: Colors.transparent,
        showDragHandle: true,
        dragHandleColor: c.onSheet.withValues(alpha: 0.3),
        shape: const RoundedRectangleBorder(
          borderRadius: BorderRadius.vertical(
            top: Radius.circular(Radii.sheet),
          ),
        ),
      ),
      dialogTheme: DialogThemeData(
        backgroundColor: c.sheet,
        surfaceTintColor: Colors.transparent,
        shape: RoundedRectangleBorder(
          borderRadius: BorderRadius.circular(Radii.card),
        ),
      ),
      snackBarTheme: SnackBarThemeData(
        backgroundColor: c.surface,
        contentTextStyle: SomanType.body.copyWith(color: c.onSurface),
        behavior: SnackBarBehavior.floating,
        shape: const StadiumBorder(),
      ),
      progressIndicatorTheme: ProgressIndicatorThemeData(
        color: c.primary,
        linearTrackColor: c.surface.withValues(alpha: 0.2),
      ),
      switchTheme: SwitchThemeData(
        thumbColor: WidgetStatePropertyAll(c.onPrimary),
        trackColor: WidgetStateProperty.resolveWith(
          (s) => s.contains(WidgetState.selected) ? c.primary : c.surface,
        ),
        trackOutlineColor: const WidgetStatePropertyAll(Colors.transparent),
      ),
      pageTransitionsTheme: const PageTransitionsTheme(
        builders: {
          TargetPlatform.android: FadeForwardsPageTransitionsBuilder(),
        },
      ),
    );
  }
}

/// System by default; Perfil can override it (saved in Phase H).
class ThemeModeNotifier extends Notifier<ThemeMode> {
  @override
  ThemeMode build() => ThemeMode.system;

  void set(ThemeMode mode) => state = mode;
}

final themeModeProvider = NotifierProvider<ThemeModeNotifier, ThemeMode>(
  ThemeModeNotifier.new,
);
