import 'package:flutter/material.dart';
import 'package:flutter_riverpod/flutter_riverpod.dart';

import '../../app/theme/soman_colors.dart';
import '../../app/theme/soman_theme.dart';
import '../../app/theme/soman_type.dart';
import '../../app/theme/spacing.dart';
import '../../ui/ui.dart';
import 'components_gallery.dart';

/// `/debug/gallery`: every widget, token and text style, in the current
/// mode. The switch at the top flips light and dark.
class GalleryScreen extends ConsumerStatefulWidget {
  const GalleryScreen({super.key});

  @override
  ConsumerState<GalleryScreen> createState() => _GalleryScreenState();
}

class _GalleryScreenState extends ConsumerState<GalleryScreen> {
  bool _components = true;

  @override
  Widget build(BuildContext context) {
    final c = context.colors;
    final isDark = Theme.of(context).brightness == Brightness.dark;

    return Scaffold(
      appBar: AppBar(
        title: const Text('Galería'),
        actions: [
          Text(isDark ? 'Oscuro' : 'Claro', style: SomanType.label),
          Switch(
            value: isDark,
            onChanged: (v) => ref
                .read(themeModeProvider.notifier)
                .set(v ? ThemeMode.dark : ThemeMode.light),
          ),
          const SizedBox(width: Space.s),
        ],
      ),
      body: ListView(
        padding: const EdgeInsets.fromLTRB(
          Space.screen,
          Space.s,
          Space.screen,
          Space.xxxl,
        ),
        children: [
          Wrap(
            spacing: Space.s,
            children: [
              SomanChip(
                label: 'Componentes',
                selected: _components,
                onSelected: (_) => setState(() => _components = true),
              ),
              SomanChip(
                label: 'Tokens',
                selected: !_components,
                onSelected: (_) => setState(() => _components = false),
              ),
            ],
          ),
          if (_components) const ComponentsGallery() else ..._tokens(c, isDark),
        ],
      ),
    );
  }

  List<Widget> _tokens(SomanColors c, bool isDark) => [
    const _Section('LOGO'),
    Text('SOMAN', style: SomanType.logo.copyWith(color: c.text)),
    Text('SOMAN', style: SomanType.logoSmall.copyWith(color: c.text)),
    const _Section('NUA'),
    Row(
      children: [
        Image.asset(c.nuaImage, height: 120),
        const SizedBox(width: Space.l),
        Expanded(
          child: Text(
            isDark ? 'Iguana amarilla (oscuro)' : 'Iguana verde (claro)',
            style: SomanType.body.copyWith(color: c.textMuted),
          ),
        ),
      ],
    ),
    const _Section('COLORES'),
    Wrap(
      spacing: Space.m,
      runSpacing: Space.m,
      children: [for (final e in c.asMap.entries) _Swatch(e.key, e.value)],
    ),
    const _Section('CONTRASTE (MÍN. 4.5)'),
    for (final (name, fg, bg) in c.textPairs) _ContrastRow(name, fg, bg),
    const _Section('TIPOGRAFÍA'),
    for (final e in SomanType.all.entries)
      if (!e.key.startsWith('logo'))
        Padding(
          padding: const EdgeInsets.only(bottom: Space.m),
          child: Column(
            crossAxisAlignment: CrossAxisAlignment.start,
            children: [
              Text(
                '${e.key} · ${e.value.fontSize!.toInt()}',
                style: SomanType.label.copyWith(color: c.textMuted),
              ),
              Text(
                e.key == 'overline' ? 'HAZ LA SEÑA' : 'Hola, soy Nua',
                style: e.value.copyWith(color: c.text),
              ),
            ],
          ),
        ),
    const _Section('ESPACIADO'),
    for (final s in Space.scale)
      Padding(
        padding: const EdgeInsets.only(bottom: Space.xs),
        child: Row(
          children: [
            SizedBox(
              width: 32,
              child: Text(
                '${s.toInt()}',
                style: SomanType.label.copyWith(color: c.textMuted),
              ),
            ),
            Container(
              width: s * 6,
              height: 12,
              decoration: BoxDecoration(
                color: c.primary,
                borderRadius: BorderRadius.circular(Radii.pill),
              ),
            ),
          ],
        ),
      ),
    const _Section('ESQUINAS'),
    Wrap(
      spacing: Space.m,
      runSpacing: Space.m,
      children: [
        for (final e in Radii.all.entries)
          Container(
            width: 96,
            height: 72,
            alignment: Alignment.center,
            decoration: BoxDecoration(
              color: c.surface,
              borderRadius: BorderRadius.circular(e.value),
            ),
            child: Text(
              '${e.key}\n${e.value.toInt()}',
              textAlign: TextAlign.center,
              style: SomanType.label.copyWith(color: c.onSurface),
            ),
          ),
      ],
    ),
    const _Section('MATERIAL (NO DEBE VERSE MORADO)'),
    Wrap(
      spacing: Space.m,
      runSpacing: Space.m,
      crossAxisAlignment: WrapCrossAlignment.center,
      children: [
        FilledButton(onPressed: () {}, child: const Text('Continuar')),
        ElevatedButton(onPressed: () {}, child: const Text('Saltar')),
        TextButton(onPressed: () {}, child: const Text('Pista')),
        const SizedBox(width: 120, child: LinearProgressIndicator(value: .6)),
      ],
    ),
  ];
}

class _Section extends StatelessWidget {
  const _Section(this.text);
  final String text;

  @override
  Widget build(BuildContext context) => Padding(
    padding: const EdgeInsets.only(top: Space.xxxl, bottom: Space.m),
    child: Text(
      text,
      style: SomanType.overline.copyWith(color: context.colors.textMuted),
    ),
  );
}

class _Swatch extends StatelessWidget {
  const _Swatch(this.name, this.color);
  final String name;
  final Color color;

  @override
  Widget build(BuildContext context) {
    final c = context.colors;
    final hex = color
        .toARGB32()
        .toRadixString(16)
        .toUpperCase()
        .padLeft(8, '0');
    return SizedBox(
      width: 100,
      child: Column(
        crossAxisAlignment: CrossAxisAlignment.start,
        children: [
          Container(
            height: 56,
            decoration: BoxDecoration(
              color: color,
              borderRadius: BorderRadius.circular(Radii.tile),
              boxShadow: [
                BoxShadow(
                  color: SomanPalette.wine.withValues(alpha: 0.15),
                  blurRadius: 8,
                  offset: const Offset(0, 3),
                ),
              ],
            ),
          ),
          const SizedBox(height: Space.xs),
          Text(name, style: SomanType.label.copyWith(color: c.text)),
          Text(
            '#$hex',
            style: SomanType.label.copyWith(
              color: c.textMuted,
              fontWeight: FontWeight.w400,
            ),
          ),
        ],
      ),
    );
  }
}

class _ContrastRow extends StatelessWidget {
  const _ContrastRow(this.name, this.fg, this.bg);
  final String name;
  final Color fg;
  final Color bg;

  @override
  Widget build(BuildContext context) {
    final ratio = contrastRatio(fg, bg);
    final ok = ratio >= 4.5;
    return Container(
      margin: const EdgeInsets.only(bottom: Space.s),
      padding: const EdgeInsets.symmetric(
        horizontal: Space.l,
        vertical: Space.m,
      ),
      decoration: BoxDecoration(
        color: bg,
        borderRadius: BorderRadius.circular(Radii.pill),
      ),
      child: Row(
        children: [
          Expanded(
            child: Text(name, style: SomanType.body.copyWith(color: fg)),
          ),
          Text(
            '${ratio.toStringAsFixed(1)}  ${ok ? 'OK' : 'BAJO'}',
            style: SomanType.label.copyWith(color: fg),
          ),
        ],
      ),
    );
  }
}
