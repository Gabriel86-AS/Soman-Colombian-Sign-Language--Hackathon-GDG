import 'package:flutter/material.dart';

import '../../app/theme/soman_colors.dart';
import '../../app/theme/soman_type.dart';
import '../../app/theme/spacing.dart';
import '../../ui/ui.dart';

/// Every shared widget with live demo state. Shown in `/debug/gallery`.
class ComponentsGallery extends StatefulWidget {
  const ComponentsGallery({super.key});

  @override
  State<ComponentsGallery> createState() => _ComponentsGalleryState();
}

class _ComponentsGalleryState extends State<ComponentsGallery> {
  int _speed = 0; // 0 lento, 1 normal
  bool _mirror = false;
  int _tab = 0;
  int _lives = 3;
  int _xp = 120;
  int _done = 3;
  double _hold = 0.4;
  double _precision = 0.82;

  void _say(String s) => ScaffoldMessenger.of(context)
    ..hideCurrentSnackBar()
    ..showSnackBar(SnackBar(content: Text(s), duration: Motion.slow * 4));

  @override
  Widget build(BuildContext context) {
    final c = context.colors;
    return Column(
      crossAxisAlignment: CrossAxisAlignment.stretch,
      children: [
        const _Label('LOGO'),
        const Row(children: [SomanLogo(small: true), Spacer()]),

        const _Label('BOTONES'),
        PrimaryPillButton(
          label: '¡Practicar ahora!',
          icon: SomanIcons.camera,
          onPressed: () => _say('Primario'),
        ),
        const SizedBox(height: Space.m),
        PrimaryPillButton(
          label: 'Continuar',
          onPressed: () => _say('Continuar'),
        ),
        const SizedBox(height: Space.m),
        const PrimaryPillButton(label: 'Desactivado', onPressed: null),
        const SizedBox(height: Space.m),
        Wrap(
          spacing: Space.s,
          runSpacing: Space.s,
          children: [
            SecondaryPillButton(
              label: 'Pista',
              icon: SomanIcons.bulb,
              onPressed: () {},
            ),
            SecondaryPillButton(label: 'Saltar', onPressed: () {}),
            SecondaryPillButton(
              label: 'Explorar categorías',
              icon: SomanIcons.grid,
              height: 40,
              onPressed: () {},
            ),
          ],
        ),
        const SizedBox(height: Space.m),
        Row(
          children: [
            CircleIconButton(
              icon: SomanIcons.back,
              semanticLabel: 'Volver',
              onPressed: () => _say('Volver'),
            ),
            const SizedBox(width: Space.s),
            CircleIconButton(
              icon: SomanIcons.close,
              semanticLabel: 'Cerrar práctica',
              onPressed: () {},
            ),
            const SizedBox(width: Space.s),
            CircleIconButton(
              icon: SomanIcons.bookmark,
              semanticLabel: 'Guardar lección',
              onPressed: () {},
            ),
          ],
        ),

        const _Label('CHIPS Y PASTILLAS'),
        Wrap(
          spacing: Space.s,
          runSpacing: Space.s,
          children: [
            SomanChip(
              label: 'Lento',
              selected: _speed == 0,
              onSelected: (_) => setState(() => _speed = 0),
            ),
            SomanChip(
              label: 'Normal',
              selected: _speed == 1,
              onSelected: (_) => setState(() => _speed = 1),
            ),
            SomanChip(
              label: 'Espejo',
              icon: SomanIcons.mirror,
              selected: _mirror,
              onSelected: (v) => setState(() => _mirror = v),
            ),
          ],
        ),
        const SizedBox(height: Space.m),
        Row(
          children: [
            Text('Hola', style: SomanType.display.copyWith(color: c.text)),
            const SizedBox(width: 14),
            const SomanTag('Saludo'),
          ],
        ),
        const SizedBox(height: Space.m),
        Wrap(
          spacing: Space.s,
          runSpacing: Space.s,
          crossAxisAlignment: WrapCrossAlignment.center,
          children: [
            StatPill.streak(5),
            GestureDetector(
              onTap: () => setState(() => _xp += 10),
              child: StatPill.xp(_xp),
            ),
            GestureDetector(
              onTap: () =>
                  setState(() => _lives = _lives == 0 ? 3 : _lives - 1),
              child: LivesPill(lives: _lives),
            ),
            Text(
              '← toca: +XP, derramar taza',
              style: SomanType.label.copyWith(color: c.textMuted),
            ),
          ],
        ),

        const _Label('BARRA DE NAVEGACIÓN'),
        FloatingNavBar(
          currentIndex: _tab,
          onTap: (i) => setState(() => _tab = i),
        ),

        const _Label('TARJETA DE UNIDAD'),
        UnitCard(
          overline: 'Saludos',
          title: 'Primeros saludos',
          subtitle: 'Saluda, preséntate y despídete.',
          onGuide: () => _say('Guía'),
        ),

        const _Label('CAMINO DE LECCIONES'),
        LessonPath(
          stops: [
            PathStop(label: 'A', state: NodeState.done, onTap: () {}),
            PathStop(label: 'B', state: NodeState.done, onTap: () {}),
            PathStop(
              label: 'Hola',
              state: NodeState.current,
              onTap: () => _say('Empezar Hola'),
            ),
            const PathStop(label: 'Gracias', state: NodeState.locked),
            const PathStop(label: 'Adiós', state: NodeState.locked),
          ],
        ),

        const _Label('CATEGORÍAS'),
        GridView.count(
          crossAxisCount: 2,
          shrinkWrap: true,
          physics: const NeverScrollableScrollPhysics(),
          mainAxisSpacing: Space.cardGap,
          crossAxisSpacing: Space.cardGap,
          childAspectRatio: 1.6,
          children: [
            CategoryTile(
              title: 'Abecedario',
              tone: SomanTone.primary,
              illustration: HandIllustration(
                size: 40,
                color: c.onPrimary,
                cuffColor: SomanPalette.cream,
              ),
              onTap: () {},
            ),
            CategoryTile(
              title: 'Números',
              tone: SomanTone.surface,
              illustration: const HandIllustration(size: 40),
              onTap: () {},
            ),
            CategoryTile(
              title: 'Saludos',
              tone: SomanTone.surfaceAlt,
              illustration: const CoffeeMug(size: 40),
              onTap: () {},
            ),
            CategoryTile(
              title: 'Repaso',
              tone: SomanTone.accent,
              illustration: const HandIllustration(size: 40),
              onTap: () {},
            ),
          ],
        ),

        const _Label('CONSEJO'),
        const TipCard(text: 'Mantén una expresión facial amable y natural.'),

        const _Label('ESTADÍSTICAS'),
        const Row(
          children: [
            Expanded(
              child: StatCard(value: '+20', label: 'XP ganados'),
            ),
            SizedBox(width: 10),
            Expanded(
              child: StatCard(
                value: '6',
                label: 'días de racha',
                tone: SomanTone.accent,
              ),
            ),
            SizedBox(width: 10),
            Expanded(
              child: StatCard(
                value: '92%',
                label: 'precisión',
                tone: SomanTone.surface,
              ),
            ),
          ],
        ),

        const _Label('PROGRESO DE LECCIÓN'),
        Row(
          children: [
            Expanded(child: SegmentProgress(total: 6, done: _done)),
            const SizedBox(width: Space.m),
            LivesPill(lives: _lives),
          ],
        ),
        Slider(
          value: _done.toDouble(),
          max: 6,
          divisions: 6,
          onChanged: (v) => setState(() => _done = v.round()),
        ),

        const _Label('CÁMARA'),
        CameraPanel(landmarks: demoHandLandmarks, precision: _precision),
        Slider(
          value: _precision,
          onChanged: (v) => setState(() => _precision = v),
        ),

        const _Label('ANILLO DE SOSTENER'),
        Center(
          child: HoldCheckButton(
            progress: _hold,
            onPressed: () => _say('Comprobar'),
          ),
        ),
        Slider(value: _hold, onChanged: (v) => setState(() => _hold = v)),

        const _Label('ILUSTRACIONES'),
        SizedBox(
          height: 120,
          child: Stack(
            clipBehavior: Clip.hardEdge,
            children: [
              const Positioned.fill(child: ConfettiDots()),
              Positioned(
                right: -40,
                bottom: -60,
                child: DecorCircle(size: 140, color: c.accent),
              ),
              const Positioned(
                left: 0,
                top: 10,
                child: Row(
                  children: [
                    HandIllustration(size: 64),
                    SizedBox(width: Space.l),
                    CoffeeMug(),
                    SizedBox(width: Space.l),
                    CoffeeBranch(),
                    SizedBox(width: Space.l),
                    NuaAvatar(size: 60),
                  ],
                ),
              ),
            ],
          ),
        ),
      ],
    );
  }
}

class _Label extends StatelessWidget {
  const _Label(this.text);
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
