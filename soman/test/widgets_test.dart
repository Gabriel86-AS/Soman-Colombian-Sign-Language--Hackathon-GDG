import 'package:flutter/material.dart';
import 'package:flutter_test/flutter_test.dart';
import 'package:soman/app/theme/soman_theme.dart';
import 'package:soman/features/debug/components_gallery.dart';
import 'package:soman/ui/ui.dart';

Widget _app(ThemeData theme, Widget child) => MaterialApp(
      theme: theme,
      home: Scaffold(body: SingleChildScrollView(child: child)),
    );

void main() {
  for (final (mode, theme) in [
    ('light', SomanTheme.light),
    ('dark', SomanTheme.dark),
  ]) {
    testWidgets('all components render without overflow ($mode)',
        (tester) async {
      tester.view.physicalSize = const Size(390, 844) * 3;
      tester.view.devicePixelRatio = 3;
      addTearDown(tester.view.reset);

      await tester.pumpWidget(_app(theme, const ComponentsGallery()));
      await tester.pumpAndSettle();
      expect(tester.takeException(), isNull);
    });
  }

  testWidgets('nav bar: every tab has a spoken label, one is selected',
      (tester) async {
    var tapped = -1;
    await tester.pumpWidget(_app(
      SomanTheme.light,
      FloatingNavBar(currentIndex: 0, onTap: (i) => tapped = i),
    ));
    for (final t in somanTabs) {
      expect(find.bySemanticsLabel(t.label), findsOneWidget);
    }
    await tester.tap(find.bySemanticsLabel('Progreso'));
    expect(tapped, 2);
  });

  testWidgets('locked lesson node does not react to taps', (tester) async {
    var taps = 0;
    await tester.pumpWidget(_app(
      SomanTheme.light,
      LessonNode(state: NodeState.locked, label: 'Adiós', onTap: () => taps++),
    ));
    await tester.tap(find.byType(LessonNode));
    expect(taps, 0);
    expect(find.bySemanticsLabel('Adiós, bloqueada'), findsOneWidget);
  });

  testWidgets('lives pill says cups, not hearts', (tester) async {
    await tester.pumpWidget(_app(SomanTheme.light, const LivesPill(lives: 2)));
    expect(find.bySemanticsLabel('2 tazas de tinto'), findsOneWidget);
  });
}
