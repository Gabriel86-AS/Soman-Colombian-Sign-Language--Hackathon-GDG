import 'package:flutter_test/flutter_test.dart';
import 'package:soman/app/theme/soman_colors.dart';

void main() {
  for (final (mode, c) in [('light', SomanColors.light), ('dark', SomanColors.dark)]) {
    group('$mode tokens', () {
      for (final (name, fg, bg) in c.textPairs) {
        test('$name contrast >= 4.5', () {
          expect(contrastRatio(fg, bg), greaterThanOrEqualTo(4.5));
        });
      }

      test('no ochre text on cream', () {
        for (final (_, fg, bg) in c.textPairs) {
          expect(
            fg == SomanPalette.ochre && bg == SomanPalette.cream,
            isFalse,
          );
        }
      });

      test('teal is never the background', () {
        expect(c.background, isNot(SomanPalette.teal));
      });
    });
  }
}
