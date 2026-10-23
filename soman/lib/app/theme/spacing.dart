import 'package:flutter/animation.dart';

/// Spacing scale from DESIGN.md.
abstract final class Space {
  static const double xs = 4;
  static const double s = 8;
  static const double m = 12;
  static const double l = 16;
  static const double xl = 20;
  static const double xxl = 24;
  static const double xxxl = 32;

  static const double screen = 20; // side padding
  static const double cardGap = 14; // 14-16 between stacked cards
  static const double navInset = 16; // floating nav bar inset

  static const scale = [xs, s, m, l, xl, xxl, xxxl];
}

/// Corner radii. Nothing in the app is square.
abstract final class Radii {
  static const double tile = 24;
  static const double card = 30;
  static const double media = 34;
  static const double sheet = 34;
  static const double pill = 999;

  static const all = {
    'tile': tile,
    'card': card,
    'media': media,
    'sheet': sheet,
    'pill': pill,
  };
}

/// Motion is smooth and calm: 250-350 ms.
abstract final class Motion {
  static const fast = Duration(milliseconds: 250);
  static const normal = Duration(milliseconds: 300);
  static const slow = Duration(milliseconds: 350);
  static const curve = Curves.easeOutCubic;
}
