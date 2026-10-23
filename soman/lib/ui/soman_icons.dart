import 'package:flutter/widgets.dart';
import 'package:flutter_svg/flutter_svg.dart';

/// One icon from the canvas, as SVG markup in a 24×24 box.
@immutable
class SomanIconData {
  const SomanIconData(this.body, {this.filled = false, this.stroke = 1.9});

  /// Inner SVG elements, drawn in black and recolored at paint time.
  final String body;

  /// Filled shapes instead of strokes.
  final bool filled;
  final double stroke;

  String get svg => filled
      ? '<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 24 24" '
            'fill="#000">$body</svg>'
      : '<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 24 24" '
            'fill="none" stroke="#000" stroke-width="$stroke" '
            'stroke-linecap="round" stroke-linejoin="round">$body</svg>';
}

/// Icons taken from the Design canvas (stroke style, rounded joins).
abstract final class SomanIcons {
  static const home = SomanIconData(
    '<path d="M3 11l9-7 9 7v9a1 1 0 0 1-1 1h-5v-6h-6v6H4a1 1 0 0 1-1-1z"/>',
  );
  static const camera = SomanIconData(
    '<path d="M4 8h3l2-3h6l2 3h3v11H4zM12 16.5a3.5 3.5 0 1 0 0-7 3.5 3.5 0 0 0 0 7z"/>',
    stroke: 1.7,
  );
  static const chart = SomanIconData(
    '<path d="M5 20V11M11 20V5M17 20v-6M3 20h18"/>',
    stroke: 1.7,
  );
  static const user = SomanIconData(
    '<path d="M12 12a4 4 0 1 0 0-8 4 4 0 0 0 0 8zM4.5 21a7.5 7.5 0 0 1 15 0"/>',
    stroke: 1.7,
  );
  static const back = SomanIconData(
    '<path d="M19 12H5M11 6l-6 6 6 6"/>',
    stroke: 2,
  );
  static const close = SomanIconData(
    '<path d="M6 6l12 12M18 6L6 18"/>',
    stroke: 2,
  );
  static const check = SomanIconData(
    '<path d="M5 12.5l4.5 4.5L19 7.5"/>',
    stroke: 2.6,
  );
  static const arrowRight = SomanIconData(
    '<path d="M5 12h14M13 6l6 6-6 6"/>',
    stroke: 2,
  );
  static const lock = SomanIconData(
    '<rect x="5" y="11" width="14" height="10" rx="3"/>'
    '<path d="M8 11V8a4 4 0 0 1 8 0v3"/>',
    stroke: 2,
  );
  static const book = SomanIconData(
    '<path d="M4 19V5a2 2 0 0 1 2-2h14v14H6a2 2 0 0 0-2 2zm0 0a2 2 0 0 0 2 2h14"/>',
  );
  static const grid = SomanIconData(
    '<rect x="4" y="4" width="7" height="7" rx="2"/>'
    '<rect x="13" y="4" width="7" height="7" rx="2"/>'
    '<rect x="4" y="13" width="7" height="7" rx="2"/>'
    '<rect x="13" y="13" width="7" height="7" rx="2"/>',
    stroke: 1.8,
  );
  static const bookmark = SomanIconData(
    '<path d="M6 3h12v18l-6-4-6 4z"/>',
    stroke: 1.8,
  );
  static const mirror = SomanIconData(
    '<path d="M12 3v18M8 7l-5 5 5 5M16 7l5 5-5 5"/>',
    stroke: 1.8,
  );
  static const bulb = SomanIconData(
    '<path d="M9 18h6M10 21h4M12 3a6 6 0 0 0-4 10.5c.7.7 1 1.5 1 2.5h6c0-1 .3-1.8 1-2.5A6 6 0 0 0 12 3z"/>',
    stroke: 1.8,
  );
  static const play = SomanIconData(
    '<path d="M8 5l11 7-11 7z"/>',
    filled: true,
  );
  static const flame = SomanIconData(
    '<path d="M12 3c.5 3.5 5 5.5 5 10.5a5 5 0 0 1-10 0c0-2.6 1.4-3.9 2.2-5.3.6 1.4 1.4 2 2.3 2.3C12 8 11.3 5.5 12 3z"/>',
    filled: true,
  );

  /// A cup of tinto: one life. Never hearts.
  static const cup = SomanIconData(
    '<path d="M3.5 10h13v4.5a5.5 5.5 0 0 1-5.5 5.5H9a5.5 5.5 0 0 1-5.5-5.5z"/>'
    '<path d="M16.5 11.5h1.2a2.6 2.6 0 0 1 0 5.2h-1.6" fill="none" stroke="#000" stroke-width="2"/>'
    '<path d="M8 3.5c-1.2 1.1 1.2 2.2 0 3.6M12 3.5c-1.2 1.1 1.2 2.2 0 3.6" fill="none" stroke="#000" stroke-width="1.6" stroke-linecap="round"/>',
    filled: true,
  );
}

/// Draws a [SomanIconData] in one color. Decorative unless [semanticLabel]
/// is given.
class SomanIcon extends StatelessWidget {
  const SomanIcon(
    this.icon, {
    super.key,
    this.size = 20,
    this.color,
    this.semanticLabel,
  });

  final SomanIconData icon;
  final double size;
  final Color? color;
  final String? semanticLabel;

  @override
  Widget build(BuildContext context) {
    final c = color ?? IconTheme.of(context).color ?? const Color(0xFF000000);
    final picture = SvgPicture.string(
      icon.svg,
      width: size,
      height: size,
      colorFilter: ColorFilter.mode(c, BlendMode.srcIn),
    );
    if (semanticLabel == null) return ExcludeSemantics(child: picture);
    return Semantics(label: semanticLabel, image: true, child: picture);
  }
}
