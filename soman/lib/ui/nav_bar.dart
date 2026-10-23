import 'package:flutter/material.dart';

import '../app/theme/soman_colors.dart';
import '../app/theme/soman_type.dart';
import '../app/theme/spacing.dart';
import 'buttons.dart';
import 'soman_icons.dart';

class NavItem {
  const NavItem(this.icon, this.label);
  final SomanIconData icon;
  final String label;
}

/// The four tabs, in order.
const somanTabs = [
  NavItem(SomanIcons.home, 'Inicio'),
  NavItem(SomanIcons.camera, 'Practicar'),
  NavItem(SomanIcons.chart, 'Progreso'),
  NavItem(SomanIcons.user, 'Perfil'),
];

/// Floating red pill, 66 high, inset 16 from the screen edges. The active
/// tab is an ochre pill with icon and label; the others show only the icon.
class FloatingNavBar extends StatelessWidget {
  const FloatingNavBar({
    super.key,
    required this.currentIndex,
    required this.onTap,
    this.items = somanTabs,
  });

  final int currentIndex;
  final ValueChanged<int> onTap;
  final List<NavItem> items;

  @override
  Widget build(BuildContext context) {
    final c = context.colors;
    return SafeArea(
      top: false,
      minimum: const EdgeInsets.all(Space.navInset),
      child: Semantics(
        container: true,
        label: 'Navegación principal',
        child: Container(
          height: 66,
          padding: const EdgeInsets.only(left: 10, right: 14),
          decoration: BoxDecoration(
            color: c.navBar,
            borderRadius: BorderRadius.circular(Radii.pill),
            boxShadow: floatShadow(0.3),
          ),
          child: Row(
            mainAxisAlignment: MainAxisAlignment.spaceBetween,
            children: [
              for (var i = 0; i < items.length; i++)
                _Tab(
                  item: items[i],
                  active: i == currentIndex,
                  onTap: () => onTap(i),
                ),
            ],
          ),
        ),
      ),
    );
  }
}

class _Tab extends StatelessWidget {
  const _Tab({required this.item, required this.active, required this.onTap});

  final NavItem item;
  final bool active;
  final VoidCallback onTap;

  @override
  Widget build(BuildContext context) {
    final c = context.colors;
    final fg = active ? c.onPrimary : c.onSurface;
    return Semantics(
      button: true,
      selected: active,
      label: item.label,
      excludeSemantics: true,
      child: Material(
        color: Colors.transparent,
        shape: const StadiumBorder(),
        clipBehavior: Clip.antiAlias,
        child: InkWell(
          onTap: onTap,
          child: AnimatedContainer(
            duration: Motion.normal,
            curve: Motion.curve,
            height: 46,
            constraints: const BoxConstraints(minWidth: 46),
            padding: EdgeInsets.symmetric(horizontal: active ? 18 : 12),
            decoration: BoxDecoration(
              color: active ? c.primary : c.primary.withValues(alpha: 0),
              borderRadius: BorderRadius.circular(Radii.pill),
            ),
            child: AnimatedSize(
              duration: Motion.normal,
              curve: Motion.curve,
              child: Row(
                mainAxisSize: MainAxisSize.min,
                children: [
                  SomanIcon(item.icon, size: active ? 20 : 22, color: fg),
                  if (active) ...[
                    const SizedBox(width: 8),
                    Text(
                      item.label,
                      style: SomanType.label.copyWith(color: fg, fontSize: 14),
                    ),
                  ],
                ],
              ),
            ),
          ),
        ),
      ),
    );
  }
}
