import 'package:flutter/material.dart';
import 'package:go_router/go_router.dart';

import '../features/debug/gallery_screen.dart';
import 'theme/soman_colors.dart';
import 'theme/soman_type.dart';
import 'theme/spacing.dart';

/// Routes grow in Phase D (welcome, tab shell, lessons).
final router = GoRouter(
  routes: [
    GoRoute(path: '/', builder: (_, _) => const _Placeholder()),
    GoRoute(path: '/debug/gallery', builder: (_, _) => const GalleryScreen()),
  ],
);

/// Temporary home until Phase D.
class _Placeholder extends StatelessWidget {
  const _Placeholder();

  @override
  Widget build(BuildContext context) {
    final c = context.colors;
    return Scaffold(
      body: Center(
        child: Column(
          mainAxisSize: MainAxisSize.min,
          children: [
            Image.asset(c.nuaImage, height: 160),
            const SizedBox(height: Space.xl),
            Text('SOMAN', style: SomanType.logo.copyWith(color: c.text)),
            const SizedBox(height: Space.xxl),
            FilledButton(
              onPressed: () => context.push('/debug/gallery'),
              child: const Text('Ver galería'),
            ),
          ],
        ),
      ),
    );
  }
}
