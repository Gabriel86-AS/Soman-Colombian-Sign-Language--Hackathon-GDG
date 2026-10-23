import 'package:flutter/material.dart';
import 'package:flutter_riverpod/flutter_riverpod.dart';

import 'app/router.dart';
import 'app/theme/soman_theme.dart';

class SomanApp extends ConsumerWidget {
  const SomanApp({super.key});

  @override
  Widget build(BuildContext context, WidgetRef ref) {
    return MaterialApp.router(
      title: 'Soman',
      debugShowCheckedModeBanner: false,
      theme: SomanTheme.light,
      darkTheme: SomanTheme.dark,
      themeMode: ref.watch(themeModeProvider),
      routerConfig: router,
    );
  }
}
