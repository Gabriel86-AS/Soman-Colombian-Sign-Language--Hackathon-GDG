# Soman · app Flutter

App Android de Soman para aprender Lengua de Señas Colombiana. La documentación completa (tecnologías, modelos, datasets, cómo ejecutar) está en el [README principal](../README.md).

```bash
flutter pub get
flutter run     # celular Android conectado por USB
flutter test
```

- `lib/app/`: router y tema (colores, tipografía, espaciado).
- `lib/ui/`: componentes compartidos (importa `ui/ui.dart`).
- `lib/features/`: pantallas. `/debug/gallery` muestra todos los componentes.

La fuente del logo (`assets/fonts/RegularAesthetic.ttf`) no está en el repositorio: pídela al equipo antes de compilar.
