// Renders the app's screens to PNG with the real fonts, for design review and the pitch deck.
//   flutter test test_screens --update-goldens      -> test_screens/goldens/*.png
// Also renders the launcher icon from the same LogoPainter the app uses (android/app/src/main/res).
import 'dart:convert';
import 'dart:io';
import 'dart:ui' as ui;

import 'package:accessmap/core/app_state.dart';
import 'package:accessmap/core/router.dart';
import 'package:accessmap/core/store.dart';
import 'package:accessmap/main.dart';
import 'package:accessmap/ui/screens/guidance.dart';
import 'package:accessmap/ui/screens/map_screen.dart';
import 'package:accessmap/ui/screens/report.dart';
import 'package:accessmap/ui/theme.dart';
import 'package:accessmap/ui/widgets/common.dart';
import 'package:flutter/material.dart';
import 'package:flutter/services.dart';
import 'package:flutter_test/flutter_test.dart';
import 'package:shared_preferences/shared_preferences.dart';

Future<void> _fonts() async {
  final jak = FontLoader('PlusJakartaSans');
  for (final w in [400, 500, 600, 700, 800]) {
    jak.addFont(Future.value(ByteData.sublistView(File('assets/fonts/PlusJakartaSans-$w.ttf').readAsBytesSync())));
  }
  await jak.load();
  final flutterRoot = Platform.environment['FLUTTER_ROOT'] ?? 'C:/dev/flutter';
  final icons = FontLoader('MaterialIcons')
    ..addFont(Future.value(ByteData.sublistView(File('$flutterRoot/bin/cache/artifacts/material_fonts/materialicons-regular.otf').readAsBytesSync())));
  await icons.load();
}

Future<void> settle(WidgetTester t, [int n = 14]) async {
  for (var k = 0; k < n; k++) {
    await t.pump(const Duration(milliseconds: 60));
  }
}

Future<void> _png(Size size, CustomPainter painter, String path) async {
  final rec = ui.PictureRecorder();
  painter.paint(Canvas(rec), size);
  final img = await rec.endRecording().toImage(size.width.round(), size.height.round());
  final bytes = await img.toByteData(format: ui.ImageByteFormat.png);
  File(path)
    ..createSync(recursive: true)
    ..writeAsBytesSync(bytes!.buffer.asUint8List());
}

void main() {
  TestWidgetsFlutterBinding.ensureInitialized();

  setUpAll(_fonts);
  setUp(() {
    for (final ch in ['flutter_tts', 'plugin.csdcorp.com/speech_to_text']) {
      TestDefaultBinaryMessengerBinding.instance.defaultBinaryMessenger
          .setMockMethodCallHandler(MethodChannel(ch), (call) async => call.method == 'initialize' ? false : 1);
    }
  });

  Future<AppState> state({bool onboarded = true, bool large = false, String profile = 'wheelchair', String theme = 'dark'}) async {
    // ignore: invalid_use_of_visible_for_testing_member
    SharedPreferences.setMockInitialValues({'onboarded': onboarded, 'profile': profile, 'voice': false, 'largeText': large, 'theme': theme});
    final dir = await Directory.systemTemp.createTemp('accessmap_shots');
    final store = MapStore(dir);
    final fx = jsonDecode(File('test/fixtures/router_demopcc.json').readAsStringSync()) as Map<String, dynamic>;
    final pack = Map<String, dynamic>.from(fx['pack'] as Map)..['name'] = 'PCC Block - Lab Corridor';
    await store.save(jsonEncode(pack), null);
    final s = AppState(store: store);
    await s.init();
    return s;
  }

  void phone(WidgetTester t) {
    t.view.physicalSize = const Size(1080, 2340);
    t.view.devicePixelRatio = 2.75;
    addTearDown(t.view.reset);
  }

  Future<void> shot(WidgetTester t, String name) async {
    await expectLater(find.byType(MaterialApp), matchesGoldenFile('goldens/$name.png'));
  }

  testWidgets('launcher icons', (t) async {
    await t.runAsync(() async {
      const sizes = {'mdpi': 48, 'hdpi': 72, 'xhdpi': 96, 'xxhdpi': 144, 'xxxhdpi': 192};
      for (final e in sizes.entries) {
        final s = e.value.toDouble();
        await _png(Size(s, s), const LogoPainter(), 'android/app/src/main/res/mipmap-${e.key}/ic_launcher.png');
        // adaptive icon foreground: 108 dp canvas, mark inside the 66 dp safe zone
        final f = s * 108 / 48;
        await _png(Size(f, f), const LogoPainter(background: false, scale: 0.62),
            'android/app/src/main/res/mipmap-${e.key}/ic_launcher_foreground.png');
      }
      await _png(const Size(512, 512), const LogoPainter(), 'assets/icon-512.png');
      await _png(const Size(1024, 1024), const LogoPainter(), 'test_screens/goldens/logo.png');
    });
  });

  testWidgets('onboarding', (t) async {
    phone(t);
    final s = await t.runAsync(() => state(onboarded: false));
    await t.pumpWidget(AccessMapApp(state: s!));
    await settle(t);
    await shot(t, '01_onboarding');
    s.dispose();
  });

  testWidgets('onboarding, large text', (t) async {
    phone(t);
    final s = await t.runAsync(() => state(onboarded: false, large: true));
    await t.pumpWidget(AccessMapApp(state: s!));
    await settle(t);
    await shot(t, '02_onboarding_large_text');
    s.dispose();
  });

  testWidgets('home', (t) async {
    phone(t);
    final s = await t.runAsync(() => state());
    await t.pumpWidget(AccessMapApp(state: s!));
    await settle(t);
    await shot(t, '03_home');
    s.dispose();
    await t.pumpWidget(const SizedBox());
  });

  testWidgets('map + route + report', (t) async {
    phone(t);
    final s = await t.runAsync(() => state());
    await t.pumpWidget(AccessMapApp(state: s!));
    await settle(t);
    await t.tap(find.text('PCC Block - Lab Corridor').first);
    await settle(t, 20);
    await shot(t, '04_map');
    await t.ensureVisible(find.widgetWithText(ActionChip, 'CG Lab'));
    await settle(t);
    await t.tap(find.widgetWithText(ActionChip, 'CG Lab'));
    await settle(t, 20);
    await shot(t, '05_route');
    final ctx = t.element(find.byType(MapScreen));
    showAccessibilityReport(ctx, s.maps.first.pack);
    await settle(t, 20);
    await shot(t, '06_report');
    s.dispose();
    await t.pumpWidget(const SizedBox());
  });

  testWidgets('guidance', (t) async {
    phone(t);
    final s = await t.runAsync(() => state());
    final m = s!.maps.first;
    final wf = Wayfinder(m.pack);
    final r = wf.ask('to CG Lab', profile: 'wheelchair', start: 'Lobby').route;
    await t.pumpWidget(AppScope(state: s, child: MaterialApp(theme: AM.theme(), home: GuidanceScreen(map: m, route: r))));
    await settle(t, 20);
    await shot(t, '07_guidance');
    s.dispose();
    await t.pumpWidget(const SizedBox());
  });

  testWidgets('light theme: home, map + route, report', (t) async {
    phone(t);
    final s = await t.runAsync(() => state(theme: 'light'));
    await t.pumpWidget(AccessMapApp(state: s!));
    await settle(t);
    await shot(t, '08_home_light');
    await t.tap(find.text('PCC Block - Lab Corridor').first);
    await settle(t, 20);
    await t.ensureVisible(find.widgetWithText(ActionChip, 'CG Lab'));
    await settle(t);
    await t.tap(find.widgetWithText(ActionChip, 'CG Lab'));
    await settle(t, 20);
    await shot(t, '09_route_light');
    showAccessibilityReport(t.element(find.byType(MapScreen)), s.maps.first.pack);
    await settle(t, 20);
    await shot(t, '10_report_light');
    s.dispose();
    await t.pumpWidget(const SizedBox());
    AM.isDark = true;
  });

  testWidgets('building: home, map with floor picker, cross-floor route', (t) async {
    phone(t);
    final s = await t.runAsync(() async {
      // ignore: invalid_use_of_visible_for_testing_member
      SharedPreferences.setMockInitialValues({'onboarded': true, 'profile': 'wheelchair', 'voice': false, 'theme': 'dark'});
      final dir = await Directory.systemTemp.createTemp('accessmap_bshots');
      final store = MapStore(dir);
      final fx = jsonDecode(File('test/fixtures/building_demo.json').readAsStringSync()) as Map<String, dynamic>;
      await store.save(jsonEncode(fx['pack']), null);
      final st = AppState(store: store);
      await st.init();
      return st;
    });
    await t.pumpWidget(AccessMapApp(state: s!));
    await settle(t);
    await shot(t, '11_building_home');
    await t.tap(find.text('Main Block'));
    await settle(t, 20);
    await shot(t, '12_building_map');
    final sc = find.descendant(of: find.byType(DraggableScrollableSheet), matching: find.byType(Scrollable)).first;
    await t.scrollUntilVisible(find.widgetWithText(ActionChip, 'Seminar Hall · 3'), 80, scrollable: sc);
    await t.ensureVisible(find.widgetWithText(ActionChip, 'Seminar Hall · 3'));
    await settle(t);
    await t.tap(find.widgetWithText(ActionChip, 'Seminar Hall · 3'));
    await settle(t, 20);
    await t.scrollUntilVisible(find.text('Take Lift 1 up to the Third floor (3 floors).'), 120, scrollable: sc);
    await settle(t);
    await shot(t, '13_building_route_steps');
    await t.tap(find.bySemanticsLabel(RegExp(r'^Third floor, on your route')));
    await settle(t, 20);
    await shot(t, '14_building_route_floor3');
    s.dispose();
    await t.pumpWidget(const SizedBox());
  });
}
