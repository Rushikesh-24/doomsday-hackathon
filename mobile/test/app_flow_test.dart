// End-to-end UI flow, fully offline: a saved map pack -> home -> map -> pick a destination -> route ->
// guidance. The server is unreachable in tests, which is exactly the offline case.
import 'dart:io';

import 'package:accessmap/core/app_state.dart';
import 'package:accessmap/core/store.dart';
import 'package:accessmap/main.dart';
import 'package:accessmap/ui/theme.dart';
import 'package:flutter/material.dart';
import 'package:flutter/services.dart';
import 'package:flutter_test/flutter_test.dart';
import 'package:shared_preferences/shared_preferences.dart';
import 'dart:convert';

/// Step the clock frame by frame (routes and sheets animate; the plan pulses forever, so no pumpAndSettle).
Future<void> settle(WidgetTester t) async {
  for (var k = 0; k < 12; k++) {
    await t.pump(const Duration(milliseconds: 60));
  }
}

void main() {
  TestWidgetsFlutterBinding.ensureInitialized();

  setUp(() {
    // TTS / speech plugins are not available in widget tests: answer their channels with no-ops.
    for (final ch in ['flutter_tts', 'plugin.csdcorp.com/speech_to_text']) {
      TestDefaultBinaryMessengerBinding.instance.defaultBinaryMessenger
          .setMockMethodCallHandler(MethodChannel(ch), (call) async => call.method == 'initialize' ? false : 1);
    }
  });

  Future<AppState> makeState({required bool onboarded}) async {
    SharedPreferences.setMockInitialValues({'onboarded': onboarded, 'profile': 'wheelchair', 'voice': false});
    final dir = await Directory.systemTemp.createTemp('accessmap_test');
    final store = MapStore(dir);
    final fx = jsonDecode(File('test/fixtures/router_demopcc.json').readAsStringSync()) as Map<String, dynamic>;
    final pack = Map<String, dynamic>.from(fx['pack'] as Map)..['name'] = 'PCC Block - Lab Corridor';
    await store.save(jsonEncode(pack), null);
    final s = AppState(store: store);
    await s.init();
    return s;
  }

  testWidgets('first run shows onboarding and saves the profile', (t) async {
    final s = await t.runAsync(() => makeState(onboarded: false));
    await t.pumpWidget(AccessMapApp(state: s!));
    await t.pump();
    expect(find.text('How do you get around?'), findsOneWidget);
    await t.tap(find.text('Low vision'));
    await t.tap(find.text('Get started'));
    for (var k = 0; k < 10; k++) {
      await t.pump(const Duration(milliseconds: 100));
    }
    expect(s.profile, 'low_vision');
    expect(s.onboarded, isTrue);
    expect(find.text('Saved maps'), findsOneWidget);
    s.dispose();
  });

  testWidgets('offline: open a saved map, route to CG Lab step-free and start guidance', (t) async {
    t.view.physicalSize = const Size(1080, 2340);
    t.view.devicePixelRatio = 2.75;
    addTearDown(t.view.reset);
    final s = await t.runAsync(() => makeState(onboarded: true));
    await t.pumpWidget(AccessMapApp(state: s!));
    await settle(t);
    expect(find.text('PCC Block - Lab Corridor'), findsOneWidget);
    expect(find.text('OFFLINE'), findsOneWidget);
    expect(find.text('79 · B'), findsOneWidget); // accessibility score from the pack

    await t.tap(find.text('PCC Block - Lab Corridor'));
    await settle(t);
    expect(find.text('Where to?'), findsOneWidget);
    expect(find.text('From: Lobby'), findsOneWidget);

    await t.ensureVisible(find.widgetWithText(ActionChip, 'CG Lab'));
    await settle(t);
    await t.tap(find.widgetWithText(ActionChip, 'CG Lab'));
    await settle(t);
    expect(find.text('Lobby → CG Lab'), findsOneWidget);
    expect(find.textContaining('About 1 minute'), findsOneWidget);
    expect(find.text('Start guidance'), findsOneWidget);

    await t.tap(find.text('Start guidance'));
    await settle(t);
    expect(find.text('Start at Lobby. Step-free route: stairs are excluded.'), findsOneWidget);
    await t.tap(find.text('Next step'));
    await settle(t);
    expect(find.text('Go straight for about 12 m, passing CC Lab.'), findsOneWidget);
    await t.tap(find.text('Next step'));
    await settle(t);
    expect(find.textContaining('You have arrived at CG Lab.'), findsOneWidget);
    expect(find.text('Done'), findsOneWidget);
    s.dispose();
    await t.pumpWidget(const SizedBox());
  });

  testWidgets('wheelchair user from the upper landing gets an honest "no step-free route"', (t) async {
    t.view.physicalSize = const Size(1080, 2340);
    t.view.devicePixelRatio = 2.75;
    addTearDown(t.view.reset);
    final s = await t.runAsync(() => makeState(onboarded: true));
    await t.pumpWidget(AccessMapApp(state: s!));
    await settle(t);
    await t.tap(find.text('PCC Block - Lab Corridor'));
    await settle(t);
    await t.tap(find.text('From: Lobby'));
    await settle(t);
    await t.tap(find.text('Upper landing').last);
    await settle(t);
    await t.ensureVisible(find.widgetWithText(ActionChip, 'CG Lab'));
    await settle(t);
    await t.tap(find.widgetWithText(ActionChip, 'CG Lab'));
    await settle(t);
    expect(find.text('No step-free route'), findsOneWidget);
    expect(find.textContaining('a route exists only via stairs'), findsOneWidget);
    s.dispose();
    await t.pumpWidget(const SizedBox());
  });

  testWidgets('2x system text on a narrow phone: no overflow on onboarding, home, map and route', (t) async {
    t.view.physicalSize = const Size(720, 1520); // 360 x 760 dp
    t.view.devicePixelRatio = 2.0;
    t.platformDispatcher.textScaleFactorTestValue = 2.0;
    addTearDown(t.view.reset);
    addTearDown(t.platformDispatcher.clearTextScaleFactorTestValue);
    final s = await t.runAsync(() => makeState(onboarded: false));
    await t.pumpWidget(AccessMapApp(state: s!));
    await settle(t);
    await t.scrollUntilVisible(find.text('Low vision'), 100, scrollable: find.byType(Scrollable).first);
    expect(find.text('Wheelchair'), findsOneWidget); // stacked profile list, readable
    await t.tap(find.text('Get started'));
    await settle(t);
    await t.scrollUntilVisible(find.text('PCC Block - Lab Corridor'), 150, scrollable: find.byType(Scrollable).first);
    await settle(t);
    await t.tap(find.text('PCC Block - Lab Corridor'));
    await settle(t);
    await t.scrollUntilVisible(find.widgetWithText(ActionChip, 'CG Lab'), 80,
        scrollable: find.descendant(of: find.byType(DraggableScrollableSheet), matching: find.byType(Scrollable)).first);
    await t.ensureVisible(find.widgetWithText(ActionChip, 'CG Lab'));
    await settle(t);
    await t.tap(find.widgetWithText(ActionChip, 'CG Lab'));
    await settle(t);
    expect(find.text('Start guidance'), findsOneWidget);
    // any RenderFlex overflow above would have failed the test
    s.dispose();
    await t.pumpWidget(const SizedBox());
  });

  testWidgets('theme switch: home toggle and Settings > Appearance change the palette', (t) async {
    t.view.physicalSize = const Size(1080, 2340);
    t.view.devicePixelRatio = 2.75;
    addTearDown(t.view.reset);
    final s = await t.runAsync(() => makeState(onboarded: true));
    await t.pumpWidget(AccessMapApp(state: s!));
    await settle(t);
    expect(s.themeMode, 'system');
    Color bg() => Theme.of(t.element(find.byType(Scaffold).first)).scaffoldBackgroundColor;
    final before = bg();
    await t.tap(find.byTooltip(AM.isDark ? 'Switch to light theme' : 'Switch to dark theme'));
    await settle(t);
    expect(bg(), isNot(before));
    expect(s.themeMode, AM.isDark ? 'dark' : 'light');
    await t.tap(find.byTooltip('Settings'));
    await settle(t);
    await t.tap(find.text('Dark'));
    await settle(t);
    expect(s.themeMode, 'dark');
    expect(AM.isDark, isTrue);
    expect(bg(), const Color(0xFF000000));
    await t.tap(find.text('Light'));
    await settle(t);
    expect(AM.isDark, isFalse);
    expect(bg(), const Color(0xFFFFFFFF));
    final prefs = await t.runAsync(() => SharedPreferences.getInstance());
    expect(prefs!.getString('theme'), 'light'); // remembered
    s.dispose();
    await t.pumpWidget(const SizedBox());
    AM.isDark = true;
  });

  testWidgets('building: floor picker, wheelchair route takes the lift across floors, guidance follows floors', (t) async {
    t.view.physicalSize = const Size(1080, 2340);
    t.view.devicePixelRatio = 2.75;
    addTearDown(t.view.reset);
    final s = await t.runAsync(() async {
      SharedPreferences.setMockInitialValues({'onboarded': true, 'profile': 'wheelchair', 'voice': false});
      final dir = await Directory.systemTemp.createTemp('accessmap_building');
      final store = MapStore(dir);
      final fx = jsonDecode(File('test/fixtures/building_demo.json').readAsStringSync()) as Map<String, dynamic>;
      await store.save(jsonEncode(fx['pack']), null);
      final st = AppState(store: store);
      await st.init();
      return st;
    });
    await t.pumpWidget(AccessMapApp(state: s!));
    await settle(t);
    expect(find.text('4 floors'), findsOneWidget);
    await t.tap(find.text('Main Block'));
    await settle(t);
    expect(find.text('From: Main entrance'), findsOneWidget);
    for (final f in ['G', '1', '2', '3']) {
      expect(find.text(f), findsWidgets);
    }
    await t.scrollUntilVisible(find.widgetWithText(ActionChip, 'Seminar Hall · 3'), 80,
        scrollable: find.descendant(of: find.byType(DraggableScrollableSheet), matching: find.byType(Scrollable)).first);
    await t.ensureVisible(find.widgetWithText(ActionChip, 'Seminar Hall · 3'));
    await settle(t);
    await t.tap(find.widgetWithText(ActionChip, 'Seminar Hall · 3'));
    await settle(t);
    expect(find.text('Main entrance → Seminar Hall'), findsOneWidget);
    await t.scrollUntilVisible(find.text('Take Lift 1 up to the Third floor (3 floors).'), 120,
        scrollable: find.descendant(of: find.byType(DraggableScrollableSheet), matching: find.byType(Scrollable)).first);
    expect(find.text('FLOOR G → 3'), findsOneWidget);
    // floor picker: show the third floor
    await t.tap(find.bySemanticsLabel(RegExp(r'^Third floor, on your route')));
    await settle(t);
    await t.scrollUntilVisible(find.text('Start guidance'), -120,
        scrollable: find.descendant(of: find.byType(DraggableScrollableSheet), matching: find.byType(Scrollable)).first);
    await t.tap(find.text('Start guidance'));
    await settle(t);
    expect(find.text('Ground floor'), findsOneWidget);           // guidance starts on the ground floor
    await t.tap(find.text('Next step'));
    await settle(t);
    await t.tap(find.text('Next step'));
    await settle(t);
    expect(find.text('Take Lift 1 up to the Third floor (3 floors).'), findsOneWidget);
    await t.tap(find.text('Next step'));
    await settle(t);
    expect(find.text('Third floor'), findsOneWidget);            // ...and switches floor after the lift
    s.dispose();
    await t.pumpWidget(const SizedBox());
  });
}
