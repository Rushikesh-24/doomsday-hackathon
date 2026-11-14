// AccessMap visual language, v3: black and white only, in a dark and a light variant.
// Dark: white on black (21:1). Light: black on white (21:1). Greys mark secondary information; meaning is
// carried by brightness, shape, icons and words, never by hue. High contrast adds thicker borders.
//
// `AM.isDark` selects the palette; the app sets it from Settings → Appearance (System / Light / Dark) before
// building the MaterialApp, so every screen, sheet and painter reads the active colours.
import 'package:flutter/material.dart';

class _Pal {
  const _Pal({
    required this.bg,
    required this.surface,
    required this.surface2,
    required this.stroke,
    required this.text,
    required this.muted,
    required this.accent2,
    required this.soft,
    required this.dim,
    required this.sevMedium,
    required this.sevLow,
    required this.floor,
    required this.wall,
    required this.thumbA,
    required this.thumbB,
    required this.thumbFloor,
    required this.heroA,
    required this.heroB,
  });
  final Color bg, surface, surface2, stroke, text, muted, accent2, soft, dim, sevMedium, sevLow;
  final Color floor, wall, thumbA, thumbB, thumbFloor, heroA, heroB;
}

const _dark = _Pal(
  bg: Color(0xFF000000),
  surface: Color(0xFF111111),
  surface2: Color(0xFF1C1C1C),
  stroke: Color(0xFF2E2E2E),
  text: Color(0xFFFFFFFF),
  muted: Color(0xFF9A9A9A),
  accent2: Color(0xFFBDBDBD),
  soft: Color(0x1FFFFFFF),
  dim: Color(0xFFD6D6D6),
  sevMedium: Color(0xFFB0B0B0),
  sevLow: Color(0xFF6E6E6E),
  floor: Color(0xFF1A1A1A),
  wall: Color(0xFFEDEDED),
  thumbA: Color(0xFF161616),
  thumbB: Color(0xFF050505),
  thumbFloor: Color(0xFF262626),
  heroA: Color(0xFF1A1A1A),
  heroB: Color(0xFF000000),
);

const _light = _Pal(
  bg: Color(0xFFFFFFFF),
  surface: Color(0xFFF4F4F4),
  surface2: Color(0xFFEAEAEA),
  stroke: Color(0xFFD4D4D4),
  text: Color(0xFF000000),
  muted: Color(0xFF616161),
  accent2: Color(0xFF424242),
  soft: Color(0x14000000),
  dim: Color(0xFF2E2E2E),
  sevMedium: Color(0xFF5C5C5C),
  sevLow: Color(0xFF9E9E9E),
  floor: Color(0xFFF0F0F0),
  wall: Color(0xFF111111),
  thumbA: Color(0xFFF7F7F7),
  thumbB: Color(0xFFE6E6E6),
  thumbFloor: Color(0xFFFFFFFF),
  heroA: Color(0xFFF0F0F0),
  heroB: Color(0xFFFFFFFF),
);

class AM {
  /// Active palette. Set by the app from the Appearance setting before the MaterialApp is built.
  static bool isDark = true;
  static _Pal get _p => isDark ? _dark : _light;

  static Color get bg => _p.bg;
  static Color get surface => _p.surface;
  static Color get surface2 => _p.surface2;
  static Color get stroke => _p.stroke;
  static Color get text => _p.text;
  static Color get muted => _p.muted;

  // accents: the "ink" colour of the theme (white in dark, black in light) and greys
  static Color get lime => _p.text;
  static Color get limeDeep => _p.dim;
  static Color get onLime => _p.bg;
  static Color get violet => _p.accent2;
  static Color get violetSoft => _p.soft;
  static Color get teal => _p.dim;
  static Color get tealDeep => _p.accent2;
  static Color get rose => _p.text;
  static Color get amber => _p.accent2;
  static Color get green => _p.text;
  static Color get blue => _p.text;

  static Color get ink => _p.bg;
  static Color get paper => _p.bg;
  static Color get line => _p.stroke;

  // map drawing
  static Color get floor => _p.floor;
  static Color get wall => _p.wall;
  static Color get thumbA => _p.thumbA;
  static Color get thumbB => _p.thumbB;
  static Color get thumbFloor => _p.thumbFloor;

  static Color place(String kind) => _p.dim;

  /// high = full ink, medium = mid grey, low = faint grey (the "!" marker and the label say which).
  static Color severity(String s) => switch (s) { 'high' => _p.text, 'medium' => _p.sevMedium, _ => _p.sevLow };

  static Color grade(int score) => _p.text;

  static LinearGradient get heroGradient =>
      LinearGradient(begin: Alignment.topLeft, end: Alignment.bottomRight, colors: [_p.heroA, _p.heroB]);

  static ThemeData theme({bool highContrast = false}) {
    final hc = highContrast;
    final dark = isDark;
    final ink = text, paperC = bg;
    final scheme = (dark ? const ColorScheme.dark() : const ColorScheme.light()).copyWith(
      primary: ink,
      onPrimary: paperC,
      secondary: hc ? ink : violet,
      onSecondary: paperC,
      surface: hc ? paperC : surface,
      onSurface: ink,
      surfaceContainerHighest: hc ? paperC : surface2,
      outline: hc ? ink : stroke,
      error: ink,
    );
    final base = ThemeData(useMaterial3: true, brightness: dark ? Brightness.dark : Brightness.light, colorScheme: scheme, fontFamily: 'PlusJakartaSans');
    final border = hc ? ink : stroke;
    return base.copyWith(
      scaffoldBackgroundColor: paperC,
      canvasColor: hc ? paperC : surface,
      dividerColor: border,
      textTheme: base.textTheme.apply(bodyColor: ink, displayColor: ink).copyWith(
            headlineLarge: base.textTheme.headlineLarge?.copyWith(fontWeight: FontWeight.w800, letterSpacing: -1.0, color: ink),
            headlineMedium: base.textTheme.headlineMedium?.copyWith(fontWeight: FontWeight.w800, letterSpacing: -0.8, color: ink),
            titleLarge: base.textTheme.titleLarge?.copyWith(fontWeight: FontWeight.w800, letterSpacing: -0.4, color: ink),
            titleMedium: base.textTheme.titleMedium?.copyWith(fontWeight: FontWeight.w700, color: ink),
          ),
      appBarTheme: AppBarTheme(
          backgroundColor: Colors.transparent, elevation: 0, scrolledUnderElevation: 0, foregroundColor: ink,
          titleTextStyle: TextStyle(fontFamily: 'PlusJakartaSans', fontWeight: FontWeight.w800, fontSize: 18, color: ink)),
      cardTheme: CardThemeData(
        color: hc ? paperC : surface,
        elevation: 0,
        margin: EdgeInsets.zero,
        shape: RoundedRectangleBorder(borderRadius: BorderRadius.circular(26), side: BorderSide(color: border, width: hc ? 2 : 1)),
      ),
      bottomSheetTheme: BottomSheetThemeData(
        backgroundColor: hc ? paperC : surface,
        modalBackgroundColor: hc ? paperC : surface,
        shape: const RoundedRectangleBorder(borderRadius: BorderRadius.vertical(top: Radius.circular(30))),
        showDragHandle: true,
        dragHandleColor: border,
      ),
      listTileTheme: ListTileThemeData(iconColor: ink, textColor: ink, subtitleTextStyle: TextStyle(color: hc ? ink : muted, fontSize: 13)),
      filledButtonTheme: FilledButtonThemeData(
        style: FilledButton.styleFrom(
          backgroundColor: ink,
          foregroundColor: paperC,
          disabledBackgroundColor: stroke,
          minimumSize: const Size(48, 56),
          padding: const EdgeInsets.symmetric(horizontal: 22),
          shape: RoundedRectangleBorder(borderRadius: BorderRadius.circular(20)),
          textStyle: const TextStyle(fontFamily: 'PlusJakartaSans', fontWeight: FontWeight.w800, fontSize: 15),
        ),
      ),
      outlinedButtonTheme: OutlinedButtonThemeData(
        style: OutlinedButton.styleFrom(
          foregroundColor: ink,
          minimumSize: const Size(48, 52),
          shape: RoundedRectangleBorder(borderRadius: BorderRadius.circular(18)),
          side: BorderSide(color: border, width: hc ? 2 : 1),
          textStyle: const TextStyle(fontFamily: 'PlusJakartaSans', fontWeight: FontWeight.w700),
        ),
      ),
      textButtonTheme: TextButtonThemeData(style: TextButton.styleFrom(foregroundColor: ink)),
      segmentedButtonTheme: SegmentedButtonThemeData(
        style: ButtonStyle(
          backgroundColor: WidgetStateProperty.resolveWith((s) => s.contains(WidgetState.selected) ? ink : Colors.transparent),
          foregroundColor: WidgetStateProperty.resolveWith((s) => s.contains(WidgetState.selected) ? paperC : ink),
          side: WidgetStateProperty.all(BorderSide(color: border)),
          textStyle: WidgetStateProperty.all(const TextStyle(fontFamily: 'PlusJakartaSans', fontWeight: FontWeight.w700)),
        ),
      ),
      switchTheme: SwitchThemeData(
        thumbColor: WidgetStateProperty.resolveWith((s) => s.contains(WidgetState.selected) ? paperC : muted),
        trackColor: WidgetStateProperty.resolveWith((s) => s.contains(WidgetState.selected) ? ink : surface2),
        trackOutlineColor: WidgetStateProperty.all(border),
      ),
      inputDecorationTheme: InputDecorationTheme(
        filled: true,
        fillColor: hc ? paperC : surface2,
        hintStyle: TextStyle(color: muted),
        border: OutlineInputBorder(borderRadius: BorderRadius.circular(18), borderSide: BorderSide(color: border)),
        enabledBorder: OutlineInputBorder(borderRadius: BorderRadius.circular(18), borderSide: BorderSide(color: border)),
        focusedBorder: OutlineInputBorder(borderRadius: BorderRadius.circular(18), borderSide: BorderSide(color: ink, width: 2)),
      ),
      chipTheme: base.chipTheme.copyWith(
        shape: const StadiumBorder(),
        side: BorderSide(color: border),
        backgroundColor: hc ? paperC : surface2,
        labelStyle: TextStyle(color: ink, fontWeight: FontWeight.w600, fontFamily: 'PlusJakartaSans'),
      ),
      snackBarTheme: SnackBarThemeData(
          behavior: SnackBarBehavior.floating,
          backgroundColor: ink,
          contentTextStyle: TextStyle(color: paperC, fontFamily: 'PlusJakartaSans')),
      progressIndicatorTheme: ProgressIndicatorThemeData(color: ink),
    );
  }
}
