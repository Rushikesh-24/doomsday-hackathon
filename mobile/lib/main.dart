import 'package:flutter/material.dart';
import 'package:flutter/services.dart';

import 'core/app_state.dart';
import 'ui/screens/home.dart';
import 'ui/screens/onboarding.dart';
import 'ui/theme.dart';

Future<void> main() async {
  WidgetsFlutterBinding.ensureInitialized();
  SystemChrome.setSystemUIOverlayStyle(
    const SystemUiOverlayStyle(statusBarColor: Colors.transparent),
  );
  final state = AppState();
  await state.init();
  runApp(AccessMapApp(state: state));
}

class AccessMapApp extends StatelessWidget {
  const AccessMapApp({super.key, required this.state});
  final AppState state;

  @override
  Widget build(BuildContext context) {
    return AppScope(
      state: state,
      child: ListenableBuilder(
        listenable: state,
        builder: (context, _) {
          // Appearance: System follows the phone's dark mode; Light / Dark are fixed.
          final sys =
              WidgetsBinding.instance.platformDispatcher.platformBrightness;
          AM.isDark =
              state.themeMode == 'dark' ||
              (state.themeMode == 'system' && sys == Brightness.dark);
          SystemChrome.setSystemUIOverlayStyle(
            SystemUiOverlayStyle(
              statusBarColor: Colors.transparent,
              statusBarIconBrightness: AM.isDark
                  ? Brightness.light
                  : Brightness.dark,
              systemNavigationBarColor: AM.bg,
              systemNavigationBarIconBrightness: AM.isDark
                  ? Brightness.light
                  : Brightness.dark,
            ),
          );
          return MaterialApp(
            title: 'AccessMap',
            debugShowCheckedModeBanner: false,
            theme: AM.theme(highContrast: state.highContrast),
            builder: (context, child) {
              final mq = MediaQuery.of(context);
              final sys = mq.textScaler.scale(1);
              return MediaQuery(
                data: mq.copyWith(
                  textScaler: TextScaler.linear(
                    state.largeText
                        ? (sys < 1.3 ? 1.3 : sys)
                        : sys.clamp(0.85, 2.0),
                  ),
                ),
                child: child!,
              );
            },
            home: state.onboarded
                ? const HomeScreen()
                : const OnboardingScreen(),
          );
        },
      ),
    );
  }
}
