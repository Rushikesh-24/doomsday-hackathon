import 'package:flutter/material.dart';

import '../../core/app_state.dart';
import '../theme.dart';
import '../widgets/common.dart';
import 'home.dart';

class OnboardingScreen extends StatefulWidget {
  const OnboardingScreen({super.key});
  @override
  State<OnboardingScreen> createState() => _OnboardingScreenState();
}

class _OnboardingScreenState extends State<OnboardingScreen> {
  String profile = 'standard';
  bool voice = true, large = false;

  @override
  Widget build(BuildContext context) {
    final s = AppScope.of(context);
    return Scaffold(
      body: Column(children: [
        Expanded(
          child: ListView(padding: EdgeInsets.zero, children: [
            Container(
              decoration: BoxDecoration(gradient: AM.heroGradient, borderRadius: BorderRadius.vertical(bottom: Radius.circular(36))),
              child: SafeArea(
                bottom: false,
                child: Padding(
                  padding: const EdgeInsets.fromLTRB(24, 28, 24, 34),
                  child: Column(crossAxisAlignment: CrossAxisAlignment.start, children: [
                    const Logo(size: 64),
                    const SizedBox(height: 26),
                    const SectionLabel('AccessMap'),
                    const SizedBox(height: 8),
                    Text.rich(
                      TextSpan(children: [
                        const TextSpan(text: 'Find your way,\n'),
                        TextSpan(text: 'your way.', style: TextStyle(color: Theme.of(context).colorScheme.primary)),
                      ]),
                      style: Theme.of(context).textTheme.headlineLarge?.copyWith(height: 1.05),
                    ),
                    const SizedBox(height: 12),
                    Text(
                      'Indoor directions built from real 3D scans of your campus: step-free routes, spoken guidance and a camera that tells you where you are. Works offline once a map is saved.',
                      style: TextStyle(color: AM.muted, height: 1.45),
                    ),
                  ]),
                ),
              ),
            ),
            Padding(
              padding: const EdgeInsets.fromLTRB(24, 26, 24, 24),
              child: Column(crossAxisAlignment: CrossAxisAlignment.start, children: [
                Text('How do you get around?', style: Theme.of(context).textTheme.titleLarge),
                const SizedBox(height: 4),
                Text('Routes and spoken help adapt to this. You can change it any time.', style: TextStyle(color: AM.muted)),
                const SizedBox(height: 18),
                ProfilePicker(
                    value: profile,
                    onChanged: (v) => setState(() {
                          profile = v;
                          if (v == 'low_vision') voice = true;
                        })),
                const SizedBox(height: 22),
                Card(
                  child: Column(children: [
                    SwitchListTile(
                      value: voice,
                      onChanged: (v) => setState(() => voice = v),
                      title: const Text('Speak directions', style: TextStyle(fontWeight: FontWeight.w700)),
                      subtitle: const Text('Read each step aloud while guiding'),
                      secondary: const Icon(Icons.record_voice_over_rounded),
                    ),
                    const Divider(height: 1),
                    SwitchListTile(
                      value: large,
                      onChanged: (v) async {
                        setState(() => large = v);
                        await s.update(largeText: v); // preview the size right away
                      },
                      title: const Text('Larger text', style: TextStyle(fontWeight: FontWeight.w700)),
                      secondary: const Icon(Icons.text_increase_rounded),
                    ),
                  ]),
                ),
              ]),
            ),
          ]),
        ),
        SafeArea(
          top: false,
          child: Padding(
            padding: const EdgeInsets.fromLTRB(24, 0, 24, 16),
            child: SizedBox(
              width: double.infinity,
              child: FilledButton(
                onPressed: () async {
                  await s.update(profile: profile, voice: voice, largeText: large, onboarded: true);
                  if (!context.mounted) return;
                  Navigator.of(context).pushReplacement(MaterialPageRoute(builder: (_) => const HomeScreen()));
                },
                child: const Text('Get started'),
              ),
            ),
          ),
        ),
      ]),
    );
  }
}
