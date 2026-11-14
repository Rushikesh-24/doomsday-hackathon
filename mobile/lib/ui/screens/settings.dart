import 'package:flutter/material.dart';

import '../../core/app_state.dart';
import '../theme.dart';
import '../widgets/common.dart';

class SettingsScreen extends StatefulWidget {
  const SettingsScreen({super.key});
  @override
  State<SettingsScreen> createState() => _SettingsScreenState();
}

class _SettingsScreenState extends State<SettingsScreen> {
  late final TextEditingController _server = TextEditingController(text: AppScope.read(context).serverUrl);
  late final TextEditingController _token = TextEditingController(text: AppScope.read(context).token);
  String? test;
  bool testing = false;

  @override
  void dispose() {
    _server.dispose();
    _token.dispose();
    super.dispose();
  }

  Future<void> _save() async {
    final s = AppScope.read(context);
    var url = _server.text.trim();
    if (url.isNotEmpty && !url.startsWith('http')) url = 'http://$url';
    _server.text = url;
    setState(() {
      testing = true;
      test = null;
    });
    await s.update(serverUrl: url, token: _token.text);
    final ok = await s.api.health();
    if (!mounted) return;
    setState(() {
      testing = false;
      test = ok ? 'Connected. Maps will sync automatically.' : 'Could not reach $url. Check the address and that the phone can reach it.';
    });
  }

  @override
  Widget build(BuildContext context) {
    final s = AppScope.of(context);
    return Scaffold(
      appBar: AppBar(title: const Text('Settings')),
      body: ListView(padding: const EdgeInsets.fromLTRB(20, 8, 20, 32), children: [
        const SectionLabel('Travelling as'),
        const SizedBox(height: 10),
        ProfilePicker(value: s.profile, onChanged: (v) => s.update(profile: v)),
        const SizedBox(height: 22),
        const SectionLabel('Appearance'),
        const SizedBox(height: 10),
        Row(children: [
          for (final (i, o) in const [
            ('system', 'System', Icons.brightness_auto_rounded),
            ('light', 'Light', Icons.light_mode_rounded),
            ('dark', 'Dark', Icons.dark_mode_rounded),
          ].indexed) ...[
            if (i > 0) const SizedBox(width: 10),
            Expanded(
              child: Semantics(
                button: true,
                selected: s.themeMode == o.$1,
                label: '${o.$2} theme',
                excludeSemantics: true,
                child: InkWell(
                  borderRadius: BorderRadius.circular(18),
                  onTap: () => s.update(themeMode: o.$1),
                  child: AnimatedContainer(
                    duration: const Duration(milliseconds: 180),
                    padding: const EdgeInsets.symmetric(vertical: 14, horizontal: 6),
                    decoration: BoxDecoration(
                      color: s.themeMode == o.$1 ? AM.text : AM.surface,
                      borderRadius: BorderRadius.circular(18),
                      border: Border.all(color: s.themeMode == o.$1 ? AM.text : AM.stroke, width: 1.5),
                    ),
                    child: Column(children: [
                      Icon(o.$3, color: s.themeMode == o.$1 ? AM.bg : AM.text),
                      const SizedBox(height: 6),
                      FittedBox(
                        fit: BoxFit.scaleDown,
                        child: Text(o.$2, style: TextStyle(fontWeight: FontWeight.w800, color: s.themeMode == o.$1 ? AM.bg : AM.text)),
                      ),
                    ]),
                  ),
                ),
              ),
            ),
          ],
        ]),
        const SizedBox(height: 22),
        const SectionLabel('Accessibility'),
        const SizedBox(height: 10),
        Card(
          child: Column(children: [
            SwitchListTile(
              value: s.voice,
              onChanged: (v) => s.update(voice: v),
              secondary: const Icon(Icons.record_voice_over_rounded),
              title: const Text('Speak directions'),
            ),
            const Divider(height: 1),
            SwitchListTile(
              value: s.largeText,
              onChanged: (v) => s.update(largeText: v),
              secondary: const Icon(Icons.text_increase_rounded),
              title: const Text('Larger text'),
            ),
            const Divider(height: 1),
            SwitchListTile(
              value: s.highContrast,
              onChanged: (v) => s.update(highContrast: v),
              secondary: const Icon(Icons.contrast_rounded),
              title: const Text('High contrast'),
            ),
          ]),
        ),
        const SizedBox(height: 22),
        const SectionLabel('Server'),
        const SizedBox(height: 10),
        Card(
          child: Padding(
            padding: const EdgeInsets.all(16),
            child: Column(crossAxisAlignment: CrossAxisAlignment.start, children: [
              const ConnPill(),
              const SizedBox(height: 14),
              TextField(
                controller: _server,
                keyboardType: TextInputType.url,
                autocorrect: false,
                decoration: const InputDecoration(labelText: 'Server address', hintText: 'http://192.168.1.20:8000'),
              ),
              const SizedBox(height: 10),
              TextField(
                controller: _token,
                autocorrect: false,
                obscureText: true,
                decoration: const InputDecoration(labelText: 'Access token (optional)'),
              ),
              const SizedBox(height: 12),
              Wrap(spacing: 10, runSpacing: 8, crossAxisAlignment: WrapCrossAlignment.center, children: [
                FilledButton(onPressed: testing ? null : _save, child: Text(testing ? 'Testing…' : 'Save & test')),
                TextButton(
                  onPressed: () {
                    _server.text = AppState.defaultServer;
                    _save();
                  },
                  child: const Text('USB default'),
                ),
              ]),
              if (test != null) Padding(padding: const EdgeInsets.only(top: 10), child: Text(test!, style: TextStyle(color: AM.muted))),
              const SizedBox(height: 10),
              Text(
                'USB: the computer runs "adb reverse tcp:8000 tcp:8000" and the app uses http://127.0.0.1:8000.\n'
                'Wi-Fi: use the computer\'s address on the same network.',
                style: TextStyle(color: AM.muted, fontSize: 12, height: 1.45),
              ),
              if (s.online) ...[
                const SizedBox(height: 10),
                Text(
                  s.assistantAvailable
                      ? 'AI assistant: available (${s.assistantModel ?? 'Groq'})'
                      : 'AI assistant: ${s.assistantReason ?? 'not configured on the server'}',
                  style: TextStyle(fontSize: 12, color: s.assistantAvailable ? AM.tealDeep : AM.amber, fontWeight: FontWeight.w600),
                ),
              ],
            ]),
          ),
        ),
        const SizedBox(height: 22),
        const SectionLabel('Offline maps'),
        const SizedBox(height: 10),
        Card(
          child: Column(children: [
            for (final m in s.maps)
              ListTile(
                title: Text(m.pack.name, style: const TextStyle(fontWeight: FontWeight.w600)),
                subtitle: Text('${(m.bytesOnDisk / 1e6).toStringAsFixed(1)} MB · version ${m.pack.version.substring(0, 6)}'),
                trailing: IconButton(
                  tooltip: 'Remove from phone',
                  icon: const Icon(Icons.delete_outline_rounded),
                  onPressed: () => s.deleteMap(m.pack.id),
                ),
              ),
            if (s.maps.isEmpty) const ListTile(title: Text('No maps saved yet')),
          ]),
        ),
        const SizedBox(height: 28),
        Center(child: Text('AccessMap · screening & guidance aid, not a safety system', style: TextStyle(color: AM.muted, fontSize: 12))),
      ]),
    );
  }
}
