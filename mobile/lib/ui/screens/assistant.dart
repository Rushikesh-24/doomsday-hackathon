// Chat with the wayfinding assistant. Online: the server's Groq assistant, which plans real routes with the
// same router. Offline (or when the server has no Groq key): on-device understanding of "take me to ..." and
// "from A to B" requests, so the core help never depends on the network.
import 'package:flutter/material.dart';

import '../../core/app_state.dart';
import '../../core/api.dart';
import '../../core/models.dart';
import '../../core/router.dart';
import '../../core/voice.dart';
import '../theme.dart';
import '../widgets/common.dart';

class _Msg {
  _Msg(this.role, this.text, {this.route, this.offline = false});
  final String role; // user | assistant
  final String text;
  final RouteResult? route;
  final bool offline;
}

class AssistantScreen extends StatefulWidget {
  const AssistantScreen({super.key, required this.mapId, required this.profile, this.start, this.route, this.seed});
  final String mapId;
  final String? seed; // a first question sent automatically (e.g. from the camera guide)
  final String profile;
  final String? start;
  final RouteResult? route;
  @override
  State<AssistantScreen> createState() => _AssistantScreenState();
}

class _AssistantScreenState extends State<AssistantScreen> {
  final _c = TextEditingController();
  final _scroll = ScrollController();
  final List<_Msg> msgs = [];
  bool busy = false;
  bool listening = false;

  @override
  void initState() {
    super.initState();
    if (widget.seed != null) WidgetsBinding.instance.addPostFrameCallback((_) => _send(widget.seed!));
  }

  @override
  void dispose() {
    Voice.instance.stop();
    Voice.instance.stopListening();
    _c.dispose();
    _scroll.dispose();
    super.dispose();
  }

  List<String> _suggestions(MapPack p) => [
        for (final pl in p.places.where((x) => !x.isStart).take(2)) 'How do I get to ${pl.name}?',
        if (Wayfinder(p).classes.contains('elevator')) 'Where is the nearest lift?',
        'Is the route step-free?',
        'What is near me?',
      ];

  Future<void> _send(String text) async {
    text = text.trim();
    if (text.isEmpty || busy) return;
    final s = AppScope.read(context);
    final m = s.mapById(widget.mapId);
    if (m == null) return;
    setState(() {
      msgs.add(_Msg('user', text));
      busy = true;
      _c.clear();
    });
    _toEnd();
    _Msg reply;
    if (s.online && s.assistantAvailable) {
      try {
        final hist = [
          for (final x in msgs.where((x) => !x.offline || x.role == 'user')) {'role': x.role, 'content': x.text}
        ];
        final r = await s.api.assistantChat(
            mapId: widget.mapId, messages: hist, profile: widget.profile, start: widget.start, route: widget.route?.toJson());
        final rt = r['route'] is Map ? RouteResult.fromJson((r['route'] as Map).cast<String, dynamic>()) : null;
        reply = _Msg('assistant', (r['reply'] as String?)?.trim().isNotEmpty == true ? r['reply'] as String : 'Here is what I found.',
            route: rt);
      } on ApiException catch (e) {
        reply = _offline(m.pack, text, note: e.status == 0 ? null : e.message);
      }
    } else {
      reply = _offline(m.pack, text);
    }
    if (!mounted) return;
    setState(() {
      msgs.add(reply);
      busy = false;
    });
    _toEnd();
    if (s.voice) Voice.instance.say(reply.text);
  }

  _Msg _offline(MapPack p, String text, {String? note}) {
    final wf = Wayfinder(p);
    final lower = text.toLowerCase();
    try {
      final a = wf.ask(text, profile: widget.profile, start: widget.start);
      final r = a.route;
      final say = r.feasible
          ? '${r.directions.map((d) => d.text).skip(1).take(3).join(' ')}${r.directions.length > 4 ? ' Open the route for every step.' : ''}'
          : 'There is no ${widget.profile == 'wheelchair' ? 'step-free ' : ''}route to ${a.goalName} from here. ${widget.profile == 'wheelchair' ? 'Please ask staff about a lift or ramp.' : ''}';
      return _Msg('assistant', say, route: r.feasible ? r : null, offline: true);
    } on RouteError {
      if (lower.contains('step') && lower.contains('free') && widget.route != null) {
        final stairs = widget.route!.warnings.contains('stairs');
        return _Msg('assistant', stairs ? 'No. This route uses stairs.' : 'Yes. This route has no stairs on the scanned path.', offline: true);
      }
      if (lower.contains('near')) {
        final names = p.places.map((x) => x.name).take(6).join(', ');
        return _Msg('assistant', 'Places on this map: $names. Ask me to take you to any of them.', offline: true);
      }
      final names = p.places.map((x) => x.name).join(', ');
      return _Msg(
          'assistant',
          '${note != null ? '$note. ' : ''}I can still guide you without internet. Try "Take me to ${p.places.where((x) => !x.isStart).firstOrNull?.name ?? 'a place'}". '
          '${names.isNotEmpty ? 'Places here: $names.' : ''}',
          offline: true);
    }
  }

  void _toEnd() => WidgetsBinding.instance.addPostFrameCallback((_) {
        if (_scroll.hasClients) _scroll.animateTo(_scroll.position.maxScrollExtent, duration: const Duration(milliseconds: 250), curve: Curves.easeOut);
      });

  Future<void> _mic() async {
    if (listening) {
      await Voice.instance.stopListening();
      setState(() => listening = false);
      return;
    }
    final ok = await Voice.instance.listen(
      onText: (t) => setState(() => _c.text = t),
      onDone: (t) {
        setState(() => listening = false);
        _send(t);
      },
    );
    if (!ok && mounted) {
      ScaffoldMessenger.of(context).showSnackBar(const SnackBar(content: Text('Speech recognition is not available on this phone.')));
    }
    setState(() => listening = ok);
  }

  @override
  Widget build(BuildContext context) {
    final s = AppScope.of(context);
    final m = s.mapById(widget.mapId);
    final live = s.online && s.assistantAvailable;
    return Scaffold(
      appBar: AppBar(
        title: Row(children: [
          Container(
            width: 34,
            height: 34,
            decoration: BoxDecoration(color: AM.lime, borderRadius: BorderRadius.circular(12)),
            child: Icon(Icons.auto_awesome_rounded, color: AM.onLime, size: 18),
          ),
          const SizedBox(width: 10),
          Column(crossAxisAlignment: CrossAxisAlignment.start, children: [
            const Text('Guide'),
            Text(live ? 'AI assistant · online' : 'On-device · works offline',
                style: TextStyle(fontSize: 11, fontWeight: FontWeight.w600, color: live ? AM.tealDeep : AM.amber)),
          ]),
        ]),
      ),
      body: Column(children: [
        Expanded(
          child: msgs.isEmpty && m != null
              ? ListView(padding: const EdgeInsets.all(20), children: [
                  Text('Ask me anything about getting around ${m.pack.name}.', style: Theme.of(context).textTheme.titleLarge),
                  const SizedBox(height: 6),
                  Text(
                      live
                          ? 'I plan real routes for your profile (${profileInfo(widget.profile).label}) and can explain them.'
                          : 'You are offline, so I answer on the phone: directions to tagged places still work.',
                      style: TextStyle(color: AM.muted, height: 1.4)),
                  const SizedBox(height: 20),
                  for (final q in _suggestions(m.pack))
                    Padding(
                      padding: const EdgeInsets.only(bottom: 10),
                      child: OutlinedButton(
                        onPressed: () => _send(q),
                        style: OutlinedButton.styleFrom(alignment: Alignment.centerLeft, padding: const EdgeInsets.all(16)),
                        child: Text(q, style: TextStyle(color: AM.text, fontWeight: FontWeight.w600)),
                      ),
                    ),
                ])
              : ListView.builder(
                  controller: _scroll,
                  padding: const EdgeInsets.fromLTRB(16, 12, 16, 12),
                  itemCount: msgs.length + (busy ? 1 : 0),
                  itemBuilder: (_, k) {
                    if (k == msgs.length) {
                      return const Align(
                        alignment: Alignment.centerLeft,
                        child: Padding(padding: EdgeInsets.all(12), child: SizedBox(width: 22, height: 22, child: CircularProgressIndicator(strokeWidth: 2.5))),
                      );
                    }
                    final x = msgs[k];
                    final me = x.role == 'user';
                    return Align(
                      alignment: me ? Alignment.centerRight : Alignment.centerLeft,
                      child: Container(
                        constraints: BoxConstraints(maxWidth: MediaQuery.sizeOf(context).width * 0.82),
                        margin: const EdgeInsets.only(bottom: 10),
                        padding: const EdgeInsets.fromLTRB(16, 12, 16, 12),
                        decoration: BoxDecoration(
                          color: me ? AM.lime : AM.surface2,
                          border: me ? null : Border.all(color: AM.stroke),
                          borderRadius: BorderRadius.only(
                            topLeft: const Radius.circular(22),
                            topRight: const Radius.circular(22),
                            bottomLeft: Radius.circular(me ? 22 : 6),
                            bottomRight: Radius.circular(me ? 6 : 22),
                          ),
                        ),
                        child: Column(crossAxisAlignment: CrossAxisAlignment.start, children: [
                          Text(x.text, style: TextStyle(color: me ? AM.onLime : AM.text, height: 1.4, fontSize: 15, fontWeight: me ? FontWeight.w600 : FontWeight.w400)),
                          if (x.route != null) ...[
                            const SizedBox(height: 10),
                            FilledButton.icon(
                              onPressed: () => Navigator.pop(context, x.route),
                              icon: const Icon(Icons.route_rounded),
                              label: Text('Show route to ${x.route!.goalName}'),
                            ),
                          ],
                          if (!me && x.offline)
                            Padding(
                              padding: EdgeInsets.only(top: 6),
                              child: Text('answered on this phone', style: TextStyle(fontSize: 11, color: AM.muted)),
                            ),
                        ]),
                      ),
                    );
                  },
                ),
        ),
        SafeArea(
          top: false,
          child: Padding(
            padding: const EdgeInsets.fromLTRB(12, 6, 12, 10),
            child: Row(children: [
              Expanded(
                child: TextField(
                  controller: _c,
                  minLines: 1,
                  maxLines: 4,
                  textInputAction: TextInputAction.send,
                  onSubmitted: _send,
                  decoration: InputDecoration(hintText: listening ? 'Listening…' : 'Ask about routes, places, access…'),
                ),
              ),
              const SizedBox(width: 8),
              RoundAction(icon: listening ? Icons.stop_rounded : Icons.mic_rounded, label: 'Speak', primary: listening, size: 50, onTap: _mic),
              const SizedBox(width: 8),
              RoundAction(icon: Icons.send_rounded, label: 'Send', primary: true, size: 50, onTap: busy ? null : () => _send(_c.text)),
            ]),
          ),
        ),
      ]),
    );
  }
}
