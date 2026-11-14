// Camera guide: point the phone down the corridor and AccessMap works out where you are and which way to go.
//
// Every few seconds a frame is sent to the server (POST /maps/{id}/locate): it is matched against the scan's
// own keyframes (DINOv2 shortlist + SIFT/RANSAC check) to get the nearest point of the walk and the phone's
// heading, and the facility detector reports what is in view (stairs, doors, obstacles; left / ahead / right).
// The route is then re-planned ON THE PHONE from that point, and lime AR chevrons point toward the next part
// of the route relative to where the camera is facing. Spoken prompts change only when the advice changes.
import 'dart:async';
import 'dart:math' as math;

import 'package:camera/camera.dart';
import 'package:flutter/material.dart';
import 'package:flutter/services.dart';

import '../../core/api.dart';
import '../../core/app_state.dart';
import '../../core/models.dart';
import '../../core/router.dart';
import '../../core/store.dart';
import '../../core/voice.dart';
import '../theme.dart';
import '../widgets/common.dart';
import 'assistant.dart';

class LocateFix {
  LocateFix(this.raw);
  final Map<String, dynamic> raw;
  String get node => raw['node'] as String;
  List<double> get position => (raw['position'] as List).map((v) => (v as num).toDouble()).toList();
  double get heading => (raw['heading'] as num).toDouble();
  String get confidence => raw['confidence'] as String? ?? 'low';
  bool get usable => confidence != 'low';
  String? get nearPlace => (raw['near_place'] as Map?)?['name'] as String?;
  double? get nearDistance => ((raw['near_place'] as Map?)?['distance'] as num?)?.toDouble();
  List<String> get hints => (((raw['scene'] as Map?)?['hints'] as List?) ?? const []).cast<String>();
  List<Map<String, dynamic>> get objects =>
      (((raw['scene'] as Map?)?['objects'] as List?) ?? const []).map((o) => (o as Map).cast<String, dynamic>()).toList();
  String? get caption => raw['caption'] as String?;
}

class CameraGuideScreen extends StatefulWidget {
  const CameraGuideScreen({super.key, required this.mapId, required this.profile, this.goal, this.goalName, this.floorLevel});
  final String mapId;
  final int? floorLevel; // buildings: the floor the traveller is on (its scan is matched)
  final String profile;
  final String? goal; // resolved goal ref (place id / class); null = just "where am I"
  final String? goalName;
  @override
  State<CameraGuideScreen> createState() => _CameraGuideScreenState();
}

class _CameraGuideScreenState extends State<CameraGuideScreen> with WidgetsBindingObserver, SingleTickerProviderStateMixin {
  CameraController? _cam;
  String? _camError;
  bool _busy = false, _auto = true, _describing = false;
  LocateFix? fix;
  RouteResult? route;
  String? error;
  double? relAngle; // radians, + = to the right of where the camera faces
  double? remaining;
  String _lastSpoken = '';
  DateTime _lastSpokenAt = DateTime(2000);
  Timer? _loop;
  late final AnimationController _flow = AnimationController(vsync: this, duration: const Duration(milliseconds: 1400))..repeat();

  @override
  void initState() {
    super.initState();
    WidgetsBinding.instance.addObserver(this);
    _start();
  }

  Future<void> _start() async {
    try {
      final cams = await availableCameras();
      final back = cams.firstWhere((c) => c.lensDirection == CameraLensDirection.back, orElse: () => cams.first);
      final c = CameraController(back, ResolutionPreset.high, enableAudio: false, imageFormatGroup: ImageFormatGroup.jpeg);
      await c.initialize();
      try {
        await c.setFlashMode(FlashMode.off);
      } catch (_) {}
      if (!mounted) {
        await c.dispose();
        return;
      }
      setState(() => _cam = c);
      _schedule(const Duration(milliseconds: 600));
    } on CameraException catch (e) {
      setState(() => _camError = e.code == 'CameraAccessDenied' ? 'Camera permission is needed for the camera guide.' : 'Camera error: ${e.description}');
    } catch (e) {
      setState(() => _camError = 'No camera available ($e).');
    }
  }

  void _schedule(Duration d) {
    _loop?.cancel();
    if (_auto) _loop = Timer(d, () => _shot());
  }

  @override
  void didChangeAppLifecycleState(AppLifecycleState state) {
    final c = _cam;
    if (c == null || !c.value.isInitialized) return;
    if (state == AppLifecycleState.inactive || state == AppLifecycleState.paused) {
      _loop?.cancel();
      c.dispose();
      _cam = null;
    } else if (state == AppLifecycleState.resumed) {
      _start();
    }
  }

  @override
  void dispose() {
    WidgetsBinding.instance.removeObserver(this);
    _loop?.cancel();
    _flow.dispose();
    _cam?.dispose();
    Voice.instance.stop();
    super.dispose();
  }

  Future<void> _shot({bool caption = false}) async {
    final c = _cam;
    final s = AppScope.read(context);
    if (c == null || !c.value.isInitialized || _busy) return;
    if (!s.online) {
      setState(() => error = 'The camera guide needs the AccessMap server. Offline, use step-by-step guidance.');
      _schedule(const Duration(seconds: 5));
      return;
    }
    _busy = true;
    if (caption) setState(() => _describing = true);
    try {
      final pic = await c.takePicture();
      final bytes = await pic.readAsBytes();
      final m = s.mapById(widget.mapId);
      final fl = widget.floorLevel == null || m == null ? null : m.pack.floors.where((f) => f.level == widget.floorLevel).firstOrNull;
      final r = await s.api.locate(fl?.scan ?? widget.mapId, bytes, caption: caption);
      if (!mounted) return;
      _apply(LocateFix(fl == null ? r : _toBuilding(r, fl, m!.pack)), s);
    } on ApiException catch (e) {
      if (mounted) setState(() => error = e.message);
    } on CameraException catch (e) {
      if (mounted) setState(() => error = 'Camera: ${e.description}');
    } finally {
      _busy = false;
      if (mounted) {
        setState(() => _describing = false);
        _schedule(const Duration(milliseconds: 2200));
      }
    }
  }

  /// A floor-scan fix -> building frame: node "camK" becomes "(scan)|camK" (its building position), heading
  /// turns by the floor's yaw.
  Map<String, dynamic> _toBuilding(Map<String, dynamic> r, BuildingFloor fl, MapPack pack) {
    final node = '${fl.scan}|${r['node']}';
    final n = pack.nodes.where((x) => x.id == node).firstOrNull;
    return {
      ...r,
      'node': node,
      if (n != null) 'position': n.pos,
      'heading': (r['heading'] as num).toDouble() + (fl.yawByScan[fl.scan] ?? fl.yawDeg) * math.pi / 180,
    };
  }

  void _apply(LocateFix f, AppState s) {
    final m = s.mapById(widget.mapId);
    if (m == null) return;
    RouteResult? rt;
    double? rel, rem;
    String? err;
    if (f.usable && widget.goal != null) {
      try {
        final wf = Wayfinder(m.pack);
        rt = m.pack.isBuilding ? wf.routeBuilding(f.node, widget.goal!, profile: widget.profile) : wf.route(f.node, widget.goal!, profile: widget.profile);
        if (rt.feasible && rt.path.length > 1) {
          // aim ~2.5 m (or 4 steps) ahead along the route
          final ahead = m.pack.metric ? 2.5 : m.pack.medianStep * 4;
          var acc = 0.0;
          var target = rt.path.last;
          for (var k = 1; k < rt.path.length; k++) {
            final a = rt.path[k - 1], b = rt.path[k];
            acc += math.sqrt(math.pow(b[0] - a[0], 2) + math.pow(b[2] - a[2], 2));
            if (acc >= ahead) {
              target = b;
              break;
            }
          }
          final bearing = math.atan2(target[2] - f.position[2], target[0] - f.position[0]);
          rel = _wrap(bearing - f.heading);
          rem = rt.length;
        }
      } on RouteError catch (e) {
        err = e.message;
      }
    }
    setState(() {
      fix = f;
      route = rt;
      relAngle = rel;
      remaining = rem;
      error = err ?? (f.usable ? null : 'Not sure where this is yet. Point the camera along the corridor, at doors or signs.');
    });
    _speak(f, rt);
  }

  static double _wrap(double a) {
    while (a > math.pi) {
      a -= 2 * math.pi;
    }
    while (a < -math.pi) {
      a += 2 * math.pi;
    }
    return a;
  }

  String get _instruction {
    final f = fix;
    if (f == null) return 'Point the camera ahead';
    if (!f.usable) return 'Looking for a match…';
    if (widget.goal == null) return f.nearPlace != null ? 'Near ${f.nearPlace}' : 'Located';
    final rt = route;
    if (rt == null || !rt.feasible) return rt?.reason != null ? 'No ${widget.profile == 'wheelchair' ? 'step-free ' : ''}route' : 'Located';
    final arrive = (remaining ?? 99) < (AppScope.read(context).mapById(widget.mapId)?.pack.metric == true ? 1.5 : 0.01);
    if (arrive) return 'You have arrived';
    final a = (relAngle ?? 0) * 180 / math.pi;
    if (a.abs() < 25) return 'Go straight';
    if (a.abs() > 140) return 'Turn around';
    return a > 0 ? (a < 60 ? 'Bear right' : 'Turn right') : (a > -60 ? 'Bear left' : 'Turn left');
  }

  void _speak(LocateFix f, RouteResult? rt) {
    if (!AppScope.read(context).voice) return;
    final hazard = f.hints.where((h) => h.startsWith('Careful')).firstOrNull;
    final parts = <String>[
      ?hazard,
      if (f.usable && widget.goal != null) _instruction,
      if (f.usable && widget.goal != null && remaining != null && (rt?.feasible ?? false) && _instruction != 'You have arrived')
        '${remaining!.round()} metres to ${widget.goalName ?? 'your destination'}.',
      if (f.usable && widget.goal == null && f.nearPlace != null) 'You are near ${f.nearPlace}.',
    ];
    final key = parts.take(2).join('|');
    final now = DateTime.now();
    if (parts.isEmpty || (key == _lastSpoken && now.difference(_lastSpokenAt).inSeconds < 12)) return;
    _lastSpoken = key;
    _lastSpokenAt = now;
    HapticFeedback.mediumImpact();
    Voice.instance.say(parts.join(' '));
  }

  Future<void> _describe() async {
    _loop?.cancel();
    await _shot(caption: true);
    final f = fix;
    if (f == null || !mounted) return;
    final words = [
      if (f.caption != null) 'I see ${f.caption}.',
      ...f.hints,
      if (f.usable && f.nearPlace != null) 'You are near ${f.nearPlace}.',
    ].join(' ');
    if (words.isNotEmpty) Voice.instance.say(words);
  }

  void _askGuide() {
    final f = fix;
    final what = [
      if (f?.caption != null) 'The camera sees: ${f!.caption}.',
      if (f != null && f.hints.isNotEmpty) 'Detected: ${f.hints.join(' ')}',
    ].join(' ');
    Navigator.of(context).push(MaterialPageRoute(
      builder: (_) => AssistantScreen(
        mapId: widget.mapId,
        profile: widget.profile,
        start: f?.usable == true ? f!.node : null,
        route: route,
        seed: f?.usable == true
            ? 'I am at the spot the camera located${f!.nearPlace != null ? ' near ${f.nearPlace}' : ''}. $what What should I do next?'
            : null,
      ),
    ));
  }

  @override
  Widget build(BuildContext context) {
    final s = AppScope.of(context);
    final m = s.mapById(widget.mapId);
    final c = _cam;
    final f = fix;
    final pad = MediaQuery.paddingOf(context);
    return Scaffold(
      backgroundColor: Colors.black,
      body: Stack(fit: StackFit.expand, children: [
        if (c != null && c.value.isInitialized)
          FittedBox(
            fit: BoxFit.cover,
            child: SizedBox(width: c.value.previewSize!.height, height: c.value.previewSize!.width, child: CameraPreview(c)),
          )
        else
          Center(
            child: Padding(
              padding: const EdgeInsets.all(32),
              child: Text(_camError ?? 'Starting camera…', textAlign: TextAlign.center, style: TextStyle(color: AM.muted)),
            ),
          ),
        // vignette for legibility
        const DecoratedBox(
          decoration: BoxDecoration(
            gradient: LinearGradient(
                begin: Alignment.topCenter,
                end: Alignment.bottomCenter,
                colors: [Color(0xAA000000), Colors.transparent, Colors.transparent, Color(0xDD000000)],
                stops: [0, 0.22, 0.5, 1]),
          ),
        ),
        // AR chevrons on the floor
        if (relAngle != null && _instruction != 'You have arrived')
          IgnorePointer(child: AnimatedBuilder(animation: _flow, builder: (_, _) => CustomPaint(painter: _Chevrons(relAngle!, _flow.value)))),
        // detected objects
        if (f != null && c != null && c.value.isInitialized)
          Positioned.fill(
            child: IgnorePointer(
              child: LayoutBuilder(builder: (context, box) {
                // the preview is shown with BoxFit.cover: map photo-normalised boxes through the same crop
                final pw = c.value.previewSize!.height, ph = c.value.previewSize!.width;
                final k = math.max(box.maxWidth / pw, box.maxHeight / ph);
                final dw = pw * k, dh = ph * k;
                final ox = (box.maxWidth - dw) / 2, oy = (box.maxHeight - dh) / 2;
                return Stack(children: [
                  for (final o in f.objects.take(5)) _ObjectBox(o: o, ox: ox, oy: oy, w: dw, h: dh),
                ]);
              }),
            ),
          ),
        // top bar
        Positioned(
          top: pad.top + 8,
          left: 12,
          right: 12,
          child: Row(children: [
            RoundAction(icon: Icons.close_rounded, label: 'Close camera guide', size: 48, onTap: () => Navigator.pop(context, fix)),
            const SizedBox(width: 10),
            Expanded(child: _StatusPill(fix: f, busy: _busy, goalName: widget.goalName)),
            const SizedBox(width: 10),
            RoundAction(
              icon: _auto ? Icons.pause_rounded : Icons.play_arrow_rounded,
              label: _auto ? 'Pause live locating' : 'Resume live locating',
              size: 48,
              onTap: () {
                setState(() => _auto = !_auto);
                if (_auto) _schedule(Duration.zero);
              },
            ),
          ]),
        ),
        // bottom card
        Positioned(
          left: 12,
          right: 12,
          bottom: pad.bottom + 12,
          child: Container(
            padding: const EdgeInsets.fromLTRB(18, 16, 18, 14),
            decoration: BoxDecoration(
              color: AM.surface.withValues(alpha: 0.94),
              borderRadius: BorderRadius.circular(28),
              border: Border.all(color: AM.text.withValues(alpha: 0.1)),
            ),
            child: Column(mainAxisSize: MainAxisSize.min, crossAxisAlignment: CrossAxisAlignment.start, children: [
              Semantics(
                liveRegion: true,
                child: Row(children: [
                  Container(
                    width: 52,
                    height: 52,
                    decoration: BoxDecoration(color: AM.lime, borderRadius: BorderRadius.circular(16)),
                    child: Icon(_icon(), color: AM.onLime, size: 30),
                  ),
                  const SizedBox(width: 14),
                  Expanded(
                    child: Column(crossAxisAlignment: CrossAxisAlignment.start, children: [
                      Text(_instruction, style: Theme.of(context).textTheme.headlineSmall?.copyWith(fontWeight: FontWeight.w800)),
                      if (remaining != null && widget.goal != null && _instruction != 'You have arrived')
                        Text('${remaining!.round()} ${m?.pack.metric == true ? 'm' : 'units'} to ${widget.goalName ?? 'goal'}',
                            style: TextStyle(color: AM.muted, fontWeight: FontWeight.w600)),
                      if (widget.goal == null && f?.usable == true && f?.nearDistance != null)
                        Text('${f!.nearDistance!.toStringAsFixed(1)} m from ${f.nearPlace}', style: TextStyle(color: AM.muted)),
                    ]),
                  ),
                ]),
              ),
              if (error != null) ...[
                const SizedBox(height: 10),
                Text(error!, style: TextStyle(color: AM.amber, height: 1.35)),
              ],
              if (f != null && (f.hints.isNotEmpty || f.caption != null)) ...[
                const SizedBox(height: 12),
                if (f.caption != null)
                  Padding(
                    padding: const EdgeInsets.only(bottom: 8),
                    child: Text('“${f.caption}”', style: TextStyle(color: AM.text, fontStyle: FontStyle.italic)),
                  ),
                Wrap(spacing: 6, runSpacing: 6, children: [
                  for (final h in f.hints)
                    Container(
                      padding: const EdgeInsets.symmetric(horizontal: 10, vertical: 6),
                      decoration: BoxDecoration(
                          color: h.startsWith('Careful') ? AM.text : AM.text.withValues(alpha: 0.08),
                          borderRadius: BorderRadius.circular(12)),
                      child: Text(h, style: TextStyle(fontSize: 13, fontWeight: FontWeight.w600, color: h.startsWith('Careful') ? AM.bg : AM.text)),
                    ),
                ]),
              ],
              const SizedBox(height: 14),
              Row(children: [
                Expanded(
                  child: OutlinedButton.icon(
                    onPressed: _describing ? null : _describe,
                    icon: _describing
                        ? const SizedBox(width: 18, height: 18, child: CircularProgressIndicator(strokeWidth: 2))
                        : const Icon(Icons.record_voice_over_rounded),
                    label: const Text('Describe'),
                  ),
                ),
                const SizedBox(width: 10),
                Expanded(
                  child: FilledButton.icon(
                    onPressed: _askGuide,
                    icon: const Icon(Icons.auto_awesome_rounded),
                    label: const Text('Ask Guide'),
                  ),
                ),
              ]),
            ]),
          ),
        ),
      ]),
    );
  }

  IconData _icon() => switch (_instruction) {
        'Go straight' => Icons.straight_rounded,
        'Turn right' => Icons.turn_right_rounded,
        'Bear right' => Icons.turn_slight_right_rounded,
        'Turn left' => Icons.turn_left_rounded,
        'Bear left' => Icons.turn_slight_left_rounded,
        'Turn around' => Icons.u_turn_left_rounded,
        'You have arrived' => Icons.flag_rounded,
        _ => Icons.center_focus_strong_rounded,
      };
}

class _StatusPill extends StatelessWidget {
  const _StatusPill({required this.fix, required this.busy, this.goalName});
  final LocateFix? fix;
  final bool busy;
  final String? goalName;
  @override
  Widget build(BuildContext context) {
    final f = fix;
    final (col, label) = f == null
        ? (AM.muted, busy ? 'Locating…' : 'Camera guide')
        : !f.usable
            ? (AM.amber, 'Searching${busy ? '…' : ''}')
            : (f.confidence == 'high' ? AM.lime : AM.teal, '${f.nearPlace != null ? 'Near ${f.nearPlace}' : 'Located'} · ${f.confidence}');
    return Container(
      padding: const EdgeInsets.symmetric(horizontal: 14, vertical: 10),
      decoration: BoxDecoration(color: Colors.black.withValues(alpha: 0.55), borderRadius: BorderRadius.circular(99)),
      child: Row(children: [
        Container(width: 9, height: 9, decoration: BoxDecoration(color: col, shape: BoxShape.circle, boxShadow: [BoxShadow(color: col, blurRadius: 8)])),
        const SizedBox(width: 8),
        Expanded(child: Text(label, maxLines: 1, overflow: TextOverflow.ellipsis, style: const TextStyle(fontWeight: FontWeight.w700, color: Colors.white))),
        if (busy) const SizedBox(width: 14, height: 14, child: CircularProgressIndicator(strokeWidth: 2)),
      ]),
    );
  }
}

class _ObjectBox extends StatelessWidget {
  const _ObjectBox({required this.o, required this.ox, required this.oy, required this.w, required this.h});
  final Map<String, dynamic> o;
  final double ox, oy, w, h;
  @override
  Widget build(BuildContext context) {
    final b = (o['bbox'] as List).map((v) => (v as num).toDouble()).toList();
    final hazard = o['hazard'] == true;
    final col = hazard ? AM.rose : AM.lime;
    return Positioned(
        left: ox + b[0] * w,
        top: oy + b[1] * h,
        width: (b[2] - b[0]) * w,
        height: (b[3] - b[1]) * h,
        child: IgnorePointer(
          child: Container(
            decoration: BoxDecoration(border: Border.all(color: col.withValues(alpha: 0.9), width: 2), borderRadius: BorderRadius.circular(12)),
            alignment: Alignment.topLeft,
            child: Container(
              margin: const EdgeInsets.all(4),
              padding: const EdgeInsets.symmetric(horizontal: 6, vertical: 2),
              decoration: BoxDecoration(color: col, borderRadius: BorderRadius.circular(6)),
              child: Text(o['label'] as String? ?? '', style: TextStyle(fontSize: 11, fontWeight: FontWeight.w800, color: AM.onLime)),
            ),
          ),
        ),
      );
  }
}

/// Lime chevrons "painted on the floor", bending toward the direction to walk.
class _Chevrons extends CustomPainter {
  _Chevrons(this.rel, this.t);
  final double rel; // radians, + = right
  final double t;
  @override
  void paint(Canvas canvas, Size size) {
    final turn = rel.clamp(-math.pi / 2, math.pi / 2);
    final base = Offset(size.width / 2, size.height * 0.9);
    for (var k = 0; k < 6; k++) {
      final f = (k + t) / 6; // 0 near, 1 far
      final depth = math.pow(f, 0.85).toDouble();
      // path curves toward the turn as it goes "into" the scene
      final x = base.dx + math.sin(turn * depth) * size.width * 0.45 * depth;
      final y = base.dy - depth * size.height * 0.42;
      final scale = 1 - depth * 0.72;
      final ang = turn * depth;
      canvas.save();
      canvas.translate(x, y);
      canvas.rotate(ang);
      canvas.scale(scale, scale * 0.55); // flatten: lying on the floor
      final w = size.width * 0.2;
      final p = Path()
        ..moveTo(-w, w * 0.35)
        ..lineTo(0, -w * 0.45)
        ..lineTo(w, w * 0.35)
        ..lineTo(w * 0.62, w * 0.62)
        ..lineTo(0, -w * 0.02)
        ..lineTo(-w * 0.62, w * 0.62)
        ..close();
      final a = (1 - (f - 0.85).clamp(0, 1) / 0.15) * (f < 0.12 ? f / 0.12 : 1);
      canvas.drawPath(p, Paint()
        ..color = AM.lime.withValues(alpha: 0.35 * a)
        ..maskFilter = const MaskFilter.blur(BlurStyle.normal, 10));
      canvas.drawPath(p, Paint()..color = AM.lime.withValues(alpha: 0.92 * a));
      canvas.restore();
    }
  }

  @override
  bool shouldRepaint(covariant _Chevrons o) => o.t != t || o.rel != rel;
}

/// Small helper so other screens can open the guide for a map.
void openCameraGuide(BuildContext context, LocalMap m, String profile, {String? goal, String? goalName}) {
  Navigator.of(context).push(MaterialPageRoute(
      builder: (_) => CameraGuideScreen(mapId: m.pack.id, profile: profile, goal: goal, goalName: goalName)));
}
