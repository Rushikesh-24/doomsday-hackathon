import 'package:flutter/material.dart';

import '../../core/api.dart';
import '../../core/app_state.dart';
import '../../core/models.dart';
import '../../core/router.dart';
import '../../core/voice.dart';
import '../theme.dart';
import '../widgets/common.dart';
import '../widgets/plan_view.dart';
import 'assistant.dart';
import 'camera_guide.dart';
import 'guidance.dart';
import 'report.dart';

const _nearest = {
  'elevator': ('Nearest lift', Icons.elevator_rounded),
  'ramp': ('Nearest ramp', Icons.accessible_rounded),
  'toilet_sign': ('Nearest toilet', Icons.wc_rounded),
  'entrance_exit': ('Nearest exit', Icons.door_front_door_rounded),
};

class MapScreen extends StatefulWidget {
  const MapScreen({super.key, required this.mapId});
  final String mapId;
  @override
  State<MapScreen> createState() => _MapScreenState();
}

class _MapScreenState extends State<MapScreen> {
  RouteResult? route;
  String? start; // resolved node id
  double? startHeading; // from the camera guide
  bool startFromCamera = false;
  String? selected;
  String? error;
  late String profile;
  bool _init = false;
  bool showIssues = true, showScan = false;
  int fit = 0;

  @override
  void didChangeDependencies() {
    super.didChangeDependencies();
    if (!_init) {
      profile = AppScope.read(context).profile;
      _init = true;
    }
  }

  Wayfinder? _wf(AppState s) {
    final m = s.mapById(widget.mapId);
    return m == null ? null : Wayfinder(m.pack);
  }

  int? floor; // buildings: the floor being shown

  String _startOf(Wayfinder wf) => start ?? (wf.pack.isBuilding ? wf.pack.startId! : wf.resolve('start'));

  void _go(Wayfinder wf, String goalRef) {
    try {
      final RouteResult r;
      if (wf.pack.isBuilding) {
        r = wf.routeBuilding(_startOf(wf), goalRef, profile: profile);
      } else {
        r = wf.route(_startOf(wf), wf.resolve(goalRef), profile: profile);
      }
      setState(() {
        route = r;
        error = null;
        selected = r.goal;
        if (r.floors.isNotEmpty) floor = r.floors.first;
      });
      if (AppScope.read(context).voice) {
        Voice.instance.say(r.feasible
            ? '${r.goalName}. ${_summary(r, wf.pack)}. Press start guidance when you are ready.'
            : 'No ${profile == 'wheelchair' ? 'step-free ' : ''}route found. ${r.reason}');
      }
    } on RouteError catch (e) {
      setState(() => error = e.message);
    }
  }

  void _ask(Wayfinder wf, String text) {
    try {
      final q = wf.parseRequest(text);
      if (q.start != null) {
        start = wf.resolve(q.start!);
        startFromCamera = false;
      }
      _go(wf, q.goal);
    } on RouteError catch (e) {
      setState(() => error = e.message);
    }
  }

  String _summary(RouteResult r, MapPack p) {
    if (r.units == 'm' && r.length != null) {
      final speed = {'wheelchair': 0.9, 'low_vision': 0.8}[r.profile] ?? 1.3;
      final min = (r.length! / speed / 60).ceil();
      return 'About $min minute${min == 1 ? '' : 's'}, ${r.length!.round()} metres';
    }
    return '${r.directions.where((d) => d.kind == 'turn').length} turns';
  }

  Future<void> _search(Wayfinder wf) async {
    final res = await showModalBottomSheet<(String, bool)>(
      context: context,
      isScrollControlled: true,
      backgroundColor: Colors.transparent,
      showDragHandle: false,
      builder: (_) => _SearchSheet(wf: wf, currentStart: _startOf(wf)),
    );
    if (res == null || !mounted) return;
    final (value, isText) = res;
    if (isText) {
      _ask(wf, value);
    } else {
      _go(wf, value);
    }
  }

  Future<void> _pickStart(Wayfinder wf) async {
    final v = await showModalBottomSheet<String>(
      context: context,
      builder: (_) => SafeArea(
        child: ListView(shrinkWrap: true, padding: const EdgeInsets.fromLTRB(16, 0, 16, 8), children: [
          const Padding(padding: EdgeInsets.fromLTRB(8, 0, 8, 8), child: SectionLabel('I am at')),
          ListTile(
            leading: CircleAvatar(backgroundColor: AM.lime, child: Icon(Icons.photo_camera_rounded, color: AM.onLime, size: 18)),
            title: const Text('Find me with the camera', style: TextStyle(fontWeight: FontWeight.w700)),
            subtitle: const Text('Point the phone along the corridor (online)'),
            onTap: () => Navigator.pop(context, '__camera'),
          ),
          for (final p in wf.pack.places)
            ListTile(
              leading: CircleAvatar(backgroundColor: AM.place(p.kind), child: Icon(Icons.place_rounded, color: AM.bg, size: 18)),
              title: Text(p.name, style: const TextStyle(fontWeight: FontWeight.w700)),
              subtitle: p.isStart ? Text(['Default start', if (p.floorName != null) p.floorName!].join(' · ')) : (p.floorName != null ? Text(p.floorName!) : null),
              onTap: () => Navigator.pop(context, p.id),
            ),
          ListTile(
            leading: CircleAvatar(backgroundColor: AM.surface2, child: Icon(Icons.flag_rounded, color: AM.green, size: 18)),
            title: const Text('Start of the scanned walk', style: TextStyle(fontWeight: FontWeight.w700)),
            onTap: () => Navigator.pop(context, 'cam0'),
          ),
        ]),
      ),
    );
    if (v == null || !mounted) return;
    if (v == '__camera') return _camera(wf);
    setState(() {
      start = v;
      startFromCamera = false;
      startHeading = null;
    });
    if (route?.goal != null) _go(wf, route!.goal!);
  }

  Future<void> _camera(Wayfinder wf) async {
    final s = AppScope.read(context);
    final m = s.mapById(widget.mapId)!;
    final fix = await Navigator.of(context).push<LocateFix>(MaterialPageRoute(
        builder: (_) => CameraGuideScreen(
            mapId: m.pack.id, profile: profile, goal: route?.goal, goalName: route?.goalName, floorLevel: m.pack.isBuilding ? _floorOf(wf) : null)));
    if (fix == null || !fix.usable || !mounted) return;
    setState(() {
      start = fix.node;
      startHeading = fix.heading;
      startFromCamera = true;
    });
    if (route?.goal != null) _go(wf, route!.goal!);
  }

  int _floorOf(Wayfinder wf) =>
      floor ?? wf.nodes[_startOf(wf)]?.floor ?? (wf.pack.floors.isEmpty ? 0 : wf.pack.floors.first.level);

  Future<void> _pinHere() async {
    final s = AppScope.read(context);
    final msg = ScaffoldMessenger.of(context);
    try {
      await s.pinHere(widget.mapId);
      msg.showSnackBar(const SnackBar(content: Text('Pinned. Phones near here will open this map first.')));
    } on ApiException catch (e) {
      msg.showSnackBar(SnackBar(content: Text(e.message)));
    }
  }

  void _issue(Issue i) {
    showModalBottomSheet<void>(
      context: context,
      builder: (_) => SafeArea(
        child: Padding(
          padding: const EdgeInsets.fromLTRB(22, 0, 22, 22),
          child: Column(mainAxisSize: MainAxisSize.min, crossAxisAlignment: CrossAxisAlignment.start, children: [
            Row(children: [
              Container(
                padding: const EdgeInsets.symmetric(horizontal: 10, vertical: 4),
                decoration: BoxDecoration(color: AM.severity(i.severity).withValues(alpha: 0.18), borderRadius: BorderRadius.circular(99)),
                child: Text('${i.severity.toUpperCase()} PRIORITY',
                    style: TextStyle(fontSize: 11, fontWeight: FontWeight.w800, letterSpacing: 0.8, color: AM.severity(i.severity))),
              ),
            ]),
            const SizedBox(height: 12),
            Text(i.title, style: Theme.of(context).textTheme.titleLarge),
            const SizedBox(height: 6),
            Text(i.detail, style: TextStyle(color: AM.muted, height: 1.45)),
            const SizedBox(height: 16),
            OutlinedButton.icon(onPressed: () => Voice.instance.say('${i.title}. ${i.detail}'), icon: const Icon(Icons.volume_up_rounded), label: const Text('Read aloud')),
          ]),
        ),
      ),
    );
  }

  Future<void> _layers() async {
    await showModalBottomSheet<void>(
      context: context,
      builder: (_) => StatefulBuilder(
        builder: (ctx, set) => SafeArea(
          child: Column(mainAxisSize: MainAxisSize.min, children: [
            const Padding(padding: EdgeInsets.fromLTRB(24, 0, 24, 8), child: Align(alignment: Alignment.centerLeft, child: SectionLabel('Map layers'))),
            SwitchListTile(
              value: showIssues,
              onChanged: (v) {
                set(() {});
                setState(() => showIssues = v);
              },
              secondary: const Icon(Icons.report_rounded),
              title: const Text('Accessibility problems', style: TextStyle(fontWeight: FontWeight.w700)),
              subtitle: const Text('Narrow spots, stairs without alternatives, obstacles'),
            ),
            SwitchListTile(
              value: showScan,
              onChanged: (v) {
                set(() {});
                setState(() => showScan = v);
              },
              secondary: const Icon(Icons.blur_on_rounded),
              title: const Text('Raw 3D scan points', style: TextStyle(fontWeight: FontWeight.w700)),
              subtitle: const Text('What the camera reconstructed, under the plan'),
            ),
            ListTile(
              leading: const Icon(Icons.push_pin_rounded),
              title: const Text('Pin this map to my location', style: TextStyle(fontWeight: FontWeight.w700)),
              subtitle: const Text('Phones here will open this map automatically'),
              onTap: () {
                Navigator.pop(ctx);
                _pinHere();
              },
            ),
            const SizedBox(height: 8),
          ]),
        ),
      ),
    );
  }

  @override
  Widget build(BuildContext context) {
    final s = AppScope.of(context);
    final m = s.mapById(widget.mapId);
    final wf = _wf(s);
    if (m == null || wf == null) {
      return Scaffold(appBar: AppBar(), body: const Center(child: Text('This map is no longer on the phone.')));
    }
    final startNode = wf.nodes[_startOf(wf)];
    final building = m.pack.isBuilding;
    final level = building ? _floorOf(wf) : null;
    final viewPack = building ? m.pack.floorView(level!) : m.pack;
    final rt = route?.feasible == true ? route! : null;
    final routeHere = rt == null
        ? null
        : !building
            ? rt.path
            : [for (var k = 0; k < rt.path.length; k++) if (k < rt.pathFloors.length && rt.pathFloors[k] == level) rt.path[k]];
    final startName = startFromCamera ? 'You (camera)' : startNode?.name ?? (startNode?.id == 'cam0' ? 'Start of the walk' : 'Your position');
    final a = m.pack.access;
    final big = bigText(context);
    final bottomInset = MediaQuery.sizeOf(context).height * (route == null ? (big ? 0.42 : 0.34) : (big ? 0.66 : 0.42));
    return Scaffold(
      body: Stack(children: [
        Positioned.fill(
          child: PlanView(
            pack: viewPack,
            planImage: building ? null : m.plan,
            route: routeHere,
            you: building && startNode?.floor != level ? null : startNode?.pos,
            youHeading: startHeading,
            selectedPlace: selected,
            onPlaceTap: (id) => _go(wf, id),
            onIssueTap: _issue,
            showIssues: showIssues,
            showScan: showScan,
            fitNonce: fit,
            padding: EdgeInsets.fromLTRB(building ? 78 : 28, 170, 76, bottomInset + 20),
          ),
        ),
        // top bar
        SafeArea(
          child: Padding(
            padding: const EdgeInsets.fromLTRB(12, 8, 12, 0),
            child: Column(children: [
              Row(children: [
                RoundAction(icon: Icons.arrow_back_rounded, label: 'Back', size: 48, onTap: () => Navigator.pop(context)),
                const SizedBox(width: 10),
                Expanded(
                  child: Semantics(
                    button: true,
                    label: 'Search where to go',
                    child: Material(
                      color: AM.surface2,
                      shape: StadiumBorder(side: BorderSide(color: AM.text.withValues(alpha: 0.12))),
                      child: InkWell(
                        borderRadius: BorderRadius.circular(99),
                        onTap: () => _search(wf),
                        child: Padding(
                          padding: const EdgeInsets.symmetric(horizontal: 18, vertical: 14),
                          child: Row(children: [
                            Icon(Icons.search_rounded, color: AM.lime),
                            const SizedBox(width: 10),
                            Expanded(
                              child: Text(route?.goalName ?? 'Where to?',
                                  overflow: TextOverflow.ellipsis,
                                  style: TextStyle(fontWeight: FontWeight.w700, color: route == null ? AM.muted : AM.text, fontSize: 15)),
                            ),
                            Icon(Icons.mic_none_rounded, color: AM.muted),
                          ]),
                        ),
                      ),
                    ),
                  ),
                ),
                if (a != null) ...[
                  const SizedBox(width: 10),
                  Semantics(
                    button: true,
                    label: 'Accessibility report, score ${a.score}',
                    child: GestureDetector(
                      onTap: () => showAccessibilityReport(context, m.pack, onIssue: _issue),
                      child: Container(
                        padding: const EdgeInsets.all(4),
                        decoration: BoxDecoration(color: AM.surface2, shape: BoxShape.circle, border: Border.all(color: AM.stroke)),
                        child: ScoreRing(score: a.score, grade: a.grade, size: 44, stroke: 4.5),
                      ),
                    ),
                  ),
                ],
              ]),
              const SizedBox(height: 10),
              Row(children: [
                Flexible(child: _Chip(icon: startFromCamera ? Icons.photo_camera_rounded : Icons.trip_origin_rounded, label: 'From: $startName', onTap: () => _pickStart(wf))),
                const SizedBox(width: 8),
                _Chip(
                  icon: profileInfo(profile).icon,
                  label: profileInfo(profile).label,
                  onTap: () {
                    final i = profiles.indexWhere((p) => p.key == profile);
                    setState(() => profile = profiles[(i + 1) % profiles.length].key);
                    if (route?.goal != null) _go(wf, route!.goal!);
                  },
                ),
              ]),
            ]),
          ),
        ),
        if (building)
          Positioned(
            left: 12,
            top: MediaQuery.paddingOf(context).top + 130,
            child: _FloorPicker(
              floors: m.pack.floors,
              current: level!,
              onRoute: rt?.floors ?? const [],
              here: startNode?.floor,
              onPick: (lv) => setState(() {
                floor = lv;
                fit++;
              }),
            ),
          ),
        // floating actions
        Positioned(
          right: 14,
          bottom: bottomInset + 16,
          child: Column(children: [
            RoundAction(
              icon: Icons.photo_camera_rounded,
              label: 'Camera guide: find where I am',
              onTap: () => _camera(wf),
            ),
            const SizedBox(height: 12),
            RoundAction(icon: Icons.layers_rounded, label: 'Map layers', onTap: _layers),
            const SizedBox(height: 12),
            RoundAction(icon: Icons.center_focus_strong_rounded, label: 'Re-centre the map', onTap: () => setState(() => fit++)),
            const SizedBox(height: 12),
            RoundAction(
              icon: Icons.auto_awesome_rounded,
              label: 'Ask the guide assistant',
              primary: true,
              onTap: () async {
                final r = await Navigator.of(context).push<RouteResult>(MaterialPageRoute(
                    builder: (_) => AssistantScreen(mapId: m.pack.id, profile: profile, start: _startOf(wf), route: route)));
                if (r != null && mounted) {
                  setState(() {
                    route = r;
                    selected = r.goal;
                  });
                }
              },
            ),
          ]),
        ),
        _Sheet(
          // a new route starts the sheet fresh: summary and "Start guidance" on top, not the old scroll offset
          key: ValueKey('${route?.goal}|${route?.profile}|${route?.startName}|${error != null}'),
          wf: wf,
          route: route,
          error: error,
          profile: profile,
          summary: route != null && route!.feasible ? _summary(route!, wf.pack) : null,
          onPick: (ref) => _go(wf, ref),
          onClear: () => setState(() {
            route = null;
            selected = null;
            error = null;
          }),
          onGuide: () => Navigator.of(context).push(MaterialPageRoute(builder: (_) => GuidanceScreen(map: m, route: route!))),
          onCamera: () => _camera(wf),
          onReport: a == null ? null : () => showAccessibilityReport(context, m.pack, onIssue: _issue),
        ),
      ]),
    );
  }
}

class _Chip extends StatelessWidget {
  const _Chip({required this.icon, required this.label, required this.onTap});
  final IconData icon;
  final String label;
  final VoidCallback onTap;
  @override
  Widget build(BuildContext context) => Material(
        color: AM.surface2,
        shape: StadiumBorder(side: BorderSide(color: AM.text.withValues(alpha: 0.12))),
        child: InkWell(
          borderRadius: BorderRadius.circular(99),
          onTap: onTap,
          child: Padding(
            padding: const EdgeInsets.symmetric(horizontal: 13, vertical: 10),
            child: Row(mainAxisSize: MainAxisSize.min, children: [
              Icon(icon, size: 16, color: AM.lime),
              const SizedBox(width: 6),
              Flexible(child: Text(label, overflow: TextOverflow.ellipsis, style: const TextStyle(fontWeight: FontWeight.w700, fontSize: 13))),
            ]),
          ),
        ),
      );
}

class _Sheet extends StatelessWidget {
  const _Sheet({
    super.key,
    required this.wf,
    required this.route,
    required this.error,
    required this.profile,
    required this.summary,
    required this.onPick,
    required this.onClear,
    required this.onGuide,
    required this.onCamera,
    required this.onReport,
  });
  final Wayfinder wf;
  final RouteResult? route;
  final String? error;
  final String profile;
  final String? summary;
  final ValueChanged<String> onPick;
  final VoidCallback onClear;
  final VoidCallback onGuide;
  final VoidCallback onCamera;
  final VoidCallback? onReport;

  @override
  Widget build(BuildContext context) {
    final r = route;
    final a = wf.pack.access;
    return DraggableScrollableSheet(
      // large text: open taller so the summary and "Start guidance" stay in view
      initialChildSize: r == null ? (bigText(context) ? 0.42 : 0.34) : (bigText(context) ? 0.66 : 0.42),
      minChildSize: 0.16,
      maxChildSize: 0.92,
      snap: true,
      builder: (context, sc) => Container(
        decoration: BoxDecoration(
          color: AM.surface,
          borderRadius: const BorderRadius.vertical(top: Radius.circular(32)),
          border: Border(top: BorderSide(color: AM.text.withValues(alpha: 0.1))),
          boxShadow: [BoxShadow(color: Colors.black.withValues(alpha: AM.isDark ? 1 : 0.12), blurRadius: 30, offset: const Offset(0, -6))],
        ),
        child: ListView(controller: sc, padding: const EdgeInsets.fromLTRB(20, 10, 20, 28), children: [
          Center(child: Container(width: 44, height: 5, decoration: BoxDecoration(color: AM.stroke, borderRadius: BorderRadius.circular(9)))),
          const SizedBox(height: 16),
          if (error != null)
            Container(
              margin: const EdgeInsets.only(bottom: 12),
              padding: const EdgeInsets.all(12),
              decoration: BoxDecoration(color: AM.rose.withValues(alpha: 0.14), borderRadius: BorderRadius.circular(16)),
              child: Text(error!, style: TextStyle(color: AM.text, fontWeight: FontWeight.w600)),
            ),
          if (r == null) ...[
            Text(wf.pack.name, style: Theme.of(context).textTheme.titleLarge),
            const SizedBox(height: 2),
            Text('${wf.pack.places.length} places · tap a pin or pick below', style: TextStyle(color: AM.muted)),
            const SizedBox(height: 18),
            const SectionLabel('Popular destinations'),
            const SizedBox(height: 10),
            Wrap(spacing: 8, runSpacing: 8, children: [
              for (final p in wf.pack.places.where((p) => !p.isStart && !const ['lift', 'stairwell', 'stairs'].contains(p.kind) || !wf.pack.isBuilding && !p.isStart))
                ActionChip(
                  avatar: Icon(Icons.place_rounded, size: 18, color: AM.place(p.kind)),
                  label: Text(p.floor != null ? '${p.name} · ${p.floor == 0 ? 'G' : p.floor}' : p.name, style: const TextStyle(fontWeight: FontWeight.w700)),
                  onPressed: () => onPick(p.id),
                ),
              for (final e in _nearest.entries.where((e) => wf.classes.contains(e.key)))
                ActionChip(avatar: Icon(e.value.$2, size: 18, color: AM.teal), label: Text(e.value.$1), onPressed: () => onPick(e.key)),
            ]),
            if (a != null && onReport != null) ...[
              const SizedBox(height: 18),
              _ReportTeaser(a: a, onTap: onReport!),
            ],
          ] else if (!r.feasible) ...[
            Row(children: [
              Icon(Icons.block_rounded, color: AM.rose),
              const SizedBox(width: 8),
              Expanded(child: Text('No ${profile == 'wheelchair' ? 'step-free ' : ''}route', style: Theme.of(context).textTheme.titleLarge)),
              IconButton(onPressed: onClear, icon: const Icon(Icons.close_rounded), tooltip: 'Clear'),
            ]),
            const SizedBox(height: 6),
            Text(r.reason ?? '', style: TextStyle(color: AM.muted, height: 1.4)),
            const SizedBox(height: 12),
            if (profile == 'wheelchair')
              Container(
                padding: const EdgeInsets.all(14),
                decoration: BoxDecoration(color: AM.violetSoft, borderRadius: BorderRadius.circular(18)),
                child: Text(
                  wf.classes.contains('elevator')
                      ? 'Try "Nearest lift", or ask staff for an accessible way around the stairs.'
                      : 'Ask staff about a lift or ramp. The scan did not find a step-free way from here.',
                  style: const TextStyle(height: 1.4),
                ),
              ),
          ] else ...[
            Row(crossAxisAlignment: CrossAxisAlignment.start, children: [
              Expanded(
                child: Column(crossAxisAlignment: CrossAxisAlignment.start, children: [
                  Text('${r.startName} → ${r.goalName}', maxLines: 2, overflow: TextOverflow.ellipsis, style: TextStyle(color: AM.muted)),
                  const SizedBox(height: 4),
                  // headline is already 24 pt: cap its growth so the action button stays reachable at 2x text
                  MediaQuery.withClampedTextScaling(
                    maxScaleFactor: 1.4,
                    child: Text(summary ?? '', style: Theme.of(context).textTheme.headlineSmall?.copyWith(fontWeight: FontWeight.w800)),
                  ),
                  Text(r.profileLabel ?? '', style: TextStyle(color: AM.lime, fontWeight: FontWeight.w600)),
                ]),
              ),
              IconButton(onPressed: onClear, icon: const Icon(Icons.close_rounded), tooltip: 'Clear route'),
            ]),
            if (r.warnings.isNotEmpty) ...[
              const SizedBox(height: 10),
              Wrap(spacing: 6, runSpacing: 6, children: [
                for (final w in r.warnings)
                  Chip(
                    label: Text(w.replaceAll('_', ' ')),
                    backgroundColor: AM.amber.withValues(alpha: 0.16),
                    side: BorderSide.none,
                    labelStyle: TextStyle(color: AM.amber, fontWeight: FontWeight.w700, fontSize: 12),
                  ),
              ]),
            ],
            const SizedBox(height: 16),
            FilledButton.icon(onPressed: onGuide, icon: const Icon(Icons.navigation_rounded), label: const Text('Start guidance')),
            const SizedBox(height: 10),
            Row(children: [
              Expanded(child: OutlinedButton.icon(onPressed: onCamera, icon: const Icon(Icons.photo_camera_rounded), label: const Text('Camera guide'))),
              const SizedBox(width: 10),
              OutlinedButton(
                onPressed: () => Voice.instance.say(r.spoken),
                style: OutlinedButton.styleFrom(minimumSize: const Size(56, 52)),
                child: const Icon(Icons.volume_up_rounded, semanticLabel: 'Read the route aloud'),
              ),
            ]),
            const SizedBox(height: 20),
            for (final (i, d) in r.directions.indexed)
              _StepRow(d: d, last: i == r.directions.length - 1, showFloor: wf.pack.isBuilding),
          ],
        ]),
      ),
    );
  }
}

class _ReportTeaser extends StatelessWidget {
  const _ReportTeaser({required this.a, required this.onTap});
  final AccessScore a;
  final VoidCallback onTap;
  @override
  Widget build(BuildContext context) {
    final high = a.issues.where((i) => i.severity == 'high').length;
    return Material(
      color: AM.surface2,
      borderRadius: BorderRadius.circular(22),
      child: InkWell(
        borderRadius: BorderRadius.circular(22),
        onTap: onTap,
        child: Padding(
          padding: const EdgeInsets.all(14),
          child: Row(children: [
            ScoreRing(score: a.score, grade: a.grade, size: 54, stroke: 5),
            const SizedBox(width: 14),
            Expanded(
              child: Column(crossAxisAlignment: CrossAxisAlignment.start, children: [
                Text(a.verdict, style: const TextStyle(fontWeight: FontWeight.w800, fontSize: 15)),
                const SizedBox(height: 2),
                Text(
                  a.issues.isEmpty ? 'No problems found on the scanned path' : '${a.issues.length} problem spot${a.issues.length == 1 ? '' : 's'}${high > 0 ? ' · $high high priority' : ''}',
                  style: TextStyle(color: AM.muted, fontSize: 13),
                ),
              ]),
            ),
            Icon(Icons.chevron_right_rounded, color: AM.muted),
          ]),
        ),
      ),
    );
  }
}

IconData stepIcon(DirectionStep d) => switch (d.kind) {
      'start' => Icons.flag_rounded,
      'arrive' => Icons.place_rounded,
      'lift' => Icons.elevator_rounded,
      'stairs' => Icons.stairs_rounded,
      'turn' => d.text.startsWith('Turn around')
          ? Icons.u_turn_left_rounded
          : (d.angle ?? 0) > 0
              ? (d.text.startsWith('Bear') ? Icons.turn_slight_right_rounded : Icons.turn_right_rounded)
              : (d.text.startsWith('Bear') ? Icons.turn_slight_left_rounded : Icons.turn_left_rounded),
      _ => Icons.straight_rounded,
    };

Color stepColor(DirectionStep d) =>
    switch (d.kind) { 'start' => AM.green, 'arrive' => AM.lime, 'lift' || 'stairs' => AM.lime, 'turn' => AM.violet, _ => AM.surface2 };

class _StepRow extends StatelessWidget {
  const _StepRow({required this.d, required this.last, this.showFloor = false});
  final DirectionStep d;
  final bool last, showFloor;
  @override
  Widget build(BuildContext context) {
    final dark = d.kind == 'arrive' || d.kind == 'start' || d.kind == 'lift' || d.kind == 'stairs';
    final change = d.kind == 'lift' || d.kind == 'stairs';
    return IntrinsicHeight(
      child: Row(crossAxisAlignment: CrossAxisAlignment.stretch, children: [
        Column(children: [
          CircleAvatar(radius: 17, backgroundColor: stepColor(d), child: Icon(stepIcon(d), size: 18, color: dark ? AM.onLime : AM.text)),
          if (!last) Expanded(child: Container(width: 2, color: AM.stroke)),
        ]),
        const SizedBox(width: 12),
        Expanded(
          child: Padding(
            padding: const EdgeInsets.only(top: 6, bottom: 16),
            child: Column(crossAxisAlignment: CrossAxisAlignment.start, children: [
              if (showFloor && d.floor != null)
                Padding(
                  padding: const EdgeInsets.only(bottom: 4),
                  child: Text(change ? 'FLOOR ${_short(d.floor!)} → ${_short(d.toFloor ?? d.floor!)}' : 'FLOOR ${_short(d.floor!)}',
                      style: TextStyle(fontSize: 10, letterSpacing: 1.2, fontWeight: FontWeight.w800, color: AM.muted)),
                ),
              Text(d.text, style: TextStyle(fontWeight: d.kind == 'turn' || change ? FontWeight.w800 : FontWeight.w500, height: 1.35)),
              for (final n in d.notes)
                Container(
                  margin: const EdgeInsets.only(top: 6),
                  padding: const EdgeInsets.symmetric(horizontal: 10, vertical: 6),
                  decoration: BoxDecoration(color: AM.amber.withValues(alpha: 0.14), borderRadius: BorderRadius.circular(10)),
                  child: Text(n, style: TextStyle(fontSize: 13, color: AM.text)),
                ),
            ]),
          ),
        ),
      ]),
    );
  }
}

String _short(int level) => level == 0 ? 'G' : '$level';

/// Lift-panel style floor buttons (top floor first). Floors the route uses get a dot; "you" get a ring.
class _FloorPicker extends StatelessWidget {
  const _FloorPicker({required this.floors, required this.current, required this.onRoute, required this.onPick, this.here});
  final List<BuildingFloor> floors;
  final int current;
  final List<int> onRoute;
  final int? here;
  final ValueChanged<int> onPick;
  @override
  Widget build(BuildContext context) {
    final sorted = [...floors]..sort((a, b) => b.level.compareTo(a.level));
    return Container(
      padding: const EdgeInsets.all(5),
      decoration: BoxDecoration(color: AM.surface2, borderRadius: BorderRadius.circular(26), border: Border.all(color: AM.text.withValues(alpha: 0.12))),
      child: Column(mainAxisSize: MainAxisSize.min, children: [
        for (final f in sorted)
          Semantics(
            button: true,
            selected: f.level == current,
            label: '${f.name}${onRoute.contains(f.level) ? ', on your route' : ''}${here == f.level ? ', you are here' : ''}',
            excludeSemantics: true,
            child: GestureDetector(
              onTap: () => onPick(f.level),
              child: Container(
                width: 44,
                height: 44,
                margin: const EdgeInsets.symmetric(vertical: 2),
                decoration: BoxDecoration(
                  color: f.level == current ? AM.text : Colors.transparent,
                  shape: BoxShape.circle,
                  border: here == f.level && f.level != current ? Border.all(color: AM.text, width: 2) : null,
                ),
                child: Stack(alignment: Alignment.center, children: [
                  Text(f.short, style: TextStyle(fontWeight: FontWeight.w800, fontSize: 16, color: f.level == current ? AM.bg : AM.text)),
                  if (onRoute.contains(f.level))
                    Positioned(
                      bottom: 5,
                      child: Container(width: 5, height: 5, decoration: BoxDecoration(color: f.level == current ? AM.bg : AM.text, shape: BoxShape.circle)),
                    ),
                ]),
              ),
            ),
          ),
      ]),
    );
  }
}

class _SearchSheet extends StatefulWidget {
  const _SearchSheet({required this.wf, required this.currentStart});
  final Wayfinder wf;
  final String currentStart;
  @override
  State<_SearchSheet> createState() => _SearchSheetState();
}

class _SearchSheetState extends State<_SearchSheet> {
  final _c = TextEditingController();
  bool listening = false;

  @override
  void dispose() {
    Voice.instance.stopListening();
    _c.dispose();
    super.dispose();
  }

  Future<void> _mic() async {
    if (listening) {
      await Voice.instance.stopListening();
      setState(() => listening = false);
      return;
    }
    final ok = await Voice.instance.listen(
      onText: (t) => setState(() => _c.text = t),
      onDone: (t) {
        if (mounted && t.trim().isNotEmpty) Navigator.pop(context, (t, true));
      },
    );
    if (!ok && mounted) {
      ScaffoldMessenger.of(context).showSnackBar(const SnackBar(content: Text('Speech recognition is not available on this phone.')));
    }
    setState(() => listening = ok);
  }

  @override
  Widget build(BuildContext context) {
    final q = _c.text.trim().toLowerCase();
    final places = widget.wf.pack.places
        .where((p) => q.isEmpty || p.name.toLowerCase().contains(q) || p.aliases.any((a) => a.toLowerCase().contains(q)))
        .toList();
    return Padding(
      padding: EdgeInsets.only(bottom: MediaQuery.viewInsetsOf(context).bottom),
      child: Container(
        constraints: BoxConstraints(maxHeight: MediaQuery.sizeOf(context).height * 0.8),
        decoration: BoxDecoration(color: AM.surface, borderRadius: BorderRadius.vertical(top: Radius.circular(30))),
        child: SafeArea(
          top: false,
          child: Column(mainAxisSize: MainAxisSize.min, children: [
            Padding(
              padding: const EdgeInsets.fromLTRB(16, 18, 16, 8),
              child: Row(children: [
                Expanded(
                  child: TextField(
                    controller: _c,
                    autofocus: true,
                    textInputAction: TextInputAction.go,
                    onChanged: (_) => setState(() {}),
                    onSubmitted: (t) => t.trim().isEmpty ? null : Navigator.pop(context, (t, true)),
                    decoration: const InputDecoration(hintText: 'e.g. Take me to the CG Lab', prefixIcon: Icon(Icons.search_rounded)),
                  ),
                ),
                const SizedBox(width: 10),
                RoundAction(icon: listening ? Icons.stop_rounded : Icons.mic_rounded, label: 'Speak destination', primary: listening, onTap: _mic),
              ]),
            ),
            if (listening) Padding(padding: EdgeInsets.all(8), child: Text('Listening…', style: TextStyle(color: AM.lime, fontWeight: FontWeight.w800))),
            Flexible(
              child: ListView(shrinkWrap: true, padding: const EdgeInsets.fromLTRB(8, 4, 8, 16), children: [
                for (final p in places)
                  ListTile(
                    leading: CircleAvatar(backgroundColor: AM.place(p.kind), child: Icon(Icons.place_rounded, color: AM.bg, size: 18)),
                    title: Text(p.name, style: const TextStyle(fontWeight: FontWeight.w800)),
                    subtitle: Text([
                      widget.wf.pack.placeKinds[p.kind] ?? p.kind,
                      if (p.aliases.isNotEmpty) p.aliases.join(', '),
                      if (p.stepFree == true) 'step-free',
                    ].join(' · ')),
                    enabled: p.id != widget.currentStart,
                    onTap: () => Navigator.pop(context, (p.id, false)),
                  ),
                for (final e in _nearest.entries.where((e) => widget.wf.classes.contains(e.key) && (q.isEmpty || e.value.$1.toLowerCase().contains(q))))
                  ListTile(
                    leading: CircleAvatar(backgroundColor: AM.teal, child: Icon(e.value.$2, color: AM.bg, size: 18)),
                    title: Text(e.value.$1, style: const TextStyle(fontWeight: FontWeight.w700)),
                    onTap: () => Navigator.pop(context, (e.key, false)),
                  ),
                if (q.isNotEmpty)
                  ListTile(
                    leading: CircleAvatar(backgroundColor: AM.violetSoft, child: Icon(Icons.keyboard_return_rounded, color: AM.violet)),
                    title: Text('Ask: "${_c.text.trim()}"'),
                    onTap: () => Navigator.pop(context, (_c.text.trim(), true)),
                  ),
              ]),
            ),
          ]),
        ),
      ),
    );
  }
}
