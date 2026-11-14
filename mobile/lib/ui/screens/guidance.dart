// Step-by-step guidance in a dark, high-contrast layout with huge type and buttons.
// There is no indoor positioning, so the traveller confirms each step ("Next") - or uses "Preview walk",
// which moves the marker along the route at walking pace and speaks each step as it is reached.
import 'dart:async';
import 'dart:math' as math;

import 'package:flutter/material.dart';
import 'package:flutter/services.dart';

import '../../core/app_state.dart';
import '../../core/models.dart';
import '../../core/store.dart';
import '../../core/voice.dart';
import '../theme.dart';
import '../widgets/plan_view.dart';
import 'map_screen.dart' show stepIcon;

class GuidanceScreen extends StatefulWidget {
  const GuidanceScreen({super.key, required this.map, required this.route});
  final LocalMap map;
  final RouteResult route;
  @override
  State<GuidanceScreen> createState() => _GuidanceScreenState();
}

class _GuidanceScreenState extends State<GuidanceScreen> {
  int i = 0;
  Timer? _tick;
  double _dist = 0; // distance travelled along the route (preview walk)
  late final List<double> _cum;
  late final Map<String, int> _nodeIndex;

  RouteResult get r => widget.route;
  List<DirectionStep> get steps => r.directions;

  @override
  void initState() {
    super.initState();
    _cum = [0];
    for (var k = 1; k < r.path.length; k++) {
      _cum.add(_cum.last + _seg(r.path[k - 1], r.path[k]));
    }
    _nodeIndex = {for (final (k, id) in r.nodes.indexed) id: k};
    WidgetsBinding.instance.addPostFrameCallback((_) => _announce());
  }

  @override
  void dispose() {
    _tick?.cancel();
    Voice.instance.stop();
    super.dispose();
  }

  static double _seg(List<double> a, List<double> b) => math.sqrt(
    math.pow(a[0] - b[0], 2) +
        math.pow(a[1] - b[1], 2) +
        math.pow(a[2] - b[2], 2),
  );

  double _distAtStep(int s) {
    final id = steps[s].kind == 'arrive' ? r.nodes.last : steps[s].at;
    final k = id == null ? 0 : (_nodeIndex[id] ?? 0);
    return _cum[math.min(k, _cum.length - 1)];
  }

  List<double> _posAt(double d) {
    if (r.path.isEmpty) return const [0, 0, 0];
    d = d.clamp(0, _cum.last);
    var k = 1;
    while (k < _cum.length - 1 && _cum[k] < d) {
      k++;
    }
    if (k >= r.path.length) return r.path.last;
    final a = r.path[k - 1], b = r.path[k];
    final seg = (_cum[k] - _cum[k - 1]) == 0 ? 1.0 : _cum[k] - _cum[k - 1];
    final t = (d - _cum[k - 1]) / seg;
    return [for (var j = 0; j < 3; j++) a[j] + (b[j] - a[j]) * t];
  }

  double? _headingAt(double d) {
    final a = _posAt(d), b = _posAt(d + math.max(0.3, _cum.last * 0.02));
    final dx = b[0] - a[0], dz = b[2] - a[2];
    if (dx.abs() + dz.abs() < 1e-9) return null;
    return math.atan2(dz, dx);
  }

  void _announce() {
    HapticFeedback.mediumImpact();
    if (AppScope.read(context).voice) Voice.instance.say(steps[i].spoken);
  }

  void _goTo(int n) {
    if (n < 0 || n >= steps.length) return;
    setState(() {
      i = n;
      if (_tick == null) _dist = _distAtStep(n);
    });
    _announce();
  }

  void _togglePreview() {
    if (_tick != null) {
      _tick!.cancel();
      setState(() => _tick = null);
      return;
    }
    final speed =
        (widget.map.pack.metric ? 1.1 : widget.map.pack.medianStep * 3) *
        1.6; // brisk preview
    if (_dist >= _cum.last - 1e-6) _dist = 0;
    _tick = Timer.periodic(const Duration(milliseconds: 50), (t) {
      setState(() => _dist = math.min(_cum.last, _dist + speed * 0.05));
      var target = i;
      while (target + 1 < steps.length &&
          _distAtStep(target + 1) <= _dist + 1e-6) {
        target++;
      }
      if (target != i) {
        i = target;
        _announce();
      }
      if (_dist >= _cum.last) {
        t.cancel();
        setState(() => _tick = null);
      }
    });
    setState(() {});
  }

  /// Building routes: the floor the traveller is on at distance [d] along the route.
  int? _floorAt(double d) {
    if (r.pathFloors.isEmpty) return null;
    var k = 0;
    while (k < _cum.length - 1 && _cum[k + 1] <= d + 1e-9) {
      k++;
    }
    return r.pathFloors[math.min(k, r.pathFloors.length - 1)];
  }

  @override
  Widget build(BuildContext context) {
    final step = steps[i];
    final last = i == steps.length - 1;
    final pos = _posAt(_dist);
    final pack = widget.map.pack;
    final lv = pack.isBuilding ? (_floorAt(_dist) ?? step.floor ?? 0) : null;
    final viewPack = lv == null ? pack : pack.floorView(lv);
    final routeHere = lv == null
        ? r.path
        : [
            for (var k = 0; k < r.path.length; k++)
              if (k < r.pathFloors.length && r.pathFloors[k] == lv) r.path[k],
          ];
    return Theme(
      data: Theme.of(context).copyWith(scaffoldBackgroundColor: AM.bg),
      child: Scaffold(
        body: SafeArea(
          child: Column(
            children: [
              Padding(
                padding: const EdgeInsets.fromLTRB(8, 4, 8, 0),
                child: Row(
                  children: [
                    IconButton(
                      tooltip: 'End guidance',
                      onPressed: () => Navigator.pop(context),
                      icon: Icon(Icons.close_rounded, color: AM.text, size: 28),
                    ),
                    Expanded(
                      child: Column(
                        children: [
                          Text(
                            'To ${r.goalName}',
                            maxLines: 1,
                            overflow: TextOverflow.ellipsis,
                            style: TextStyle(
                              color: AM.text,
                              fontWeight: FontWeight.w700,
                              fontSize: 16,
                            ),
                          ),
                          Text(
                            r.profileLabel ?? '',
                            style: TextStyle(color: AM.muted, fontSize: 12),
                          ),
                        ],
                      ),
                    ),
                    IconButton(
                      tooltip: 'Repeat step',
                      onPressed: () => Voice.instance.say(step.spoken),
                      icon: Icon(
                        Icons.replay_rounded,
                        color: AM.text,
                        size: 26,
                      ),
                    ),
                  ],
                ),
              ),
              Expanded(
                flex: 5,
                child: Padding(
                  padding: const EdgeInsets.fromLTRB(16, 8, 16, 8),
                  child: ClipRRect(
                    borderRadius: BorderRadius.circular(28),
                    child: Stack(
                      children: [
                        Positioned.fill(
                          child: PlanView(
                            key: ValueKey('floor$lv'),
                            pack: viewPack,
                            planImage: lv == null ? widget.map.plan : null,
                            route: routeHere,
                            you: pos,
                            youHeading: _headingAt(_dist),
                            selectedPlace: r.goal,
                            showIssues: false,
                            padding: const EdgeInsets.all(28),
                          ),
                        ),
                        if (lv != null)
                          Positioned(
                            left: 12,
                            top: 12,
                            child: Container(
                              padding: const EdgeInsets.symmetric(
                                horizontal: 12,
                                vertical: 7,
                              ),
                              decoration: BoxDecoration(
                                color: AM.text,
                                borderRadius: BorderRadius.circular(99),
                              ),
                              child: Text(
                                pack.floors
                                    .firstWhere(
                                      (f) => f.level == lv,
                                      orElse: () => pack.floors.first,
                                    )
                                    .name,
                                style: TextStyle(
                                  color: AM.bg,
                                  fontWeight: FontWeight.w800,
                                ),
                              ),
                            ),
                          ),
                      ],
                    ),
                  ),
                ),
              ),
              // progress dots
              Padding(
                padding: const EdgeInsets.symmetric(horizontal: 24),
                child: Row(
                  children: [
                    for (var k = 0; k < steps.length; k++)
                      Expanded(
                        child: AnimatedContainer(
                          duration: const Duration(milliseconds: 250),
                          height: 6,
                          margin: const EdgeInsets.symmetric(horizontal: 2),
                          decoration: BoxDecoration(
                            color: k <= i
                                ? AM.lime
                                : AM.text.withValues(alpha: 0.15),
                            borderRadius: BorderRadius.circular(9),
                          ),
                        ),
                      ),
                  ],
                ),
              ),
              Expanded(
                flex: 6,
                child: Padding(
                  padding: const EdgeInsets.fromLTRB(16, 14, 16, 8),
                  child: AnimatedSwitcher(
                    duration: const Duration(milliseconds: 250),
                    child: Container(
                      key: ValueKey(i),
                      width: double.infinity,
                      padding: const EdgeInsets.all(22),
                      decoration: BoxDecoration(
                        gradient: step.kind == 'arrive'
                            ? LinearGradient(
                                colors: [AM.surface2, AM.surface],
                                begin: Alignment.topLeft,
                                end: Alignment.bottomRight,
                              )
                            : LinearGradient(
                                colors: [AM.surface2, AM.surface],
                                begin: Alignment.topLeft,
                                end: Alignment.bottomRight,
                              ),
                        border: Border.all(
                          color: step.kind == 'arrive' ? AM.lime : AM.stroke,
                          width: 1.5,
                        ),
                        borderRadius: BorderRadius.circular(30),
                      ),
                      child: SingleChildScrollView(
                        child: Semantics(
                          liveRegion: true,
                          label:
                              'Step ${i + 1} of ${steps.length}. ${step.spoken}',
                          child: ExcludeSemantics(
                            child: Column(
                              crossAxisAlignment: CrossAxisAlignment.start,
                              children: [
                                Row(
                                  children: [
                                    Container(
                                      width: 64,
                                      height: 64,
                                      decoration: BoxDecoration(
                                        color: AM.lime,
                                        borderRadius: BorderRadius.circular(20),
                                      ),
                                      child: Icon(
                                        stepIcon(step),
                                        color: AM.onLime,
                                        size: 40,
                                      ),
                                    ),
                                    const Spacer(),
                                    Text(
                                      '${i + 1} / ${steps.length}',
                                      style: TextStyle(
                                        color: AM.muted,
                                        fontWeight: FontWeight.w700,
                                        fontSize: 16,
                                      ),
                                    ),
                                  ],
                                ),
                                const SizedBox(height: 18),
                                Text(
                                  step.text,
                                  style: TextStyle(
                                    color: AM.text,
                                    fontSize: 26,
                                    height: 1.2,
                                    fontWeight: FontWeight.w800,
                                    letterSpacing: -0.4,
                                  ),
                                ),
                                for (final n in step.notes)
                                  Container(
                                    margin: const EdgeInsets.only(top: 12),
                                    padding: const EdgeInsets.all(12),
                                    decoration: BoxDecoration(
                                      color: AM.amber,
                                      borderRadius: BorderRadius.circular(16),
                                    ),
                                    child: Row(
                                      crossAxisAlignment:
                                          CrossAxisAlignment.start,
                                      children: [
                                        Icon(
                                          Icons.warning_amber_rounded,
                                          color: AM.ink,
                                        ),
                                        const SizedBox(width: 8),
                                        Expanded(
                                          child: Text(
                                            n,
                                            style: TextStyle(
                                              color: AM.ink,
                                              fontWeight: FontWeight.w700,
                                              fontSize: 16,
                                              height: 1.3,
                                            ),
                                          ),
                                        ),
                                      ],
                                    ),
                                  ),
                              ],
                            ),
                          ),
                        ),
                      ),
                    ),
                  ),
                ),
              ),
              Padding(
                padding: const EdgeInsets.fromLTRB(16, 4, 16, 16),
                child: Row(
                  children: [
                    _Big(
                      icon: Icons.chevron_left_rounded,
                      label: 'Previous step',
                      onTap: i > 0 ? () => _goTo(i - 1) : null,
                    ),
                    const SizedBox(width: 10),
                    _Big(
                      icon: _tick != null
                          ? Icons.pause_rounded
                          : Icons.play_arrow_rounded,
                      label: _tick != null ? 'Pause preview' : 'Preview walk',
                      onTap: _togglePreview,
                    ),
                    const SizedBox(width: 10),
                    Expanded(
                      child: SizedBox(
                        height: 72,
                        child: FilledButton(
                          style: FilledButton.styleFrom(
                            backgroundColor: AM.lime,
                            foregroundColor: AM.onLime,
                            shape: RoundedRectangleBorder(
                              borderRadius: BorderRadius.circular(24),
                            ),
                          ),
                          onPressed: last
                              ? () => Navigator.pop(context)
                              : () => _goTo(i + 1),
                          child: Text(
                            last ? 'Done' : 'Next step',
                            style: const TextStyle(
                              fontSize: 20,
                              fontWeight: FontWeight.w800,
                            ),
                          ),
                        ),
                      ),
                    ),
                  ],
                ),
              ),
            ],
          ),
        ),
      ),
    );
  }
}

class _Big extends StatelessWidget {
  const _Big({required this.icon, required this.label, required this.onTap});
  final IconData icon;
  final String label;
  final VoidCallback? onTap;
  @override
  Widget build(BuildContext context) => Semantics(
    button: true,
    label: label,
    child: Tooltip(
      message: label,
      child: Material(
        color: AM.text.withValues(alpha: onTap == null ? 0.05 : 0.1),
        borderRadius: BorderRadius.circular(24),
        child: InkWell(
          borderRadius: BorderRadius.circular(24),
          onTap: onTap,
          child: SizedBox(
            width: 72,
            height: 72,
            child: Icon(
              icon,
              color: onTap == null ? AM.text.withValues(alpha: 0.25) : AM.text,
              size: 36,
            ),
          ),
        ),
      ),
    ),
  );
}
