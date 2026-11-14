// Top-down floor plan, drawn as vectors from the map pack (no blurry point-cloud image needed):
//   floor + walls measured from the scan (pack.floor), the walked centre line, tagged places, accessibility
//   problem spots, the route (lime, animated chevrons) and "you are here" with heading.
// The raw scan density image is an optional faint layer. Local frame is +Y up: plan x = local x, y = local z.
import 'dart:math' as math;
import 'dart:typed_data';

import 'package:flutter/material.dart';

import '../../core/models.dart';
import '../theme.dart';

class PlanView extends StatefulWidget {
  const PlanView({
    super.key,
    required this.pack,
    this.planImage,
    this.route,
    this.you,
    this.youHeading,
    this.selectedPlace,
    this.onPlaceTap,
    this.onIssueTap,
    this.showScan = false,
    this.showIssues = true,
    this.fitNonce = 0,
    this.padding = EdgeInsets.zero,
    this.interactive = true,
  });
  final MapPack pack;
  final Uint8List? planImage;
  final List<List<double>>? route;
  final List<double>? you;
  final double? youHeading; // radians on the plan, 0 = +x
  final String? selectedPlace;
  final ValueChanged<String>? onPlaceTap;
  final ValueChanged<Issue>? onIssueTap;
  final bool showScan, showIssues, interactive;
  final int fitNonce; // change to re-fit the view
  final EdgeInsets padding;

  @override
  State<PlanView> createState() => _PlanViewState();
}

class _PlanViewState extends State<PlanView> with SingleTickerProviderStateMixin {
  final _tc = TransformationController();
  late final AnimationController _pulse = AnimationController(vsync: this, duration: const Duration(milliseconds: 1800))..repeat();
  Size? _viewport;

  Offset _p(List<double> v) {
    final g = widget.pack.plan;
    return Offset((v[0] - g.originX) * g.pxPerUnit, (v[2] - g.originZ) * g.pxPerUnit);
  }

  Offset _q(List<double> v) {
    final g = widget.pack.plan;
    return Offset((v[0] - g.originX) * g.pxPerUnit, (v[1] - g.originZ) * g.pxPerUnit);
  }

  void _fit(Size vp) {
    final g = widget.pack.plan;
    final inner = Size(math.max(40, vp.width - widget.padding.horizontal), math.max(40, vp.height - widget.padding.vertical));
    var rect = Offset.zero & Size(g.width, g.height);
    final f = widget.pack.floor;
    final pts = <Offset>[
      if (f != null) ...[...f.left.map(_q), ...f.right.map(_q)] else ...widget.pack.path.map(_p),
      if (widget.route != null) ...widget.route!.map(_p),
      for (final pl in widget.pack.places) _p(pl.position),
    ];
    if (pts.isNotEmpty) {
      var minX = pts.first.dx, maxX = minX, minY = pts.first.dy, maxY = minY;
      for (final o in pts) {
        minX = math.min(minX, o.dx);
        maxX = math.max(maxX, o.dx);
        minY = math.min(minY, o.dy);
        maxY = math.max(maxY, o.dy);
      }
      final pad = math.max(50.0, 0.1 * math.max(maxX - minX, maxY - minY));
      rect = Rect.fromLTRB(minX - pad, minY - pad, maxX + pad, maxY + pad);
    }
    final s = math.min(inner.width / rect.width, inner.height / rect.height).clamp(0.05, 8.0);
    final dx = widget.padding.left + (inner.width - rect.width * s) / 2 - rect.left * s;
    final dy = widget.padding.top + (inner.height - rect.height * s) / 2 - rect.top * s;
    _tc.value = Matrix4.identity()
      ..translateByDouble(dx, dy, 0, 1)
      ..scaleByDouble(s, s, 1, 1);
  }

  @override
  void didUpdateWidget(covariant PlanView old) {
    super.didUpdateWidget(old);
    if (_viewport != null &&
        (old.route != widget.route || old.pack.id != widget.pack.id || old.fitNonce != widget.fitNonce || old.padding != widget.padding)) {
      _fit(_viewport!);
    }
  }

  @override
  void dispose() {
    _pulse.dispose();
    _tc.dispose();
    super.dispose();
  }

  @override
  Widget build(BuildContext context) {
    final g = widget.pack.plan;
    return LayoutBuilder(builder: (context, c) {
      final vp = Size(c.maxWidth, c.maxHeight);
      if (_viewport != vp) {
        _viewport = vp;
        WidgetsBinding.instance.addPostFrameCallback((_) => mounted ? _fit(vp) : null);
      }
      final map = InteractiveViewer(
        transformationController: _tc,
        constrained: false,
        minScale: 0.05,
        maxScale: 14,
        panEnabled: widget.interactive,
        scaleEnabled: widget.interactive,
        boundaryMargin: const EdgeInsets.all(3000),
        child: SizedBox(
          width: g.width,
          height: g.height,
          child: Stack(children: [
            if (widget.showScan && widget.planImage != null)
              Positioned.fill(
                child: Opacity(
                  opacity: 0.35,
                  child: ColorFiltered(
                    colorFilter: const ColorFilter.matrix([-1, 0, 0, 0, 255, 0, -1, 0, 0, 255, 0, 0, -1, 0, 255, 0, 0, 0, 1, 0]),
                    child: Image.memory(widget.planImage!, fit: BoxFit.fill, gaplessPlayback: true),
                  ),
                ),
              ),
            Positioned.fill(
              child: RepaintBoundary(
                child: AnimatedBuilder(
                  animation: _pulse,
                  builder: (_, _) => CustomPaint(painter: _PlanPainter(widget, _p, _q, _pulse.value)),
                ),
              ),
            ),
            if (widget.interactive) ..._hits(),
          ]),
        ),
      );
      return ClipRect(
        child: Stack(children: [
          Positioned.fill(child: CustomPaint(painter: _GridPainter(_tc))),
          Positioned.fill(child: map),
          if (widget.interactive && widget.pack.metric)
            Positioned(
              left: 16,
              bottom: widget.padding.bottom - 40 > 16 ? widget.padding.bottom - 40 : 16,
              child: IgnorePointer(child: _ScaleBar(tc: _tc, pxPerMetre: g.pxPerUnit)),
            ),
        ]),
      );
    });
  }

  List<Widget> _hits() {
    final unit = math.max(widget.pack.plan.width, widget.pack.plan.height) / 100;
    return [
      for (final p in widget.pack.places)
        Positioned(
          left: _p(p.position).dx - unit * 2.4,
          top: _p(p.position).dy - unit * 4.2,
          width: unit * 4.8,
          height: unit * 5,
          child: Semantics(
            button: true,
            label: 'Place ${p.name}',
            child: GestureDetector(behavior: HitTestBehavior.opaque, onTap: () => widget.onPlaceTap?.call(p.id)),
          ),
        ),
      if (widget.showIssues)
        for (final i in widget.pack.access?.issues ?? const <Issue>[])
          Positioned(
            left: _p(i.position).dx - unit * 2,
            top: _p(i.position).dy - unit * 2,
            width: unit * 4,
            height: unit * 4,
            child: Semantics(
              button: true,
              label: 'Accessibility problem: ${i.title}',
              child: GestureDetector(behavior: HitTestBehavior.opaque, onTap: () => widget.onIssueTap?.call(i)),
            ),
          ),
    ];
  }
}

/// Screen-space dot grid: gives depth and a sense of scale without competing with the map.
class _GridPainter extends CustomPainter {
  _GridPainter(this.tc) : super(repaint: tc);
  final TransformationController tc;
  @override
  void paint(Canvas canvas, Size size) {
    canvas.drawRect(Offset.zero & size, Paint()..color = AM.bg);
    final m = tc.value;
    final s = m.getMaxScaleOnAxis();
    var step = 28.0;
    while (step * s < 18) {
      step *= 2;
    }
    while (step * s > 56) {
      step /= 2;
    }
    final gap = step * s;
    final ox = m.getTranslation().x % gap, oy = m.getTranslation().y % gap;
    final dot = Paint()..color = AM.text.withValues(alpha: 0.07);
    for (var x = ox; x < size.width; x += gap) {
      for (var y = oy; y < size.height; y += gap) {
        canvas.drawCircle(Offset(x, y), 1.1, dot);
      }
    }
  }

  @override
  bool shouldRepaint(covariant _GridPainter o) => false;
}

class _ScaleBar extends StatelessWidget {
  const _ScaleBar({required this.tc, required this.pxPerMetre});
  final TransformationController tc;
  final double pxPerMetre;
  @override
  Widget build(BuildContext context) => ValueListenableBuilder<Matrix4>(
        valueListenable: tc,
        builder: (_, m, _) {
          final screenPerMetre = pxPerMetre * m.getMaxScaleOnAxis();
          var metres = 1.0;
          for (final c in const [1.0, 2.0, 5.0, 10.0, 20.0, 50.0]) {
            metres = c;
            if (c * screenPerMetre >= 50) break;
          }
          final w = (metres * screenPerMetre).clamp(20.0, 160.0);
          return Container(
            padding: const EdgeInsets.fromLTRB(8, 4, 8, 6),
            decoration: BoxDecoration(color: AM.surface.withValues(alpha: 0.85), borderRadius: BorderRadius.circular(10)),
            child: Column(crossAxisAlignment: CrossAxisAlignment.start, mainAxisSize: MainAxisSize.min, children: [
              Text('${metres.toStringAsFixed(0)} m',
                  textScaler: TextScaler.noScaling, style: TextStyle(fontSize: 11, fontWeight: FontWeight.w700, color: AM.text)),
              const SizedBox(height: 3),
              Container(
                  width: w,
                  height: 4,
                  decoration: BoxDecoration(
                      border: Border(left: BorderSide(color: AM.text, width: 2), right: BorderSide(color: AM.text, width: 2), bottom: BorderSide(color: AM.text, width: 2)))),
            ]),
          );
        },
      );
}

class _PlanPainter extends CustomPainter {
  _PlanPainter(this.w, this.p, this.q, this.t);
  final PlanView w;
  final Offset Function(List<double>) p;
  final Offset Function(List<double>) q;
  final double t;

  Path _line(List<Offset> o) {
    final path = Path()..moveTo(o.first.dx, o.first.dy);
    for (final x in o.skip(1)) {
      path.lineTo(x.dx, x.dy);
    }
    return path;
  }

  void _dashed(Canvas canvas, Path path, Paint paint, double on, double off) {
    for (final m in path.computeMetrics()) {
      for (var d = 0.0; d < m.length; d += on + off) {
        canvas.drawPath(m.extractPath(d, math.min(d + on, m.length)), paint);
      }
    }
  }

  @override
  void paint(Canvas canvas, Size size) {
    final g = w.pack.plan;
    final unit = math.max(g.width, g.height) / 100;
    final metre = w.pack.metric ? g.pxPerUnit : unit * 1.4;
    final pieces = w.pack.floorSegments.isNotEmpty ? w.pack.floorSegments : [if (w.pack.floor != null) w.pack.floor!];

    // ---------------------------------------------------------------- floor + walls (one piece per walk on this floor)
    for (final f in pieces.where((f) => f.left.length > 1)) {
      final l = f.left.map(q).toList(), r = f.right.map(q).toList();
      final poly = _line(l);
      for (final o in r.reversed) {
        poly.lineTo(o.dx, o.dy);
      }
      poly.close();
      canvas.drawPath(poly, Paint()
        ..color = AM.violet.withValues(alpha: 0.18)
        ..maskFilter = MaskFilter.blur(BlurStyle.normal, metre * 0.6));
      canvas.drawPath(poly, Paint()..color = AM.floor);
      // wall runs: solid where the scan saw a wall, dashed where that side was open / unknown
      final wall = Paint()
        ..color = AM.wall
        ..style = PaintingStyle.stroke
        ..strokeWidth = math.max(metre * 0.12, unit * 0.28)
        ..strokeCap = StrokeCap.round
        ..strokeJoin = StrokeJoin.round;
      final open = Paint()
        ..color = AM.muted.withValues(alpha: 0.7)
        ..style = PaintingStyle.stroke
        ..strokeWidth = wall.strokeWidth * 0.6;
      for (final (pts, flags) in [(l, f.openLeft), (r, f.openRight)]) {
        var i = 0;
        while (i < pts.length - 1) {
          final isOpen = i < flags.length && flags[i];
          var j = i + 1;
          while (j < pts.length - 1 && (j < flags.length && flags[j]) == isOpen) {
            j++;
          }
          final seg = _line(pts.sublist(i, j + 1));
          if (isOpen) {
            _dashed(canvas, seg, open, metre * 0.25, metre * 0.25);
          } else {
            canvas.drawPath(seg, wall);
          }
          i = j;
        }
      }
      // centre line (the walked line)
      if (f.centre.length > 1) {
        _dashed(canvas, _line(f.centre.map(q).toList()), Paint()
          ..color = AM.text.withValues(alpha: 0.14)
          ..style = PaintingStyle.stroke
          ..strokeWidth = metre * 0.05, metre * 0.3, metre * 0.3);
      }
    }
    if (pieces.every((f) => f.left.length < 2) && w.pack.path.length > 1) {
      final path = _line(w.pack.path.map(p).toList());
      canvas.drawPath(path, Paint()
        ..color = AM.floor
        ..style = PaintingStyle.stroke
        ..strokeWidth = math.max(unit * 2.2, metre * 1.6)
        ..strokeCap = StrokeCap.round
        ..strokeJoin = StrokeJoin.round);
    }

    // ---------------------------------------------------------------- route
    final rt = w.route;
    if (rt != null && rt.length > 1) {
      final path = _line(rt.map(p).toList());
      final width = math.max(unit * 0.9, metre * 0.45);
      canvas.drawPath(path, Paint()
        ..color = AM.lime.withValues(alpha: 0.35)
        ..style = PaintingStyle.stroke
        ..strokeWidth = width * 2.4
        ..strokeCap = StrokeCap.round
        ..strokeJoin = StrokeJoin.round
        ..maskFilter = MaskFilter.blur(BlurStyle.normal, width));
      canvas.drawPath(path, Paint()
        ..color = AM.lime
        ..style = PaintingStyle.stroke
        ..strokeWidth = width
        ..strokeCap = StrokeCap.round
        ..strokeJoin = StrokeJoin.round);
      // chevrons flowing toward the goal
      final chev = Paint()
        ..color = AM.onLime
        ..style = PaintingStyle.stroke
        ..strokeWidth = width * 0.28
        ..strokeCap = StrokeCap.round
        ..strokeJoin = StrokeJoin.round;
      final step = width * 3.2;
      for (final m in path.computeMetrics()) {
        for (var d = (t * step) % step + step * 0.5; d < m.length - step * 0.3; d += step) {
          final tg = m.getTangentForOffset(d);
          if (tg == null) continue;
          final a = tg.angle, o = tg.position, s = width * 0.36;
          final dir = Offset(math.cos(-a), math.sin(-a)), nrm = Offset(-dir.dy, dir.dx);
          canvas.drawPath(
              Path()
                ..moveTo(o.dx - dir.dx * s + nrm.dx * s, o.dy - dir.dy * s + nrm.dy * s)
                ..lineTo(o.dx + dir.dx * s * 0.6, o.dy + dir.dy * s * 0.6)
                ..lineTo(o.dx - dir.dx * s - nrm.dx * s, o.dy - dir.dy * s - nrm.dy * s),
              chev);
        }
      }
      _ring(canvas, p(rt.first), unit * 0.95, AM.text);
      _flag(canvas, p(rt.last), unit);
    } else if (w.pack.path.isNotEmpty) {
      _ring(canvas, p(w.pack.path.first), unit * 0.8, AM.green);
    }

    // ---------------------------------------------------------------- places
    for (final pl in w.pack.places) {
      final o = p(pl.position);
      final col = AM.place(pl.kind);
      final sel = w.selectedPlace == pl.id;
      if (sel) canvas.drawCircle(o, unit * (1.8 + 1.6 * t), Paint()..color = col.withValues(alpha: 0.3 * (1 - t)));
      canvas.drawCircle(o, unit * 0.95, Paint()..color = AM.bg);
      canvas.drawCircle(o, unit * 0.7, Paint()..color = col);
      final fs = unit * (sel ? 1.55 : 1.3);
      final tp = TextPainter(
        text: TextSpan(
            text: pl.name,
            style: TextStyle(fontFamily: 'PlusJakartaSans', fontSize: fs, fontWeight: FontWeight.w800, color: sel ? AM.onLime : AM.text)),
        textDirection: TextDirection.ltr,
        textScaler: TextScaler.noScaling,
      )..layout();
      final box = RRect.fromRectAndRadius(
          Rect.fromLTWH(o.dx - (tp.width + unit * 2.6) / 2, o.dy - unit * 1.6 - tp.height - unit * 0.9, tp.width + unit * 2.6, tp.height + unit * 0.9),
          Radius.circular(fs));
      canvas.drawRRect(box.shift(Offset(0, unit * 0.25)), Paint()..color = Colors.black.withValues(alpha: AM.isDark ? 0.5 : 0.1));
      canvas.drawRRect(box, Paint()..color = sel ? AM.lime : AM.surface2);
      canvas.drawRRect(box, Paint()
        ..color = sel ? AM.lime : col.withValues(alpha: 0.7)
        ..style = PaintingStyle.stroke
        ..strokeWidth = unit * 0.14);
      canvas.drawCircle(Offset(box.left + unit * 0.95, box.center.dy), unit * 0.38, Paint()..color = sel ? AM.onLime : col);
      tp.paint(canvas, Offset(box.left + unit * 1.7, box.top + unit * 0.45));
    }

    // ---------------------------------------------------------------- issues (on top of labels)
    if (w.showIssues) {
      for (final i in w.pack.access?.issues ?? const <Issue>[]) {
        final o = p(i.position), col = AM.severity(i.severity), r = unit * 1.1;
        canvas.drawCircle(o, r * 1.25, Paint()..color = AM.bg);
        canvas.drawCircle(o, r * (1.6 + 0.8 * t), Paint()..color = col.withValues(alpha: 0.22 * (1 - t)));
        final d = Path()
          ..moveTo(o.dx, o.dy - r)
          ..lineTo(o.dx + r, o.dy)
          ..lineTo(o.dx, o.dy + r)
          ..lineTo(o.dx - r, o.dy)
          ..close();
        canvas.drawPath(d, Paint()..color = col);
        _text(canvas, '!', o, r * 1.2, AM.bg, center: true);
      }
    }

    // ---------------------------------------------------------------- you are here
    final you = w.you;
    if (you != null) {
      final o = p(you);
      canvas.drawCircle(o, unit * (2.0 + 2.0 * t), Paint()..color = AM.blue.withValues(alpha: 0.25 * (1 - t)));
      if (w.youHeading != null) {
        canvas.save();
        canvas.translate(o.dx, o.dy);
        canvas.rotate(w.youHeading!);
        final cone = Path()
          ..moveTo(0, 0)
          ..lineTo(unit * 5.5, -unit * 2.6)
          ..arcToPoint(Offset(unit * 5.5, unit * 2.6), radius: Radius.circular(unit * 6))
          ..close();
        canvas.drawPath(cone, Paint()
          ..shader = RadialGradient(colors: [AM.blue.withValues(alpha: 0.55), AM.blue.withValues(alpha: 0)])
              .createShader(Rect.fromCircle(center: Offset.zero, radius: unit * 6)));
        canvas.restore();
      }
      canvas.drawCircle(o, unit * 1.15, Paint()..color = AM.bg);
      canvas.drawCircle(o, unit * 0.85, Paint()..color = AM.text);
      canvas.drawCircle(o, unit * 0.32, Paint()..color = AM.bg);
    }
  }

  void _text(Canvas c, String s, Offset at, double size, Color col, {bool center = false}) {
    final tp = TextPainter(
      text: TextSpan(text: s, style: TextStyle(fontFamily: 'PlusJakartaSans', fontSize: size, fontWeight: FontWeight.w900, color: col)),
      textDirection: TextDirection.ltr,
      textScaler: TextScaler.noScaling,
    )..layout();
    tp.paint(c, center ? at - Offset(tp.width / 2, tp.height / 2) : at);
  }

  void _ring(Canvas c, Offset o, double r, Color col) {
    c.drawCircle(o, r * 1.35, Paint()..color = AM.bg);
    c.drawCircle(o, r, Paint()
      ..color = col
      ..style = PaintingStyle.stroke
      ..strokeWidth = r * 0.45);
  }

  void _flag(Canvas c, Offset o, double unit) {
    c.drawCircle(o, unit * 1.5, Paint()..color = AM.lime.withValues(alpha: 0.25));
    c.drawCircle(o, unit * 0.95, Paint()..color = AM.lime);
    c.drawCircle(o, unit * 0.4, Paint()..color = AM.onLime);
  }

  @override
  bool shouldRepaint(covariant _PlanPainter o) => true;
}

/// Small static preview of a map for cards (floor outline, route-free).
class MapThumb extends StatelessWidget {
  const MapThumb({super.key, required this.pack});
  final MapPack pack;
  @override
  Widget build(BuildContext context) => CustomPaint(painter: _ThumbPainter(pack), child: const SizedBox.expand());
}

class _ThumbPainter extends CustomPainter {
  _ThumbPainter(this.pack) : dark = AM.isDark;
  final MapPack pack;
  final bool dark;
  @override
  void paint(Canvas canvas, Size size) {
    canvas.drawRect(Offset.zero & size, Paint()
      ..shader = LinearGradient(begin: Alignment.topLeft, end: Alignment.bottomRight, colors: [AM.thumbA, AM.thumbB])
          .createShader(Offset.zero & size));
    if (pack.isBuilding) return _stack(canvas, size);
    final f = pack.floor;
    final pts = <List<double>>[
      if (f != null) ...[...f.left, ...f.right] else ...pack.path.map((v) => [v[0], v[2]]),
    ];
    if (pts.length < 2) return;
    var minX = pts.first[0], maxX = minX, minY = pts.first[1], maxY = minY;
    for (final v in pts) {
      minX = math.min(minX, v[0]);
      maxX = math.max(maxX, v[0]);
      minY = math.min(minY, v[1]);
      maxY = math.max(maxY, v[1]);
    }
    final s = math.min((size.width - 40) / math.max(maxX - minX, 1e-3), (size.height - 36) / math.max(maxY - minY, 1e-3));
    Offset m(double x, double y) =>
        Offset((x - minX) * s + (size.width - (maxX - minX) * s) / 2, (y - minY) * s + (size.height - (maxY - minY) * s) / 2);
    if (f != null) {
      final poly = Path()..moveTo(m(f.left.first[0], f.left.first[1]).dx, m(f.left.first[0], f.left.first[1]).dy);
      for (final v in f.left.skip(1)) {
        poly.lineTo(m(v[0], v[1]).dx, m(v[0], v[1]).dy);
      }
      for (final v in f.right.reversed) {
        poly.lineTo(m(v[0], v[1]).dx, m(v[0], v[1]).dy);
      }
      poly.close();
      canvas.drawPath(poly, Paint()
        ..color = AM.lime.withValues(alpha: 0.25)
        ..maskFilter = const MaskFilter.blur(BlurStyle.normal, 8));
      canvas.drawPath(poly, Paint()..color = AM.thumbFloor);
      canvas.drawPath(poly, Paint()
        ..color = AM.lime.withValues(alpha: 0.9)
        ..style = PaintingStyle.stroke
        ..strokeWidth = 1.6);
    } else {
      final path = Path()..moveTo(m(pack.path.first[0], pack.path.first[2]).dx, m(pack.path.first[0], pack.path.first[2]).dy);
      for (final v in pack.path.skip(1)) {
        path.lineTo(m(v[0], v[2]).dx, m(v[0], v[2]).dy);
      }
      canvas.drawPath(path, Paint()
        ..color = AM.lime
        ..style = PaintingStyle.stroke
        ..strokeWidth = 4
        ..strokeCap = StrokeCap.round);
    }
    for (final pl in pack.places) {
      final o = m(pl.position[0], pl.position[2]);
      canvas.drawCircle(o, 5, Paint()..color = AM.bg);
      canvas.drawCircle(o, 3.6, Paint()..color = AM.place(pl.kind));
    }
  }

  /// Buildings: every floor's outline, stacked with an offset (lowest floor at the bottom).
  void _stack(Canvas canvas, Size size) {
    final floors = pack.floors..sort((a, b) => a.level.compareTo(b.level));
    final pts = [for (final f in floors) ...f.floor.left, for (final f in floors) ...f.floor.right];
    if (pts.length < 2) return;
    var minX = pts.first[0], maxX = minX, minY = pts.first[1], maxY = minY;
    for (final v in pts) {
      minX = math.min(minX, v[0]);
      maxX = math.max(maxX, v[0]);
      minY = math.min(minY, v[1]);
      maxY = math.max(maxY, v[1]);
    }
    final n = floors.length;
    final step = math.min(16.0, (size.height - 30) / (n + 2));
    final s = math.min((size.width - 60) / math.max(maxX - minX, 1e-3), (size.height - 24 - step * (n - 1)) / math.max(maxY - minY, 1e-3) * 2.2);
    for (final (i, f) in floors.indexed) {
      final oy = size.height - 14 - (maxY - minY) * s * 0.45 - i * step - 6;
      final ox = (size.width - (maxX - minX) * s) / 2 + i * 4;
      Offset m(List<double> v) => Offset(ox + (v[0] - minX) * s, oy + (v[1] - minY) * s * 0.45);
      final poly = Path()..moveTo(m(f.floor.left.first).dx, m(f.floor.left.first).dy);
      for (final v in f.floor.left.skip(1)) {
        poly.lineTo(m(v).dx, m(v).dy);
      }
      for (final v in f.floor.right.reversed) {
        poly.lineTo(m(v).dx, m(v).dy);
      }
      poly.close();
      canvas.drawPath(poly, Paint()..color = AM.thumbFloor);
      canvas.drawPath(poly, Paint()
        ..color = AM.lime.withValues(alpha: 0.4 + 0.6 * (i + 1) / n)
        ..style = PaintingStyle.stroke
        ..strokeWidth = 1.4);
    }
  }

  @override
  bool shouldRepaint(covariant _ThumbPainter o) => o.pack.version != pack.version || o.dark != dark;
}
