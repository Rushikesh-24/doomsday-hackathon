import 'dart:math' as math;

import 'package:flutter/material.dart';

import '../../core/app_state.dart';
import '../theme.dart';

const profiles = [
  (key: 'standard', label: 'Walking', hint: 'Shortest route', icon: Icons.directions_walk_rounded),
  (key: 'wheelchair', label: 'Wheelchair', hint: 'Step-free only', icon: Icons.accessible_forward_rounded),
  (key: 'low_vision', label: 'Low vision', hint: 'Handrails, spoken steps', icon: Icons.visibility_rounded),
];

({String key, String label, String hint, IconData icon}) profileInfo(String k) =>
    profiles.firstWhere((p) => p.key == k, orElse: () => profiles.first);

String ago(DateTime? t) {
  if (t == null) return 'never';
  final s = DateTime.now().difference(t).inSeconds;
  if (s < 45) return 'just now';
  if (s < 3600) return '${(s / 60).round()} min ago';
  if (s < 86400) return '${(s / 3600).round()} h ago';
  return '${t.day}/${t.month}/${t.year}';
}

/// True when the user's text is large enough that side-by-side layouts stop fitting.
bool bigText(BuildContext c) => MediaQuery.textScalerOf(c).scale(14) / 14 > 1.25;

// ------------------------------------------------------------------ logo
/// The AccessMap mark: a white location pin with a navigation arrow cut out of it, standing on a grey
/// "you are here" ring. Drawn in code so the in-app logo and the launcher icon are identical.
class LogoPainter extends CustomPainter {
  const LogoPainter({this.background = true, this.scale = 1.0});
  final bool background;
  final double scale; // < 1 keeps the mark inside Android's adaptive-icon safe zone

  @override
  void paint(Canvas canvas, Size size) {
    final s = size.shortestSide;
    canvas.save();
    canvas.translate((size.width - s) / 2, (size.height - s) / 2);
    if (background) {
      final r = RRect.fromRectAndRadius(Offset.zero & Size(s, s), Radius.circular(s * 0.28));
      canvas.drawRRect(
          r,
          Paint()
            ..shader = const RadialGradient(
              center: Alignment(-0.4, -0.6),
              radius: 1.2,
              colors: [Color(0xFF2A2A2A), Color(0xFF000000)],
            ).createShader(Offset.zero & Size(s, s)));
    }
    canvas.translate(s / 2, s / 2);
    canvas.scale(scale);
    canvas.translate(-s / 2, -s / 2);
    // ground ring
    final ring = Rect.fromCenter(center: Offset(s * 0.5, s * 0.8), width: s * 0.46, height: s * 0.13);
    canvas.drawOval(ring, Paint()..color = Colors.white.withValues(alpha: 0.18));
    canvas.drawOval(ring, Paint()
      ..color = const Color(0xFF8C8C8C)
      ..style = PaintingStyle.stroke
      ..strokeWidth = s * 0.022);
    // pin
    final c = Offset(s * 0.5, s * 0.4);
    final rad = s * 0.25;
    final tip = Offset(s * 0.5, s * 0.8);
    final a = math.asin(rad / (tip.dy - c.dy));
    final pin = Path()
      ..moveTo(tip.dx, tip.dy)
      ..lineTo(c.dx - rad * math.cos(a), c.dy + rad * math.sin(a))
      ..arcTo(Rect.fromCircle(center: c, radius: rad), math.pi - a, math.pi + 2 * a, false)
      ..close();
    canvas.drawPath(
        pin,
        Paint()
          ..shader = const LinearGradient(begin: Alignment.topLeft, end: Alignment.bottomRight, colors: [Color(0xFFFFFFFF), Color(0xFFF2F2F2), Color(0xFFCFCFCF)])
              .createShader(Rect.fromCircle(center: c, radius: rad * 1.6)));
    // navigation arrow cut-out
    final w = rad * 0.62;
    final arrow = Path()
      ..moveTo(c.dx, c.dy - w * 1.05)
      ..lineTo(c.dx + w * 0.82, c.dy + w * 0.85)
      ..lineTo(c.dx, c.dy + w * 0.42)
      ..lineTo(c.dx - w * 0.82, c.dy + w * 0.85)
      ..close();
    canvas.drawPath(arrow, Paint()..color = const Color(0xFF000000));
    canvas.restore();
  }

  @override
  bool shouldRepaint(covariant LogoPainter o) => o.background != background || o.scale != scale;
}

class Logo extends StatelessWidget {
  const Logo({super.key, this.size = 48});
  final double size;
  @override
  Widget build(BuildContext context) =>
      Semantics(label: 'AccessMap', image: true, child: CustomPaint(size: Size.square(size), painter: const LogoPainter()));
}

// ------------------------------------------------------------------ small pieces
/// Online / offline pill used across screens.
class ConnPill extends StatelessWidget {
  const ConnPill({super.key});
  @override
  Widget build(BuildContext context) {
    final s = AppScope.of(context);
    final (color, label) = switch (s.conn) {
      Conn.online => (AM.lime, s.syncing ? 'Syncing…' : 'Online · synced ${ago(s.lastSync)}'),
      Conn.offline => (AM.amber, 'Offline · saved maps'),
      Conn.unknown => (AM.muted, 'Connecting…'),
    };
    return Semantics(
      label: label,
      child: Container(
        padding: const EdgeInsets.symmetric(horizontal: 12, vertical: 7),
        decoration: BoxDecoration(
          color: AM.text.withValues(alpha: 0.06),
          borderRadius: BorderRadius.circular(99),
          border: Border.all(color: AM.text.withValues(alpha: 0.12)),
        ),
        child: Row(mainAxisSize: MainAxisSize.min, children: [
          Container(
              width: 8,
              height: 8,
              decoration: BoxDecoration(color: color, shape: BoxShape.circle, boxShadow: [BoxShadow(color: color.withValues(alpha: 0.6), blurRadius: 8)])),
          const SizedBox(width: 8),
          Flexible(
            child: Text(label,
                maxLines: 1, overflow: TextOverflow.ellipsis, style: TextStyle(fontSize: 12, fontWeight: FontWeight.w600, color: AM.text)),
          ),
        ]),
      ),
    );
  }
}

class RoundAction extends StatelessWidget {
  const RoundAction({super.key, required this.icon, required this.label, required this.onTap, this.primary = false, this.size = 52});
  final IconData icon;
  final String label;
  final VoidCallback? onTap;
  final bool primary;
  final double size;
  @override
  Widget build(BuildContext context) {
    final c = Theme.of(context).colorScheme;
    return Semantics(
      button: true,
      label: label,
      child: Tooltip(
        message: label,
        child: Material(
          color: primary ? c.primary : AM.surface2.withValues(alpha: 0.96),
          shape: CircleBorder(side: BorderSide(color: primary ? Colors.transparent : AM.text.withValues(alpha: 0.12))),
          elevation: 0,
          child: InkWell(
            customBorder: const CircleBorder(),
            onTap: onTap,
            child: SizedBox(width: size, height: size, child: Icon(icon, color: primary ? c.onPrimary : AM.text)),
          ),
        ),
      ),
    );
  }
}

class SectionLabel extends StatelessWidget {
  const SectionLabel(this.text, {super.key, this.color});
  final String text;
  final Color? color;
  @override
  Widget build(BuildContext context) => Text(text.toUpperCase(),
      style: TextStyle(fontSize: 11, letterSpacing: 1.6, fontWeight: FontWeight.w800, color: color ?? Theme.of(context).colorScheme.primary));
}

/// Score ring with the grade, e.g. "79 · B".
class ScoreRing extends StatelessWidget {
  const ScoreRing({super.key, required this.score, required this.grade, this.size = 64, this.stroke = 7});
  final int score;
  final String grade;
  final double size, stroke;
  @override
  Widget build(BuildContext context) {
    final col = AM.grade(score);
    return Semantics(
      label: 'Accessibility score $score out of 100, grade $grade',
      child: SizedBox(
        width: size,
        height: size,
        child: Stack(alignment: Alignment.center, children: [
          SizedBox.expand(
            child: CircularProgressIndicator(
              value: score / 100,
              strokeWidth: stroke,
              strokeCap: StrokeCap.round,
              color: col,
              backgroundColor: AM.text.withValues(alpha: 0.08),
            ),
          ),
          Column(mainAxisSize: MainAxisSize.min, children: [
            Text('$score',
                textScaler: TextScaler.noScaling,
                style: TextStyle(fontSize: size * 0.32, fontWeight: FontWeight.w800, height: 1, color: AM.text)),
            Text(grade, textScaler: TextScaler.noScaling, style: TextStyle(fontSize: size * 0.17, fontWeight: FontWeight.w800, color: col)),
          ]),
        ]),
      ),
    );
  }
}

/// Compact pill for cards: "♿ 79 · B".
class ScorePill extends StatelessWidget {
  const ScorePill({super.key, required this.score, required this.grade});
  final int score;
  final String grade;
  @override
  Widget build(BuildContext context) {
    final col = AM.grade(score);
    return Semantics(
      label: 'Accessibility $score, grade $grade',
      child: Container(
        padding: const EdgeInsets.symmetric(horizontal: 9, vertical: 4),
        decoration: BoxDecoration(color: col.withValues(alpha: 0.16), borderRadius: BorderRadius.circular(99), border: Border.all(color: col.withValues(alpha: 0.5))),
        child: Row(mainAxisSize: MainAxisSize.min, children: [
          Icon(Icons.accessible_rounded, size: 14, color: col),
          const SizedBox(width: 4),
          Text('$score · $grade', style: TextStyle(fontSize: 12, fontWeight: FontWeight.w800, color: col)),
        ]),
      ),
    );
  }
}

/// Three travel profiles. Side-by-side tiles normally; a stacked list when text is large or the screen is
/// narrow, so labels are never squeezed or cut off.
class ProfilePicker extends StatelessWidget {
  const ProfilePicker({super.key, required this.value, required this.onChanged, this.dense = false});
  final String value;
  final ValueChanged<String> onChanged;
  final bool dense;

  @override
  Widget build(BuildContext context) {
    return LayoutBuilder(builder: (context, c) {
      final stacked = bigText(context) || c.maxWidth < 330;
      final tiles = [for (final p in profiles) _tile(context, p, stacked)];
      if (stacked) {
        return Column(children: [
          for (final (i, t) in tiles.indexed) ...[t, if (i < tiles.length - 1) const SizedBox(height: 10)],
        ]);
      }
      return IntrinsicHeight(
        child: Row(crossAxisAlignment: CrossAxisAlignment.stretch, children: [
          for (final (i, t) in tiles.indexed) ...[Expanded(child: t), if (i < tiles.length - 1) const SizedBox(width: 10)],
        ]),
      );
    });
  }

  Widget _tile(BuildContext context, ({String key, String label, String hint, IconData icon}) p, bool stacked) {
    final c = Theme.of(context).colorScheme;
    final sel = value == p.key;
    final fg = sel ? c.onPrimary : AM.text;
    final icon = Container(
      width: stacked ? 46 : null,
      height: stacked ? 46 : null,
      decoration: stacked ? BoxDecoration(color: sel ? AM.bg.withValues(alpha: 0.15) : AM.surface, borderRadius: BorderRadius.circular(14)) : null,
      child: Icon(p.icon, size: dense ? 22 : 28, color: fg),
    );
    final label = Text(p.label, style: TextStyle(fontWeight: FontWeight.w800, fontSize: 14, color: fg));
    final hint = Text(p.hint,
        textAlign: stacked ? TextAlign.start : TextAlign.center,
        style: TextStyle(fontSize: 12, height: 1.25, color: sel ? c.onPrimary.withValues(alpha: 0.75) : AM.muted));
    return Semantics(
      selected: sel,
      button: true,
      label: '${p.label}: ${p.hint}',
      excludeSemantics: true,
      child: InkWell(
        borderRadius: BorderRadius.circular(22),
        onTap: () => onChanged(p.key),
        child: AnimatedContainer(
          duration: const Duration(milliseconds: 200),
          padding: stacked ? const EdgeInsets.all(12) : EdgeInsets.symmetric(vertical: dense ? 12 : 18, horizontal: 8),
          decoration: BoxDecoration(
            color: sel ? c.primary : AM.surface2,
            borderRadius: BorderRadius.circular(22),
            border: Border.all(color: sel ? c.primary : AM.stroke, width: 1.5),
            boxShadow: sel ? [BoxShadow(color: c.primary.withValues(alpha: 0.35), blurRadius: 22, offset: const Offset(0, 8))] : null,
          ),
          child: stacked
              ? Row(children: [
                  icon,
                  const SizedBox(width: 14),
                  Expanded(child: Column(crossAxisAlignment: CrossAxisAlignment.start, children: [label, if (!dense) hint])),
                  if (sel) Icon(Icons.check_circle_rounded, color: fg),
                ])
              : Column(mainAxisAlignment: MainAxisAlignment.center, children: [
                  icon,
                  const SizedBox(height: 8),
                  FittedBox(fit: BoxFit.scaleDown, child: label),
                  if (!dense) ...[const SizedBox(height: 3), hint],
                ]),
        ),
      ),
    );
  }
}
