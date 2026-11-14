// Accessibility report of a map: overall score, the six weighted categories with their reasons, and the list
// of problem spots (tap one to see it on the map). Same numbers as the web Explorer, computed by the server
// from the scan (vision/navigation/score.py) and stored in the offline pack.
import 'package:flutter/material.dart';

import '../../core/models.dart';
import '../../core/voice.dart';
import '../theme.dart';
import '../widgets/common.dart';

Future<void> showAccessibilityReport(BuildContext context, MapPack pack, {ValueChanged<Issue>? onIssue}) {
  final a = pack.access!;
  return showModalBottomSheet<void>(
    context: context,
    isScrollControlled: true,
    builder: (ctx) => DraggableScrollableSheet(
      expand: false,
      initialChildSize: 0.75,
      maxChildSize: 0.95,
      builder: (ctx, sc) => ListView(controller: sc, padding: const EdgeInsets.fromLTRB(22, 0, 22, 28), children: [
        Row(children: [
          ScoreRing(score: a.score, grade: a.grade, size: 92, stroke: 8),
          const SizedBox(width: 18),
          Expanded(
            child: Column(crossAxisAlignment: CrossAxisAlignment.start, children: [
              const SectionLabel('Accessibility score'),
              const SizedBox(height: 4),
              Text(a.verdict, style: Theme.of(ctx).textTheme.titleLarge),
              const SizedBox(height: 2),
              Text(pack.name, style: TextStyle(color: AM.muted)),
            ]),
          ),
          IconButton(
            tooltip: 'Read the report aloud',
            onPressed: () => Voice.instance.say([
              '${pack.name}: accessibility score ${a.score} out of 100, grade ${a.grade}. ${a.verdict}.',
              for (final i in a.issues) '${i.title}.',
            ].join(' ')),
            icon: const Icon(Icons.volume_up_rounded),
          ),
        ]),
        const SizedBox(height: 22),
        for (final c in a.categories) ...[
          Row(children: [
            Expanded(child: Text(c.label, style: const TextStyle(fontWeight: FontWeight.w800))),
            Text(c.points == null ? 'unknown' : '${c.points!.round()} / ${c.weight}',
                style: TextStyle(color: AM.muted, fontWeight: FontWeight.w700, fontFeatures: [FontFeature.tabularFigures()])),
          ]),
          const SizedBox(height: 6),
          ClipRRect(
            borderRadius: BorderRadius.circular(99),
            child: LinearProgressIndicator(
              value: c.score ?? 0,
              minHeight: 8,
              backgroundColor: AM.surface2,
              color: c.score == null ? AM.surface2 : AM.grade(((c.score ?? 0) * 100).round()),
            ),
          ),
          const SizedBox(height: 6),
          Text(c.reason, style: TextStyle(color: AM.muted, fontSize: 13, height: 1.35)),
          const SizedBox(height: 16),
        ],
        const SizedBox(height: 4),
        const SectionLabel('Where it is not accessible'),
        const SizedBox(height: 10),
        if (a.issues.isEmpty)
          Container(
            padding: const EdgeInsets.all(14),
            decoration: BoxDecoration(color: AM.lime.withValues(alpha: 0.12), borderRadius: BorderRadius.circular(18)),
            child: Text('Nothing flagged along the scanned path.', style: TextStyle(color: AM.lime, fontWeight: FontWeight.w700)),
          ),
        for (final i in a.issues)
          Padding(
            padding: const EdgeInsets.only(bottom: 10),
            child: Material(
              color: AM.surface2,
              borderRadius: BorderRadius.circular(20),
              child: InkWell(
                borderRadius: BorderRadius.circular(20),
                onTap: onIssue == null
                    ? null
                    : () {
                        Navigator.pop(ctx);
                        onIssue(i);
                      },
                child: Padding(
                  padding: const EdgeInsets.all(14),
                  child: Row(crossAxisAlignment: CrossAxisAlignment.start, children: [
                    Container(
                      width: 34,
                      height: 34,
                      decoration: BoxDecoration(color: AM.severity(i.severity), borderRadius: BorderRadius.circular(11)),
                      child: Icon(Icons.priority_high_rounded, color: AM.bg, size: 20),
                    ),
                    const SizedBox(width: 12),
                    Expanded(
                      child: Column(crossAxisAlignment: CrossAxisAlignment.start, children: [
                        Text(i.title, style: const TextStyle(fontWeight: FontWeight.w800)),
                        const SizedBox(height: 2),
                        Text(i.detail, style: TextStyle(color: AM.muted, fontSize: 13, height: 1.35)),
                      ]),
                    ),
                  ]),
                ),
              ),
            ),
          ),
        const SizedBox(height: 8),
        Text(a.note, style: TextStyle(color: AM.muted, fontSize: 12, height: 1.4)),
      ]),
    ),
  );
}
