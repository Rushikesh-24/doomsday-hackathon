// The on-device router must answer exactly like the server (vision/navigation). Fixtures come from
// scripts/export_router_fixtures.py.
import 'dart:convert';
import 'dart:io';

import 'package:accessmap/core/models.dart';
import 'package:accessmap/core/router.dart';
import 'package:flutter_test/flutter_test.dart';

void main() {
  final files = Directory('test/fixtures').listSync().whereType<File>().where((f) => f.path.contains('router_'));
  test('fixtures exist', () => expect(files, isNotEmpty));

  for (final f in files) {
    final data = jsonDecode(f.readAsStringSync()) as Map<String, dynamic>;
    final pack = MapPack((data['pack'] as Map).cast<String, dynamic>());
    final wf = Wayfinder(pack);
    for (final c in (data['cases'] as List).cast<Map<String, dynamic>>()) {
      test('${pack.id}: "${c['text']}" [${c['profile']}]', () {
        if (c.containsKey('error')) {
          expect(() => wf.ask(c['text'] as String, profile: c['profile'] as String, start: c['start'] as String?),
              throwsA(isA<RouteError>()));
          return;
        }
        final r = wf.ask(c['text'] as String, profile: c['profile'] as String, start: c['start'] as String?);
        final exp = (c['expected'] as Map).cast<String, dynamic>();
        expect(r.start, c['resolved_start']);
        final got = r.route;
        expect(got.feasible, exp['feasible']);
        expect(got.reason, exp['reason']);
        if (exp['feasible'] != true) return;
        expect(got.goal, exp['goal']);
        expect(got.nodes, (exp['nodes'] as List).cast<String>());
        expect(got.length, closeTo((exp['length'] as num).toDouble(), 1e-9));
        expect(got.units, exp['units']);
        expect(got.warnings, (exp['warnings'] as List).cast<String>());
        expect(got.startName, exp['start_name']);
        expect(got.goalName, exp['goal_name']);
        final ed = (exp['directions'] as List).cast<Map<String, dynamic>>();
        expect(got.directions.map((d) => d.text).toList(), ed.map((d) => d['text']).toList());
        expect(got.directions.map((d) => d.kind).toList(), ed.map((d) => d['kind']).toList());
        expect(got.directions.map((d) => d.notes).toList(),
            ed.map((d) => ((d['notes'] as List?) ?? const []).cast<String>()).toList());
        expect(got.directions.map((d) => d.angle).toList(), ed.map((d) => d['angle']).toList());
      });
    }
  }

  // multi-floor buildings: cross-floor routes with lift / stairs steps (scripts/export_router_fixtures.py --building)
  final bfiles = Directory('test/fixtures').listSync().whereType<File>().where((f) => f.path.contains('building_'));
  test('building fixtures exist', () => expect(bfiles, isNotEmpty));
  for (final f in bfiles) {
    final data = jsonDecode(f.readAsStringSync()) as Map<String, dynamic>;
    final pack = MapPack((data['pack'] as Map).cast<String, dynamic>());
    final wf = Wayfinder(pack);
    test('building pack loads', () {
      expect(pack.isBuilding, isTrue);
      expect(pack.floors.map((x) => x.level).toList(), [0, 1, 2, 3]);
      expect(pack.floorView(2).places.map((p) => p.name).toSet(), containsAll(['CG Lab', 'Toilet', 'Lift 1']));
    });
    for (final c in (data['cases'] as List).cast<Map<String, dynamic>>()) {
      test('building: ${c['start'] ?? 'entrance'} -> ${c['goal']} [${c['profile']}]', () {
        if (c.containsKey('error')) {
          expect(() => wf.routeBuilding(c['start'] as String?, c['goal'] as String, profile: c['profile'] as String),
              throwsA(isA<RouteError>().having((e) => e.message, 'message', c['error'])));
          return;
        }
        final got = wf.routeBuilding(c['start'] as String?, c['goal'] as String, profile: c['profile'] as String);
        final exp = (c['expected'] as Map).cast<String, dynamic>();
        expect(got.feasible, exp['feasible']);
        expect(got.reason, exp['reason']);
        if (exp['feasible'] != true) return;
        expect(got.nodes, (exp['nodes'] as List).cast<String>());
        expect(got.length, closeTo((exp['length'] as num).toDouble(), 1e-9));
        expect(got.startName, exp['start_name']);
        expect(got.goalName, exp['goal_name']);
        expect(got.floors, (exp['floors'] as List).cast<int>());
        expect(got.pathFloors, (exp['path_floors'] as List).cast<int>());
        final ed = (exp['directions'] as List).cast<Map<String, dynamic>>();
        expect(got.directions.map((d) => d.text).toList(), ed.map((d) => d['text']).toList());
        expect(got.directions.map((d) => d.kind).toList(), ed.map((d) => d['kind']).toList());
        expect(got.directions.map((d) => d.floor).toList(), ed.map((d) => d['floor']).toList());
        expect(got.directions.map((d) => d.toFloor).toList(), ed.map((d) => d['to_floor']).toList());
        expect(got.directions.map((d) => d.notes).toList(),
            ed.map((d) => ((d['notes'] as List?) ?? const []).cast<String>()).toList());
      });
    }
  }

  test('python rounding helpers', () {
    expect(pyRound(2.5), 2);
    expect(pyRound(3.5), 4);
    expect(pyRound(12.175), 12);
    expect(pyFixed1(1.25), '1.2');
    expect(pyFixed1(2.96), '3.0');
  });
}
