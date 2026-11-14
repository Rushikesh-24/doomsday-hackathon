// Map pack model: mirrors vision/navigation/mappack.py (format 2: + floor walls, accessibility score, location).
import 'dart:convert';

class GNode {
  GNode(this.id, this.kind, this.pos, {this.name, this.className, this.placeKind, this.floor});
  final String id;
  final String kind; // trajectory | facility | place
  final List<double> pos;
  final String? name;
  final String? className;
  final String? placeKind;
  final int? floor; // building packs: level of the floor this node is on

  factory GNode.fromJson(Map<String, dynamic> j) => GNode(
        j['id'] as String,
        j['kind'] as String,
        (j['pos'] as List).map((v) => (v as num).toDouble()).toList(),
        name: j['name'] as String?,
        className: j['class_name'] as String?,
        placeKind: j['place_kind'] as String?,
        floor: (j['floor'] as num?)?.toInt(),
      );
}

class GEdge {
  GEdge(this.a, this.b, this.length, this.tags, {this.connector, this.connectorName, this.connectorKind});
  final String a, b;
  final double length;
  final List<String> tags;
  final String? connector, connectorName, connectorKind; // lifts / stairs between floors
  factory GEdge.fromJson(Map<String, dynamic> j) => GEdge(
      j['a'] as String, j['b'] as String, (j['length'] as num).toDouble(), (j['tags'] as List).cast<String>(),
      connector: j['connector'] as String?, connectorName: j['connector_name'] as String?, connectorKind: j['connector_kind'] as String?);
}

class Place {
  Place(this.raw);
  final Map<String, dynamic> raw;
  String get id => raw['id'] as String;
  String get name => raw['name'] as String;
  String get kind => raw['kind'] as String;
  List<double> get position => (raw['position'] as List).map((v) => (v as num).toDouble()).toList();
  List<String> get aliases => ((raw['aliases'] as List?) ?? const []).cast<String>();
  String? get description => raw['description'] as String?;
  bool get isStart => raw['is_start'] == true;
  Map<String, dynamic> get access => (raw['access'] as Map?)?.cast<String, dynamic>() ?? const {};
  bool? get stepFree => access['step_free'] as bool?;
  int? get floor => (raw['floor'] as num?)?.toInt();
  String? get floorName => raw['floor_name'] as String?;
  String? get accessNotes => access['notes'] as String?;
}

class PlanGeom {
  PlanGeom(this.width, this.height, this.pxPerUnit, this.originX, this.originZ, this.image);
  final double width, height, pxPerUnit, originX, originZ;
  final String image;
  factory PlanGeom.fromJson(Map<String, dynamic> j) => PlanGeom(
        (j['width'] as num?)?.toDouble() ?? 1000,
        (j['height'] as num?)?.toDouble() ?? 600,
        (j['px_per_unit'] as num?)?.toDouble() ?? 30,
        ((j['origin'] as List?)?[0] as num?)?.toDouble() ?? 0,
        ((j['origin'] as List?)?[1] as num?)?.toDouble() ?? 0,
        j['image'] as String? ?? '',
      );
}

List<List<double>> _pts(dynamic v) =>
    ((v as List?) ?? const []).map((p) => (p as List).map((x) => (x as num).toDouble()).toList()).toList();

/// Measured corridor: left / right wall polylines on the floor ([x, z], local frame).
class FloorGeom {
  FloorGeom(Map<String, dynamic> j)
      : centre = _pts(j['centre']),
        left = _pts(j['left']),
        right = _pts(j['right']),
        openLeft = ((j['open_left'] as List?) ?? const []).cast<bool>(),
        openRight = ((j['open_right'] as List?) ?? const []).cast<bool>(),
        widthM = ((j['width_m'] as List?) ?? const []).map((v) => (v as num?)?.toDouble()).toList();
  final List<List<double>> centre, left, right;
  final List<bool> openLeft, openRight;
  final List<double?> widthM;
}

class Issue {
  Issue(this.raw);
  final Map<String, dynamic> raw;
  String get id => raw['id'] as String;
  String get kind => raw['kind'] as String;
  String get severity => raw['severity'] as String; // high | medium | low
  String get title => raw['title'] as String;
  String get detail => raw['detail'] as String? ?? '';
  List<double> get position => (raw['position'] as List).map((v) => (v as num).toDouble()).toList();
}

class ScoreCategory {
  ScoreCategory(this.raw);
  final Map<String, dynamic> raw;
  String get label => raw['label'] as String;
  int get weight => (raw['weight'] as num).toInt();
  double? get score => (raw['score'] as num?)?.toDouble();
  double? get points => (raw['points'] as num?)?.toDouble();
  String get reason => raw['reason'] as String? ?? '';
}

class AccessScore {
  AccessScore(Map<String, dynamic> j)
      : score = (j['score'] as num).toInt(),
        grade = j['grade'] as String,
        note = j['note'] as String? ?? '',
        categories = ((j['categories'] as List?) ?? const []).map((c) => ScoreCategory((c as Map).cast<String, dynamic>())).toList(),
        issues = ((j['issues'] as List?) ?? const []).map((c) => Issue((c as Map).cast<String, dynamic>())).toList();
  final int score;
  final String grade;
  final String note;
  final List<ScoreCategory> categories;
  final List<Issue> issues;
  String get verdict => score >= 85 ? 'Highly accessible' : score >= 70 ? 'Mostly accessible' : score >= 55 ? 'Partly accessible' : 'Hard to access';
}

class GeoPin {
  GeoPin(this.lat, this.lng, this.radius);
  final double lat, lng, radius;
  static GeoPin? fromJson(dynamic j) => j is Map && j['lat'] != null
      ? GeoPin((j['lat'] as num).toDouble(), (j['lng'] as num).toDouble(), ((j['radius'] as num?) ?? 80).toDouble())
      : null;
}

/// One floor of a building pack (geometry already in the building frame).
class BuildingFloor {
  BuildingFloor(Map<String, dynamic> j)
      : level = (j['level'] as num).toInt(),
        name = j['name'] as String,
        label = j['label'] as String? ?? j['name'] as String,
        scan = j['scan'] as String?,
        yawDeg = (j['yaw_deg'] as num?)?.toDouble() ?? 0,
        floor = FloorGeom((j['floor'] as Map).cast<String, dynamic>()),
        segments = ((j['segments'] as List?) ?? const []).map((x) => FloorGeom((x as Map).cast<String, dynamic>())).toList(),
        yawByScan = ((j['yaw_by_scan'] as Map?) ?? const {}).map((k, v) => MapEntry(k as String, (v as num).toDouble())),
        path = _pts(j['path']);
  final int level;
  final String name, label;
  final String? scan; // the floor's own scan (camera guide locates against it)
  final double yawDeg; // floor frame -> building frame rotation
  final FloorGeom floor;
  final List<FloorGeom> segments; // every walk's corridor piece on this floor
  final Map<String, double> yawByScan;
  final List<List<double>> path;
  String get short => level == 0 ? 'G' : '$level';
}

class MapPack {
  MapPack(this.raw)
      : nodes = (raw['nodes'] as List).map((n) => GNode.fromJson(n as Map<String, dynamic>)).toList(),
        edges = (raw['edges'] as List).map((e) => GEdge.fromJson(e as Map<String, dynamic>)).toList(),
        places = ((raw['places'] as List?) ?? const []).map((p) => Place((p as Map).cast<String, dynamic>())).toList(),
        path = ((raw['path'] as List?) ?? const [])
            .map((p) => (p as List).map((v) => (v as num).toDouble()).toList())
            .toList(),
        plan = PlanGeom.fromJson((raw['plan'] as Map).cast<String, dynamic>()),
        floor = raw['floor'] is Map ? FloorGeom((raw['floor'] as Map).cast<String, dynamic>()) : null,
        access = raw['accessibility'] is Map ? AccessScore((raw['accessibility'] as Map).cast<String, dynamic>()) : null;

  final Map<String, dynamic> raw;
  final List<GNode> nodes;
  final List<GEdge> edges;
  final List<Place> places;
  final List<List<double>> path;
  final PlanGeom plan;
  final FloorGeom? floor; // null in format-1 packs
  /// Buildings: all corridor pieces of the floor being drawn (a floor can be several walks).
  List<FloorGeom> get floorSegments =>
      ((raw['floor_segments'] as List?) ?? const []).map((x) => FloorGeom((x as Map).cast<String, dynamic>())).toList();
  final AccessScore? access;
  GeoPin? get location => GeoPin.fromJson(raw['location']);

  bool get isBuilding => raw['kind'] == 'building';
  List<BuildingFloor> get floors =>
      ((raw['floors'] as List?) ?? const []).map((f) => BuildingFloor((f as Map).cast<String, dynamic>())).toList();
  String? get startId => raw['start'] as String?;

  String get id => raw['id'] as String;
  String get name => raw['name'] as String;
  String? get description => raw['description'] as String?;
  String get version => raw['version'] as String;
  String get units => raw['units'] as String;
  String get scaleStatus => raw['scale_status'] as String;
  double get medianStep => (raw['median_step'] as num?)?.toDouble() ?? 1e-6;
  bool get metric => scaleStatus != 'relative';

  /// profile -> tag -> multiplier (null = forbidden), plus 'label'
  Map<String, Map<String, dynamic>> get profiles =>
      (raw['profiles'] as Map).map((k, v) => MapEntry(k as String, (v as Map).cast<String, dynamic>()));
  Map<String, Map<String, String>> get help => (raw['help'] as Map)
      .map((k, v) => MapEntry(k as String, (v as Map).map((a, b) => MapEntry(a as String, b as String))));
  Set<String> get nearClasses => ((raw['near_classes'] as List?) ?? const []).cast<String>().toSet();
  Map<String, List<String>> get classWords =>
      (raw['class_words'] as Map).map((k, v) => MapEntry(k as String, (v as List).cast<String>()));
  Map<String, String> get placeKinds => ((raw['place_kinds'] as Map?) ?? const {}).cast<String, String>();

  Place? get defaultStart => places.where((p) => p.isStart).firstOrNull;

  /// The same pack restricted to one floor, for drawing that floor (walls, path, places).
  MapPack floorView(int level) {
    final f = floors.firstWhere((x) => x.level == level);
    return MapPack({
      ...raw,
      'kind': 'floor_view',
      'nodes': [for (final n in raw['nodes'] as List) if ((n as Map)['floor'] == level) n],
      'edges': const [],
      'places': [for (final p in raw['places'] as List) if ((p as Map)['floor'] == level) p],
      'path': f.path,
      'floor': {
        'centre': f.floor.centre, 'left': f.floor.left, 'right': f.floor.right,
        'open_left': f.floor.openLeft, 'open_right': f.floor.openRight, 'width_m': const [],
      },
      'accessibility': null,
      'floor_segments': [
        for (final sg in f.segments)
          {'centre': sg.centre, 'left': sg.left, 'right': sg.right, 'open_left': sg.openLeft, 'open_right': sg.openRight, 'width_m': const []}
      ],
    });
  }

  String encode() => jsonEncode(raw);
  static MapPack decode(String s) => MapPack((jsonDecode(s) as Map).cast<String, dynamic>());
}

class DirectionStep {
  DirectionStep(this.kind, this.text, {this.notes = const [], this.length, this.angle, this.at, this.to, this.floor, this.toFloor});
  final String kind; // start | walk | turn | arrive | lift | stairs (lift / stairs: change floor)
  final int? floor, toFloor;
  final String text;
  final List<String> notes;
  final double? length;
  final int? angle;
  final String? at, to;

  String get spoken => notes.isEmpty ? text : '$text ${notes.join(' ')}';

  Map<String, dynamic> toJson() => {
        'kind': kind,
        'text': text,
        if (notes.isNotEmpty) 'notes': notes,
        if (length != null) 'length': length,
        if (angle != null) 'angle': angle,
        if (at != null) 'at': at,
        if (to != null) 'to': to,
        if (floor != null) 'floor': floor,
        if (toFloor != null) 'to_floor': toFloor,
      };

  DirectionStep withFloor(int? f) =>
      DirectionStep(kind, text, notes: notes, length: length, angle: angle, at: at, to: to, floor: f, toFloor: toFloor);

  factory DirectionStep.fromJson(Map<String, dynamic> j) => DirectionStep(
        j['kind'] as String,
        j['text'] as String,
        notes: ((j['notes'] as List?) ?? const []).cast<String>(),
        length: (j['length'] as num?)?.toDouble(),
        angle: (j['angle'] as num?)?.toInt(),
        at: j['at'] as String?,
        to: j['to'] as String?,
        floor: (j['floor'] as num?)?.toInt(),
        toFloor: (j['to_floor'] as num?)?.toInt(),
      );
}

class RouteResult {
  RouteResult({
    required this.feasible,
    required this.profile,
    this.profileLabel,
    this.reason,
    this.goal,
    this.path = const [],
    this.nodes = const [],
    this.length,
    this.units,
    this.warnings = const [],
    this.directions = const [],
    this.startName,
    this.goalName,
    this.floors = const [],
    this.pathFloors = const [],
  });
  final bool feasible;
  final String profile;
  final String? profileLabel, reason, goal, units, startName, goalName;
  final List<List<double>> path;
  final List<String> nodes;
  final double? length;
  final List<String> warnings;
  final List<DirectionStep> directions;
  final List<int> floors; // building routes: floors walked on, in order
  final List<int> pathFloors; // floor of each path point

  String get spoken => feasible ? directions.map((d) => d.spoken).join(' ') : 'No route. ${reason ?? ''}';

  Map<String, dynamic> toJson() => {
        'feasible': feasible,
        'profile': profile,
        'profile_label': profileLabel,
        'reason': reason,
        'goal': goal,
        'path': path,
        'nodes': nodes,
        'length': length,
        'units': units,
        'warnings': warnings,
        'directions': directions.map((d) => d.toJson()).toList(),
        'start_name': startName,
        'goal_name': goalName,
        if (floors.isNotEmpty) 'floors': floors,
        if (pathFloors.isNotEmpty) 'path_floors': pathFloors,
      };

  factory RouteResult.fromJson(Map<String, dynamic> j) => RouteResult(
        feasible: j['feasible'] == true,
        profile: j['profile'] as String? ?? 'standard',
        profileLabel: j['profile_label'] as String?,
        reason: j['reason'] as String?,
        goal: j['goal'] as String?,
        path: ((j['path'] as List?) ?? const [])
            .map((p) => (p as List).map((v) => (v as num).toDouble()).toList())
            .toList(),
        nodes: ((j['nodes'] as List?) ?? const []).cast<String>(),
        length: (j['length'] as num?)?.toDouble(),
        units: j['units'] as String?,
        warnings: ((j['warnings'] as List?) ?? const []).cast<String>(),
        directions: ((j['directions'] as List?) ?? const [])
            .map((d) => DirectionStep.fromJson((d as Map).cast<String, dynamic>()))
            .toList(),
        startName: j['start_name'] as String?,
        goalName: j['goal_name'] as String?,
        floors: ((j['floors'] as List?) ?? const []).map((v) => (v as num).toInt()).toList(),
        pathFloors: ((j['path_floors'] as List?) ?? const []).map((v) => (v as num).toInt()).toList(),
      );
}

class MapSummary {
  MapSummary(this.raw);
  final Map<String, dynamic> raw;
  String get id => raw['id'] as String;
  String get name => raw['name'] as String;
  String get version => raw['version'] as String;
  String? get description => raw['description'] as String?;
  int get places => (raw['places'] as num?)?.toInt() ?? 0;
  List<String> get placeNames => ((raw['place_names'] as List?) ?? const []).cast<String>();
  double? get length => (raw['length'] as num?)?.toDouble();
  String? get units => raw['units'] as String?;
  GeoPin? get location => GeoPin.fromJson(raw['location']);
  int? get score => ((raw['accessibility'] as Map?)?['score'] as num?)?.toInt();
  bool get isBuilding => raw['kind'] == 'building';
  String? get grade => (raw['accessibility'] as Map?)?['grade'] as String?;
}
