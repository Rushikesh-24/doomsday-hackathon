// "Which building am I at?" from the phone's GPS, matched against the location pinned on each map.
// GPS works outdoors / near windows only; inside the building the camera guide takes over.
import 'dart:math' as math;

import 'package:geolocator/geolocator.dart';

import 'models.dart';

class GeoFix {
  GeoFix(this.lat, this.lng, this.accuracy, this.at);
  final double lat, lng, accuracy;
  final DateTime at;
}

enum GeoStatus { unknown, off, denied, ok }

class Geo {
  /// Great-circle distance in metres.
  static double distance(double lat1, double lng1, double lat2, double lng2) {
    const r = 6371000.0;
    final p1 = lat1 * math.pi / 180, p2 = lat2 * math.pi / 180;
    final dp = p2 - p1, dl = (lng2 - lng1) * math.pi / 180;
    final a = math.sin(dp / 2) * math.sin(dp / 2) + math.cos(p1) * math.cos(p2) * math.sin(dl / 2) * math.sin(dl / 2);
    return 2 * r * math.atan2(math.sqrt(a), math.sqrt(1 - a));
  }

  /// The map whose pinned area contains the fix (or is closest to it, within its radius + GPS accuracy).
  static ({T map, double metres})? nearest<T>(GeoFix fix, Iterable<T> maps, GeoPin? Function(T) pin) {
    ({T map, double metres})? best;
    for (final m in maps) {
      final g = pin(m);
      if (g == null) continue;
      final d = distance(fix.lat, fix.lng, g.lat, g.lng);
      if (d > g.radius + math.min(fix.accuracy, 150)) continue;
      if (best == null || d < best.metres) best = (map: m, metres: d);
    }
    return best;
  }

  static Future<(GeoStatus, GeoFix?)> current({bool ask = true}) async {
    try {
      if (!await Geolocator.isLocationServiceEnabled()) return (GeoStatus.off, null);
      var perm = await Geolocator.checkPermission();
      if (perm == LocationPermission.denied && ask) perm = await Geolocator.requestPermission();
      if (perm == LocationPermission.denied || perm == LocationPermission.deniedForever) return (GeoStatus.denied, null);
      Position? pos = await Geolocator.getLastKnownPosition();
      if (pos == null || DateTime.now().difference(pos.timestamp).inMinutes > 3) {
        pos = await Geolocator.getCurrentPosition(
            locationSettings: const LocationSettings(accuracy: LocationAccuracy.high, timeLimit: Duration(seconds: 12)));
      }
      return (GeoStatus.ok, GeoFix(pos.latitude, pos.longitude, pos.accuracy, pos.timestamp));
    } catch (_) {
      return (GeoStatus.unknown, null);
    }
  }
}
