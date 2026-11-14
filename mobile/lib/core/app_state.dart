// App-wide state: settings, connectivity, offline maps and sync with the AccessMap server.
//
// Offline-first: the UI always reads maps from the on-device store. When the server is reachable the app
// downloads new or changed map packs (versioned by content hash, 304 when unchanged) in the background,
// and the Groq assistant becomes available. Nothing breaks when the network goes away.
import 'dart:async';

import 'package:flutter/widgets.dart';
import 'package:shared_preferences/shared_preferences.dart';

import 'api.dart';
import 'geo.dart';
import 'models.dart';
import 'store.dart';

enum Conn { unknown, online, offline }

class AppState extends ChangeNotifier with WidgetsBindingObserver {
  AppState({MapStore? store}) : store = store ?? MapStore();

  final MapStore store;
  SharedPreferences? _prefs;

  // settings
  String serverUrl = defaultServer;
  String token = '';
  String profile = 'standard';
  bool onboarded = false;
  bool voice = true;
  bool largeText = false;
  bool highContrast = false;
  String themeMode = 'system'; // system | light | dark

  static const defaultServer = 'http://127.0.0.1:8000'; // via `adb reverse tcp:8000 tcp:8000` over USB

  // live state
  Conn conn = Conn.unknown;
  bool syncing = false;
  String? syncError;
  DateTime? lastSync;
  bool assistantAvailable = false;
  String? assistantModel;
  String? assistantReason;
  List<LocalMap> maps = [];
  List<MapSummary> remote = [];
  final Set<String> updating = {};
  Timer? _timer;

  // GPS: which saved map (building) the phone is at
  GeoStatus geoStatus = GeoStatus.unknown;
  GeoFix? fix;
  String? nearbyMapId;
  double? nearbyMetres;
  bool locating = false;

  Api get api => Api(serverUrl, token: token);
  bool get online => conn == Conn.online;

  Future<void> init() async {
    _prefs = await SharedPreferences.getInstance();
    final p = _prefs!;
    serverUrl = p.getString('server') ?? defaultServer;
    token = p.getString('token') ?? '';
    profile = p.getString('profile') ?? 'standard';
    onboarded = p.getBool('onboarded') ?? false;
    voice = p.getBool('voice') ?? true;
    largeText = p.getBool('largeText') ?? false;
    highContrast = p.getBool('highContrast') ?? false;
    themeMode = p.getString('theme') ?? 'system';
    final ls = p.getString('lastSync');
    lastSync = ls != null ? DateTime.tryParse(ls) : null;
    maps = await store.list();
    notifyListeners();
    WidgetsBinding.instance.addObserver(this);
    unawaited(sync());
    _timer = Timer.periodic(const Duration(seconds: 60), (_) => sync(quiet: true));
  }

  /// The phone switched between light and dark: matters when Appearance is "System".
  @override
  void didChangePlatformBrightness() {
    if (themeMode == 'system') notifyListeners();
  }

  @override
  void didChangeAppLifecycleState(AppLifecycleState state) {
    if (state == AppLifecycleState.resumed) unawaited(sync(quiet: true));
  }

  bool _disposed = false;

  @override
  void notifyListeners() {
    if (!_disposed) super.notifyListeners(); // background sync may finish after the UI is gone
  }

  @override
  void dispose() {
    _disposed = true;
    _timer?.cancel();
    WidgetsBinding.instance.removeObserver(this);
    super.dispose();
  }

  Future<void> update({
    String? serverUrl,
    String? token,
    String? profile,
    bool? onboarded,
    bool? voice,
    bool? largeText,
    bool? highContrast,
    String? themeMode,
  }) async {
    final p = _prefs ?? await SharedPreferences.getInstance();
    if (serverUrl != null) {
      this.serverUrl = serverUrl.trim().isEmpty ? defaultServer : serverUrl.trim();
      await p.setString('server', this.serverUrl);
    }
    if (token != null) await p.setString('token', this.token = token.trim());
    if (profile != null) await p.setString('profile', this.profile = profile);
    if (onboarded != null) await p.setBool('onboarded', this.onboarded = onboarded);
    if (voice != null) await p.setBool('voice', this.voice = voice);
    if (largeText != null) await p.setBool('largeText', this.largeText = largeText);
    if (highContrast != null) await p.setBool('highContrast', this.highContrast = highContrast);
    if (themeMode != null && const ['system', 'light', 'dark'].contains(themeMode)) {
      await p.setString('theme', this.themeMode = themeMode);
    }
    notifyListeners();
    if (serverUrl != null || token != null) unawaited(sync());
  }

  LocalMap? mapById(String id) => maps.where((m) => m.pack.id == id).firstOrNull;

  /// Pinned location of a map: the latest server list wins (it can change without a new pack).
  GeoPin? pinOf(String id) =>
      remote.where((r) => r.id == id).firstOrNull?.location ?? mapById(id)?.pack.location;

  /// Read GPS and pick the saved map whose pinned area we are in. [ask] shows the permission prompt.
  Future<void> locate({bool ask = true}) async {
    if (locating) return;
    locating = true;
    notifyListeners();
    try {
      final (st, f) = await Geo.current(ask: ask);
      geoStatus = st;
      if (f != null) fix = f;
      _matchNearby();
    } finally {
      locating = false;
      notifyListeners();
    }
  }

  void _matchNearby() {
    final f = fix;
    if (f == null) return;
    final hit = Geo.nearest<LocalMap>(f, maps, (m) => pinOf(m.pack.id));
    nearbyMapId = hit?.map.pack.id;
    nearbyMetres = hit?.metres;
  }

  /// Pin [mapId] to where the phone is now (online; uses the write token if one is set).
  Future<void> pinHere(String mapId) async {
    final (st, f) = await Geo.current();
    geoStatus = st;
    if (f == null) throw ApiException(0, st == GeoStatus.off ? 'Turn on location first.' : 'Location permission is needed.');
    fix = f;
    await api.setLocation(mapId, f.lat, f.lng, radius: (f.accuracy * 2).clamp(60, 300).toDouble());
    await sync();
    _matchNearby();
    notifyListeners();
  }

  /// Check the server, refresh the remote list and download new / changed packs.
  Future<void> sync({bool quiet = false}) async {
    if (syncing) return;
    syncing = true;
    if (!quiet) syncError = null;
    notifyListeners();
    final a = api;
    try {
      if (!await a.health()) {
        conn = Conn.offline;
        return;
      }
      conn = Conn.online;
      notifyListeners();
      remote = await a.maps();
      final local = {for (final m in maps) m.pack.id: m.pack.version};
      for (final m in remote) {
        if (local[m.id] == m.version) continue;
        updating.add(m.id);
        notifyListeners();
        try {
          final pj = await a.pack(m.id, haveVersion: local[m.id]);
          if (pj != null) {
            final pack = MapPack.decode(pj);
            List<int>? png;
            try {
              png = pack.plan.image.isEmpty ? null : await a.bytes(pack.plan.image); // buildings have none
            } catch (_) {
              png = null; // the map still works without its plan image
            }
            await store.save(pj, png);
          }
        } finally {
          updating.remove(m.id);
        }
      }
      maps = await store.list();
      _matchNearby();
      lastSync = DateTime.now();
      await _prefs?.setString('lastSync', lastSync!.toIso8601String());
      unawaited(_assistantStatus(a));
    } on ApiException catch (e) {
      syncError = e.message;
      if (e.status == 0) conn = Conn.offline;
    } catch (e) {
      syncError = 'Sync failed: $e';
    } finally {
      syncing = false;
      notifyListeners();
    }
  }

  Future<void> _assistantStatus(Api a) async {
    try {
      final s = await a.assistantStatus();
      assistantAvailable = s['available'] == true;
      assistantModel = s['model'] as String?;
      assistantReason = s['reason'] as String?;
    } catch (_) {
      assistantAvailable = false;
    }
    notifyListeners();
  }

  Future<void> deleteMap(String id) async {
    await store.delete(id);
    maps = await store.list();
    notifyListeners();
  }

  Future<void> clearMaps() async {
    await store.clear();
    maps = [];
    notifyListeners();
  }
}

/// Makes AppState reachable anywhere below it without extra packages.
class AppScope extends InheritedNotifier<AppState> {
  const AppScope({super.key, required AppState state, required super.child}) : super(notifier: state);
  static AppState of(BuildContext c) => c.dependOnInheritedWidgetOfExactType<AppScope>()!.notifier!;
  static AppState read(BuildContext c) => c.getInheritedWidgetOfExactType<AppScope>()!.notifier!;
}
