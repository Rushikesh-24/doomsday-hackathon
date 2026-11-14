// Offline storage: every downloaded map lives in <app documents>/maps/<id>/ as pack.json + plan.png,
// so the app works with no network at all. Writes go to a temp file first and are renamed (atomic).
import 'dart:convert';
import 'dart:io';
import 'dart:typed_data';

import 'package:path_provider/path_provider.dart';

import 'models.dart';

class LocalMap {
  LocalMap(this.pack, this.plan, this.bytesOnDisk);
  final MapPack pack;
  final Uint8List? plan;
  final int bytesOnDisk;
}

class MapStore {
  MapStore([this._rootOverride]);
  final Directory? _rootOverride;
  Directory? _root;

  Future<Directory> _dir() async {
    if (_root != null) return _root!;
    final base = _rootOverride ?? await getApplicationDocumentsDirectory();
    _root = Directory('${base.path}${Platform.pathSeparator}maps');
    await _root!.create(recursive: true);
    return _root!;
  }

  static bool _safeId(String id) => RegExp(r'^[A-Za-z0-9_-]{1,64}$').hasMatch(id);

  Future<File> _f(String id, String name) async {
    if (!_safeId(id)) throw ArgumentError('bad map id');
    final d = Directory('${(await _dir()).path}${Platform.pathSeparator}$id');
    await d.create(recursive: true);
    return File('${d.path}${Platform.pathSeparator}$name');
  }

  Future<void> _atomicWrite(File f, List<int> bytes) async {
    final tmp = File('${f.path}.part');
    await tmp.writeAsBytes(bytes, flush: true);
    if (await f.exists()) await f.delete();
    await tmp.rename(f.path);
  }

  Future<void> save(String packJson, List<int>? planPng) async {
    final pack = MapPack.decode(packJson); // validates before anything is replaced
    if (planPng != null) await _atomicWrite(await _f(pack.id, 'plan.png'), planPng);
    await _atomicWrite(await _f(pack.id, 'pack.json'), utf8.encode(packJson));
  }

  Future<List<LocalMap>> list() async {
    final out = <LocalMap>[];
    final d = await _dir();
    await for (final e in d.list()) {
      if (e is! Directory) continue;
      final pf = File('${e.path}${Platform.pathSeparator}pack.json');
      if (!await pf.exists()) continue;
      try {
        final pack = MapPack.decode(await pf.readAsString());
        final img = File('${e.path}${Platform.pathSeparator}plan.png');
        final plan = await img.exists() ? await img.readAsBytes() : null;
        out.add(LocalMap(pack, plan, (await pf.length()) + (plan?.length ?? 0)));
      } catch (_) {
        // a corrupt pack is skipped (and replaced on the next sync)
      }
    }
    out.sort((a, b) => a.pack.name.toLowerCase().compareTo(b.pack.name.toLowerCase()));
    return out;
  }

  Future<void> delete(String id) async {
    if (!_safeId(id)) return;
    final d = Directory('${(await _dir()).path}${Platform.pathSeparator}$id');
    if (await d.exists()) await d.delete(recursive: true);
  }

  Future<void> clear() async {
    final d = await _dir();
    if (await d.exists()) await d.delete(recursive: true);
    _root = null;
  }
}
