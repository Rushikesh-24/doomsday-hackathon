import 'package:flutter/material.dart';

import '../../core/app_state.dart';
import '../../core/geo.dart';
import '../../core/store.dart';
import '../theme.dart';
import '../widgets/common.dart';
import '../widgets/plan_view.dart';
import 'map_screen.dart';
import 'settings.dart';

class HomeScreen extends StatefulWidget {
  const HomeScreen({super.key});
  @override
  State<HomeScreen> createState() => _HomeScreenState();
}

class _HomeScreenState extends State<HomeScreen> {
  @override
  void initState() {
    super.initState();
    // quietly check where we are if permission was already granted (no prompt on launch)
    WidgetsBinding.instance.addPostFrameCallback((_) => AppScope.read(context).locate(ask: false));
  }

  @override
  Widget build(BuildContext context) {
    final s = AppScope.of(context);
    final prof = profileInfo(s.profile);
    final notDownloaded = s.remote.where((r) => s.mapById(r.id) == null).toList();
    final here = s.nearbyMapId != null ? s.mapById(s.nearbyMapId!) : null;
    final others = s.maps.where((m) => m.pack.id != here?.pack.id).toList();
    return Scaffold(
      body: RefreshIndicator(
        onRefresh: () => s.sync(),
        child: CustomScrollView(slivers: [
          SliverToBoxAdapter(
            child: Container(
              decoration: BoxDecoration(gradient: AM.heroGradient, borderRadius: BorderRadius.vertical(bottom: Radius.circular(36))),
              child: SafeArea(
                bottom: false,
                child: Padding(
                  padding: const EdgeInsets.fromLTRB(20, 10, 10, 26),
                  child: Column(crossAxisAlignment: CrossAxisAlignment.start, children: [
                    Row(children: [
                      const Logo(size: 40),
                      const SizedBox(width: 10),
                      const Flexible(child: Align(alignment: Alignment.centerLeft, child: ConnPill())),
                      const SizedBox(width: 4),
                      IconButton(
                        tooltip: 'Sync now',
                        onPressed: s.syncing ? null : () => s.sync(),
                        icon: s.syncing
                            ? const SizedBox(width: 20, height: 20, child: CircularProgressIndicator(strokeWidth: 2))
                            : const Icon(Icons.sync_rounded),
                      ),
                      IconButton(
                        tooltip: AM.isDark ? 'Switch to light theme' : 'Switch to dark theme',
                        onPressed: () => s.update(themeMode: AM.isDark ? 'light' : 'dark'),
                        icon: Icon(AM.isDark ? Icons.light_mode_rounded : Icons.dark_mode_rounded),
                      ),
                      IconButton(
                        tooltip: 'Settings',
                        onPressed: () => Navigator.of(context).push(MaterialPageRoute(builder: (_) => const SettingsScreen())),
                        icon: const Icon(Icons.tune_rounded),
                      ),
                    ]),
                    const SizedBox(height: 26),
                    Text.rich(
                      TextSpan(children: [
                        const TextSpan(text: 'Where are we\n'),
                        TextSpan(text: 'going today?', style: TextStyle(color: Theme.of(context).colorScheme.primary)),
                      ]),
                      style: Theme.of(context).textTheme.headlineLarge?.copyWith(height: 1.05),
                    ),
                    const SizedBox(height: 16),
                    InkWell(
                      borderRadius: BorderRadius.circular(99),
                      onTap: () => Navigator.of(context).push(MaterialPageRoute(builder: (_) => const SettingsScreen())),
                      child: Container(
                        padding: const EdgeInsets.symmetric(horizontal: 14, vertical: 9),
                        decoration: BoxDecoration(color: AM.text.withValues(alpha: 0.07), borderRadius: BorderRadius.circular(99)),
                        child: Row(mainAxisSize: MainAxisSize.min, children: [
                          Icon(prof.icon, size: 18),
                          const SizedBox(width: 8),
                          Flexible(
                            child: Text('Travelling as ${prof.label}',
                                maxLines: 1, overflow: TextOverflow.ellipsis, style: const TextStyle(fontWeight: FontWeight.w700)),
                          ),
                          const SizedBox(width: 4),
                          Icon(Icons.expand_more_rounded, color: AM.muted, size: 18),
                        ]),
                      ),
                    ),
                  ]),
                ),
              ),
            ),
          ),
          SliverPadding(padding: const EdgeInsets.fromLTRB(20, 20, 20, 0), sliver: SliverToBoxAdapter(child: _HereCard(here: here))),
          SliverPadding(
            padding: const EdgeInsets.fromLTRB(20, 24, 20, 10),
            sliver: SliverToBoxAdapter(
              child: Wrap(crossAxisAlignment: WrapCrossAlignment.center, spacing: 8, runSpacing: 6, children: [
                Text(here != null ? 'Other maps' : 'Saved maps', style: Theme.of(context).textTheme.titleLarge),
                Container(
                  padding: const EdgeInsets.symmetric(horizontal: 9, vertical: 2),
                  decoration: BoxDecoration(color: AM.surface2, borderRadius: BorderRadius.circular(99)),
                  child: Text('${others.length}', style: const TextStyle(fontWeight: FontWeight.w800)),
                ),
                Row(mainAxisSize: MainAxisSize.min, children: [
                  Icon(Icons.offline_pin_rounded, size: 16, color: AM.lime),
                  SizedBox(width: 4),
                  Text('Work offline', style: TextStyle(fontSize: 12, color: AM.muted)),
                ]),
              ]),
            ),
          ),
          if (s.syncError != null)
            SliverPadding(
              padding: const EdgeInsets.fromLTRB(20, 0, 20, 10),
              sliver: SliverToBoxAdapter(
                child: Container(
                  padding: const EdgeInsets.all(12),
                  decoration: BoxDecoration(color: AM.amber.withValues(alpha: 0.12), borderRadius: BorderRadius.circular(16)),
                  child: Text(s.syncError!, style: TextStyle(color: AM.amber)),
                ),
              ),
            ),
          if (s.maps.isEmpty)
            const SliverPadding(padding: EdgeInsets.fromLTRB(20, 0, 20, 20), sliver: SliverToBoxAdapter(child: _EmptyMaps()))
          else
            SliverPadding(
              padding: const EdgeInsets.fromLTRB(20, 0, 20, 8),
              sliver: SliverList.separated(
                itemCount: others.length,
                separatorBuilder: (_, _) => const SizedBox(height: 14),
                itemBuilder: (_, i) => MapCard(m: others[i]),
              ),
            ),
          if (notDownloaded.isNotEmpty)
            SliverPadding(
              padding: const EdgeInsets.fromLTRB(20, 4, 20, 0),
              sliver: SliverToBoxAdapter(
                child: Text('Downloading ${notDownloaded.length} more map(s)…', style: TextStyle(color: AM.muted)),
              ),
            ),
          const SliverToBoxAdapter(child: SizedBox(height: 32)),
        ]),
      ),
    );
  }
}

/// "You're at (building)" from GPS, or a prompt to find the building.
class _HereCard extends StatelessWidget {
  const _HereCard({required this.here});
  final LocalMap? here;
  @override
  Widget build(BuildContext context) {
    final s = AppScope.of(context);
    final h = here;
    if (h != null) {
      return Column(crossAxisAlignment: CrossAxisAlignment.start, children: [
        Row(children: [
          Icon(Icons.my_location_rounded, size: 16, color: AM.lime),
          const SizedBox(width: 6),
          Flexible(
            child: Text(
              s.nearbyMetres != null && s.nearbyMetres! > 30 ? 'You are near (${s.nearbyMetres!.round()} m)' : 'You are here',
              style: TextStyle(color: AM.lime, fontWeight: FontWeight.w800, letterSpacing: 0.3),
            ),
          ),
        ]),
        const SizedBox(height: 10),
        MapCard(m: h, highlight: true),
      ]);
    }
    final (title, sub, action) = switch (s.geoStatus) {
      GeoStatus.off => ('Location is off', 'Turn on location to find the building you are in.', 'Try again'),
      GeoStatus.denied => ('Location not allowed', 'Allow location so AccessMap can open the right map for you.', 'Allow'),
      GeoStatus.ok => (
          'No saved map here',
          s.maps.isEmpty ? 'Sync maps first.' : 'None of your maps is pinned near this spot. Open a map and pin it from its menu.',
          'Check again'
        ),
      GeoStatus.unknown => ('Which building are you in?', 'Use your location to open the right map automatically.', 'Find my building'),
    };
    return Container(
      padding: const EdgeInsets.all(16),
      decoration: BoxDecoration(
        color: AM.surface,
        borderRadius: BorderRadius.circular(24),
        border: Border.all(color: AM.stroke),
      ),
      child: Row(children: [
        Container(
          width: 48,
          height: 48,
          decoration: BoxDecoration(color: AM.lime.withValues(alpha: 0.14), borderRadius: BorderRadius.circular(16)),
          child: Icon(Icons.travel_explore_rounded, color: AM.lime),
        ),
        const SizedBox(width: 14),
        Expanded(
          child: Column(crossAxisAlignment: CrossAxisAlignment.start, children: [
            Text(title, style: const TextStyle(fontWeight: FontWeight.w800, fontSize: 15)),
            const SizedBox(height: 2),
            Text(sub, style: TextStyle(color: AM.muted, fontSize: 13, height: 1.3)),
            const SizedBox(height: 10),
            FilledButton.icon(
              onPressed: s.locating ? null : () => s.locate(),
              style: FilledButton.styleFrom(minimumSize: const Size(0, 44)),
              icon: s.locating
                  ? SizedBox(width: 16, height: 16, child: CircularProgressIndicator(strokeWidth: 2, color: AM.bg))
                  : const Icon(Icons.my_location_rounded, size: 18),
              label: Text(action),
            ),
          ]),
        ),
      ]),
    );
  }
}

class MapCard extends StatelessWidget {
  const MapCard({super.key, required this.m, this.highlight = false});
  final LocalMap m;
  final bool highlight;
  @override
  Widget build(BuildContext context) {
    final s = AppScope.of(context);
    final p = m.pack;
    final updating = s.updating.contains(p.id);
    final remote = s.remote.where((r) => r.id == p.id).firstOrNull;
    final stale = remote != null && remote.version != p.version;
    final a = p.access;
    return Semantics(
      button: true,
      label: 'Open map ${p.name}, ${p.places.length} places${a != null ? ', accessibility ${a.score}' : ''}',
      child: Container(
        decoration: BoxDecoration(
          borderRadius: BorderRadius.circular(28),
          border: Border.all(color: highlight ? AM.lime : AM.stroke, width: highlight ? 2 : 1),
          boxShadow: highlight ? [BoxShadow(color: AM.lime.withValues(alpha: 0.18), blurRadius: 30, offset: const Offset(0, 10))] : null,
        ),
        child: Material(
          color: AM.surface,
          borderRadius: BorderRadius.circular(27),
          clipBehavior: Clip.antiAlias,
          child: InkWell(
            onTap: () => Navigator.of(context).push(MaterialPageRoute(builder: (_) => MapScreen(mapId: p.id))),
            child: Column(crossAxisAlignment: CrossAxisAlignment.start, children: [
              SizedBox(
                height: 132,
                width: double.infinity,
                child: Stack(fit: StackFit.expand, children: [
                  MapThumb(pack: p),
                  Positioned(
                    top: 12,
                    left: 12,
                    child: updating
                        ? const SizedBox(width: 18, height: 18, child: CircularProgressIndicator(strokeWidth: 2))
                        : Container(
                            padding: const EdgeInsets.symmetric(horizontal: 9, vertical: 4),
                            decoration: BoxDecoration(color: stale ? AM.amber : AM.bg.withValues(alpha: 0.75), borderRadius: BorderRadius.circular(99)),
                            child: Row(mainAxisSize: MainAxisSize.min, children: [
                              Icon(stale ? Icons.download_rounded : Icons.offline_pin_rounded, size: 13, color: stale ? AM.onLime : AM.lime),
                              const SizedBox(width: 4),
                              Text(stale ? 'UPDATE' : 'OFFLINE',
                                  style: TextStyle(fontSize: 10, fontWeight: FontWeight.w800, letterSpacing: 0.8, color: stale ? AM.onLime : AM.text)),
                            ]),
                          ),
                  ),
                  if (a != null) Positioned(top: 10, right: 12, child: ScorePill(score: a.score, grade: a.grade)),
                  if (p.isBuilding)
                    Positioned(
                      top: 10,
                      right: 12,
                      child: Container(
                        padding: const EdgeInsets.symmetric(horizontal: 9, vertical: 4),
                        decoration: BoxDecoration(color: AM.bg.withValues(alpha: 0.75), borderRadius: BorderRadius.circular(99), border: Border.all(color: AM.stroke)),
                        child: Row(mainAxisSize: MainAxisSize.min, children: [
                          Icon(Icons.layers_rounded, size: 14, color: AM.text),
                          const SizedBox(width: 4),
                          Text('${p.floors.length} floors', style: TextStyle(fontSize: 12, fontWeight: FontWeight.w800, color: AM.text)),
                        ]),
                      ),
                    ),
                ]),
              ),
              Padding(
                padding: const EdgeInsets.fromLTRB(16, 14, 16, 16),
                child: Column(crossAxisAlignment: CrossAxisAlignment.start, children: [
                  Row(children: [
                    Expanded(
                      child: Text(p.name, maxLines: 2, overflow: TextOverflow.ellipsis, style: Theme.of(context).textTheme.titleMedium?.copyWith(fontWeight: FontWeight.w800)),
                    ),
                    Icon(Icons.arrow_forward_rounded, color: AM.lime),
                  ]),
                  const SizedBox(height: 10),
                  Wrap(spacing: 6, runSpacing: 6, children: [
                    for (final pl in p.places.take(5))
                      Container(
                        padding: const EdgeInsets.symmetric(horizontal: 10, vertical: 4),
                        decoration: BoxDecoration(color: AM.place(pl.kind).withValues(alpha: 0.14), borderRadius: BorderRadius.circular(99)),
                        child: Text(pl.name, style: TextStyle(fontSize: 12, fontWeight: FontWeight.w700, color: AM.place(pl.kind))),
                      ),
                    if (p.places.isEmpty) Text('No rooms tagged yet', style: TextStyle(color: AM.muted, fontSize: 12)),
                  ]),
                  const SizedBox(height: 10),
                  Text(
                    [
                      '${p.places.length} places',
                      if (a != null && a.issues.isNotEmpty) '${a.issues.length} access issue${a.issues.length == 1 ? '' : 's'}',
                      p.metric ? 'metric' : 'relative scale',
                      '${(m.bytesOnDisk / 1e6).toStringAsFixed(1)} MB',
                    ].join(' · '),
                    style: TextStyle(fontSize: 12, color: AM.muted),
                  ),
                ]),
              ),
            ]),
          ),
        ),
      ),
    );
  }
}

class _EmptyMaps extends StatelessWidget {
  const _EmptyMaps();
  @override
  Widget build(BuildContext context) {
    final s = AppScope.of(context);
    return Card(
      child: Padding(
        padding: const EdgeInsets.all(22),
        child: Column(crossAxisAlignment: CrossAxisAlignment.start, children: [
          Container(
            width: 48,
            height: 48,
            decoration: BoxDecoration(color: AM.lime.withValues(alpha: 0.14), borderRadius: BorderRadius.circular(16)),
            child: Icon(Icons.map_rounded, color: AM.lime),
          ),
          const SizedBox(height: 14),
          Text('No maps on this phone yet', style: Theme.of(context).textTheme.titleMedium),
          const SizedBox(height: 6),
          Text(
            s.online
                ? 'The server has no finished scans yet. Process a walkthrough video in AI Studio, then pull down to sync.'
                : 'Connect once to download maps; after that they work offline.\n\n'
                    'Wi-Fi: set the computer\'s address in Settings (e.g. http://192.168.1.20:8000).\n'
                    'USB: run "adb reverse tcp:8000 tcp:8000" on the computer.',
            style: TextStyle(color: AM.muted, height: 1.45),
          ),
          const SizedBox(height: 14),
          Wrap(spacing: 10, runSpacing: 10, children: [
            FilledButton.icon(onPressed: () => s.sync(), icon: const Icon(Icons.sync_rounded), label: const Text('Try again')),
            OutlinedButton(
              onPressed: () => Navigator.of(context).push(MaterialPageRoute(builder: (_) => const SettingsScreen())),
              child: const Text('Server settings'),
            ),
          ]),
        ]),
      ),
    );
  }
}
