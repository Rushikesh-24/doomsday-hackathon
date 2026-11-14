# AccessMap mobile (Android, Flutter)

Accessible indoor wayfinding on the phone, built from the 3D scans processed by the AccessMap server.
**Offline-first:** maps are downloaded once and then work with no network, including routing, turn-by-turn
directions, spoken guidance and the "Guide" assistant (on-device mode).

## What it does

* **Saved maps.** Every finished scan on the server shows up automatically (`GET /maps`) and is stored on the
  phone as a ~25 KB map pack plus its floor-plan image. Packs are versioned by content hash: the app re-downloads
  only what changed, on start, every minute and when you come back to the app. Rename or hide a map from the web
  Explorer → Details → *Phone app*.
* **Travelling as Walking / Wheelchair / Low vision.** Routes, warnings and spoken help adapt: step-free only,
  handrail and obstacle advice, and arrival hints from each tagged place.
* **Where to?** Tap a pin or chip, type, or say it ("from CC lab to the graphics lab", "nearest lift").
* **Guidance mode.** A dark, high-contrast view with huge step cards, haptics, voice, and Next/Previous buttons.
  *Preview walk* animates your position along the route and speaks each step as it is reached.
  (There is no indoor positioning, so the traveller confirms each step.)
* **Guide (assistant).** When online, the server's Groq-powered assistant answers questions and plans real
  routes with the same router. Offline, the app answers on the phone. The Groq key never leaves the server.
* **You are here.** GPS picks the building: each map can be pinned to a location (web Explorer → Details, or the
  map's layers menu → *Pin this map to my location*), and the home screen opens the matching map.
* **Camera guide (online).** Point the phone along the corridor: the photo is matched against the scan's keyframes
  (DINOv2 + SIFT/RANSAC on the server) to find where you are and which way you face; the route is re-planned on the
  phone and lime AR chevrons point the way. Detected stairs / doors / obstacles are spoken ("Careful: stairs
  ahead"), *Describe* reads a caption, *Ask Guide* hands the scene to the Groq assistant.
* **Accessibility score.** Each map shows its 0-100 score and grade, the six weighted categories and the problem
  spots on the plan (same numbers as the website).
* **Accessibility.** Larger text, high-contrast theme, screen-reader labels and live regions, 48 dp+ targets,
  and the system text-scale setting is respected.

Routing runs **on the device** as a 1:1 Dart port of the server's Python router (`lib/core/router.dart`).
`test/router_parity_test.dart` checks it word-for-word against fixtures produced by the Python implementation
(`scripts/export_router_fixtures.py`).

## Run on a phone over USB

1. On the phone: Settings → About → tap *Build number* 7 times; then Developer options → enable **USB debugging**.
   Plug in and accept the "Allow USB debugging" prompt.
2. On the computer, with the AccessMap API running on port 8000:

```bash
adb reverse tcp:8000 tcp:8000
```
```bash
flutter run --release
```

The app's default server is `http://127.0.0.1:8000`, which `adb reverse` maps to the computer. Over Wi-Fi, set
the computer's LAN address in Settings (e.g. `http://192.168.1.20:8000`) instead.

## Build

```bash
flutter build apk --release --split-per-abi
```

Release signing: create `android/key.properties` (git-ignored) with `storeFile`, `storePassword`, `keyAlias`
and `keyPassword`. Without it, release builds use the debug key, which is fine for testing but not for the Play
Store.

## Tests

```bash
flutter test
```

* `router_parity_test.dart`: the on-device router equals the server (10 real requests on the demo scan).
* `app_flow_test.dart`: offline UI flows, including 2x system text on a 360 dp phone with no overflow. Onboarding; opening a saved map; routing to CG Lab step-free and
  stepping through guidance; and a wheelchair user at the top of the stairs getting an honest "no step-free
  route".

Regenerate the fixtures after changing routing on either side:

```bash
../.venv/Scripts/python ../scripts/export_router_fixtures.py demopcc
```

Screenshots of every screen (and the launcher icon, drawn from the same `LogoPainter` as the app):

```bash
flutter test test_screens --update-goldens
```

## Layout

```
lib/core/   models.dart (map pack), router.dart (wayfinding), api.dart, store.dart (offline files),
            app_state.dart (settings, connectivity, sync), voice.dart (TTS + speech input)
lib/ui/     theme.dart, widgets/ (plan view, common), screens/ (onboarding, home, map, guidance, assistant,
            settings)
```
