// HTTP client for the AccessMap server (FastAPI). All calls time out quickly so the app falls back to its
// offline copy instead of hanging when the server is unreachable.
import 'dart:async';
import 'dart:convert';
import 'dart:io';

import 'package:http/http.dart' as http;

import 'models.dart';

class ApiException implements Exception {
  ApiException(this.status, this.message);
  final int status;
  final String message;
  @override
  String toString() => message;
}

class Api {
  Api(this.baseUrl, {this.token = '', http.Client? client}) : _http = client ?? http.Client();
  final String baseUrl;
  final String token;
  final http.Client _http;

  Uri _u(String path) => Uri.parse('${baseUrl.replaceAll(RegExp(r'/+$'), '')}$path');
  Map<String, String> get _headers => {
        'accept': 'application/json',
        if (token.isNotEmpty) 'x-accessmap-token': token,
      };

  Future<http.Response> _get(String path, {Duration timeout = const Duration(seconds: 8), Map<String, String>? extra}) async {
    try {
      return await _http.get(_u(path), headers: {..._headers, ...?extra}).timeout(timeout);
    } on TimeoutException {
      throw ApiException(0, 'The server did not answer in time.');
    } on SocketException {
      throw ApiException(0, 'Cannot reach the server.');
    } on http.ClientException catch (e) {
      throw ApiException(0, 'Cannot reach the server (${e.message}).');
    }
  }

  dynamic _json(http.Response r) {
    dynamic body;
    try {
      body = r.body.isEmpty ? null : jsonDecode(utf8.decode(r.bodyBytes));
    } catch (_) {
      body = null;
    }
    if (r.statusCode >= 400) {
      final d = body is Map ? body['detail'] : null;
      throw ApiException(r.statusCode, d is String ? d : 'Server error ${r.statusCode}');
    }
    return body;
  }

  Future<bool> health() async {
    try {
      final r = await _get('/vision/health', timeout: const Duration(seconds: 3));
      return r.statusCode == 200;
    } catch (_) {
      return false;
    }
  }

  Future<List<MapSummary>> maps() async {
    final j = _json(await _get('/maps')) as Map<String, dynamic>;
    return (j['maps'] as List).map((m) => MapSummary((m as Map).cast<String, dynamic>())).toList();
  }

  /// Returns null when the server says our version is current (304).
  Future<String?> pack(String id, {String? haveVersion}) async {
    final r = await _get('/maps/$id/pack',
        timeout: const Duration(seconds: 30), extra: haveVersion != null ? {'if-none-match': '"$haveVersion"'} : null);
    if (r.statusCode == 304) return null;
    _json(r);
    return utf8.decode(r.bodyBytes);
  }

  Future<List<int>> bytes(String path) async {
    final r = await _get(path, timeout: const Duration(seconds: 30));
    if (r.statusCode >= 400) throw ApiException(r.statusCode, 'Download failed (${r.statusCode})');
    return r.bodyBytes;
  }

  Future<Map<String, dynamic>> assistantStatus() async =>
      (_json(await _get('/assistant/status', timeout: const Duration(seconds: 6))) as Map).cast<String, dynamic>();

  Future<Map<String, dynamic>> assistantChat({
    required String mapId,
    required List<Map<String, String>> messages,
    required String profile,
    String? start,
    Map<String, dynamic>? route,
  }) async {
    try {
      final r = await _http
          .post(_u('/assistant/chat'),
              headers: {..._headers, 'content-type': 'application/json'},
              body: jsonEncode({'map_id': mapId, 'messages': messages, 'profile': profile, 'start': start, 'route': route}))
          .timeout(const Duration(seconds: 40));
      return (_json(r) as Map).cast<String, dynamic>();
    } on TimeoutException {
      throw ApiException(0, 'The assistant took too long to answer.');
    } on SocketException {
      throw ApiException(0, 'Cannot reach the server.');
    } on http.ClientException {
      throw ApiException(0, 'Cannot reach the server.');
    }
  }

  /// Camera guide: where on [mapId] was this photo taken, and what is in front of the camera.
  Future<Map<String, dynamic>> locate(String mapId, List<int> jpeg, {bool scene = true, bool caption = false}) async {
    try {
      final req = http.MultipartRequest('POST', _u('/maps/$mapId/locate'))
        ..headers.addAll(_headers)
        ..fields['scene'] = '$scene'
        ..fields['caption'] = '$caption'
        ..files.add(http.MultipartFile.fromBytes('file', jpeg, filename: 'camera.jpg'));
      final r = await http.Response.fromStream(await _http.send(req).timeout(const Duration(seconds: 25)));
      return (_json(r) as Map).cast<String, dynamic>();
    } on TimeoutException {
      throw ApiException(0, 'Locating took too long.');
    } on SocketException {
      throw ApiException(0, 'Cannot reach the server.');
    } on http.ClientException {
      throw ApiException(0, 'Cannot reach the server.');
    }
  }

  /// Pin a map to a GPS position (so phones can tell which building they are at).
  Future<void> setLocation(String mapId, double lat, double lng, {double radius = 80}) async {
    try {
      final r = await _http
          .patch(_u('/maps/$mapId'),
              headers: {..._headers, 'content-type': 'application/json'},
              body: jsonEncode({'location': {'lat': lat, 'lng': lng, 'radius': radius}}))
          .timeout(const Duration(seconds: 10));
      _json(r);
    } on TimeoutException {
      throw ApiException(0, 'The server did not answer in time.');
    } on SocketException {
      throw ApiException(0, 'Cannot reach the server.');
    }
  }

  void close() => _http.close();
}
