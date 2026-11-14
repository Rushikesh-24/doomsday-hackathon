// Speech out (flutter_tts, uses the phone's offline TTS voices) and speech in (speech_to_text; Android's
// recogniser works offline when the language pack is installed).
import 'package:flutter_tts/flutter_tts.dart';
import 'package:speech_to_text/speech_to_text.dart';

class Voice {
  Voice._();
  static final Voice instance = Voice._();

  final FlutterTts _tts = FlutterTts();
  final SpeechToText _stt = SpeechToText();
  bool _ttsReady = false;
  bool? _sttReady;

  Future<void> _initTts() async {
    if (_ttsReady) return;
    try {
      await _tts.setSpeechRate(0.47);
      await _tts.setPitch(1.0);
      await _tts.awaitSpeakCompletion(false);
    } catch (_) {}
    _ttsReady = true;
  }

  Future<void> say(String text) async {
    if (text.trim().isEmpty) return;
    await _initTts();
    try {
      await _tts.stop();
      await _tts.speak(text);
    } catch (_) {}
  }

  Future<void> stop() async {
    try {
      await _tts.stop();
    } catch (_) {}
  }

  Future<bool> get canListen async {
    if (_sttReady != null) return _sttReady!;
    try {
      _sttReady = await _stt.initialize(onError: (_) {}, onStatus: (_) {});
    } catch (_) {
      _sttReady = false;
    }
    return _sttReady!;
  }

  bool get listening => _stt.isListening;

  /// Listens once; calls [onText] with partial results and [onDone] with the final text.
  Future<bool> listen({required void Function(String) onText, required void Function(String) onDone}) async {
    if (!await canListen) return false;
    await stop();
    await _stt.listen(
      onResult: (r) {
        onText(r.recognizedWords);
        if (r.finalResult) onDone(r.recognizedWords);
      },
      listenOptions: SpeechListenOptions(
        partialResults: true,
        cancelOnError: true,
        listenMode: ListenMode.search,
        listenFor: const Duration(seconds: 12),
        pauseFor: const Duration(seconds: 3),
      ),
    );
    return true;
  }

  Future<void> stopListening() async {
    try {
      await _stt.stop();
    } catch (_) {}
  }
}
