# Flutter plugins keep their own rules; keep TTS / speech services reachable via reflection.
-keep class com.tundralabs.fluttertts.** { *; }
-keep class com.csdcorp.speech_to_text.** { *; }
