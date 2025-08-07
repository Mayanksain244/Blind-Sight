import 'package:flutter/material.dart';
import 'package:flutter_tts/flutter_tts.dart';

class TTSService {
  static final FlutterTts _flutterTts = FlutterTts();

  static Future<void> initialize() async {
    await _flutterTts.setLanguage('en-US');
    await _flutterTts.setSpeechRate(0.5);
    await _flutterTts.setVolume(1.0);
    await _flutterTts.setPitch(1.0);
  }

  static Future<void> speak(String text) async {
    if (text.isEmpty) return;

    try {
      await _flutterTts.speak(text);
    } catch (e) {
      debugPrint('TTS Speaking Error: $e');
    }
  }

  static Future<void> stop() async {
    try {
      await _flutterTts.stop();
    } catch (e) {
      debugPrint('TTS Stop Error: $e');
    }
  }

  static Future<void> pause() async {
    try {
      await _flutterTts.pause();
    } catch (e) {
      debugPrint('TTS Pause Error: $e');
    }
  }

  static Future<void> setLanguage(String language) async {
    try {
      await _flutterTts.setLanguage(language);
    } catch (e) {
      debugPrint('TTS Language Setting Error: $e');
    }
  }

  static Future<void> setSpeechRate(double rate) async {
    try {
      await _flutterTts.setSpeechRate(rate);
    } catch (e) {
      debugPrint('TTS Speech Rate Setting Error: $e');
    }
  }

  static Future<void> setVolume(double volume) async {
    try {
      await _flutterTts.setVolume(volume);
    } catch (e) {
      debugPrint('TTS Volume Setting Error: $e');
    }
  }

  static Future<void> setPitch(double pitch) async {
    try {
      await _flutterTts.setPitch(pitch);
    } catch (e) {
      debugPrint('TTS Pitch Setting Error: $e');
    }
  }
}
