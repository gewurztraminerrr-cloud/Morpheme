import 'dart:io' show Platform, File;
import 'dart:convert' show base64Encode, jsonDecode;
import 'package:flutter/material.dart';
import 'package:flutter/services.dart';
import 'package:webview_flutter/webview_flutter.dart';
import 'package:webview_flutter_android/webview_flutter_android.dart';
import 'package:webview_flutter_wkwebview/webview_flutter_wkwebview.dart';
import 'package:file_picker/file_picker.dart';
import 'package:flutter_soloud/flutter_soloud.dart';
import 'package:audio_session/audio_session.dart';

void main() {
  WidgetsFlutterBinding.ensureInitialized();
  // Start with edge-to-edge system bars visible
  SystemChrome.setEnabledSystemUIMode(SystemUiMode.edgeToEdge);
  SystemChrome.setSystemUIOverlayStyle(const SystemUiOverlayStyle(
    statusBarColor: Colors.transparent,
    statusBarIconBrightness: Brightness.light,
    statusBarBrightness: Brightness.dark,
    systemNavigationBarColor: Colors.transparent,
    systemNavigationBarDividerColor: Colors.transparent,
    systemNavigationBarIconBrightness: Brightness.light,
  ));

  // After 3 seconds, transition to immersiveSticky mode: navigation buttons disappear
  // and swipe-up from bottom temporarily reveals them for a few seconds
  Future.delayed(const Duration(seconds: 3), () {
    SystemChrome.setEnabledSystemUIMode(SystemUiMode.immersiveSticky);
  });

  runApp(const MorphemeApp());
}

class MorphemeApp extends StatelessWidget {
  const MorphemeApp({super.key});

  @override
  Widget build(BuildContext context) {
    return MaterialApp(
      title: 'Morpheme Word Game',
      debugShowCheckedModeBanner: false,
      theme: ThemeData(
        brightness: Brightness.dark,
        scaffoldBackgroundColor: const Color(0xFF0D1117), // Morpheme Dark BG
      ),
      home: const GameScreen(),
    );
  }
}

class GameScreen extends StatefulWidget {
  const GameScreen({super.key});

  @override
  State<GameScreen> createState() => _GameScreenState();
}

class _GameScreenState extends State<GameScreen> {
  late final WebViewController controller;
  final Map<int, AudioSource> _tileSources = {};
  final Map<String, AudioSource> _bellSources = {};
  AudioSource? _successSource;
  AudioSource? _failureSource;
  AudioSource? _keepAliveSource;
  SoundHandle? _keepAliveHandle;
  bool _soLoudInitialized = false;

  void _playSound(AudioSource? source) {
    if (source == null || !_soLoudInitialized) return;
    try {
      SoLoud.instance.play(source);
    } catch (e) {
      debugPrint("Error playing sound: $e");
    }
  }

  @override
  void dispose() {
    SoLoud.instance.deinit();
    super.dispose();
  }

  Future<void> _initAudio() async {
    try {
      final soloud = SoLoud.instance;
      // Set a small buffer size (512) for ultra-low latency
      await soloud.init(bufferSize: 512);

      // Load all sources
      for (int i = 1; i <= 16; i++) {
        final source = await soloud.loadAsset('assets/sounds/tile_$i.wav');
        _tileSources[i] = source;
      }
      _successSource = await soloud.loadAsset('assets/sounds/success.wav');
      _failureSource = await soloud.loadAsset('assets/sounds/failure.wav');

      // Load bell/beep warning chimes
      for (final type in ['bell1', 'bell2', 'bell3', 'beep1', 'beep2', 'beep3']) {
        final source = await soloud.loadAsset('assets/sounds/$type.wav');
        _bellSources[type] = source;
      }

      // Load keep-alive source
      _keepAliveSource = await soloud.loadAsset('assets/sounds/keep_alive.wav');

      setState(() {
        _soLoudInitialized = true;
      });
      debugPrint("SoLoud audio engine initialized successfully with low latency buffer");

      // Listen to audio session routing changes
      final session = await AudioSession.instance;
      session.devicesChangedEventStream.listen((event) {
        debugPrint("[AudioSession] Audio devices changed. Reconfiguring session...");
        _configureAudioSession();
      });

      // Initial session configuration
      await _configureAudioSession();
    } catch (e) {
      debugPrint("Error initializing SoLoud: $e");
    }
  }

  Future<void> _configureAudioSession() async {
    try {
      final session = await AudioSession.instance;
      final devices = await session.getDevices();
      bool hasBluetooth = false;
      for (final device in devices) {
        if (device.type == AudioDeviceType.bluetoothA2dp ||
            device.type == AudioDeviceType.bluetoothSco ||
            device.type == AudioDeviceType.bluetoothLe) {
          hasBluetooth = true;
          break;
        }
      }

      if (hasBluetooth) {
        debugPrint("[AudioSession] Bluetooth device detected. Setting category to playAndRecord and mode to voiceChat...");
        await session.configure(AudioSessionConfiguration(
          avAudioSessionCategory: AVAudioSessionCategory.playAndRecord,
          avAudioSessionCategoryOptions: AVAudioSessionCategoryOptions.allowBluetooth |
              AVAudioSessionCategoryOptions.defaultToSpeaker,
          avAudioSessionMode: AVAudioSessionMode.voiceChat,
          avAudioSessionRouteSharingPolicy: AVAudioSessionRouteSharingPolicy.defaultPolicy,
          androidAudioAttributes: const AndroidAudioAttributes(
            contentType: AndroidAudioContentType.music,
            usage: AndroidAudioUsage.game,
          ),
          androidAudioFocusGainType: AndroidAudioFocusGainType.gain,
        ));

        // Start native 30Hz keep-alive to keep Bluetooth active
        _updateKeepAlive(true);
      } else {
        debugPrint("[AudioSession] Standard playback detected. Setting category to playback...");
        await session.configure(AudioSessionConfiguration(
          avAudioSessionCategory: AVAudioSessionCategory.playback,
          avAudioSessionCategoryOptions: AVAudioSessionCategoryOptions.mixWithOthers,
          avAudioSessionMode: AVAudioSessionMode.defaultMode,
          avAudioSessionRouteSharingPolicy: AVAudioSessionRouteSharingPolicy.defaultPolicy,
          androidAudioAttributes: const AndroidAudioAttributes(
            contentType: AndroidAudioContentType.music,
            usage: AndroidAudioUsage.game,
          ),
          androidAudioFocusGainType: AndroidAudioFocusGainType.gain,
        ));

        // Stop native 30Hz keep-alive to save battery
        _updateKeepAlive(false);
      }
      await session.setActive(true);
    } catch (e) {
      debugPrint("Error configuring AudioSession: $e");
    }
  }

  void _updateKeepAlive(bool enable) {
    if (!_soLoudInitialized || _keepAliveSource == null) return;
    try {
      if (enable) {
        if (_keepAliveHandle == null) {
          _keepAliveHandle = SoLoud.instance.play(_keepAliveSource!, looping: true, volume: 0.02);
          debugPrint("[SoLoud] Native Bluetooth keep-alive loop started.");
        }
      } else {
        if (_keepAliveHandle != null) {
          SoLoud.instance.stop(_keepAliveHandle!);
          _keepAliveHandle = null;
          debugPrint("[SoLoud] Native Bluetooth keep-alive loop stopped.");
        }
      }
    } catch (e) {
      debugPrint("Error updating keep-alive: $e");
    }
  }

  @override
  void initState() {
    super.initState();
    _initAudio();

    controller = WebViewController()
      ..setJavaScriptMode(JavaScriptMode.unrestricted)
      ..setBackgroundColor(const Color(0xFF0D1117))
      ..addJavaScriptChannel(
        'MorphemeFilePicker',
        onMessageReceived: (JavaScriptMessage message) async {
          final inputId = message.message;
          final isImage = inputId.contains('image') || inputId.contains('avatar');
          try {
            final result = await FilePicker.pickFiles(
              allowMultiple: false,
              type: isImage ? FileType.image : FileType.any,
            );
            if (result != null && result.files.isNotEmpty && result.files.first.path != null) {
              final file = File(result.files.first.path!);
              final bytes = await file.readAsBytes();
              final base64Data = base64Encode(bytes);
              final filename = result.files.first.name;
              final mimeType = isImage ? 'image/jpeg' : 'text/plain';
              final jsCode = "window.setFileInputBlob('$inputId', '$base64Data', '$filename', '$mimeType');";
              await controller.runJavaScript(jsCode);
            }
          } catch (e) {
            debugPrint("Error in MorphemeFilePicker channel: $e");
          }
        },
      )
      ..addJavaScriptChannel(
        'MorphemeAudioBridge',
        onMessageReceived: (JavaScriptMessage message) {
          try {
            final data = jsonDecode(message.message);
            final String sound = data['sound'] ?? '';
            if (sound == 'tile') {
              int pathLen = data['pathLen'] ?? 1;
              if (pathLen < 1) pathLen = 1;
              if (pathLen > 16) pathLen = 16;
              final source = _tileSources[pathLen];
              if (source != null) {
                _playSound(source);
              }
            } else if (sound == 'success') {
              _playSound(_successSource);
            } else if (sound == 'failure') {
              _playSound(_failureSource);
            } else if (sound == 'bell') {
              final String type = data['type'] ?? 'bell1';
              final source = _bellSources[type];
              if (source != null) {
                _playSound(source);
              }
            }
          } catch (e) {
            debugPrint("Error in MorphemeAudioBridge channel: $e");
          }
        },
      )
      ..loadRequest(Uri.parse('https://morpheme.games/'));

    if (Platform.isAndroid) {
      final androidController = controller.platform as AndroidWebViewController;
      androidController.setOnShowFileSelector((
        FileSelectorParams params,
      ) async {
        try {
          final isImage = params.acceptTypes.any(
            (type) =>
                type.contains('image') ||
                type.endsWith('.jpg') ||
                type.endsWith('.png') ||
                type.endsWith('.jpeg') ||
                type.endsWith('.gif'),
          );
          final result = await FilePicker.pickFiles(
            allowMultiple: params.mode == FileSelectorMode.openMultiple,
            type: isImage ? FileType.image : FileType.any,
          );

          if (result != null && result.files.isNotEmpty) {
            return result.files
                .where((file) => file.path != null)
                .map((file) => Uri.file(file.path!).toString())
                .toList();
          }
        } catch (e) {
          debugPrint("Error picking files in webview: $e");
        }
        return [];
      });
    } else if (Platform.isIOS) {
      if (controller.platform is WebKitWebViewController) {
        (controller.platform as WebKitWebViewController).setInspectable(true);
      }
    }
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      body: SafeArea(child: WebViewWidget(controller: controller)),
    );
  }
}
