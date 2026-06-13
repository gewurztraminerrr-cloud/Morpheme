import 'dart:io' show Platform, File;
import 'dart:convert' show base64Encode, jsonDecode;
import 'package:flutter/material.dart';
import 'package:flutter/services.dart';
import 'package:webview_flutter/webview_flutter.dart';
import 'package:webview_flutter_android/webview_flutter_android.dart';
import 'package:webview_flutter_wkwebview/webview_flutter_wkwebview.dart';
import 'package:file_picker/file_picker.dart';
import 'package:audioplayers/audioplayers.dart';

void main() {
  WidgetsFlutterBinding.ensureInitialized();
  // Hide top status bar for an immersive full-screen game experience
  SystemChrome.setEnabledSystemUIMode(SystemUiMode.immersiveSticky);
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
  final Map<int, AudioPlayer> _tilePlayers = {};
  late final AudioPlayer _successPlayer;
  late final AudioPlayer _failurePlayer;

  void _playSound(AudioPlayer player) {
    player.stop().then((_) {
      player.resume();
    });
  }

  @override
  void dispose() {
    for (final player in _tilePlayers.values) {
      player.dispose();
    }
    _successPlayer.dispose();
    _failurePlayer.dispose();
    super.dispose();
  }

  @override
  void initState() {
    super.initState();

    // Preload audio players for low latency
    for (int i = 1; i <= 16; i++) {
      final player = AudioPlayer();
      player.setReleaseMode(ReleaseMode.stop);
      player.setSource(AssetSource('sounds/tile_$i.wav'));
      _tilePlayers[i] = player;
    }
    _successPlayer = AudioPlayer();
    _successPlayer.setReleaseMode(ReleaseMode.stop);
    _successPlayer.setSource(AssetSource('sounds/success.wav'));

    _failurePlayer = AudioPlayer();
    _failurePlayer.setReleaseMode(ReleaseMode.stop);
    _failurePlayer.setSource(AssetSource('sounds/failure.wav'));

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
              final player = _tilePlayers[pathLen];
              if (player != null) {
                _playSound(player);
              }
            } else if (sound == 'success') {
              _playSound(_successPlayer);
            } else if (sound == 'failure') {
              _playSound(_failurePlayer);
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
