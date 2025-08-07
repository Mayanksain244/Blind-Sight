// import 'package:flutter/material.dart';
// import 'package:camera/camera.dart';
// import 'dart:io';
// import '../services/image_caption_service.dart'; // Make sure to create this path

// class CameraScreen extends StatefulWidget {
//   const CameraScreen({Key? key}) : super(key: key);

//   @override
//   _CameraScreenState createState() => _CameraScreenState();
// }

// class _CameraScreenState extends State<CameraScreen> {
//   CameraController? _controller;
//   List<CameraDescription> cameras = [];
//   XFile? capturedImage;
//   bool isCapturing = false;
//   String bottomText = "Tap to capture an image"; // Default instruction
//   final ImageCaptionService _captionService = ImageCaptionService();

//   @override
//   void initState() {
//     super.initState();
//     _initializeCamera();
//   }

//   Future<void> _initializeCamera() async {
//     try {
//       cameras = await availableCameras();
//       _controller = CameraController(
//         cameras[0],
//         ResolutionPreset.high,
//         imageFormatGroup: ImageFormatGroup.jpeg,
//       );
//       await _controller!.initialize();
//       if (mounted) {
//         setState(() {});
//       }
//     } catch (e) {
//       print('Error initializing camera: $e');
//       setState(() {
//         bottomText = 'Failed to initialize camera';
//       });
//     }
//   }

//   @override
//   void dispose() {
//     _controller?.dispose();
//     super.dispose();
//   }

//   Future<void> _takePhoto() async {
//     if (_controller == null ||
//         !_controller!.value.isInitialized ||
//         isCapturing) {
//       return;
//     }

//     setState(() {
//       isCapturing = true;
//       bottomText = "Generating description...";
//     });

//     try {
//       final XFile file = await _controller!.takePicture();
//       setState(() {
//         capturedImage = file;
//       });

//       // Generate caption
//       try {
//         final caption = await _captionService.uploadImage(File(file.path));
//         setState(() {
//           bottomText = caption;
//         });
//       } catch (e) {
//         setState(() {
//           bottomText = "Failed to generate description: ${e.toString()}";
//         });
//       }

//       // Reset to camera view after a delay
//       await Future.delayed(const Duration(seconds: 3));
//       setState(() {
//         capturedImage = null;
//         isCapturing = false;
//       });
//     } catch (e) {
//       print('Error taking picture: $e');
//       setState(() {
//         isCapturing = false;
//         bottomText = "Failed to capture image";
//       });
//     }
//   }

//   @override
//   Widget build(BuildContext context) {
//     if (_controller == null || !_controller!.value.isInitialized) {
//       return Scaffold(
//         appBar: PreferredSize(
//           preferredSize: const Size.fromHeight(40),
//           child: AppBar(title: const Text("Blind Sight"), centerTitle: true),
//         ),
//         body: const Center(
//           child: CircularProgressIndicator(color: Colors.white),
//         ),
//       );
//     }

//     // Calculate the 4:3 aspect ratio for the camera preview
//     final screenSize = MediaQuery.of(context).size;
//     final previewWidth = screenSize.width;
//     final previewHeight = previewWidth * 4 / 3;

//     // Ensure the preview doesn't exceed the available height
//     final appBarHeight = 40.0;
//     final bottomAreaHeight = 120.0;
//     final maxAvailableHeight =
//         screenSize.height - appBarHeight - bottomAreaHeight;

//     final finalPreviewHeight =
//         previewHeight > maxAvailableHeight ? maxAvailableHeight : previewHeight;

//     return Scaffold(
//       appBar: PreferredSize(
//         preferredSize: const Size.fromHeight(40),
//         child: AppBar(
//           title: const Text(
//             "Blind Sight",
//             style: TextStyle(color: Colors.white),
//           ),
//           centerTitle: true,
//         ),
//       ),
//       body: Container(
//         color: Colors.black,
//         child: Column(
//           children: [
//             SizedBox(height: 20),

//             // Camera preview with 4:3 aspect ratio
//             SizedBox(
//               width: previewWidth,
//               height: finalPreviewHeight,
//               child: ClipRRect(
//                 borderRadius: BorderRadius.circular(12),
//                 child:
//                     capturedImage == null
//                         ? CameraPreview(_controller!)
//                         : Image.file(
//                           File(capturedImage!.path),
//                           fit: BoxFit.cover,
//                         ),
//               ),
//             ),

//             // Expanded bottom area for text and tap detection
//             Expanded(
//               child: GestureDetector(
//                 onTap: _takePhoto,
//                 child: Container(
//                   width: double.infinity,
//                   color: Colors.black,
//                   padding: const EdgeInsets.all(16),
//                   alignment: Alignment.center,
//                   child: Text(
//                     bottomText,
//                     style: const TextStyle(
//                       fontSize: 18,
//                       fontWeight: FontWeight.bold,
//                       color: Colors.white,
//                     ),
//                     textAlign: TextAlign.center,
//                   ),
//                 ),
//               ),
//             ),

//             // Loading indicator when capturing
//             if (isCapturing)
//               const Center(
//                 child: CircularProgressIndicator(color: Colors.white),
//               ),
//           ],
//         ),
//       ),
//     );
//   }
// }

import 'package:flutter/material.dart';
import 'package:camera/camera.dart';
import 'dart:io';
import '../services/image_caption_service.dart';
import '../services/tts_service.dart';

class CameraScreen extends StatefulWidget {
  const CameraScreen({Key? key}) : super(key: key);

  @override
  _CameraScreenState createState() => _CameraScreenState();
}

class _CameraScreenState extends State<CameraScreen> {
  CameraController? _controller;
  List<CameraDescription> cameras = [];
  XFile? capturedImage;
  bool isCapturing = false;
  String bottomText = "Tap to capture an image"; // Default instruction
  final ImageCaptionService _captionService = ImageCaptionService();

  @override
  void initState() {
    super.initState();
    _initializeCamera();
    _initializeTTS();
  }

  Future<void> _initializeCamera() async {
    try {
      cameras = await availableCameras();
      _controller = CameraController(
        cameras[0],
        ResolutionPreset.high,
        imageFormatGroup: ImageFormatGroup.jpeg,
      );
      await _controller!.initialize();
      if (mounted) {
        setState(() {});
      }
    } catch (e) {
      print('Error initializing camera: $e');
      setState(() {
        bottomText = 'Failed to initialize camera';
      });
    }
  }

  Future<void> _initializeTTS() async {
    try {
      await TTSService.initialize();
    } catch (e) {
      print('Error initializing TTS: $e');
    }
  }

  @override
  void dispose() {
    _controller?.dispose();
    TTSService.stop(); // Stop any ongoing speech when disposing
    super.dispose();
  }

  Future<void> _takePhoto() async {
    if (_controller == null ||
        !_controller!.value.isInitialized ||
        isCapturing) {
      return;
    }

    setState(() {
      isCapturing = true;
      bottomText = "Generating description...";
    });

    try {
      final XFile file = await _controller!.takePicture();
      setState(() {
        capturedImage = file;
      });

      // Generate caption
      try {
        final caption = await _captionService.uploadImage(File(file.path));
        setState(() {
          bottomText = caption;
        });

        // Automatically speak the caption
        await _speakCaption(caption);
      } catch (e) {
        setState(() {
          bottomText = "Failed to generate description: ${e.toString()}";
        });
      }

      // Reset to camera view after a delay
      await Future.delayed(const Duration(seconds: 3));
      setState(() {
        capturedImage = null;
        isCapturing = false;
      });
    } catch (e) {
      print('Error taking picture: $e');
      setState(() {
        isCapturing = false;
        bottomText = "Failed to capture image";
      });
    }
  }

  Future<void> _speakCaption(String caption) async {
    try {
      // Stop any ongoing speech
      await TTSService.stop();

      // Speak the caption
      await TTSService.speak(caption);
    } catch (e) {
      print('Error speaking caption: $e');
    }
  }

  @override
  Widget build(BuildContext context) {
    if (_controller == null || !_controller!.value.isInitialized) {
      return Scaffold(
        appBar: PreferredSize(
          preferredSize: const Size.fromHeight(40),
          child: AppBar(title: const Text("Blind Sight"), centerTitle: true),
        ),
        body: const Center(
          child: CircularProgressIndicator(color: Colors.white),
        ),
      );
    }

    // Calculate the 4:3 aspect ratio for the camera preview
    final screenSize = MediaQuery.of(context).size;
    final previewWidth = screenSize.width;
    final previewHeight = previewWidth * 4 / 3;

    // Ensure the preview doesn't exceed the available height
    final appBarHeight = 40.0;
    final bottomAreaHeight = 120.0;
    final maxAvailableHeight =
        screenSize.height - appBarHeight - bottomAreaHeight;

    final finalPreviewHeight =
        previewHeight > maxAvailableHeight ? maxAvailableHeight : previewHeight;

    return Scaffold(
      appBar: PreferredSize(
        preferredSize: const Size.fromHeight(40),
        child: AppBar(
          title: const Text(
            "Blind Sight",
            style: TextStyle(color: Colors.white),
          ),
          centerTitle: true,
          actions: [
            // Add a button to manually replay the last caption
            IconButton(
              icon: const Icon(Icons.volume_up),
              onPressed:
                  bottomText.isNotEmpty &&
                          bottomText != "Tap to capture an image" &&
                          bottomText != "Generating description..." &&
                          bottomText != "Failed to generate description"
                      ? () => _speakCaption(bottomText)
                      : null,
            ),
          ],
        ),
      ),
      body: Container(
        color: Colors.black,
        child: Column(
          children: [
            const SizedBox(height: 20),

            // Camera preview with 4:3 aspect ratio
            SizedBox(
              width: previewWidth,
              height: finalPreviewHeight,
              child: ClipRRect(
                borderRadius: BorderRadius.circular(12),
                child:
                    capturedImage == null
                        ? CameraPreview(_controller!)
                        : Image.file(
                          File(capturedImage!.path),
                          fit: BoxFit.cover,
                        ),
              ),
            ),

            // Expanded bottom area for text and tap detection
            Expanded(
              child: GestureDetector(
                onTap: _takePhoto,
                child: Container(
                  width: double.infinity,
                  color: Colors.black,
                  padding: const EdgeInsets.all(16),
                  alignment: Alignment.center,
                  child: Text(
                    bottomText,
                    style: const TextStyle(
                      fontSize: 18,
                      fontWeight: FontWeight.bold,
                      color: Colors.white,
                    ),
                    textAlign: TextAlign.center,
                  ),
                ),
              ),
            ),

            // Loading indicator when capturing
            if (isCapturing)
              const Center(
                child: CircularProgressIndicator(color: Colors.white),
              ),
          ],
        ),
      ),
    );
  }
}
