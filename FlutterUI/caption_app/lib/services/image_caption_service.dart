import 'dart:io';
import 'package:http/http.dart' as http;
import 'package:http_parser/http_parser.dart';
import 'package:mime/mime.dart';
import 'dart:convert';

class ImageCaptionService {
  // Replace with your server's IP and port
  static const String baseUrl = 'http://192.168.137.1:8000';

  /// Upload an image file and retrieve its caption
  Future<String> uploadImage(File imageFile) async {
    try {
      // Create multipart request
      var request = http.MultipartRequest(
        'POST',
        Uri.parse('$baseUrl/upload/'),
      );

      // request.headers['bypass-tunnel-reminder'] = 'true';

      // Determine the mime type of the file
      final mimeType = lookupMimeType(imageFile.path);
      final mimeTypeSegments =
          mimeType?.split('/') ?? ['application', 'octet-stream'];

      // Add the file to the multipart request
      request.files.add(
        await http.MultipartFile.fromPath(
          'file',
          imageFile.path,
          contentType: MediaType(mimeTypeSegments[0], mimeTypeSegments[1]),
        ),
      );

      // Send the request
      final streamedResponse = await request.send();

      // Get the response
      final response = await http.Response.fromStream(streamedResponse);

      // Check the response
      if (response.statusCode == 200) {
        // Parse the JSON response
        final jsonResponse = json.decode(response.body);

        // Return the caption
        if (jsonResponse.containsKey('caption')) {
          return jsonResponse['caption'];
        } else if (jsonResponse.containsKey('error')) {
          throw Exception('Server error: ${jsonResponse['error']}');
        } else {
          throw Exception('Unexpected response format');
        }
      } else {
        throw Exception(
          'Failed to upload image. Status code: ${response.statusCode}',
        );
      }
    } catch (e) {
      // Rethrow any exceptions for the caller to handle
      rethrow;
    }
  }
}
