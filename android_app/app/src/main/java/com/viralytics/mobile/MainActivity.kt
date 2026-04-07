package com.viralytics.mobile

import android.Manifest
import android.content.Context
import android.content.pm.PackageManager
import android.graphics.Bitmap
import android.graphics.BitmapFactory
import android.os.Bundle
import android.util.Base64
import android.widget.Toast
import androidx.activity.result.contract.ActivityResultContracts
import androidx.appcompat.app.AppCompatActivity
import androidx.core.content.ContextCompat
import com.viralytics.mobile.databinding.ActivityMainBinding
import okhttp3.MediaType.Companion.toMediaType
import okhttp3.MultipartBody
import okhttp3.OkHttpClient
import okhttp3.Request
import okhttp3.RequestBody.Companion.toRequestBody
import org.json.JSONArray
import org.json.JSONObject
import java.io.ByteArrayOutputStream

class MainActivity : AppCompatActivity() {

    private lateinit var binding: ActivityMainBinding
    private val httpClient = OkHttpClient()
    private var currentSessionId: String? = null
    private val detectedCategories = mutableListOf<String>()

    private val cameraPermissionLauncher =
        registerForActivityResult(ActivityResultContracts.RequestPermission()) { granted ->
            if (granted) {
                cameraLauncher.launch(null)
            } else {
                setStatus("Camera permission denied.")
            }
        }

    private val cameraLauncher =
        registerForActivityResult(ActivityResultContracts.TakePicturePreview()) { bitmap ->
            if (bitmap == null) {
                setStatus("Capture cancelled.")
                return@registerForActivityResult
            }
            uploadScan(bitmap)
        }

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        binding = ActivityMainBinding.inflate(layoutInflater)
        setContentView(binding.root)

        binding.serverUrlInput.setText(loadServerUrl())
        setStatus("Ready.")
        updateSessionLabel()

        binding.captureButton.setOnClickListener {
            saveServerUrl(binding.serverUrlInput.text.toString())
            launchCamera()
        }

        binding.sendChatButton.setOnClickListener {
            saveServerUrl(binding.serverUrlInput.text.toString())
            sendChat()
        }
    }

    private fun launchCamera() {
        if (ContextCompat.checkSelfPermission(this, Manifest.permission.CAMERA) == PackageManager.PERMISSION_GRANTED) {
            cameraLauncher.launch(null)
        } else {
            cameraPermissionLauncher.launch(Manifest.permission.CAMERA)
        }
    }

    private fun uploadScan(bitmap: Bitmap) {
        setStatus("Uploading scan...")
        val baseUrl = normalizedBaseUrl() ?: return
        val jpegBytes = bitmap.toJpegBytes()

        val imageBody = jpegBytes.toRequestBody("image/jpeg".toMediaType())
        val multipartBody = MultipartBody.Builder()
            .setType(MultipartBody.FORM)
            .addFormDataPart("file", "scan.jpg", imageBody)
            .build()

        val request = Request.Builder()
            .url("$baseUrl/api/mobile/scan")
            .post(multipartBody)
            .build()

        Thread {
            try {
                httpClient.newCall(request).execute().use { response ->
                    val bodyText = response.body?.string().orEmpty()
                    if (!response.isSuccessful) {
                        runOnUiThread {
                            setStatus("Scan failed: HTTP ${response.code}")
                            binding.chatReplyText.text = bodyText.ifBlank { "No error body returned." }
                        }
                        return@use
                    }

                    val json = JSONObject(bodyText)
                    currentSessionId = json.optString("session_id").ifBlank { null }
                    updateDetections(json.optJSONArray("detections"))
                    updateRecommendations(json.optJSONArray("recommendations"))
                    updateAnnotatedImage(json.optString("annotated_frame"))

                    runOnUiThread {
                        updateSessionLabel("Vision-led")
                        binding.chatReplyText.text = "Scan complete. You can now refine the search in chat."
                        setStatus("Scan complete.")
                    }
                }
            } catch (exc: Exception) {
                runOnUiThread {
                    setStatus("Scan request failed.")
                    binding.chatReplyText.text = exc.message ?: "Unknown error"
                }
            }
        }.start()
    }

    private fun sendChat() {
        val message = binding.chatInput.text.toString().trim()
        if (message.isBlank()) {
            toast("Enter a refinement message first.")
            return
        }

        val baseUrl = normalizedBaseUrl() ?: return
        setStatus("Sending refinement...")

        val payload = JSONObject().apply {
            put("message", message)
            put("session_id", currentSessionId)
            put("replace_vision", binding.replaceVisionSwitch.isChecked)
            put("detected_categories", JSONArray(detectedCategories))
            put("history", JSONArray())
            put("recommendations", JSONArray())
        }

        val request = Request.Builder()
            .url("$baseUrl/api/chat")
            .post(payload.toString().toRequestBody("application/json".toMediaType()))
            .build()

        Thread {
            try {
                httpClient.newCall(request).execute().use { response ->
                    val bodyText = response.body?.string().orEmpty()
                    if (!response.isSuccessful) {
                        runOnUiThread {
                            setStatus("Chat failed: HTTP ${response.code}")
                            binding.chatReplyText.text = bodyText.ifBlank { "No error body returned." }
                        }
                        return@use
                    }

                    val json = JSONObject(bodyText)
                    currentSessionId = json.optString("session_id").ifBlank { currentSessionId }
                    updateRecommendations(json.optJSONArray("results"))

                    runOnUiThread {
                        val mode = if (binding.replaceVisionSwitch.isChecked) "Search-led override" else "Vision + search"
                        updateSessionLabel(mode)
                        binding.chatReplyText.text = json.optString("reply", "No reply returned.")
                        setStatus("Refinement complete.")
                        binding.chatInput.text?.clear()
                    }
                }
            } catch (exc: Exception) {
                runOnUiThread {
                    setStatus("Chat request failed.")
                    binding.chatReplyText.text = exc.message ?: "Unknown error"
                }
            }
        }.start()
    }

    private fun updateDetections(detections: JSONArray?) {
        detectedCategories.clear()
        if (detections == null || detections.length() == 0) {
            runOnUiThread {
                binding.detectionsText.text = "No clothing detections came back from the server yet."
            }
            return
        }

        val lines = mutableListOf<String>()
        for (i in 0 until detections.length()) {
            val det = detections.optJSONObject(i) ?: continue
            val name = det.optString("class_name", "unknown")
            val confidence = det.optDouble("confidence", 0.0)
            detectedCategories.add(name)
            lines += "- ${name.replace("_", " ")} (${String.format("%.0f%%", confidence * 100)})"
        }

        runOnUiThread {
            binding.detectionsText.text = "Detected pieces\n" + lines.joinToString("\n")
        }
    }

    private fun updateRecommendations(items: JSONArray?) {
        if (items == null || items.length() == 0) {
            runOnUiThread {
                binding.recommendationsText.text = "No recommendations yet. Run a scan or refine with chat."
            }
            return
        }

        val lines = mutableListOf<String>()
        for (i in 0 until items.length()) {
            val item = items.optJSONObject(i) ?: continue
            val name = item.optString("name", "Unnamed item")
            val category = item.optString("category", "item")
            val price = item.optString("price", "N/A")
            val reason = item.optString("reason", "")
            val detailLine = buildString {
                append(name)
                append("\n")
                append(category.replace("_", " ").replaceFirstChar { it.uppercase() })
                append(" | ")
                append(price)
                if (reason.isNotBlank()) {
                    append("\n")
                    append(reason)
                }
            }
            lines += detailLine
        }

        runOnUiThread {
            binding.recommendationsText.text = "Recommended next steps\n" + lines.joinToString("\n\n")
        }
    }

    private fun updateAnnotatedImage(base64Image: String?) {
        if (base64Image.isNullOrBlank()) {
            return
        }
        try {
            val bytes = Base64.decode(base64Image, Base64.DEFAULT)
            val bitmap = BitmapFactory.decodeByteArray(bytes, 0, bytes.size)
            runOnUiThread {
                binding.resultImage.setImageBitmap(bitmap)
            }
        } catch (_: IllegalArgumentException) {
            runOnUiThread {
                binding.resultImage.setImageBitmap(null)
            }
        }
    }

    private fun normalizedBaseUrl(): String? {
        val raw = binding.serverUrlInput.text.toString().trim().removeSuffix("/")
        if (raw.isBlank()) {
            toast("Enter the PC server URL first.")
            return null
        }
        return raw
    }

    private fun setStatus(message: String) {
        binding.statusText.text = "Status: $message"
    }

    private fun updateSessionLabel(mode: String? = null) {
        val sessionId = currentSessionId
        binding.sessionText.text = if (sessionId.isNullOrBlank()) {
            "Session: waiting for scan"
        } else {
            val modeSuffix = if (mode.isNullOrBlank()) "" else " | $mode"
            "Session: ${sessionId.take(8)}$modeSuffix"
        }
    }

    private fun toast(message: String) {
        Toast.makeText(this, message, Toast.LENGTH_SHORT).show()
    }

    private fun loadServerUrl(): String {
        return getSharedPreferences("viralytics_mobile", Context.MODE_PRIVATE)
            .getString("server_url", "http://192.168.1.100:8000")
            .orEmpty()
    }

    private fun saveServerUrl(url: String) {
        getSharedPreferences("viralytics_mobile", Context.MODE_PRIVATE)
            .edit()
            .putString("server_url", url.trim())
            .apply()
    }

    private fun Bitmap.toJpegBytes(): ByteArray {
        val stream = ByteArrayOutputStream()
        compress(Bitmap.CompressFormat.JPEG, 90, stream)
        return stream.toByteArray()
    }
}
