package com.example.attivit_progettuale

import android.Manifest
import android.content.Context
import android.content.Intent
import android.content.pm.PackageManager
import android.graphics.*
import android.os.Build
import android.os.Bundle
import android.util.Log
import android.util.Size
import android.widget.Toast
import androidx.activity.result.contract.ActivityResultContracts
import androidx.appcompat.app.AppCompatActivity
import androidx.camera.core.*
import androidx.camera.core.ImageCapture.CAPTURE_MODE_MAXIMIZE_QUALITY
import androidx.camera.lifecycle.ProcessCameraProvider
import androidx.core.content.ContextCompat
import com.example.attivit_progettuale.databinding.ActivityMainBinding
import kotlinx.serialization.decodeFromString
import kotlinx.serialization.json.Json
import org.tensorflow.lite.support.image.TensorImage
import org.tensorflow.lite.task.core.BaseOptions
import org.tensorflow.lite.task.processor.NearestNeighbor
import org.tensorflow.lite.task.processor.SearcherOptions
import org.tensorflow.lite.task.vision.searcher.ImageSearcher
import java.io.*
import java.util.*
import java.util.concurrent.ExecutorService
import java.util.concurrent.Executors
import kotlin.collections.HashMap


lateinit var imageSearcher: ImageSearcher

class MainActivity : AppCompatActivity() {
    private lateinit var viewBinding: ActivityMainBinding

    private var imageCapture: ImageCapture? = null

    private lateinit var cameraExecutor: ExecutorService

    private val activityResultLauncher =
        registerForActivityResult(
            ActivityResultContracts.RequestMultiplePermissions())
        { permissions ->
            // Handle Permission granted/rejected
            var permissionGranted = true
            permissions.entries.forEach {
                if (it.key in REQUIRED_PERMISSIONS && !it.value)
                    permissionGranted = false
            }
            if (!permissionGranted) {
                Toast.makeText(baseContext,
                    "Permission request denied",
                    Toast.LENGTH_SHORT).show()
            } else {
                startCamera()
            }
        }

    override fun onResume(){
        super.onResume()
        viewBinding.imageCaptureButton.isClickable = true

    }

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        viewBinding = ActivityMainBinding.inflate(layoutInflater)
        setContentView(viewBinding.root)

        // Request camera permissions
        if (allPermissionsGranted()) {
            startCamera()
        } else {
            requestPermissions()
        }

        val options: ImageSearcher.ImageSearcherOptions = ImageSearcher.ImageSearcherOptions.builder()
            .setBaseOptions(BaseOptions.builder().build())
            .setSearcherOptions(
                SearcherOptions.builder().setMaxResults(10).setL2Normalize(true).build()
            )
            .build()
        imageSearcher =
            ImageSearcher.createFromFileAndOptions(this, "searcher.tflite", options)


        // Set up the listeners for take photo and video capture buttons
        viewBinding.imageCaptureButton.setOnClickListener { takePhoto() }

        cameraExecutor = Executors.newSingleThreadExecutor()

        var jsonString = getJsonDataFromAsset(this, "monete_commemotative.json")
        val listMonetaComm = jsonString?.let { Json.decodeFromString<List<MonetaCommemorativa>>(it) }
        jsonString = getJsonDataFromAsset(this, "monete_nazionali.json")
        val listMonetaNaz= jsonString?.let { Json.decodeFromString<List<MonetaNazionale>>(it) }
        jsonString = getJsonDataFromAsset(this, "monete_fronte.json")
        val listMonetaFronte= jsonString?.let { Json.decodeFromString<List<MonetaFronte>>(it) }
        CoinListHolder.addItemsComm(listMonetaComm!!)
        CoinListHolder.addItemsNaz(listMonetaNaz!!)
        CoinListHolder.addItemsFront(listMonetaFronte!!)
    }

    private fun takePhoto() {
        // Get a stable reference of the modifiable image capture use case
        val imageCapture = imageCapture ?: return
        viewBinding.imageCaptureButton.isClickable = false

        // Set up image capture listener, which is triggered after photo has
        // been taken
        imageCapture.takePicture(
            ContextCompat.getMainExecutor(this),
            object : ImageCapture.OnImageCapturedCallback() {
                override fun onError(exc: ImageCaptureException) {
                    Log.e(TAG, "Photo capture failed: ${exc.message}", exc)
                }


                override fun onCaptureSuccess(image: ImageProxy){
                    val data = ByteArray(image.planes[0].buffer.remaining())
                    image.planes[0].buffer.get(data)


                    Log.d(TAG, "Dimensioni foto: ${image.height}x${image.width}")
                    val photo = BitmapFactory.decodeByteArray(
                        data,
                        0,
                        data.size,
                        BitmapFactory.Options().also { it.inScaled = false })

                    val photoSquared = Bitmap.createBitmap(
                        photo,
                        (photo.width - photo.height) / 2,
                        0,
                        photo.height,
                        photo.height
                    )
                    val photoRotated = photoSquared.rotateBitmap(90f)
                    val photoCropped = Bitmap.createBitmap(
                        photoRotated,
                        270,
                        270,
                        540,
                        540
                    )
                    val photoForInference = Bitmap.createScaledBitmap(photoCropped, 300, 300, true)
                    val results : List<NearestNeighbor> = imageSearcher.search(TensorImage.fromBitmap(photoForInference))
                    val resultsParsed = results.associate { String(it.metadata.array()) to it.distance } as HashMap<String, Float>


                    results.forEach { Log.d(TAG, String(it.metadata.array()) +" "+ it.distance.toString())}
                    image.close()
                    try {
                        //Write file
                        val filename = "bitmap.png"
                        val stream: FileOutputStream = openFileOutput(filename, MODE_PRIVATE)
                        photoCropped.compress(Bitmap.CompressFormat.PNG, 90, stream)

                        //Cleanup
                        stream.close()
                        photoCropped.recycle()
                        //Pop intent
                        val myIntent = Intent(applicationContext, CoinResultActivity::class.java)
                        myIntent.putExtra("bitmapName",filename)
                        myIntent.putExtra("results", resultsParsed)
                        this@MainActivity.startActivity(myIntent)
                    } catch (e: java.lang.Exception) {
                        e.printStackTrace()
                    }
                }

            }
        )
    }

    fun Bitmap.rotateBitmap(angle: Float): Bitmap {
        val matrix = Matrix()
        matrix.postRotate(angle)
        return Bitmap.createBitmap(this, 0, 0, this.width, this.height, matrix, true)
    }

    private fun startCamera() {
        val cameraProviderFuture = ProcessCameraProvider.getInstance(this)

        cameraProviderFuture.addListener({
            // Used to bind the lifecycle of cameras to the lifecycle owner
            val cameraProvider: ProcessCameraProvider = cameraProviderFuture.get()

            // Preview
            val preview = Preview.Builder()
                .build()
                .also {
                    it.setSurfaceProvider(viewBinding.viewFinder.surfaceProvider)
                }
            imageCapture = ImageCapture.Builder().setCaptureMode(CAPTURE_MODE_MAXIMIZE_QUALITY).setTargetResolution(
                Size(640, 640)
            ).build()


            // Select back camera as a default
            val cameraSelector = CameraSelector.DEFAULT_BACK_CAMERA

            try {
                // Unbind use cases before rebinding
                cameraProvider.unbindAll()

                // Bind use cases to camera
                cameraProvider.bindToLifecycle(
                    this, cameraSelector, preview, imageCapture)

            } catch(exc: Exception) {
                Log.e(TAG, "Use case binding failed", exc)
            }

        }, ContextCompat.getMainExecutor(this))

    }

    private fun requestPermissions() {
        activityResultLauncher.launch(REQUIRED_PERMISSIONS)
    }

    private fun allPermissionsGranted() = REQUIRED_PERMISSIONS.all {
        ContextCompat.checkSelfPermission(
            baseContext, it) == PackageManager.PERMISSION_GRANTED
    }

    override fun onDestroy() {
        super.onDestroy()
        cameraExecutor.shutdown()
    }

    companion object {
        private const val TAG = "CameraXApp"
        private val REQUIRED_PERMISSIONS =
            mutableListOf (
                Manifest.permission.CAMERA,
                Manifest.permission.RECORD_AUDIO
            ).apply {
                if (Build.VERSION.SDK_INT <= Build.VERSION_CODES.P) {
                    add(Manifest.permission.WRITE_EXTERNAL_STORAGE)
                }
            }.toTypedArray()
    }

    private fun getJsonDataFromAsset(context: Context, fileName: String): String? {
        val jsonString: String
        try {
            jsonString = context.assets.open(fileName).bufferedReader().use { it.readText() }
        } catch (ioException: IOException) {
            ioException.printStackTrace()
            return null
        }
        return jsonString
    }


    object CoinListHolder {
        private val listComm = mutableListOf<MonetaCommemorativa>()
        private val listFronte = mutableListOf<MonetaFronte>()
        private val listNaz = mutableListOf<MonetaNazionale>()

        fun addItemsComm(items: List<MonetaCommemorativa>) {
            listComm.addAll(items)
        }
        fun addItemsFront(items: List<MonetaFronte>) {
            listFronte.addAll(items)
        }
        fun addItemsNaz(items: List<MonetaNazionale>) {
            listNaz.addAll(items)
        }

        fun getCoinFromKey(key : String) : Moneta? {
            for(coin in listComm){
                if(coin.key == key) return coin
            }
            for(coin in listFronte){
                if(coin.key == key) return coin
            }
            for(coin in listNaz){
                if(coin.key == key) return coin
            }
            return null
        }
    }

}

