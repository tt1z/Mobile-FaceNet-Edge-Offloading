/*
 * Copyright 2023 Shubham Panchal
 * Licensed under the Apache License, Version 2.0 (the "License");
 * You may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */
package com.ml.quaterion.facenetdetection

import android.Manifest
import android.content.Context
import android.content.Intent
import android.content.SharedPreferences
import android.content.pm.PackageManager
import android.graphics.Bitmap
import android.graphics.Color
import android.graphics.PixelFormat
import android.media.ExifInterface
import android.net.Uri
import android.os.Build
import android.os.Bundle
import android.os.Environment
import android.provider.DocumentsContract
import android.text.InputType
import android.text.method.ScrollingMovementMethod
import android.util.Size
import android.view.View
import android.view.WindowInsets
import android.widget.AdapterView
import android.widget.ArrayAdapter
import android.widget.Button
import android.widget.EditText
import android.widget.LinearLayout
import android.widget.RadioButton
import android.widget.RadioGroup
import android.widget.Spinner
import android.widget.TextView
import android.widget.Toast
import androidx.activity.result.contract.ActivityResultContracts
import androidx.appcompat.app.AlertDialog
import androidx.appcompat.app.AppCompatActivity
import androidx.camera.core.CameraSelector
import androidx.camera.core.ImageAnalysis
import androidx.camera.core.Preview
import androidx.camera.lifecycle.ProcessCameraProvider
import androidx.camera.view.PreviewView
import androidx.core.app.ActivityCompat
import androidx.core.content.ContextCompat
import androidx.documentfile.provider.DocumentFile
import androidx.lifecycle.LifecycleOwner
import com.google.common.util.concurrent.ListenableFuture
import com.ml.quaterion.facenetdetection.databinding.ActivityMainBinding
import com.ml.quaterion.facenetdetection.model.FaceNetModel
import com.ml.quaterion.facenetdetection.model.Models
import kotlinx.coroutines.CoroutineScope
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.launch
import kotlinx.coroutines.withContext
import java.io.*
import java.text.SimpleDateFormat
import java.util.Date
import java.util.Locale
import java.util.concurrent.Executors


class MainActivity : AppCompatActivity() {

    private var isSerializedDataStored = false

    // Serialized data will be stored ( in app's private storage ) with this filename.
    private val SERIALIZED_DATA_FILENAME = "image_data"

    // Shared Pref keys.
    private val SHARED_PREF_IS_DATA_STORED_KEY = "is_data_stored"
    private val PREF_SERVER_URL = "offload_server_url"
    private val PREF_JPEG_QUALITY = "offload_jpeg_quality"
    private val PREF_NET_PROFILE = "offload_net_profile"

    private lateinit var activityMainBinding : ActivityMainBinding
    private lateinit var previewView : PreviewView
    private lateinit var frameAnalyser  : FrameAnalyser
    private lateinit var faceNetModel : FaceNetModel
    private lateinit var fileReader : FileReader
    private lateinit var offloadClient : OffloadClient
    private lateinit var decisionEngine : PartitionDecisionEngine
    private lateinit var cameraProviderFuture : ListenableFuture<ProcessCameraProvider>
    private lateinit var sharedPreferences: SharedPreferences

    // <----------------------- User controls --------------------------->

    // Use the device's GPU to perform faster computations (can be toggled at runtime in HUD).
    private var useGpu = true

    // Use XNNPack to accelerate inference.
    private val useXNNPack = true

    // Default is Models.FACENET ; Quantized models are faster
    private val modelInfo = Models.FACENET

    // Camera Facing
    private val cameraFacing = CameraSelector.LENS_FACING_FRONT

    // CSV recording state (writes to app-private external Documents dir, no runtime permission needed).
    private var isRecordingCsv = false
    private var csvWriter: BufferedWriter? = null
    private var csvOutputFile: File? = null
    private var csvRecordedFrames = 0

    // <---------------------------------------------------------------->


    companion object {

        lateinit var logTextView : TextView

        fun setMessage( message : String ) {
            logTextView.text = message
        }

    }

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)

        // Remove the status bar to have a full screen experience
        if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.R) {
            window.decorView.windowInsetsController!!
                .hide( WindowInsets.Type.statusBars() or WindowInsets.Type.navigationBars())
        }
        else {
            window.decorView.systemUiVisibility = View.SYSTEM_UI_FLAG_FULLSCREEN
        }
        activityMainBinding = ActivityMainBinding.inflate( layoutInflater )
        setContentView( activityMainBinding.root )

        previewView = activityMainBinding.previewView
        logTextView = activityMainBinding.logTextview
        logTextView.movementMethod = ScrollingMovementMethod()

        // Keep BoundingBoxOverlay above PreviewView's SurfaceView, while staying BELOW normal HUD Views.
        val boundingBoxOverlay = activityMainBinding.bboxOverlay
        boundingBoxOverlay.cameraFacing = cameraFacing
        boundingBoxOverlay.setWillNotDraw( false )
        boundingBoxOverlay.setZOrderMediaOverlay( true )
        boundingBoxOverlay.holder.setFormat( PixelFormat.TRANSLUCENT )

        sharedPreferences = getSharedPreferences( getString( R.string.app_name ) , Context.MODE_PRIVATE )

        // Restore persisted offloading settings (Server URL, JPEG quality, controllable network profile).
        val savedUrl = sharedPreferences.getString( PREF_SERVER_URL , OffloadClient.DEFAULT_SERVER_URL )
            ?: OffloadClient.DEFAULT_SERVER_URL
        val savedQuality = sharedPreferences.getInt( PREF_JPEG_QUALITY , 75 )
        val savedProfile = NetProfile.fromHeader( sharedPreferences.getString( PREF_NET_PROFILE , "REAL" ) )

        offloadClient = OffloadClient(
            baseUrl = savedUrl,
            jpegQuality = savedQuality,
            netProfile = savedProfile
        )
        decisionEngine = PartitionDecisionEngine( offloadClient )
        faceNetModel = FaceNetModel( this , modelInfo , useGpu , useXNNPack )
        decisionEngine.onLocalBackendChanged( faceNetModel.isGpuEnabled )
        frameAnalyser = FrameAnalyser( this , boundingBoxOverlay , faceNetModel , offloadClient , decisionEngine )
        fileReader = FileReader( faceNetModel )

        decisionEngine.startNetworkProbe()
        setupHudAndControls()


        if ( ActivityCompat.checkSelfPermission( this , Manifest.permission.CAMERA ) != PackageManager.PERMISSION_GRANTED ) {
            requestCameraPermission()
        }
        else {
            startCameraPreview()
        }

        isSerializedDataStored = sharedPreferences.getBoolean( SHARED_PREF_IS_DATA_STORED_KEY , false )
        if ( !isSerializedDataStored ) {
            Logger.log( "No serialized data was found. Select the images directory.")
            showSelectDirectoryDialog()
        }
        else {
            val alertDialog = AlertDialog.Builder( this ).apply {
                setTitle( "Serialized Data")
                setMessage( "Existing image data was found on this device. Would you like to load it?" )
                setCancelable( false )
                setNegativeButton( "LOAD") { dialog, _ ->
                    dialog.dismiss()
                    frameAnalyser.faceList = loadSerializedImageData()
                    Logger.log( "Serialized data loaded. Edge server: ${offloadClient.baseUrl}")
                }
                setPositiveButton( "RESCAN") { dialog, _ ->
                    dialog.dismiss()
                    launchChooseDirectoryIntent()
                }
                create()
            }
            alertDialog.show()
        }

    }

    private fun setupHudAndControls() {
        val hudTextView = activityMainBinding.hudTextview
        val modeSpinner = activityMainBinding.modeSpinner
        val gpuSwitch = activityMainBinding.gpuSwitch
        val csvButton = activityMainBinding.csvRecordButton
        val settingsButton = activityMainBinding.settingsButton

        // 1. Bind per-frame metrics callback to live HUD + optional CSV writer.
        frameAnalyser.onMetricsUpdated = { metrics ->
            hudTextView.text = metrics.toHudText()
            if (isRecordingCsv) {
                try {
                    csvWriter?.apply {
                        write(metrics.toCsvRow())
                        newLine()
                        csvRecordedFrames += 1
                        if (csvRecordedFrames % 10 == 0) {
                            flush()
                        }
                    }
                    csvButton.text = "STOP($csvRecordedFrames)"
                } catch (e: Exception) {
                    Logger.log("CSV write error: ${e.message}")
                    stopCsvRecording()
                }
            }
        }

        // 2. Partitioning mode selector Spinner.
        val modes = PartitionMode.values()
        val adapter = ArrayAdapter(this, R.layout.spinner_item_hud, modes)
        adapter.setDropDownViewResource(R.layout.spinner_dropdown_hud)
        modeSpinner.adapter = adapter
        modeSpinner.setSelection(0, false)
        modeSpinner.onItemSelectedListener = object : AdapterView.OnItemSelectedListener {
            override fun onItemSelected(
                parent: AdapterView<*>?,
                view: View?,
                position: Int,
                id: Long
            ) {
                val selectedMode = modes[position]
                frameAnalyser.currentMode = selectedMode
                frameAnalyser.resetFpsWindow()
                Logger.log("Partition mode -> ${selectedMode.displayName}")
            }

            override fun onNothingSelected(parent: AdapterView<*>?) {}
        }

        // 3. Hardware acceleration toggle (GPU Delegate vs pure 4-thread CPU).
        gpuSwitch.isChecked = faceNetModel.isGpuEnabled
        gpuSwitch.text = faceNetModel.activeBackend
        gpuSwitch.setOnCheckedChangeListener { _, isChecked ->
            useGpu = isChecked
            gpuSwitch.isEnabled = false
            gpuSwitch.text = "..."
            CoroutineScope(Dispatchers.Default).launch {
                try {
                    val newModel = FaceNetModel(this@MainActivity, modelInfo, isChecked, useXNNPack)
                    frameAnalyser.updateFaceNetModel(newModel)
                    fileReader.updateModel(newModel)
                    decisionEngine.onLocalBackendChanged(newModel.isGpuEnabled)
                    withContext(Dispatchers.Main) {
                        faceNetModel = newModel
                        gpuSwitch.text = newModel.activeBackend
                        gpuSwitch.isEnabled = true
                        Toast.makeText(
                            this@MainActivity,
                            "Switched FaceNet backend -> ${newModel.activeBackend}",
                            Toast.LENGTH_SHORT
                        ).show()
                    }
                } catch (e: Exception) {
                    withContext(Dispatchers.Main) {
                        Logger.log("Failed to switch backend: ${e.message}")
                        gpuSwitch.text = faceNetModel.activeBackend
                        gpuSwitch.isEnabled = true
                    }
                }
            }
        }

        // 4. Local CSV profiling recorder toggle.
        csvButton.setOnClickListener {
            if (!isRecordingCsv) {
                startCsvRecording()
            } else {
                stopCsvRecording()
            }
        }

        // 5. Edge offloading settings dialog (Server URL, Net Profile, JPEG Quality, Test Ping).
        settingsButton.setOnClickListener {
            showOffloadSettingsDialog()
        }
    }

    private fun showOffloadSettingsDialog() {
        val pad = (16 * resources.displayMetrics.density).toInt()
        val container = LinearLayout(this).apply {
            orientation = LinearLayout.VERTICAL
            setPadding(pad, pad, pad, pad / 2)
        }

        // Server URL input
        val urlLabel = TextView(this).apply {
            text = "Edge Server URL (USB reverse: http://127.0.0.1:8000 | WiFi: http://<PC_IP>:8000)"
            textSize = 12f
        }
        val urlInput = EditText(this).apply {
            setText(offloadClient.baseUrl)
            inputType = InputType.TYPE_CLASS_TEXT or InputType.TYPE_TEXT_VARIATION_URI
            isSingleLine = true
        }
        container.addView(urlLabel)
        container.addView(urlInput)

        // Quick Ping row
        val pingRow = LinearLayout(this).apply {
            orientation = LinearLayout.HORIZONTAL
            setPadding(0, 8, 0, 12)
        }
        val pingBtn = Button(this).apply { text = "Test Ping" }
        val usbBtn = Button(this).apply { text = "Use USB 127.0.0.1" }
        val pingStatus = TextView(this).apply {
            text = "  Not tested"
            textSize = 12f
        }
        pingRow.addView(pingBtn)
        pingRow.addView(usbBtn)
        container.addView(pingRow)
        container.addView(pingStatus)

        // Network Emulation Profile selector
        val profileLabel = TextView(this).apply {
            text = "\nNetwork Profile (Real WiFi/5G or Controllable Emulation):"
            textSize = 12f
        }
        val profiles = NetProfile.values()
        val profileSpinner = Spinner(this).apply {
            adapter = ArrayAdapter(
                this@MainActivity,
                android.R.layout.simple_spinner_dropdown_item,
                profiles
            )
            setSelection(profiles.indexOf(offloadClient.netProfile).coerceAtLeast(0))
        }
        container.addView(profileLabel)
        container.addView(profileSpinner)

        // JPEG Quality RadioGroup
        val qualityLabel = TextView(this).apply {
            text = "\nJPEG Compression Quality (for C1/C2/C3 payload trade-off):"
            textSize = 12f
        }
        val qualityGroup = RadioGroup(this).apply {
            orientation = RadioGroup.HORIZONTAL
        }
        val qualities = listOf(50, 75, 95)
        qualities.forEach { q ->
            val rb = RadioButton(this).apply {
                text = "Q$q"
                tag = q
                id = View.generateViewId()
            }
            qualityGroup.addView(rb)
            if (q == offloadClient.jpegQuality) {
                qualityGroup.check(rb.id)
            }
        }
        container.addView(qualityLabel)
        container.addView(qualityGroup)

        usbBtn.setOnClickListener {
            urlInput.setText(OffloadClient.DEFAULT_SERVER_URL)
        }

        pingBtn.setOnClickListener {
            val testUrl = urlInput.text.toString().trim()
            val selectedProfile = profiles[profileSpinner.selectedItemPosition]
            pingStatus.text = "  Pinging $testUrl (${selectedProfile.headerValue})..."
            pingBtn.isEnabled = false
            CoroutineScope(Dispatchers.IO).launch {
                val tempClient = OffloadClient(
                    baseUrl = testUrl,
                    jpegQuality = offloadClient.jpegQuality,
                    netProfile = selectedProfile
                )
                val res = tempClient.ping()
                withContext(Dispatchers.Main) {
                    pingBtn.isEnabled = true
                    if (res.ok) {
                        pingStatus.text = String.format(
                            Locale.US,
                            "  OK! RTT = %.1f ms",
                            res.rttMs
                        )
                        pingStatus.setTextColor(Color.parseColor("#2E7D32"))
                    } else {
                        pingStatus.text = "  FAIL: ${res.detail}"
                        pingStatus.setTextColor(Color.parseColor("#C62828"))
                    }
                }
            }
        }

        AlertDialog.Builder(this)
            .setTitle("Edge Offloading Settings (Part B)")
            .setView(container)
            .setPositiveButton("SAVE") { dialog, _ ->
                val newUrl = urlInput.text.toString().trim().ifEmpty { OffloadClient.DEFAULT_SERVER_URL }
                val newProfile = profiles[profileSpinner.selectedItemPosition]
                val checkedRadio = qualityGroup.findViewById<RadioButton>(qualityGroup.checkedRadioButtonId)
                val newQuality = (checkedRadio?.tag as? Int) ?: 75

                offloadClient.baseUrl = newUrl
                offloadClient.netProfile = newProfile
                offloadClient.jpegQuality = newQuality
                frameAnalyser.resetFpsWindow()
                decisionEngine.triggerImmediateProbe()

                sharedPreferences.edit()
                    .putString(PREF_SERVER_URL, newUrl)
                    .putString(PREF_NET_PROFILE, newProfile.headerValue)
                    .putInt(PREF_JPEG_QUALITY, newQuality)
                    .apply()

                Logger.log("Edge config -> $newUrl | ${newProfile.headerValue} | Q$newQuality")
                dialog.dismiss()
            }
            .setNegativeButton("CANCEL") { dialog, _ -> dialog.dismiss() }
            .show()
    }

    private fun startCsvRecording() {
        val dir = getExternalFilesDir(Environment.DIRECTORY_DOCUMENTS) ?: filesDir
        if (!dir.exists()) {
            dir.mkdirs()
        }
        val ts = SimpleDateFormat("yyyyMMdd_HHmmss", Locale.US).format(Date())
        val file = File(dir, "sa_perf_$ts.csv")
        try {
            val writer = BufferedWriter(FileWriter(file, false))
            writer.write(FrameMetrics.CSV_HEADER)
            writer.newLine()
            writer.flush()

            csvWriter = writer
            csvOutputFile = file
            csvRecordedFrames = 0
            isRecordingCsv = true

            activityMainBinding.csvRecordButton.apply {
                text = "STOP(0)"
                setTextColor(Color.parseColor("#FF6B6B"))
            }
            Logger.log("CSV recording started: ${file.name}")
        } catch (e: Exception) {
            Logger.log("Could not start CSV recording: ${e.message}")
            Toast.makeText(this, "CSV start failed: ${e.message}", Toast.LENGTH_SHORT).show()
        }
    }

    private fun stopCsvRecording() {
        if (!isRecordingCsv) return
        isRecordingCsv = false
        val savedFrames = csvRecordedFrames
        val savedPath = csvOutputFile?.absolutePath ?: ""
        try {
            csvWriter?.flush()
            csvWriter?.close()
        } catch (_: Exception) {
        } finally {
            csvWriter = null
            csvOutputFile = null
        }

        activityMainBinding.csvRecordButton.apply {
            text = "REC"
            setTextColor(Color.WHITE)
        }
        Logger.log("CSV saved ($savedFrames frames): $savedPath")
        Toast.makeText(this, "Saved $savedFrames frames to CSV", Toast.LENGTH_SHORT).show()
    }

    override fun onDestroy() {
        stopCsvRecording()
        if (::decisionEngine.isInitialized) {
            decisionEngine.stopNetworkProbe()
        }
        if (::faceNetModel.isInitialized) {
            faceNetModel.close()
        }
        super.onDestroy()
    }

    // ---------------------------------------------- //

    private fun startCameraPreview() {
        cameraProviderFuture = ProcessCameraProvider.getInstance( this )
        cameraProviderFuture.addListener({
            val cameraProvider = cameraProviderFuture.get()
            bindPreview(cameraProvider) },
            ContextCompat.getMainExecutor(this) )
    }

    private fun bindPreview(cameraProvider : ProcessCameraProvider) {
        val preview : Preview = Preview.Builder().build()
        val cameraSelector : CameraSelector = CameraSelector.Builder()
            .requireLensFacing( cameraFacing )
            .build()
        preview.setSurfaceProvider( previewView.surfaceProvider )
        val imageFrameAnalysis = ImageAnalysis.Builder()
            .setTargetResolution(Size( 480, 640 ) )
            .setBackpressureStrategy(ImageAnalysis.STRATEGY_KEEP_ONLY_LATEST)
            .setOutputImageFormat(ImageAnalysis.OUTPUT_IMAGE_FORMAT_RGBA_8888)
            .build()
        imageFrameAnalysis.setAnalyzer(Executors.newSingleThreadExecutor(), frameAnalyser )
        cameraProvider.bindToLifecycle(this as LifecycleOwner, cameraSelector, preview , imageFrameAnalysis  )
    }

    private fun requestCameraPermission() {
        cameraPermissionLauncher.launch( Manifest.permission.CAMERA )
    }

    private val cameraPermissionLauncher = registerForActivityResult( ActivityResultContracts.RequestPermission() ) {
        isGranted ->
        if ( isGranted ) {
            startCameraPreview()
        }
        else {
            val alertDialog = AlertDialog.Builder( this ).apply {
                setTitle( "Camera Permission")
                setMessage( "The app couldn't function without the camera permission." )
                setCancelable( false )
                setPositiveButton( "ALLOW" ) { dialog, _ ->
                    dialog.dismiss()
                    requestCameraPermission()
                }
                setNegativeButton( "CLOSE" ) { dialog, _ ->
                    dialog.dismiss()
                    finish()
                }
                create()
            }
            alertDialog.show()
        }

    }

    // ---------------------------------------------- //

    private fun showSelectDirectoryDialog() {
        val alertDialog = AlertDialog.Builder( this ).apply {
            setTitle( "Select Images Directory")
            setMessage( "As mentioned in the project\'s README file, please select a directory which contains the images." )
            setCancelable( false )
            setPositiveButton( "SELECT") { dialog, _ ->
                dialog.dismiss()
                launchChooseDirectoryIntent()
            }
            create()
        }
        alertDialog.show()
    }


    private fun launchChooseDirectoryIntent() {
        val intent = Intent( Intent.ACTION_OPEN_DOCUMENT_TREE )
        directoryAccessLauncher.launch( intent )
    }


    private val directoryAccessLauncher = registerForActivityResult( ActivityResultContracts.StartActivityForResult() ) {
        val dirUri = it.data?.data ?: return@registerForActivityResult
        val childrenUri =
            DocumentsContract.buildChildDocumentsUriUsingTree(
                dirUri,
                DocumentsContract.getTreeDocumentId( dirUri )
            )
        val tree = DocumentFile.fromTreeUri(this, childrenUri)
        val images = ArrayList<Pair<String,Bitmap>>()
        var errorFound = false
        if ( tree!!.listFiles().isNotEmpty()) {
            for ( doc in tree.listFiles() ) {
                if ( doc.isDirectory && !errorFound ) {
                    val name = doc.name!!
                    for ( imageDocFile in doc.listFiles() ) {
                        try {
                            images.add( Pair( name , getFixedBitmap( imageDocFile.uri ) ) )
                        }
                        catch ( e : Exception ) {
                            errorFound = true
                            Logger.log( "Could not parse an image in $name directory. Make sure that the file structure is " +
                                    "as described in the README of the project and then restart the app." )
                            break
                        }
                    }
                    Logger.log( "Found ${doc.listFiles().size} images in $name directory" )
                }
                else {
                    errorFound = true
                    Logger.log( "The selected folder should contain only directories. Make sure that the file structure is " +
                            "as described in the README of the project and then restart the app." )
                }
            }
        }
        else {
            errorFound = true
            Logger.log( "The selected folder doesn't contain any directories. Make sure that the file structure is " +
                    "as described in the README of the project and then restart the app." )
        }
        if ( !errorFound ) {
            fileReader.run( images , fileReaderCallback )
            Logger.log( "Detecting faces in ${images.size} images ..." )
        }
        else {
            val alertDialog = AlertDialog.Builder( this ).apply {
                setTitle( "Error while parsing directory")
                setMessage( "There were some errors while parsing the directory. Please see the log below. Make sure that the file structure is " +
                        "as described in the README of the project and then tap RESELECT" )
                setCancelable( false )
                setPositiveButton( "RESELECT") { dialog, _ ->
                    dialog.dismiss()
                    launchChooseDirectoryIntent()
                }
                setNegativeButton( "CANCEL" ){ dialog , _ ->
                    dialog.dismiss()
                    finish()
                }
                create()
            }
            alertDialog.show()
        }
    }


    private fun getFixedBitmap( imageFileUri : Uri ) : Bitmap {
        var imageBitmap = BitmapUtils.getBitmapFromUri( contentResolver , imageFileUri )
        val exifInterface = ExifInterface( contentResolver.openInputStream( imageFileUri )!! )
        imageBitmap =
            when (exifInterface.getAttributeInt( ExifInterface.TAG_ORIENTATION ,
                ExifInterface.ORIENTATION_UNDEFINED )) {
                ExifInterface.ORIENTATION_ROTATE_90 -> BitmapUtils.rotateBitmap( imageBitmap , 90f )
                ExifInterface.ORIENTATION_ROTATE_180 -> BitmapUtils.rotateBitmap( imageBitmap , 180f )
                ExifInterface.ORIENTATION_ROTATE_270 -> BitmapUtils.rotateBitmap( imageBitmap , 270f )
                else -> imageBitmap
            }
        return imageBitmap
    }

    // ---------------------------------------------- //

    private val fileReaderCallback = object : FileReader.ProcessCallback {
        override fun onProcessCompleted(data: ArrayList<Pair<String, FloatArray>>, numImagesWithNoFaces: Int) {
            frameAnalyser.faceList = data
            saveSerializedImageData( data )
            Logger.log( "Images parsed. Found $numImagesWithNoFaces images with no faces." )
        }
    }


    private fun saveSerializedImageData(data : ArrayList<Pair<String,FloatArray>> ) {
        val serializedDataFile = File( filesDir , SERIALIZED_DATA_FILENAME )
        ObjectOutputStream( FileOutputStream( serializedDataFile )  ).apply {
            writeObject( data )
            flush()
            close()
        }
        sharedPreferences.edit().putBoolean( SHARED_PREF_IS_DATA_STORED_KEY , true ).apply()
    }


    private fun loadSerializedImageData() : ArrayList<Pair<String,FloatArray>> {
        val serializedDataFile = File( filesDir , SERIALIZED_DATA_FILENAME )
        val objectInputStream = ObjectInputStream( FileInputStream( serializedDataFile ) )
        val data = objectInputStream.readObject() as ArrayList<Pair<String,FloatArray>>
        objectInputStream.close()
        return data
    }


}
