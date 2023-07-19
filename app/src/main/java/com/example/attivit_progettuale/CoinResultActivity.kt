package com.example.attivit_progettuale

import android.graphics.BitmapFactory
import android.os.Bundle
import android.util.Log
import androidx.appcompat.app.AppCompatActivity
import androidx.recyclerview.widget.LinearLayoutManager
import com.example.attivit_progettuale.databinding.ActivityResultBinding
import java.io.FileInputStream
import java.util.*


class CoinResultActivity : AppCompatActivity() {
    private lateinit var binding: ActivityResultBinding

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        binding = ActivityResultBinding.inflate(layoutInflater)

        val intent = intent
        val filename = getIntent().getStringExtra("bitmapName")
        try {
            val inputStream : FileInputStream = openFileInput(filename)
            val bmp = BitmapFactory.decodeStream(inputStream)
            inputStream.close()
            binding.imageView.setImageBitmap(bmp)

        } catch (e: Exception) {
            e.printStackTrace()
        }

        val results = intent.getSerializableExtra("results") as HashMap<String, Float>
        Log.d("CameraXApp", "SORTED: $results")
        val resultsSorted = results.toList().sortedBy { (_,value)-> value }
        val customAdapter = CustomAdapter(resultsSorted, this)

        binding.recyclerView.layoutManager = LinearLayoutManager(this)
        binding.recyclerView.adapter = customAdapter

        setContentView(binding.root)

    }

}