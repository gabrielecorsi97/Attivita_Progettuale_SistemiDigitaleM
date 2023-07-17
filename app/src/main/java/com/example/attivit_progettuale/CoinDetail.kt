package com.example.attivit_progettuale

import android.graphics.BitmapFactory
import androidx.appcompat.app.AppCompatActivity
import android.os.Bundle
import android.util.Log
import com.example.attivit_progettuale.databinding.ActivityCoinDetailBinding


class CoinDetail : AppCompatActivity() {
    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        val binding = ActivityCoinDetailBinding.inflate(layoutInflater)
        val intent = intent
        val key = intent.getStringExtra("key")
        if (key != null) {
            Log.d("PROVA", key)
            val coin  = MainActivity.CoinListHolder.getCoinFromKey("$key.jpg")
            if(coin is MonetaNazionale){
                val bitmap = this.assets.open("reference/$key.jpg")
                val bit = BitmapFactory.decodeStream(bitmap)
                binding.imageDetail.setImageBitmap(bit)

            }
            if(coin is MonetaFronte){
                val bitmap = this.assets.open("reference/$key.jpg")
                val bit = BitmapFactory.decodeStream(bitmap)
                binding.imageDetail.setImageBitmap(bit)
            }
            if(coin is MonetaCommemorativa){
                binding.titleDetail.text = coin.feature
                val bitmap = this.assets.open("reference/$key.jpg")
                val bit = BitmapFactory.decodeStream(bitmap)
                binding.imageDetail.setImageBitmap(bit)
                binding.tiraturaDetail.text =  "Issuing volume: "+coin.tiratura
                binding.dataDetail.text = "Issuing date: "+coin.data
                binding.descriptionDetail.text = coin.description

            }
        }
        setContentView(binding.root)



    }
}