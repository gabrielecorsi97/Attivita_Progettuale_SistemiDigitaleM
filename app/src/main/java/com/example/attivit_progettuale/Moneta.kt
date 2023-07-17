package com.example.attivit_progettuale


@kotlinx.serialization.Serializable
open class Moneta () {
}

@kotlinx.serialization.Serializable
class MonetaCommemorativa (val feature : String,
                           val description : String,
                           val tiratura: String,
                           val data: String,
                            val key: String) : Moneta() {

}

@kotlinx.serialization.Serializable
class MonetaFronte(val diameter : String,
                    val thickness: String,
                    val weight: String,
                    val shape: String,
                    val colour: String,
                    val composition: String,
                    val edge: String,
                   val  key: String) : Moneta() {


}

@kotlinx.serialization.Serializable
class MonetaNazionale(val description: String,val key: String) : Moneta() {


}