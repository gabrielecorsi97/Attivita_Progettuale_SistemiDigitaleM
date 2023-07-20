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