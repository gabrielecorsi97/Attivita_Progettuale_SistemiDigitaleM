package com.example.attivit_progettuale

import android.graphics.Rect
import android.os.Parcelable
import kotlinx.parcelize.Parcelize
import org.pytorch.demo.objectdetection.PrePostProcessor

@Parcelize
class Result(var classIndex: Int, var score: Float, var rect: Rect) : Parcelable{


    var left : Int
    var top : Int
    var right : Int
    var bottom : Int

    init{

        this.top = rect.top
        this.left = rect.left
        this.right = rect.right
        this.bottom = rect.bottom

    }

    override fun toString(): String {
        return "ClassIndex: ${PrePostProcessor.mClasses[classIndex]}, Score: $score, Rect: ${left},${top},${right},${bottom}"
    }


}
