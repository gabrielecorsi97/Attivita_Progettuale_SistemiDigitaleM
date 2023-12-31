package com.example.attivit_progettuale

import android.content.Context
import android.content.Intent
import android.graphics.BitmapFactory
import android.view.LayoutInflater
import android.view.View
import android.view.ViewGroup
import android.widget.ImageView
import android.widget.LinearLayout
import android.widget.TextView
import androidx.recyclerview.widget.RecyclerView


class CustomAdapter(private val dataSet: List<Pair<String, Float>>, private val context: Context) :
    RecyclerView.Adapter<CustomAdapter.ViewHolder>() {

    /**
     * Provide a reference to the type of views that you are using
     * (custom ViewHolder)
     */
    class ViewHolder(view: View) : RecyclerView.ViewHolder(view) {
        val textView: TextView
        val imageView: ImageView
        val linearLayout: LinearLayout
        val textDistance: TextView
        init {
            // Define click listener for the ViewHolder's View
            textView = view.findViewById(R.id.reference_nome_moneta)
            imageView = view.findViewById(R.id.referenceMoneta)
            linearLayout = view.findViewById(R.id.linear_layout)
            textDistance = view.findViewById(R.id.distance_text)
        }
    }

    // Create new views (invoked by the layout manager)
    override fun onCreateViewHolder(viewGroup: ViewGroup, viewType: Int): ViewHolder {
        // Create a new view, which defines the UI of the list item
        val view = LayoutInflater.from(viewGroup.context)
            .inflate(R.layout.card, viewGroup, false)

        return ViewHolder(view)
    }

    // Replace the contents of a view (invoked by the layout manager)
    override fun onBindViewHolder(viewHolder: ViewHolder, position: Int) {

        // Get element from your dataset at this position and replace the
        // contents of the view with that element
        val key = dataSet[position].first
        viewHolder.textView.text = key

        val bitmap = context.assets.open("reference/$key.jpg")
        val bit = BitmapFactory.decodeStream(bitmap)
        if(position == 0){
            viewHolder.linearLayout.background = context.getDrawable(R.drawable.customborder)
        }
        viewHolder.imageView.setImageBitmap(bit)
        viewHolder.textDistance.text = dataSet[position].second.format(4)
        viewHolder.itemView.setOnClickListener {
            val myIntent = Intent(context, CoinDetail::class.java)
            myIntent.putExtra("key",key)
            context.startActivity(myIntent)
        }
    }

    // Return the size of your dataset (invoked by the layout manager)
    override fun getItemCount() = dataSet.size
    private fun Float.format(digits: Int) = "%.${digits}f".format(this)

}