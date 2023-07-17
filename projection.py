import os
import webbrowser
import cv2
import supervision as sv
import tensorflow as tf
import tensorflow_similarity as tfsim
import base64
import plotly.graph_objects as go
import numpy as np
import umap
import time
from sklearn.manifold import TSNE
from typing import Dict
from IPython.display import display, HTML
print(umap.__version__)
def image_to_data_uri(image_path):
    with open(image_path, "rb") as image_file:
        encoded_image = base64.b64encode(image_file.read()).decode("utf-8")
    return "data:image/jpeg;base64," + encoded_image


TRAIN_DIR = "/home/gab/PycharmProjects/tf_similarity/index_v4 (senza 2e,1e,50c)"
#TRAIN_DIR = "/home/gab/PycharmProjects/tf_similarity/index_v3"

save_path = "/home/gab/PycharmProjects/tf_similarity/model_efficientNet"
# reload the model
reloaded_model = tf.keras.models.load_model(
    save_path,
    custom_objects={"SimilarityModel": tfsim.models.SimilarityModel},
)
reloaded_model.summary()

labels = []
train = []
images = []
image_paths = []

class_ids = sorted(os.listdir(TRAIN_DIR))
NUM_EXAMPLES = 40
for class_id in class_ids:
    source_subdir = os.path.join(TRAIN_DIR, class_id)
    for image_path in sv.list_files_with_extensions(source_subdir)[:NUM_EXAMPLES]:
        image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
        labels.append(class_id)
        images.append(image)
        image_paths.append(str(image_path))
        embedding = reloaded_model.call(tf.convert_to_tensor(image), training=None, mask=None)
        train.append(embedding)

# class associated with image
labels = np.array(labels)
# features extracted from image
train = np.array(train)
# local image path
image_paths = np.array(image_paths)
# cached images
image_data_uris = {path: image_to_data_uri(path) for path in image_paths}

print(train.shape)
start = time.time()
tsne = TSNE(n_components=3, random_state=0, metric="euclidean")
projections = tsne.fit_transform(np.squeeze(train))
end = time.time()
print(f"generating projections with T-SNE took: {(end - start):.2f} sec")


def display_projections(
        labels: np.ndarray,
        projections: np.ndarray,
        image_paths: np.ndarray,
        image_data_uris: Dict[str, str],
        show_legend: bool = False,
        show_markers_with_text: bool = True
) -> None:
    # Create a separate trace for each unique label
    unique_labels = np.unique(labels)
    print(unique_labels)
    traces = []
    i = 0
    color = np.linspace(0, 100, len(unique_labels))
    for unique_label in unique_labels:
        mask = labels == unique_label
        customdata_masked = image_paths[mask]
        trace = go.Scatter3d(
            x=projections[mask][:, 0],
            y=projections[mask][:, 1],
            z=projections[mask][:, 2],
            mode='markers+text' if show_markers_with_text else 'markers',
            # text=labels[mask],
            customdata=customdata_masked,
            name=str(unique_label),
            marker=dict(size=8,
                        color=[int(color[i])] * NUM_EXAMPLES,
                        colorscale="rainbow",
                        cmax=100,
                        cmin=0),
            hovertemplate="<b>class: %{text}</b><br>path: %{customdata}<extra></extra>"
        )
        traces.append(trace)
        i = i + 1
    # Create the 3D scatter plot
    fig = go.Figure(data=traces)
    fig.update_layout(
        scene=dict(xaxis_title='X', yaxis_title='Y', zaxis_title='Z'),
        width=1000,
        height=1000,
        showlegend=show_legend
    )

    # Convert the chart to an HTML div string and add an ID to the div
    plotly_div = fig.to_html(full_html=False, include_plotlyjs=False, div_id="scatter-plot-3d")

    # Define your JavaScript code for copying text on point click
    javascript_code = f"""
    <script>
        function displayImage(imagePath) {{
            var imageElement = document.getElementById('image-display');
            var placeholderText = document.getElementById('placeholder-text');
            var imageDataURIs = {image_data_uris};
            imageElement.src = imageDataURIs[imagePath];
            imageElement.style.display = 'block';
            placeholderText.style.display = 'none';
        }}

        // Get the Plotly chart element by its ID
        var chartElement = document.getElementById('scatter-plot-3d');

        // Add a click event listener to the chart element
        chartElement.on('plotly_click', function(data) {{
            var customdata = data.points[0].customdata;
            displayImage(customdata);
        }});
    </script>
    """

    # Create an HTML template including the chart div and JavaScript code
    html_template = f"""
    <!DOCTYPE html>
    <html>
        <head>
            <script src="https://cdn.plot.ly/plotly-latest.min.js"></script>
            <style>
                #image-container {{
                    position: fixed;
                    top: 0;
                    left: 0;
                    width: 200px;
                    height: 200px;
                    padding: 5px;
                    border: 1px solid #ccc;
                    background-color: white;
                    z-index: 1000;
                    box-sizing: border-box;
                    display: flex;
                    align-items: center;
                    justify-content: center;
                    text-align: center;
                }}
                #image-display {{
                    width: 100%;
                    height: 100%;
                    object-fit: contain;
                }}
            </style>
        </head>
        <body>
            {plotly_div}
            <div id="image-container">
                <img id="image-display" src="" alt="Selected image" style="display: none;" />
                <p id="placeholder-text">Click on a data entry to display an image</p>
            </div>
            {javascript_code}
        </body>
    </html>
    """

    # Display the HTML template in the Jupyter Notebook
    html = HTML(html_template)
    display(html)
    with open('/home/gab/PycharmProjects/tf_similarity/projection.html', 'wb') as f:  # Use some reasonable temp name
        f.write(html_template.encode("UTF-8"))

    # open an HTML file on my own (Windows) computer
    url = r'/home/gab/PycharmProjects/tf_similarity/projection.html'
    webbrowser.open(url, new=2)

display_projections(
    labels=labels,
    projections=projections,
    image_paths=image_paths,
    image_data_uris=image_data_uris
)


start = time.time()
projections = umap.UMAP(n_neighbors=50, n_components=3, metric="euclidean").fit_transform(np.squeeze(train))
end = time.time()
print(f"generating projections with UMAP took: {(end - start):.2f} sec")

display_projections(
    labels=labels,
    projections=projections,
    image_paths=image_paths,
    image_data_uris=image_data_uris
)
start = time.time()
projections = umap.UMAP(n_neighbors=50, n_components=3, metric="cosine").fit_transform(np.squeeze(train))
end = time.time()

display_projections(
    labels=labels,
    projections=projections,
    image_paths=image_paths,
    image_data_uris=image_data_uris
)
