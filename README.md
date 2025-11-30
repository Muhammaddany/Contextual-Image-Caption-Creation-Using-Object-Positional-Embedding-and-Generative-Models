# Contextual-Image-Caption-Creation-Using-Object-Positional-Embedding-and-Generative-Models

## Overview
This project presents a novel approach to image captioning by integrating object detection, spatial relationship modeling, and advanced language generation. The system leverages YOLOv5 for precise object detection, constructs scene graphs to capture spatial relationships, and utilizes Generative model to generate contextually rich captions. In our scenario, caption generation is the assigned task. Multiple models are integrated with the state-of-the-art language generator transformer. an  expert survey was conducted to assess caption fluency and relevance. The survey link is included at the end of this repository.
Getting Started
To start working with this project, follow these steps:

•	Clone the repository:

    git clone https://github.com/yourusername contextual-image-caption.git cd contextual-image-caption

•	Install the required packages:

    pip install -r requirements.txt

•	Download the YOLOv5 weights and configuration files

    mkdir -p model
    cd model
    wget https://github.com/AlexeyAB/darknet/releases/download/darknet_yolo_v4_optimal/yolov5.weights
    wget https://raw.githubusercontent.com/AlexeyAB/darknet/master/cfg/yolov5.cfg
    wget https://raw.githubusercontent.com/AlexeyAB/darknet/master/data/coco.names

## Features
* **Object Detection:** Utilizes YOLOv5 for accurate object detection. 

* **Scene Graph Generation:** Constructs spatial relationships between detected objects. 

* **Caption Generation:** Employs GPT models for contextually rich captions. 

* **Cross-Platform Compatibility:** Supports Windows, macOS, and Linux environments.

## Project Structure
.  
├── model/  
│   ├── yolov5.weights  
│   ├── yolov5.cfg  
│   └── coco.names  
├── src/  
│   ├── object_detection.py  
│   ├── scene_graph.py  
│   ├── caption_generator.py  
│   └── utils.py  
├── output/  
│   ├── detections.txt  
│   ├── detections.csv  
│   └── scene_graph.json  
├── requirements.txt  
└── README.md


## Installation

### On Linux/macOS
    git clone https://github.com/yourusername/contextual-image-caption.git
    cd contextual-image-caption
    pip install -r requirements.txt
    mkdir -p model && cd model
    wget https://github.com/AlexeyAB/darknet/releases/download/darknet_yolo_v4_optimal/yolov5.weights
    wget https://raw.githubusercontent.com/AlexeyAB/darknet/master/cfg/yolov5.cfg
    wget https://raw.githubusercontent.com/AlexeyAB/darknet/master/data/coco.names


### On Windows

•	Install Git and clone the repository:

    git clone https://github.com/yourusername/contextual-image-caption.git cd contextual-image-caption

•	Install the required packages using pip:

    pip install -r requirements.txt

•	Download the necessary files:

    mkdir model
    cd model
    curl -O https://github.com/AlexeyAB/darknet/releases/download/darknet_yolo_v4_optimal/yolov5.weights
    curl -O https://raw.githubusercontent.com/AlexeyAB/darknet/master/cfg/yolov5.cfg
    curl -O https://raw.githubusercontent.com/AlexeyAB/darknet/master/data/coco.names

## Dependencies
numpy==1.21.0  
opencv-python==4.5.3  
matplotlib==3.4.3  
torch==1.9.0  
torchvision==0.10.0  
networkx==2.6.3  
openai==0.27.0  
cython  
yacs  
tqdm

## Implementation Details

### Object Detection Module

* **Technology:** YOLOv5

* **Image Preprocessing:**

   * Resize to 416x416

   * Blob conversion

   * BGR to RGB conversion
*  **Parameters:**

     * Confidence threshold: 0.5
     * Non-maximum suppression threshold: 0.4

## Scene Graph Generation
    
* **Relationship Prediction:** Predict spatial and contextual relationships between objects.
* **Graph Construction:** Use NetworkX to create a directed graph representing the scene.

## Caption Generation
* **Technology:** GPT model (gpt-4-turbo)
* **Parameters:** 
    * Temperature: 0.6
    * Top_p: 1
    * Max tokens: 200
    * Frequency penalty: 0
    * Presence penalty: 0




## Usage Instructions

**1. Upload Dataset**

    import cv2
    import matplotlib.pyplot as plt
    from google.colab import files

    uploaded = files.upload()

    filename = list(uploaded.keys())[0]  
    # Get the uploaded file name dynamically

    # Read the image file
    test_img = cv2.imread(filename)  
    # Using the same variable as before
    img = cv2.cvtColor(test_img, cv2.COLOR_BGR2RGB)

    # Function to plot the image
    def plot_image(img, cmap=None):
    plt.imshow(img, cmap=cmap)
    plt.xticks([])
    plt.yticks([])

    # Display the image
    plot_image(img)
    plt.show()

**2.	Download YOLOv5 Weights and Configuration Files**

    # first create a directory to store the model
    %mkdir model

    # enter the directory and download the necessary files 
    %cd model
    !wget https://github.com/AlexeyAB/darknet/releases/download/darknet_yolo_v4_optimal/yolov5.weights
    
    !wget https://raw.githubusercontent.com/AlexeyAB/darknet/master/cfg/yolov5.cfg
    
    !wget https://raw.githubusercontent.com/AlexeyAB/darknet/master/data/coco.names
    %cd ..
    
    # Image Preprocessing: Convert Image to Blob

    scalefactor = 1.0/255.0
    new_size = (416, 416)
    blob = cv2.dnn.blobFromImage(test_img, scalefactor, new_size, swapRB=True, crop=False)

    # define class labels
    class_labels_path = "model/coco.names"
    class_labels = open(class_labels_path).read().strip().split("\n")
    class_labels

    # declare repeating bounding box colors for each class 
    # 1st: create a list colors as an RGB string array
    # Example: Red, Green, Blue, Yellow, Magenda
    class_colors = ["255,0,0","0,255,0","0,0,255","255,155,0","255,0, 255"]
 
    #2nd: split the array on comma-separated strings and for change each string type to integer
    class_colors = [np.array(every_color.split(",")).astype("int") for every_color in class_colors]
 
    #3rd: convert the array or arrays to a numpy array
    class_colors = np.array(class_colors)

    #4th: tile this to get 80 class colors, i.e. as many as the classes(16 rows of 5cols each). 
    # If you want unique colors for each class you may randomize the color generation or set them manually
    class_colors = np.tile(class_colors,(16,1))

    def colored(r, g, b, text):
    return "\033[38;2;{};{};{}m{} \033[38;2;255;255;255m".format(r, g, b, text)
 
    for i in range(16):
     line = ""
     for j in range(5):
        class_id = i*5 + j
        class_id_str = str(class_id)
        text = "class" + class_id_str
        colored_text = colored(class_colors[class_id][0], class_colors[class_id][1], class_colors[class_id][2], text)
        line += colored_text
    print(line)

    # or select the colors randomly
    class_colors = np.random.randint(0, 255, size=(len(class_labels), 3), dtype="uint8")    

**3.	Object Detection**

    # Load the pre-trained model 
    yolo_model = cv2.dnn.readNetFromDarknet('model/   yolov5.cfg','model/yolov5.weights')

    # Read the network layers/components. The YOLO neural network has 379 components. They consist of convolutional layers (conv), rectifier linear units (relu) etc.:
    model_layers = yolo_model.getLayerNames()
    print("number of network components: " + str(len(model_layers))) 
    # print(model_layers)

    # extract the output layers in the code that follows:
    # - model_layer[0]: returns the index of each output layer in the range of 1 to 379
    # - model_layer[0] - 1: corrects  this to the range of 0 to 378
    # - model_layers[model_layer[0] - 1]: returns the indexed layer name 
    output_layers = [model_layers[model_layer - 1] for model_layer in yolo_model.getUnconnectedOutLayers()]
 
    # YOLOv5 deploys the same YOLO head as YOLOv4 for detection with the anchor based detection steps, and three levels of detection granularity. 
    print(output_layers) 

    # input pre-processed blob into the model
    yolo_model.setInput(blob)
 
    # compute the forward pass for the input, storing the results per output layer in a list
    obj_detections_in_layers = yolo_model.forward(output_layers)
 
    # verify the number of sets of detections
    print("number of sets of detections: " + str(len(obj_detections_in_layers)))
    def object_detection_analysis(test_image, obj_detections_in_layers, confidence_threshold): 
 
    # get the image dimensions  
    img_height = test_img.shape[0]
    img_width = test_img.shape[1]
 
    result = test_image.copy()
  
    # loop over each output layer 
    for object_detections_in_single_layer in  obj_detections_in_layers:
    # loop over the detections in each layer
      for object_detection in object_detections_in_single_layer:  
        # obj_detection[1]: bbox center pt_x
        # obj_detection[2]: bbox center pt_y
        # obj_detection[3]: bbox width
        # obj_detection[4]: bbox height
        # obj_detection[5]: confidence scores for all detections within the bbox 
 
        # get the confidence scores of all objects detected with the bounding box
        prediction_scores = object_detection[5:]
         # consider the highest score being associated with the winning class
        # get the class ID from the index of the highest score 
        predicted_class_id = np.argmax(prediction_scores)
        # get the prediction confidence
        prediction_confidence = prediction_scores[predicted_class_id]
    
        # consider object detections with confidence score higher than threshold
        if prediction_confidence > confidence_threshold:
            # get the predicted label
            predicted_class_label = class_labels[predicted_class_id]
            # compute the bounding box coordinates scaled for the input image 
            # scaling is a multiplication of the float coordinate with the appropriate  image dimension
            bounding_box = object_detection[0:4] * np.array([img_width, img_height, img_width, img_height])
            # get the bounding box centroid (x,y), width and height as integers
            (box_center_x_pt, box_center_y_pt, box_width, box_height) = bounding_box.astype("int")
            # to get the start x and y coordinates we to subtract from the centroid half the width and half the height respectively 
            # for even values of width and height of bboxes adjacent to the  image border
            #  this may generate a -1 which is prevented by the max() operator below  
            start_x_pt = max(0, int(box_center_x_pt - (box_width / 2)))
            start_y_pt = max(0, int(box_center_y_pt - (box_height / 2)))
            end_x_pt = start_x_pt + box_width
            end_y_pt = start_y_pt + box_height

            box_color = class_colors[predicted_class_id]
            
            # convert the color numpy array as a list and apply to text and box
            box_color = [int(c) for c in box_color]
            
            # print the prediction in console
            predicted_class_label = "{}: {:.2f}%".format(predicted_class_label, prediction_confidence * 100)
            print("predicted object {}".format(predicted_class_label))
            
            # draw the rectangle and text in the image
            cv2.rectangle(result, (start_x_pt, start_y_pt), (end_x_pt, end_y_pt), box_color, 1)
            cv2.putText(result, predicted_class_label, (start_x_pt, start_y_pt-5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, box_color, 1)
    return result
 
    confidence_threshold = 0.2
    result_raw = object_detection_analysis(test_img, obj_detections_in_layers, confidence_threshold)
 
    result_img = cv2.cvtColor(result_raw, cv2.COLOR_BGR2RGB)
    plt.imshow(result_img)
    plt.show()

    class_ids_list = []
    boxes_list = []
    confidences_list = []

    def object_detection_attributes(test_image, obj_detections_in_layers, confidence_threshold):
    # get the image dimensions  
    img_height = test_img.shape[0]
    img_width = test_img.shape[1]
  
    # loop over each output layer 
    for object_detections_in_single_layer in obj_detections_in_layers:
    # loop over the detections in each layer
    for object_detection in object_detections_in_single_layer:  
      # get the confidence scores of all objects detected with the bounding box
      prediction_scores = object_detection[5:]
      # consider the highest score being associated with the winning class
      # get the class ID from the index of the highest score 
      predicted_class_id = np.argmax(prediction_scores)
      # get the prediction confidence
      prediction_confidence = prediction_scores[predicted_class_id]
      
      # consider object detections with confidence score higher than threshold
      if prediction_confidence > confidence_threshold:
        # get the predicted label
        predicted_class_label = class_labels[predicted_class_id]
        # compute the bounding box coordinates scaled for the input image
        bounding_box = object_detection[0:4] * np.array([img_width, img_height, img_width, img_height])
        (box_center_x_pt, box_center_y_pt, box_width, box_height) = bounding_box.astype("int")
        start_x_pt = max(0, int(box_center_x_pt - (box_width / 2)))
        start_y_pt = max(0, int(box_center_y_pt - (box_height / 2)))
        
        # update the 3 lists for nms processing
        # - confidence is needed as a float 
        # - the bbox info has the openCV Rect format
        class_ids_list.append(predicted_class_id)
        confidences_list.append(float(prediction_confidence))
        boxes_list.append([int(start_x_pt), int(start_y_pt), int(box_width), int(box_height)])
    score_threshold = 0.5
    object_detection_attributes(test_img, obj_detections_in_layers, score_threshold)
    # NMS for a set of overlapping bboxes returns the ID of the one with highest 
    # confidence score while suppressing all others (non maxima)
    # - score_threshold: a threshold used to filter boxes by score 
    # - nms_threshold: a threshold used in non maximum suppression. 
 
    score_threshold = 0.5
    nms_threshold = 0.4
    winner_ids = cv2.dnn.NMSBoxes(boxes_list, confidences_list, score_threshold, nms_threshold)

    # loop through the final set of detections remaining after NMS and draw bounding box and write text
    for winner_id in winner_ids:
    max_class_id = winner_id
    box = boxes_list[max_class_id]
    start_x_pt = box[0]
    start_y_pt = box[1]
    box_width = box[2]
    box_height = box[3]
    
    #get the predicted class id and label
    predicted_class_id = class_ids_list[max_class_id]
    predicted_class_label = class_labels[predicted_class_id]
    prediction_confidence = confidences_list[max_class_id]

     #obtain the bounding box end coordinates
    end_x_pt = start_x_pt + box_width
    end_y_pt = start_y_pt + box_height
    
    #get a random mask color from the numpy array of colors
    box_color = class_colors[predicted_class_id]
    
    #convert the color numpy array as a list and apply to text and box
    box_color = [int(c) for c in box_color]
    
    # print the prediction in console
    predicted_class_label = "{}: {:.2f}%".format(predicted_class_label, prediction_confidence * 100)
    print("predicted object {}".format(predicted_class_label))
    
    # draw rectangle and text in the image
    cv2.rectangle(test_img, (start_x_pt, start_y_pt), (end_x_pt, end_y_pt), box_color, 2)
    cv2.putText(test_img, predicted_class_label, (start_x_pt, start_y_pt-5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, box_color, 2)
    
    test_imgz = cv2.cvtColor(test_img, cv2.COLOR_BGR2RGB)

    plt.imshow(test_imgz)
    plt.show()


**4. Scene Graph Generation**

    # Install dependencies
    !git clone https://github.com/KaihuaTang/Scene-Graph-Benchmark.pytorch.git
    %cd Scene-Graph-Benchmark.pytorch
    !pip install -r requirements.txt
    !python setup.py build develop

    # Load detected objects
    detected_objects = [
    {"label": "dog", "bbox": [200, 300, 400, 500]},
    {"label": "person", "bbox": [50, 100, 250, 400]},
    {"label": "person", "bbox": [400, 150, 600, 500]}
    ]

    # Save detections in JSON format (SGG model input)
    with open("detected_objects.json", "w") as f:
    json.dump(detected_objects, f)

    print("Saved detected objects for scene graph generation!")

    !pip install torch torchvision
    !pip install cython yacs matplotlib tqdm
    !apt-get install ninja-bui

    # Clone the repository
    !git clone https://github.com/KaihuaTang/Scene-Graph-Benchmark.pytorch.git
    %cd Scene-Graph-Benchmark.pytorch

    # Install required Python packages
    !pip install -r requirements.txt

    # Build the project
    !python setup.py build develop

    from scene_graph_benchmark.demo.predictor import SceneGraphPredictor

    # Load model (pre-trained on Visual Genome dataset)
    model = SceneGraphPredictor("checkpoints/motifnet_sgdet.tar")

    # Load detected objects
    with open("detected_objects.json") as f:
    detected_objects = json.load(f)

    # Predict relationships
    scene_graph = model.predict(detected_objects)

    # Save scene graph
    with open("scene_graph.json", "w") as f:
    json.dump(scene_graph, f, indent=4)

    print("Scene Graph Generated & Saved!")

    import networkx as nx
    import matplotlib.pyplot as plt

    # Load scene graph
    with open("scene_graph.json") as f:
    scene_graph = json.load(f)

    # Create graph
    G = nx.DiGraph()

    # Add nodes (objects)
    for obj in scene_graph["objects"]:
    G.add_node(obj)

    # Add edges (relationships)
    for rel in scene_graph["relationships"]:
    G.add_edge(rel["subject"], rel["object"], label=rel["predicate"])

    # Draw graph
    pos = nx.spring_layout(G)
    plt.figure(figsize=(8, 6))
    nx.draw(G, pos, with_labels=True, node_size=3000, node_color="lightblue", font_size=10, edge_color="gray")
    edge_labels = nx.get_edge_attributes(G, 'label')
    nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels, font_color="red")
    plt.title("Generated Scene Graph")
    plt.show()


**5. Caption Generation**

    def GPT_Completion(texts) :
    ## Call the API key under your account (in a secure way)
    response = GPT_Completion(scene_graph_text)
    print("Generated Caption:", response)

#  Results
The system generates:
    
1. Object Detection Visualization: Image with bounding boxes and labels.
2.	Scene Graph Visualization: Graph showing spatial relationships.
3.	Natural Language Captions: Context-aware descriptions.

**Evaluation**  
Survery form: https://docs.google.com/forms/d/e/1FAIpQLSfkLlBj6UdrTU6Ixpb0UWGRRitybHly4YKqsxQ8nPMpE-chcA/viewform 

