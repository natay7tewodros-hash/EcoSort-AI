EcoSort AI ♻️
Image-Based Waste Classification Using Deep Learning
Author: Natay Tewodros
Framework: PyTorch
Model Architecture: MobileNetV2
Project Type: Computer Vision · Transfer Learning · Image Classification
________________________________________
1. Project Overview
EcoSort AI is a machine-learning project that uses computer vision to classify waste images into six categories: cardboard, glass, metal, paper, plastic, and trash.
The project explores how deep learning can support waste identification and improve understanding of automated waste-sorting systems. A user can provide an image of a waste item, and the trained model predicts its most likely category.
The project uses transfer learning with MobileNetV2, a convolutional neural network architecture pretrained on ImageNet and adapted for waste classification.
EcoSort AI was developed as my first end-to-end AI/ML project. It gave me practical experience with dataset preparation, model training, evaluation, image preprocessing, and inference using PyTorch.
Current scope: EcoSort AI is a trained machine-learning model with a working single-image prediction workflow. It is not currently a deployed website or mobile application.
2. Project Objectives
•	Develop an image-classification model for common waste categories.
•	Apply transfer learning to a real-world environmental problem.
•	Compare different model-training approaches using evaluation metrics.
•	Evaluate performance on a held-out test dataset.
•	Build a reproducible project structure containing the model, source code, results, and documentation.
•	Identify limitations that must be addressed before real-world deployment.
3. Waste Categories
The model predicts one of six classes.
Category	Label
Cardboard	cardboard
Glass	glass
Metal	metal
Paper	paper
Plastic	plastic
Trash	trash
These labels follow the categories used in the dataset. They do not represent a complete waste-management or recycling decision system.
4. Dataset
EcoSort AI uses the TrashNet waste-image dataset, originally created by Gary Thung and Mindy Yang.
After excluding two identified duplicate images across dataset splits, the project used 2,525 images.
Dataset split	Images
Training	1,768
Validation	377
Testing	380
Total	2,525
The dataset was organized into training, validation, and test subsets. The validation set supported model selection, while the held-out test set was used for final evaluation.
Dataset limitations
•	The classes are imbalanced, particularly the trash class.
•	Many images have relatively clean backgrounds.
•	The dataset does not include a separate organic-waste category.
•	Performance on the dataset does not guarantee the same performance on everyday waste photographs.
See 05_Dataset_Info/ for the available dataset documentation and related records.
5. Technical Approach
Model development
The project explored multiple training approaches, including a simple CNN baseline and transfer-learning variants.
The final selected model uses MobileNetV2 with an unfrozen fine-tuning approach. Its pretrained feature representations were adapted to the six waste categories.
Image preprocessing
Input images are processed using the following configuration:
•	Color mode: RGB
•	Input resolution: 224 × 224 pixels
•	Pixel scaling: values converted to the range 0–1 before normalization
•	Normalization mean: [0.485, 0.456, 0.406]
•	Normalization standard deviation: [0.229, 0.224, 0.225]
The inference pipeline must use preprocessing consistent with the selected model.
Prediction workflow
1.	Load a waste-item image.
2.	Convert and preprocess the image.
3.	Pass the image through the trained MobileNetV2 model.
4.	Obtain scores for the six categories.
5.	Identify the category with the highest predicted score.
A predicted category is a model output, not a guarantee that an item is recyclable or accepted by a particular local recycling facility.
6. Evaluation Results
The selected model was evaluated on a held-out test set of 380 images.
Metric	Result
Test accuracy	89.2%
Correct test predictions	339 / 380
Macro F1-score	87.8%
Weighted F1-score	89.2%
Best validation accuracy	90.7%
Interpreting the results
•	Test accuracy measures the proportion of test images classified correctly.
•	Macro F1-score gives equal weight to each class when averaging its F1-score, making it useful when classes have different sample counts.
•	Weighted F1-score averages class F1-scores according to their support in the test set.
•	Validation accuracy was used during model development and is separate from final test performance.
The results show promising performance on the selected dataset. They do not establish equivalent accuracy on images from different environments.
Evaluation artifacts
The 04_Results/ directory contains the available evaluation outputs, including:
•	Confusion matrix
•	Classification report
•	Per-class F1-score visualization
•	Model comparison results
•	Final evaluation metrics
These artifacts provide more detail than overall accuracy alone.
7. Project Structure
EcoSort_AI/
│
├── 01_Project_Report/
│   └── Project documentation
│
├── 02_Source_Code/
│   └── Training, preprocessing, and prediction code
│
├── 03_Trained_Model/
│   ├── best_model.pt
│   ├── class_labels.txt
│   └── model_info.json
│
├── 04_Results/
│   ├── classification_report.csv
│   ├── class_f1_scores.png
│   ├── confusion_matrix.csv
│   ├── confusion_matrix.png
│   ├── evaluation_results.csv
│   ├── final_metrics.json
│   ├── model_comparison.csv
│   └── model_comparison.png
│
├── 05_Dataset_Info/
│   └── Dataset documentation and records
│
├── 06_Presentation/
│   └── Project presentation
│
├── README.md
└── requirements.txt
The directory descriptions summarize the intended contents. Individual files should be included only when they exist in the final submission.
8. Installation and Environment
Python and the dependencies listed in requirements.txt are required to run the project code.
Step 1: Clone or copy the project
git clone <your-repository-url>
cd EcoSort_AI
If the project is provided as a downloaded folder, open a terminal in that folder instead.
Step 2: Create a virtual environment
python -m venv .venv
Activate it on Windows PowerShell:
.\.venv\Scripts\Activate.ps1
Step 3: Install dependencies
python -m pip install --upgrade pip
pip install -r requirements.txt
Step 4: Review the source code
Start with the prediction script in 02_Source_Code/ and the model metadata in 03_Trained_Model/.
The exact prediction command depends on the arguments supported by the included script. Use its implemented interface rather than assuming that a particular command-line option exists.
Important: A successful installation also requires a compatible PyTorch environment. Consult the official PyTorch installation instructions if the installed PyTorch build does not match your system.
9. Model Files
The 03_Trained_Model/ directory contains the selected model checkpoint and supporting metadata.
•	best_model.pt — trained model checkpoint.
•	class_labels.txt — class names and their intended order.
•	model_info.json — model architecture, preprocessing configuration, and evaluation metadata.
A PyTorch checkpoint is a binary file; it should be loaded by compatible Python code rather than opened in a word processor.
10. Limitations
EcoSort AI is a prototype intended to demonstrate image classification, not a production waste-management system.
Current limitations include:
1.	Dataset domain: the training dataset may not represent local waste, lighting, backgrounds, or camera conditions.
2.	Class imbalance: some categories have fewer examples than others.
3.	Six-category scope: the model cannot identify categories that are absent from its training labels.
4.	Prediction uncertainty: a high model score does not guarantee correctness.
5.	No deployed application or hardware: the project currently provides a model and prediction workflow rather than an integrated consumer product.
6.	No demonstrated operational impact: the evaluation does not establish that the model improves recycling outcomes in real-world settings.
11. Future Development
Potential future improvements include:
•	Expanding the dataset with diverse, locally relevant waste images.
•	Improving performance on underrepresented classes.
•	Evaluating the model on a separately collected real-world dataset.
•	Improving confidence analysis and error inspection.
•	Exploring an application or hardware integration after additional testing.
•	Assessing local waste-sorting requirements before providing disposal recommendations.
These are future directions, not features claimed to be implemented in the current version.
12. Dataset Attribution and Licensing
EcoSort AI uses the TrashNet dataset associated with Gary Thung and Mindy Yang. Consult the original dataset repository and accompanying documentation for attribution and licensing terms.
The project source code, trained model, and third-party dataset images may have different licensing conditions. Do not assume that one license automatically covers every component. Include a project license only when its scope has been verified.
13. Acknowledgements
I acknowledge the creators of TrashNet for making the waste-image dataset available for research and learning. I also acknowledge the PyTorch and torchvision communities for the tools used in model development.
14. Author
Natay Tewodros
EcoSort AI is my first AI/ML project. I developed it while learning the practical workflow of building a machine-learning system, from dataset preparation through model evaluation and single-image prediction.
The project represents a foundation for continued learning in computer vision, deep learning, and environmentally focused AI.
________________________________________
EcoSort AI — exploring how computer vision can contribute to better waste identification.


