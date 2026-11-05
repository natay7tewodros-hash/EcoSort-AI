EcoSort AI ♻️

Image-Based Waste Classification Using Deep Learning

Author: Natay Tewodros Framework: PyTorch Model architecture: MobileNetV2 Project type: Computer Vision · Transfer Learning · Image Classification · Streamlit Web App

Live demo: https://ecosortai-4qr6kznx.manus.space/ (a separate web version of the project, hosted on Manus)

Project Overview

EcoSort AI is a machine-learning project that uses computer vision to classify waste images into six categories: cardboard, glass, metal, paper, plastic, and trash. The project explores how deep learning can support waste identification and improve understanding of automated waste-sorting systems.

A user uploads an image of a waste item, and the trained model predicts its most likely category and shows how confident it is.

The project uses transfer learning with MobileNetV2, a convolutional neural network pretrained on ImageNet and adapted for waste classification.

EcoSort AI was developed as my first end-to-end AI/ML project. It gave me practical experience with dataset preparation, model training, evaluation, image preprocessing, inference with PyTorch, and building a simple web app.

Current scope: EcoSort AI is a trained model with a working web app for single-image prediction. It is a learning prototype. It is not a mobile app, it does not control a sorting machine, and it is not official recycling guidance.

Project Objectives
Develop an image-classification model for common waste categories.
Apply transfer learning to a real-world environmental problem.
Compare different model-training approaches using evaluation metrics.
Evaluate performance on a held-out test dataset.
Build a simple app so people can try the model with their own photos.
Identify limitations that must be addressed before real-world use.
Waste Categories

The model predicts one of six classes.

Category	Label
Cardboard	cardboard
Glass	glass
Metal	metal
Paper	paper
Plastic	plastic
Trash	trash

These labels follow the categories used in the dataset. They do not represent a complete waste-management or recycling decision system.

The App

The code in this repository is a Streamlit app. It is a separate version from the live demo above, with a simpler design, and it runs on your own computer.

It is built with Streamlit and lives in the EcoSortAI APP/app/ folder.

What it does

The user uploads one image (jpg, jpeg, or png).
The app shows the image and the most likely category with its confidence.
The app lists the top 3 predictions with progress bars.
If the top confidence is below 60%, the app shows a warning that the model is unsure and suggests trying a clearer photo.

App files

EcoSortAI APP/app/app.py — the Streamlit page (upload, results, warnings).
EcoSortAI APP/app/predictor.py — loads the trained model once and predicts one image at a time.
Run the app locally

You need Python 3.10–3.12.

powershell
git clone https://github.com/natay7tewodros-hash/EcoSort-AI.git
cd EcoSort-AI
py -3.12 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
streamlit run "EcoSortAI APP/app/app.py"

The browser page should open automatically. If it does not, open the local URL printed in the terminal.

If your installed Python version is different, replace 3.12 in the py command with that version. On macOS or Linux, use python3 -m venv .venv and source .venv/bin/activate instead.

Note: predictor.py looks for the model file best_model_unfrozen.pt one folder above the app/ folder, which is EcoSortAI APP/. Keep the model file there.

Dataset

EcoSort AI uses the TrashNet waste-image dataset, originally created by Gary Thung and Mindy Yang. After excluding two identified duplicate images across dataset splits, the project used 2,525 images.

Dataset split	Images
Training	1,768
Validation	377
Testing	380
Total	2,525

The validation set supported model selection, while the held-out test set was used once for final evaluation.

Dataset limitations

The classes are imbalanced, particularly the trash class.
Many images have relatively clean backgrounds.
The dataset does not include a separate organic-waste category.
Performance on the dataset does not guarantee the same performance on everyday waste photographs.
Technical Approach
Model development

The project explored multiple training approaches, including a simple CNN baseline and transfer-learning variants. The final selected model uses MobileNetV2 with an unfrozen fine-tuning approach: the last blocks of the pretrained network were trained further to adapt to the six waste categories.

Image preprocessing

Input images are processed with the following configuration:

Color mode: RGB
Input resolution: 224 × 224 pixels
Pixel scaling: values converted to the range 0–1 before normalization
Normalization mean: [0.485, 0.456, 0.406]
Normalization standard deviation: [0.229, 0.224, 0.225]

The inference pipeline must use preprocessing consistent with the selected model.

Prediction workflow
Load a waste-item image.
Convert it to RGB and preprocess it.
Pass it through the trained MobileNetV2 model.
Convert the outputs to probabilities for the six categories.
Return the categories with the highest probabilities.

A predicted category is a model output, not a guarantee that an item is recyclable or accepted by a particular local recycling facility.

Evaluation Results

The selected model was evaluated on a held-out test set of 380 images.

Metric	Result
Test accuracy	89.2%
Correct test predictions	339 / 380
Macro F1-score	87.8%
Weighted F1-score	89.2%
Best validation accuracy	90.7%

Interpreting the results

Test accuracy measures the proportion of test images classified correctly.
Macro F1-score gives equal weight to each class, which is useful when classes have different sizes.
Weighted F1-score averages class F1-scores according to their size in the test set.
Validation accuracy was used during model development and is separate from final test performance.

The results show promising performance on the selected dataset. They do not establish equivalent accuracy on images from different environments.

Project Structure
EcoSort-AI/
├── EcoSortAI APP/
│   ├── app/
│   │   ├── app.py
│   │   └── predictor.py
│   ├── best_model_unfrozen.pt   (model used by the app)
│   └── requirements.txt
├── best_model_unfrozen.pt     (final model used by the app)
├── best_model.pt              (earlier simple model)
├── train_unfrozen.py          (trains the final model)
├── evaluate_test.py           (test-set evaluation)
├── predict_image.py           (predict one image from the terminal)
├── class_labels.txt
├── model_info.json
├── final_metrics.json
├── classification_report.csv
├── confusion_matrix.png
├── model_comparison.png
├── requirements.txt
└── README.md

Other files in the repo are the project's data checks, training experiments, results, and documents.

best_model_unfrozen.pt — the trained MobileNetV2 checkpoint (final selected model).
requirements.txt — Python libraries needed to run the app.

A PyTorch checkpoint is a binary file. It should be loaded by compatible Python code, not opened in a text editor.

Limitations

EcoSort AI is a prototype intended to demonstrate image classification, not a production waste-management system.

Dataset domain: the training dataset may not represent local waste, lighting, backgrounds, or camera conditions.
Class imbalance: some categories have fewer examples than others.
Six-category scope: the model cannot identify categories that are absent from its training labels.
Prediction uncertainty: a high model score does not guarantee correctness, and a low score means the result should not be trusted.
No hardware integration: the app does not control any physical sorting machine.
No demonstrated operational impact: the evaluation does not show that the model improves recycling outcomes in real-world settings.
Future Development
Expand the dataset with diverse, locally relevant waste images.
Improve performance on underrepresented classes such as trash.
Evaluate the model on a separately collected real-world dataset.
Improve confidence analysis and error inspection.
Explore hardware integration only after additional testing and a safe reject path.
Assess local waste-sorting requirements before giving disposal recommendations.

These are future directions, not features of the current version.

Dataset Attribution and Licensing

EcoSort AI uses the TrashNet dataset associated with Gary Thung and Mindy Yang. Consult the original dataset repository and accompanying documentation for attribution and licensing terms.

The project source code, trained model, and third-party dataset images may have different licensing conditions. Do not assume that one license automatically covers every component. Add a project license only after its scope has been verified.

Acknowledgements

I acknowledge the creators of TrashNet for making the waste-image dataset available for research and learning. I also acknowledge the PyTorch, torchvision, and Streamlit communities for the tools used to build this project.

Author

Natay Tewodros

EcoSort AI is my first AI/ML project. I developed it while learning the practical workflow of building a machine-learning system, from dataset preparation through model evaluation, single-image prediction, and a simple web app. It is a foundation for continued learning in computer vision, deep learning, and environmentally focused AI.

EcoSort AI — exploring how computer vision can contribute to better waste identification.
