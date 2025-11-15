🧠 Edge Classifier ML Project

Welcome!
This project trains a small CNN (Convolutional Neural Network) to classify images of open vs. closed objects (for example: eyes, hands, doors, etc.).
You’ll run everything directly inside GitHub Codespaces — no local setup required.

📁 Project structure
.
├── class_open/      ← put your "open" images here
├── class_closed/    ← put your "closed" images here
├── ML_model.py      ← main training script
├── requirements.txt ← Python dependencies
├── Dockerfile       ← environment setup
├── .devcontainer/
│   └── devcontainer.json
└── README.md        ← this guide

🚀 1. Start your Codespace

Open this repository in GitHub.

Click “Code” → “Create Codespace on main”.

Wait until the container builds and dependencies are installed.

The environment installs:

Python 3.10

TensorFlow, scikit-learn, OpenCV, Matplotlib

Git LFS (for large image files)

🖼️ 2. Add your own images

You will train the model on your own pictures.

Open the folders:

class_open/

class_closed/

Upload your images (PNG, JPG, or JPEG).

Keep at least a few samples (10–20+ per class is recommended).

Delete img000.png in both folders.

In the terminal, track your images with Git LFS and commit:

git lfs track "*.jpg" "*.png" "*.jpeg"
git add .gitattributes
git add class_open/ class_closed/
git commit -m "Add my training images"
git push origin main


✅ Tip: Pushing your images saves them safely to your GitHub repository using LFS.

🧹 3. Stop and restart the Codespace

After uploading and pushing your images:

In the Codespace menu, click “Codespaces → Stop current Codespace”.

Then reopen it again with “Open in Codespace”.

This ensures that everything (Docker image, LFS files, Python dependencies) is freshly initialized.

🧠 4. Run the Machine Learning model

Once the environment is ready again:

python ML_model.py


The script will:

Load your class_open/ and class_closed/ images

Train a CNN using 5-fold cross-validation

Save the best model as:

best_edge_classifier_model.h5


Generate metric plots (Accuracy, Loss, AUC) for each fold

Print a confusion matrix and classification report

📊 5. View and download your results

After training finishes, check:

Plots: plot_fold_1.png, plot_fold_2.png, …

Model file: best_edge_classifier_model.h5

Metrics: printed in the terminal

You can download the trained model and plots:

# Download trained model and metrics to your computer
zip results.zip best_edge_classifier_model.h5 plot_fold_*.png


Then use the “Download” button in the Files panel.

🧩 6. (Optional) Start over

If you want to redo the training with new images:

rm -rf class_open/* class_closed/*
git add .
git commit -m "Reset image folders"
git push


Then add new pictures and repeat the steps above.