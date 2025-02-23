# Sign Language Gesture Translation

This project implements a system for continuous Indian Sign Language (ISL) translation and recognition using PyTorch. It extracts keypoints from sign language videos using MediaPipe, preprocesses the data, trains an LSTM-based model (with an optional attention mechanism) to recognize gestures, and provides tools for evaluation and real-time inference.

## Table of Contents
- [Project Structure](#project-structure)
- [Installation](#installation)
- [Dataset](#dataset)
- [Data Preprocessing](#data-preprocessing)
- [Model Architecture](#model-architecture)
- [Training](#training)
- [Evaluation](#evaluation)
- [Inference](#inference)
- [Testing (Batch and Live)](#testing-batch-and-live)
- [Known Issues and Future Improvements](#known-issues-and-future-improvements)
- [License](#license)

## Project Structure

- **preprocessing.py**  
  Extracts keypoints from videos stored in subfolders (each subfolder represents a gesture).  
  - Uses MediaPipe to extract pose and hand landmarks.
  - Applies optional data augmentation.
  - Splits the data into training and test sets (with stratification) and saves the results as a compressed `.npz` file.

- **model.py**  
  Defines the LSTM-based model with:
  - A bidirectional LSTM to process variable-length sequences.
  - An optional attention mechanism to focus on important frames.
  - A final fully connected layer for classifying gestures into one of 98 classes.

- **train.py**  
  Loads the preprocessed data, wraps variable-length sequences in a custom Dataset and collate function (with padding), and trains the model.
  - Implements early stopping and learning rate scheduling.
  - Saves the best model checkpoint (`best_model.pth`).

- **evaluate.py**  
  Evaluates the trained model on the test set.
  - Displays accuracy and a detailed classification report using scikit-learn metrics.

- **inference.py**  
  Runs inference on a single video input.
  - Extracts keypoints from the video and predicts the corresponding gesture.
  - Translates the predicted label into English text using the saved label map.

- **test.py**  
  A comprehensive script that allows:
  - Batch inference on a video file.
  - Live inference using a webcam.
  - Loads the trained model and label map to output the translated gesture.

- **requirements.txt**  
  Lists all the dependencies required to run the project.

## Installation

1. **Clone the Repository:**
   ```bash
   git clone <repository_url>
   cd Continuous-Sign-Language-Translation-and-Recognition
   ```

2. **Create and Activate a Virtual Environment:**
   ```bash
   python -m venv venv
   # On Linux/Mac:
   source venv/bin/activate
   # On Windows:
   venv\Scripts\activate
   ```

3. **Install Dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

## Dataset

- The dataset should be organized as follows:
  - A root dataset folder containing subfolders.
  - Each subfolder is named after a gesture (e.g., `are you free today`, `can you repeat that please`, etc.).
  - Each subfolder contains 1–5 videos (supported formats: `.mp4`, `.avi`, `.mov`) showing the gesture.

- **Note:**  
  The Dataset i worked on was taken from [Indian Sign Language Dataset for Continuous Sign Language Translation and Recognition](https://data.mendeley.com/datasets/kcmpdxky7p/1)

## Data Preprocessing

Extract keypoints and prepare the dataset by running:
```bash
python preprocessing.py --dataset_path path/to/dataset --output_path preprocessed_data.npz --augment
```
- This script:
  - Iterates through each gesture folder.
  - Extracts keypoints from each video using MediaPipe.
  - Optionally applies Gaussian noise as data augmentation.
  - Splits the data into training and test sets (stratified by gesture).
  - Saves the preprocessed data (including a label mapping) into a compressed `.npz` file.

## Model Architecture

The model defined in **model.py** includes:
- **Input Layer:** Accepts keypoints extracted from videos (dimension 258: 33×4 for pose + 21×3 for left hand + 21×3 for right hand).
- **LSTM Layers:** A bidirectional LSTM (default 2 layers) that handles variable-length sequences.
- **Attention (Optional):** An attention mechanism to aggregate LSTM outputs.
- **Output Layer:** A fully connected layer that classifies the input into one of the 98 gesture classes.

## Training

Train the model with:
```bash
python train.py --data_path preprocessed_data.npz --batch_size 16 --epochs 50
```
- **Training Process:**
  - Loads preprocessed training and validation data.
  - Uses a custom collate function to pad sequences.
  - Trains the LSTM model while showing progress via tqdm.
  - Implements early stopping and learning rate scheduling.
  - Saves the best model checkpoint as `best_model.pth`.

## Evaluation

Evaluate the model on the test set:
```bash
python evaluate.py --data_path preprocessed_data.npz --model_path best_model.pth --batch_size 16
```
- **Evaluation Outputs:**
  - Test accuracy.
  - A detailed classification report including precision, recall, and F1-score for each class.

## Inference

To predict the gesture from a single video file, run:
```bash
python inference.py --video_path path/to/video.mp4 --model_path best_model.pth --label_map_path preprocessed_data.npz
```
- The script:
  - Extracts keypoints from the video.
  - Loads the trained model and label mapping.
  - Outputs the predicted English text corresponding to the gesture.

## Testing (Batch and Live)

Test the complete system using **test.py**:

- **Batch Inference on a Video File:**
  ```bash
  python test.py --video_path path/to/video.mp4 --label_map_path preprocessed_data.npz --model_path best_model.pth
  ```

- **Live Inference via Webcam:**
  ```bash
  python test.py --live --label_map_path preprocessed_data.npz --model_path best_model.pth
  ```
  - The live inference script displays the webcam feed and prints the predicted gesture every few frames.

## Known Issues and Future Improvements

- **Warnings During Evaluation:**  
  The classification report may warn about undefined precision for classes with no predicted samples. This is expected when some classes are rarely predicted.

- **Dataset Variability:**  
  Ensure that the dataset is well-organized and that video files are correctly named and formatted.

- **Future Work:**
  - Explore more advanced data augmentation techniques.
  - Experiment with deeper or alternative model architectures.
  - Optimize the live inference pipeline for real-time applications.

## License

This project is released under the [MIT License](LICENSE).
