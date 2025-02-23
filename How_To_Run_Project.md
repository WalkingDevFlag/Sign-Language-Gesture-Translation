## How to Run the Project

1. **Data Preprocessing:**  
   Make sure your dataset is organized as follows:  
   ```
   raw_dataset/
       gesture1/
           video1.mp4
           video2.mp4
           ...
       gesture2/
           video1.mp4
           ...
       ...
   ```  
   Then run:
   ```bash
   python data_preprocessing.py --dataset_dir raw_dataset --output_dir preprocessed_data --augment
   ```
   This will create a folder (`preprocessed_data`) containing subfolders for each gesture with corresponding `.npy` files.

2. **Training:**  
   Ensure that you have created (or saved) a label mapping (the dataset class builds it during training). You may want to save the mapping (for example, using `np.save("label_mapping.npy", dataset.label2idx)` after dataset creation). Then run:
   ```bash
   python train.py --data_dir preprocessed_data --num_epochs 50 --batch_size 16 --learning_rate 1e-3
   ```
   The script will save the best model as `best_model.pth`.

3. **Evaluation:**  
   To evaluate the model on a test set (a separate folder with the same structure as training data), run:
   ```bash
   python evaluate.py --data_dir path_to_test_data --model_path best_model.pth --batch_size 16
   ```

4. **Inference:**  
   For a single video file:
   ```bash
   python inference.py --video_path path/to/video.mp4 --model_path best_model.pth --label_mapping label_mapping.npy
   ```
   This will output the predicted gesture (English text).

5. **Testing the Entire Pipeline:**  
   The `test.py` script can run inference on either a provided video file or via live webcam input:
   - To test on a video file:
     ```bash
     python test.py --video_path path/to/video.mp4 --model_path best_model.pth --label_mapping label_mapping.npy
     ```
   - To test using the webcam (simply omit the `--video_path` parameter):
     ```bash
     python test.py --model_path best_model.pth --label_mapping label_mapping.npy
     ```