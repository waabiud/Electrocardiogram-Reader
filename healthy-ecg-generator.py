import ECG_CNN as cnn
import cv2
import os
from datetime import datetime

def save_healthy_ecg_from_folder(model_path, input_folder, output_folder, ecg_rows=3, ecg_cols=7):
    x_distance, y_distance = 50, 50
    width, height = ecg_cols * x_distance, ecg_rows * y_distance
    
    os.makedirs(output_folder, exist_ok=True)

    for filename in os.listdir(input_folder):
        if filename.lower().endswith(('.png', '.jpg', '.jpeg')):
            img_path = os.path.join(input_folder, filename)
            img = cv2.imread(img_path)
            if img is None:
                print(f"Warning: Couldn't read {img_path}")
                continue

            img = cv2.resize(img, (width, height), interpolation=cv2.INTER_AREA)

            for row in range(ecg_rows):
                for col in range(ecg_cols):
                    x_start, y_start = col * x_distance, row * y_distance
                    x_end, y_end = x_start + x_distance, y_start + y_distance
                    cut_img = img[y_start:y_end, x_start:x_end]

                    prediction = int(cnn.prediction(model_path, cut_img))
                    if prediction != 0:  # Model falsely detects issue
                        timestamp = datetime.now().strftime("%Y%m%d-%H%M%S-%f")
                        out_path = os.path.join(output_folder, f'sano-{timestamp}.jpg')
                        cv2.imwrite(out_path, cut_img)

# usage
save_healthy_ecg_from_folder(
    model_path='models/ondas.hdf5',
    input_folder='src/full/',
    output_folder='test/generador-sano/',
    ecg_rows=3,
    ecg_cols=7
)
