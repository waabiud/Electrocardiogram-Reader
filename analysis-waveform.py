import ECG_CNN as cnn
import cv2
import matplotlib.pyplot as plt

def analisis(model_path, img, ecg_rows, ecg_cols):
    width, height = 1920, 1080
    img = cv2.resize(img, (width, height), interpolation=cv2.INTER_AREA)
    
    x_step = width // ecg_cols
    y_step = height // ecg_rows

    pt1 = [0, 0]
    pt2 = [x_step, y_step]

    risk_found = False

    # RGB colors for classes: 0=sano, 1=S, 2=T, 3=Q
    risk_colors = [None, (255,0,0), (255,255,0), (0,0,255)]
    risk_text = ['Sano', 'Onda-S', 'Onda-T', 'Onda-Q']

    for row in range(ecg_rows):
        for col in range(ecg_cols):
            roi = img[pt1[1]:pt2[1], pt1[0]:pt2[0]]
            prediction = int(cnn.prediction(model_path, roi))

            if prediction != 0:
                risk_found = True
                cv2.rectangle(img, tuple(pt1), tuple(pt2), risk_colors[prediction], 3)
                cv2.putText(img, risk_text[prediction], (pt1[0], pt1[1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, risk_colors[prediction], 2)

            pt1[0] += x_step
            pt2[0] += x_step

        pt1[0] = 0
        pt1[1] += y_step
        pt2[0] = x_step
        pt2[1] += y_step

    return img if risk_found else None

# Usage
img_path = 'src/full/edges.jpg'
model_path = 'models/ondas-buff.hdf5'
input_img = cv2.imread(img_path)

risk_img = analisis(model_path, input_img, 3, 7)

if risk_img is None:
    print('Electrocardiograma completamente sano')
else:
    plt.imshow(cv2.cvtColor(risk_img, cv2.COLOR_BGR2RGB))
    plt.axis('off')
    plt.show()
