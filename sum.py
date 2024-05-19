import cv2
from roboflow import Roboflow
rf = Roboflow(api_key="oazXHCZnSbmqd489bd4G")
project = rf.workspace().project("pakistani-notes-counter")
model = project.version(1).model

# infer on a local image
predictions = model.predict(r"demo4.jpg", confidence=40, overlap=30).json()

# visualize your prediction
model.predict(r"demo4.jpg", confidence=40, overlap=30).save("prediction.jpg")



total_sum = 0  

for i in range(len(predictions['predictions'])):
    classes = predictions["predictions"][i]["class"]
    classes = int(classes)
    total_sum += classes

print("Total Amount is:", total_sum)  

image_path = 'prediction.jpg'

image = cv2.imread(image_path)

text = f'Total Amount is: {total_sum}'
                                            # text sum details
font = cv2.FONT_HERSHEY_SIMPLEX
org = (50, 50)  
font_scale = 1
color = (0, 0, 255) 
thickness = 2
image_with_text = cv2.putText(image, text, org, font, font_scale, color, thickness, cv2.LINE_AA)

cv2.imshow('Image with Total Sum', image_with_text)
cv2.waitKey(0)
cv2.destroyAllWindows()