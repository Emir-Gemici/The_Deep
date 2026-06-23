import cv2
from ultralytics import YOLO

# Nihai 5-sinifli model
model = YOLO("runs/detect/final_yolo26s-5/weights/best.pt")

# Kamerayi baslat (0 = varsayilan kamera)
cap = cv2.VideoCapture(0)

while cap.isOpened():
    success, frame = cap.read()
    if not success:
        print("Kameradan goruntu alinamadi.")
        break

    # imgsz=960 -> kucuk nesneler (baret/yelek/palet) icin daha iyi
    sonuclar = model(frame, imgsz=960, conf=0.25, verbose=False)
    gorsellestirilmis_kare = sonuclar[0].plot()

    cv2.imshow("Derineeeee derine daha derine", gorsellestirilmis_kare)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
