from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.models import load_model
from imutils.video import VideoStream
import numpy as np
import imutils
import cv2

def detect_mask(frame, faceNet, maskNet):
	#Captura as dimensões da moldura e constroi um blod.
	(height, width) = frame.shape[:2]
	blob = cv2.dnn.blobFromImage(frame, 1.0, (300, 300),(104.0, 177.0, 123.0))
	#Detecta o rosto e passa o blod pela rede.
	faceNet.setInput(blob)
	detections = faceNet.forward()
	#Inicializa as listas de rosto,locais e previsões.
	faces, locs, preds = [], [], []

	for i in range(0, detections.shape[2]): #Loop de detecções
		#Obtem a confiança na detecção.
		confidence = detections[0, 0, i, 2]
		#Filtra e garante uma maior confiança.
		if confidence > 0.5:
			#Calcula X,Y da box
			box = detections[0, 0, i, 3:7] * np.array([width, height, width, height])
			(startX, startY, endX, endY) = box.astype("int")
			#Abstrai o rosto e converte de BGR para RGB, redimencionando para 224px e pré-processa.
			face = frame[startY:endY, startX:endX]
			face = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)
			face = cv2.resize(face, (224, 224))
			face = img_to_array(face)
			face = preprocess_input(face)
			#Adiciona a box delimitadora e o rosto na lista.
			faces.append(face)
			locs.append((startX, startY, endX, endY))

	#Prever se há pelo menos um rosto.
	if len(faces) > 0:
		#Previsoes  de multiplos rostos ao mesmo tempo.
		faces = np.array(faces, dtype="float32")
		preds = maskNet.predict(faces, batch_size=32)
		#Retorna as tuplas com as localizações dos rostos e suas posições.
	return (locs, preds)

prototxtPath = r"face_detector\deploy.prototxt"
weightsPath = r"face_detector\res10_300x300_ssd_iter_140000.caffemodel"
faceNet = cv2.dnn.readNet(prototxtPath, weightsPath)
maskNet = load_model("mask_detector.model") #Carrega o modelo de detecção.

print("[INFO] Iniciando captura de video...")
vs = VideoStream(src=0).start()

while True:
	frame = vs.read()
	frame = imutils.resize(frame, width=800) #Define o tamanho do frame.
	(locs, preds) = detect_mask(frame, faceNet, maskNet)

	for (box, pred) in zip(locs, preds):
		#Desempacota a box delimitadora e as previsões.
		(startX, startY, endX, endY) = box
		(mask, withoutMask) = pred

		label = "Com Mascara" if mask > withoutMask else "Sem Mascara"#Define o rótulo par a box.
		color = (0, 255, 0) if label == "Com Mascara" else (0, 0, 255)#Define as cores para a box.
		label = "{}: {:.2f}%".format(label, max(mask, withoutMask) * 100)#Adiciona a probabilidade dos dados.
		cv2.putText(frame, label, (startX, startY - 10),cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 2)
		cv2.rectangle(frame, (startX, startY), (endX, endY), color, 2)

	cv2.imshow("Frame", frame) #Mostra o video stream.
	if cv2.waitKey(1) & 0xFF == ord("q"):  #Define o botão para encerrar.
		break

#Realiza a limpeza dos bits.
cv2.destroyAllWindows()
vs.stop()