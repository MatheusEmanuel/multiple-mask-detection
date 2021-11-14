from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.layers import AveragePooling2D
from tensorflow.keras.layers import Dropout
from tensorflow.keras.layers import Flatten
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Input
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.preprocessing.image import load_img
from tensorflow.keras.utils import to_categorical
from sklearn.preprocessing import LabelBinarizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from imutils import paths
import matplotlib.pyplot as plt
import numpy as np
import argparse
import os

#Exemplo de execução: python training.py --dataset database

#Analisador de argumentos.
arg_parse = argparse.ArgumentParser()
arg_parse.add_argument("-d", "--dataset", required=True, help="Path de entrada para conjunto de dados")
arg_parse.add_argument("-p", "--plot", type=str, default="plot.pdf", help="Path para saida do grafico de perda/precisao")
arg_parse.add_argument("-m", "--model", type=str, default="mask_detector.model", help="Caminho de saida para modelo de detectcao facial")

args = vars(arg_parse.parse_args())

INIT_LR=1e-4 #Parametros de aprendizado inicial,
EPOCHS=20 #Numero de períodos para treinar
BS=32 #Tamanho do lote


#Pré-processamento de dados para o treinamento.
print("[INFO] Carregando imagens...")
image_paths=list(paths.list_images(args["dataset"]))
data,labels=[],[]


for image_paths in image_paths: #loop sobre path das imagens
    label=image_paths.split(os.path.sep)[-2] #extrai rotudo de classe do arquivo
    image=load_img(image_paths, target_size=(224,224)) #carrega img(224x224) de entrada e pre-processa
    image=img_to_array(image)
    image=preprocess_input(image)
    #atualiza lista de dados e labels
    data.append(image)
    labels.append(label)

#converte dados e labels para matrizes numpy
data=np.array(data,dtype="float32")
labels=np.array(labels)

#codifica one-hot nas labels
lb=LabelBinarizer()
labels=lb.fit_transform(labels)
labels=to_categorical(labels)

#Usa scikit-learn para particionar os dados em 80% para treinamento e 20% para testes restantes
(trainX, testX, trainY, testY) = train_test_split(data, labels,	test_size=0.20, stratify=labels, random_state=42)

#controi um gerador de imagens para o aumento de dados no treinamento
aug=ImageDataGenerator(rotation_range=0, zoom_range=0.15, width_shift_range=0.2,height_shift_range=0.2, shear_range=0.15,
                       horizontal_flip=True, fill_mode="nearest")

#carrega a rede MobileNetV2
base_model=MobileNetV2(weights="imagenet", include_top=False, input_tensor=Input(shape=(224,224,3)))

#constroi um modelo que vai ser colocado como modelo básico
head_model=base_model.output
head_model=AveragePooling2D(pool_size=(7,7))(head_model)
head_model=Flatten(name="flatten")(head_model)
head_model=Dense(128, activation="relu")(head_model)
head_model=Dropout(0.5)(head_model)
head_model=Dense(2,activation="softmax")(head_model)

#coloca o modelo em cima do basico onde sera utilizado como o real para treinamento
model=Model(inputs=base_model.input, outputs=head_model)

#loop sobre todas as camadas e freeza para nao ser atualizado durante o primeiro treinamento
for layer in base_model.layers:
    layer.trainable=False

#compila o modelo
print("[CRITICAL INFO] Compilando o modelo...")
opt=Adam(lr=INIT_LR,decay=INIT_LR/EPOCHS)
model.compile(loss="binary_crossentropy",optimizer=opt, metrics="accuracy")

#Treinando rede
print("[CRITICAL INFO] Treinando rede...")
H=model.fit(aug.flow(trainX,trainY,batch_size=BS),steps_per_epoch=len(trainX)//BS,validation_data=(testX,testY),
            validation_steps=len(testX)//BS,epochs=EPOCHS)


print("[CRITICAL INFO] Mapeando perda e precisao...")
predIdxs = model.predict(testX, batch_size=BS)
predIdxs = np.argmax(predIdxs, axis=1)
print(classification_report(testY.argmax(axis=1), predIdxs,	target_names=lb.classes_))

print("[CRITICAL INFO] Salvando modelo...")
model.save(args["model"], save_format="h5")

N = EPOCHS
plt.style.use("ggplot")
plt.figure()
plt.plot(np.arange(0, N), H.history["loss"], label="loss_of_training")
plt.plot(np.arange(0, N), H.history["val_loss"], label="loss_of_validation")
plt.plot(np.arange(0, N), H.history["accuracy"], label="training_accuracy")
plt.plot(np.arange(0, N), H.history["val_accuracy"], label="validation_accuracy")
plt.title("PERDA E PRECISÃO DO TREINAMENTO")
plt.xlabel("Período #")
plt.ylabel("Perda/Precisão")
plt.legend(loc="lower left")
plt.savefig(args["plot"])

#salvar em pdf a qualidade fica melhor
#https://futurestud.io/tutorials/matplotlib-save-plots-as-file

print("[INFO] Treinamento finalizado.")