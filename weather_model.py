import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras import regularizers
from tensorflow.keras.metrics import Precision, Recall, AUC
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import confusion_matrix, classification_report
import matplotlib.pyplot as plt
import seaborn as sns

#daten einlesen
data = pd.read_csv('weatherAUS.csv')
#print(data.head())


#daten aufräumen!!!--------------------------------------------------------------------------------
#alle zeilen mit keinem GroundTruth-Wert werden entfernt
data = data.dropna(subset=['RainTomorrow'])

#spalten mit stringwerten entfernen
string_spalten = ["Location", "WindGustDir","WindDir9am","WindDir3pm"]
data = data.drop(columns=string_spalten)

#spalten mit zu vielen fehlenden werten entfernen
na_spalten = ["Sunshine", "Evaporation", "Cloud9am", "Cloud3pm"]
data = data.drop(columns=na_spalten)
#print(data.head())

# was machen mit anderen nan-werten?
    # https://www.geeksforgeeks.org/machine-learning/handling-missing-values-machine-learning/

data = data.fillna(data.median(numeric_only=True))
#print(data.head(20))

#konvertiere GroundTruth + RainToday in 0 und 1
data['RainTomorrow'] = data['RainTomorrow'].map({'No': 0, 'Yes': 1})
data['RainToday'] = data['RainToday'].map({'No': 0, 'Yes': 1})

#avg_rain_today = data['RainToday'].mean()
#print (f"Durchschnittlicher Wert von RainToday: {avg_rain_today}") 

# avg_rain_today = 0.223 --> auffüllen der fehlenden Werte mit 0
data['RainToday'] = data['RainToday'].fillna(0)

# Datum zu Month und Day umwandeln
data['Date'] = pd.to_datetime(data['Date'], format='%Y-%m-%d')

# in Datensatz hinzufügen
data['Month'] = data['Date'].dt.month.astype(np.float32)  
data['Day'] = data['Date'].dt.day.astype(np.float32)      

data = data.drop(columns=['Date'])

# alle werte zu float32 umwandeln
data = data.astype(np.float32)
#print(data.dtypes)
#print(data.head())





# dataset aufspalten in features und GroundTruth
X = data.drop(columns=['RainTomorrow'])
y = data['RainTomorrow']

#check
print(X.isnull().sum())
print(y.isnull().sum())
print(X.shape)
print(y.shape)

#--------------------------------------------------------------------------------------------------

#train und test set aufteilen
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.1, random_state=42)
# erste wahl für test_size = 0.1, da datensatz groß

#daten skalieren

#https://www.geeksforgeeks.org/deep-learning/implementing-neural-networks-using-tensorflow/
scaler = StandardScaler()
numerical_columns = X_train.columns.tolist()

#unterrepräsentierte klasse wird höher gewichtet (~77% kein regen, 23% regen)
X_train[numerical_columns] = scaler.fit_transform(X_train[numerical_columns])
X_val[numerical_columns] = scaler.transform(X_val[numerical_columns])

class_weights = compute_class_weight('balanced', classes=np.unique(y_train), y=y_train)
class_weight_dict = {0: class_weights[0], 1: class_weights[1]}

#--------------------------------------------------------------------------------------------------

#lernrate anpassen (von AI)
def cosine_decay(epoch):
    initial_lrate = 0.01
    final_lrate = 1e-6
    max_epochs = 100
    cosine_lr = final_lrate + 0.5 * (initial_lrate - final_lrate) * (
        1 + np.cos(np.pi * epoch / max_epochs)
    )
    return cosine_lr

lr_scheduler = tf.keras.callbacks.LearningRateScheduler(cosine_decay, verbose=1)



#modell erstellen
model = tf.keras.Sequential([
    tf.keras.layers.Dense(64, activation='relu',kernel_regularizer=regularizers.l2(0.001), input_shape=(X_train.shape[1],)),
    tf.keras.layers.BatchNormalization(), 
    tf.keras.layers.Dense(32,kernel_regularizer=regularizers.l2(0.001), activation='relu'),
    tf.keras.layers.BatchNormalization(), 
    tf.keras.layers.Dense(16,kernel_regularizer=regularizers.l2(0.001), activation='relu'),
    tf.keras.layers.BatchNormalization(), 
    tf.keras.layers.Dense(1, activation='sigmoid')  # sigmoid für 0/1 klassifikation --> kein softmax
])
model.summary()

#modell kompilieren
model.compile(optimizer=tf.keras.optimizers.legacy.Adam(learning_rate=0.01), # Adam optimizer legacy weil Apple Sicilicon M1/M2 besser auf den alten optimizern läuft
                loss='binary_crossentropy',  # binary_crossentropy = logistic loss für 0/1 klassifikation
                metrics=[
                    'accuracy',
                    tf.keras.metrics.Precision(name='precision'),
                    tf.keras.metrics.Recall(name='recall'),
                    tf.keras.metrics.AUC(name='auc')
    ])

#modell trainieren
ergebnisse = model.fit(
    X_train, y_train,
    epochs=100,                 
    batch_size=256,              
    validation_data=(X_val, y_val),
    verbose=0,
    class_weight=class_weight_dict,
    callbacks=[
        lr_scheduler , # ← Cosinus Learing rate decay
        tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=20, restore_best_weights=True)
    ]
    )



plt.figure(figsize=(15, 5))

plt.subplot(1, 3, 1)
plt.plot(ergebnisse.history['loss'], label='train_loss')
plt.plot(ergebnisse.history['val_loss'], label='val_loss')
plt.title('Loss')
plt.legend()

plt.subplot(1, 3, 2)
plt.plot(ergebnisse.history['accuracy'], label='train_acc')
plt.plot(ergebnisse.history['val_accuracy'], label='val_acc')
plt.title('Accuracy')
plt.legend()


plt.tight_layout()
plt.show()

print("validation set ergebnisse:")
test_results = model.evaluate(X_val, y_val, verbose=0)
print(f"Test Accuracy: {test_results[1]:.3f}")
print(f"Test Precision:{test_results[2]:.3f}")
print(f"Test Recall:   {test_results[3]:.3f}")
print(f"Test AUC:      {test_results[4]:.3f}")


y_pred = (model.predict(X_val) > 0.5).astype(int)


#von AI 
# Confusion Matrix berechnen und plotten
cm = confusion_matrix(y_val, y_pred)
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
            xticklabels=['Kein Regen', 'Regen'], 
            yticklabels=['Kein Regen', 'Regen'])
plt.title('Confusion Matrix\n(AUC=0.859)')
plt.ylabel('True Label')
plt.xlabel('Predicted Label')
plt.tight_layout()
plt.show()

# Zusätzlich: Absolute Zahlen
print("Confusion Matrix (TP/FP/FN/TN):")
tn, fp, fn, tp = cm.ravel()
print(f"TN: {tn:>6} | FP: {fp:>6}")
print(f"FN: {fn:>6} | TP: {tp:>6}")

print(X.shape)
print(y.shape)
