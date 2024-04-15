import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.neighbors import LocalOutlierFactor
from uniqed.runners.tof_run import detect_outlier
import tensorflow as tf
from keras import layers, Model, Input, backend as be
import matplotlib.pyplot as plt

print("Hello, TDK")

## Adatok beolvasása
# Az adattáblák elérési útja kicserélhető letöltés után a saját elérési útra
cgm_path = "/Users/dezsenyigyorgy/Desktop/tdk-lstm/data/HDeviceCGM.txt"
bgm_path = "/Users/dezsenyigyorgy/Desktop/tdk-lstm/data/HDeviceBGM.txt"
patients_path = "/Users/dezsenyigyorgy/Desktop/tdk-lstm/data/HPtRoster.txt"

df_CGM = pd.read_csv(cgm_path, sep= '|')
print('CGM adatok beolvasva')

df_BGM = pd.read_csv(bgm_path, sep='|')
print('BGM adatok beolvasva')

df_patients = pd.read_csv(patients_path, sep= '|')
print('Páciens adatok beolvasva')

## Dátummező kialakítása az adattáblák rendezéséhez

def preprocess (df, base_date = pd.Timestamp('2015-05-22')):
    df['DeviceDateCombined'] = pd.to_timedelta(df['DeviceDtTmDaysFromEnroll'], unit='D') + base_date
    df['DeviceDateCombined'] += pd.to_timedelta(df['DeviceTm'].astype(str))
    return df

## Rendezés előkészítése

df_CGM = preprocess(df_CGM)
df_BGM = preprocess(df_BGM)

## Rendezés (páciens azonosítója alapján)

def order_by (df, column1 = 'PtID', column2='DeviceDateCombined'):
    df = df.sort_values([column1, column2])
    return df

df_CGM = order_by(df_CGM)
df_BGM = order_by(df_BGM)

print('Rendezés kész!')

## Használt oszlopok kiválasztása
cgm_selectedColumns = df_CGM[['PtID', 'DeviceDateCombined', 'GlucoseValue']]
bgm_selectedColumns = df_BGM[df_BGM['RecordType']== 'BGM'][['PtID', 'DeviceDateCombined','GlucoseValue']]

## Betegek szeparációja vizsgálati csoport alapján
df_cgmBgmPatients = df_patients[df_patients['TrtGroup'] == 'CGM+BGM']['PtID'].unique()

## Adatok normalizálása
scaler = MinMaxScaler()
cgm_selectedColumns['GlucoseValueNormalized'] = scaler.fit_transform(cgm_selectedColumns[['GlucoseValue']].values.reshape(-1,1))
bgm_selectedColumns['GlucoseValueNormalized'] = scaler.fit_transform(bgm_selectedColumns[['GlucoseValue']].values.reshape(-1,1))

## Idősorok előállítása betegenként külön-külön
def create_sequences(data, sequence_length):
    sequences = []
    for i in range(len(data) - sequence_length):
        seq = data[i:(i + sequence_length)]
        sequences.append(seq)
    return np.array(sequences)

sequence_length = 6

patients_sequences = {}

for patient_id in cgm_selectedColumns['PtID'].unique():
    patient_data = cgm_selectedColumns[cgm_selectedColumns['PtID'] == patient_id]['GlucoseValueNormalized'].values
    # Adathalmaz csökkentése
    slice_index = int(len(patient_data)*0.1)
    patient_data_reduced = patient_data[:slice_index]
    
    patients_sequences[patient_id] = create_sequences(patient_data_reduced, sequence_length)

print('Sorok legenerálva!')

## Splittelés
train_test_data = {}

for patient_id, sequences in patients_sequences.items():
    sequences = np.array(sequences)

    X_train, X_test = train_test_split(sequences, test_size=0.2, random_state=42)

    train_test_data[patient_id] = {'X_train': X_train, 'X_test': X_test}

print('Splittelés kész!')

## MARD értékek számolása
def find_nearest_cgm(bgm_timestamp, cgm_timestamps):
    """Találjuk meg a legközelebbi CGM értéket (időérték alapján)"""
    deltas = np.abs(cgm_timestamps - bgm_timestamp)
    return np.argmin(deltas)

# Időbélyegek átalakítása összehasonlítható formátumba
cgm_selectedColumns['TimestampComparable'] = pd.to_datetime(cgm_selectedColumns['DeviceDateCombined']).astype('int64') // 1e9
bgm_selectedColumns['TimestampComparable'] = pd.to_datetime(bgm_selectedColumns['DeviceDateCombined']).astype('int64') // 1e9

mard_values_normalized = {}

for patient_id in df_cgmBgmPatients:
    cgm_data = cgm_selectedColumns[cgm_selectedColumns['PtID'] == patient_id]
    bgm_data = bgm_selectedColumns[bgm_selectedColumns['PtID'] == patient_id]
    
    matched_cgm_values = []
    
    for bgm_row in bgm_data.itertuples():
        nearest_cgm_index = find_nearest_cgm(bgm_row.TimestampComparable, cgm_data['TimestampComparable'].values)
        matched_cgm_values.append(cgm_data.iloc[nearest_cgm_index]['GlucoseValueNormalized'])
    
    # MARD kiszámítása
    differences = np.abs(bgm_data['GlucoseValueNormalized'].values - matched_cgm_values)
    mard = np.mean(differences)
    mard_values_normalized[patient_id] = mard

print('MARD értékek kiszámolva!')

## Temporal Outlier Factor és Local Outlier Factor
def tof_lof(dataFrame, indices):
    # TOF
    tof_values= pd.DataFrame(dataFrame)
    results_df = detect_outlier(tof_values, cutoff_n=1, in_percent=True)
    # LOF
    clf = LocalOutlierFactor(n_neighbors=5)
    predictions = clf.fit_predict(dataFrame)

    # TOF plottolása
    fig_TOF, axs_TOF = plt.subplots(2, 1, sharex=True)
    fig_TOF.suptitle('TOF anomaly detection')

    axs_TOF[0].plot(results_df[0], color='blue', label='CGM data')
    axs_TOF[0].plot(results_df.query("TOF==1")[0], lw=0, marker='o',
         color='orange', label='TOF detections')
    axs_TOF[0].set_ylabel('values')
    axs_TOF[0].legend(loc='upper left', framealpha=1)


    axs_TOF[1].plot(results_df['TOF_score'], color='k', label='TOF score')
    axs_TOF[1].plot(results_df.query("TOF==1")['TOF_score'], lw=0, marker='o',
         color='orange', label='TOF')
    axs_TOF[1].set_ylabel('TOF score')
    axs_TOF[1].set_xlabel('t')
    axs_TOF[1].legend(['TOF score', 'TOF detections'],
              loc='upper left',
              framealpha=1)

    axs_TOF[0].grid(True)
    axs_TOF[1].grid(True)

    fig_TOF.tight_layout(rect=[0, 0, 1, 1], pad=1, h_pad=0, w_pad=0)
    plt.show()
    
    # LOF plottolása
    predictions_df = pd.DataFrame(predictions)
    anomalies_lof = predictions_df[0]==-1
    
    plt.plot(tof_values, color='black', label='CGM data')
    plt.scatter(indices[anomalies_lof], tof_values[anomalies_lof], color='orange', label='Anomalies', edgecolors='black')
    plt.xlabel('Data point index')
    plt.ylabel('CGM value')
    plt.legend()
    plt.show()

    
    

## Modell építése
# Mintavételi réteg
class Sampling(layers.Layer):
    def call(self, inputs):
        z_mean, z_log_var = inputs
        batch = tf.shape(z_mean)[0]
        dim = tf.shape(z_mean)[1]
        epsilon = tf.random.normal(shape=(batch, dim))
        return z_mean + tf.exp(0.5 * z_log_var) * epsilon

# Rekonstrukciós hiba komponens    
class ReconstructionLossLayer(layers.Layer):
    def call(self, inputs):
        encoder_inputs, vae_outputs = inputs
        reconstruction_loss = tf.reduce_mean(tf.reduce_sum(tf.keras.losses.mean_squared_error(encoder_inputs, vae_outputs), axis=1))
        self.add_loss(reconstruction_loss)
        return vae_outputs

def lstm_vae_model(input_shape, latent_dim=2):
    # Enkóder
    encoder_inputs = Input(shape=input_shape)
    x = layers.LSTM(64, return_sequences=True, activation='relu')(encoder_inputs)
    x = layers.LSTM(32, activation='relu')(x)
    z_mean = layers.Dense(latent_dim)(x)
    z_log_var = layers.Dense(latent_dim)(x)
    z = Sampling()([z_mean, z_log_var])
    encoder = Model(encoder_inputs, [z_mean, z_log_var, z], name="encoder")

    # Dekóder
    latent_inputs = Input(shape=(latent_dim,), name='z_sampling')
    x = layers.RepeatVector(input_shape[0])(latent_inputs)
    x = layers.LSTM(32, return_sequences=True, activation='relu')(x)
    x = layers.LSTM(64, return_sequences=True, activation='relu')(x)
    decoder_outputs = layers.TimeDistributed(layers.Dense(input_shape[1]))(x)
    decoder = Model(latent_inputs, decoder_outputs, name="decoder")

    # VAE
    vae_outputs = decoder(encoder(encoder_inputs)[2])
    vae_outputs_with_loss = ReconstructionLossLayer()([encoder_inputs, vae_outputs])
    vae = Model(encoder_inputs, vae_outputs_with_loss, name='lstm_vae')

    vae.compile(optimizer = 'adam')  
    return vae

input_shape = (6, 1)  # Idősorok dimenziói
latent_dim = 2  

## Modell tanítása betegenként perszonalizálva
models_by_patient = {}
for patient_id in train_test_data.keys():
    print(f"Model tanítása {patient_id} azonosítójú beteg számára...")
    vae = lstm_vae_model(input_shape, latent_dim=latent_dim)
    
    X_train = train_test_data[patient_id]['X_train']
    X_test = train_test_data[patient_id]['X_test']
    
    vae.fit(X_train, X_train, epochs=12, batch_size=18, validation_split=0.2)
    
    models_by_patient[patient_id] = vae

    # Anomália-detekció
    reconstructed_X_test = np.squeeze(vae.predict(X_test))
    reconstruction_errors = np.mean(np.power(X_test - reconstructed_X_test, 2), axis=-1)

    mard_values_mse = np.mean(np.square(np.array(list(mard_values_normalized.values()))))

    threshold_basic = np.m