
import sys
import time
import re
import traceback
import os

import requests
import pandas as pd
from joblib import load
import numpy as np
import random

import sys
from PyQt6.QtWidgets import QApplication, QLabel, QMainWindow
from PyQt6.QtGui import QPixmap
from PyQt6.QtCore import Qt

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = BASE_DIR.strip("src\python") + '\model\\rf_model.sav'

URL_FETCH = "http://192.168.4.1/"
URL_DELETE = f"{URL_FETCH}delete"

TIMESTEPS = 20
NUM_BOARDS = 4
SENSORS = ['A', 'G']
AXES = ['x', 'y', 'z']

EXPECTED_ESP_IDS = [str(i + 1) for i in range(NUM_BOARDS)]
EXPECTED_DATA_COLUMNS = len(SENSORS) * len(AXES) * NUM_BOARDS

MAX_FETCH_ATTEMPTS = 2
RETRY_DELAY_SECONDS = 2
counter = 0
counter2 = 0

def export_(f, m):
    map = m.readlines()
    map = [line.strip() for line in map]
    kl = []
    for i in f:
        kicks = i.strip()
        for j in kicks:
            kl.append(map[int(j)])
    
    return kl

class UnsufficientSamples(Exception):
    """Exception raised when the number of samples received is less than expected"""
    def __init__(self, e_samples, r_samples, msg="Samples received is less than expected"):
        self.message = (
            f"{msg}: Received {r_samples}, Expected {e_samples}. "
        )
        super().__init__(self.message)

class UncompleteData(Exception):
    """Exception raised when at least one of the ESP32 didn't send any data"""
    def __init__(self, e_ids, r_ids, msg=" of the ESP32 didn't send any data"):
        self.empty_ids = [esp_id for esp_id in e_ids if esp_id not in r_ids]
        self.message = (
            f"{len(self.empty_ids)}{msg}. IDs of the boards: {self.empty_ids}. "
        )
        super().__init__(self.message)

class DataProcessor:
    def __init__(self, url_delete, timesteps, num_boards, sensors, axes, expected_esp_ids):
        self.url_delete = url_delete
        self.timesteps = timesteps
        self.num_boards = num_boards
        self.sensors = sensors
        self.axes = axes
        self.expected_esp_ids = expected_esp_ids

    def generate_column_names(self):
        column_names = []
        for board_id in range(1, self.num_boards + 1):
            for sensor_type in self.sensors:
                for axis in self.axes:
                    col_name = f"{sensor_type}{board_id}{axis}"
                    column_names.append(col_name)
        return column_names

    def parse(self, raw_data):
        number_pattern = r'-?\d+\.?\d*'
        esp_data = []
        
        # Parses all data and stores them in a list
        for esp_id in self.expected_esp_ids:
            last_start_idx = raw_data.rfind(f"Start{esp_id};")
            last_end_idx = raw_data.find(f"End{esp_id};", last_start_idx + len(f"Start{esp_id};"))
            if last_end_idx != -1 and last_start_idx != -1:
                esp_block = raw_data[last_start_idx + len(f"Start{esp_id};") : last_end_idx]
                esp_block = esp_block.replace(f"ID{esp_id}", "")
                esp_data.append([float(num_str) for num_str in re.findall(number_pattern, esp_block)])
            else: esp_data.append([])

        return esp_data
    
    def format(self, esp_data):
        # Verifies data completeness and number of samples
        active_esp_ids = [str(esp_id + 1) for esp_id in range(len(esp_data)) if esp_data[esp_id] != []]
        samples_counts = [(len(esp_data[int(esp_id) - 1]) / 6) for esp_id in active_esp_ids]
        min_samples = int(min(samples_counts))
        if self.timesteps > 0 and len(samples_counts) != self.num_boards: 
            raise UncompleteData(e_ids=self.expected_esp_ids, r_ids=active_esp_ids)
        elif min_samples < self.timesteps:
            raise UnsufficientSamples(e_samples=self.timesteps, r_samples=min_samples)

        # Eliminates excessive data
        for esp in range(self.num_boards):
            data_to_eliminate = int(-6 * (samples_counts[int(esp)] - min_samples) - 1)
            del esp_data[esp][-1:data_to_eliminate:-1]

        # Conversion in list of lists format, where each sublist contains a single sample
        data = []
        for i in range(min_samples):
            sample = []
            for j in range(self.num_boards):
                start = i * len(self.sensors) * len(self.axes)
                end = start + len(self.sensors) * len(self.axes)
                sample.extend(esp_data[j][start:end])
            data.append(sample)

        return data
    
    def delete_data_on_master(self):
        try:
            #print("Sending DELETE request at", self.url_delete)
            response = requests.get(self.url_delete, timeout=5)
            if response.status_code == 200 and "OK" in response.text:
                return True
            else:
                print(f"Error sending DELETE command: Status {response.status_code}, Response: {response.text}")
                return False
        except requests.exceptions.Timeout:
            print(f"Timeout during DELETE request at {self.url_delete}")
            return False
        except requests.exceptions.RequestException as e:
            print(f"Error during communication with Master (DELETE): {e}")
            return False
        except Exception as e:
            print(f"Generic error during DELETE: {e}")
            return False

class Predictor:
    def __init__(self, model, url_fetch, max_fetch_attempts, retry_delay_seconds, counter, counter2, data_processor: DataProcessor):
        self.model = model
        self.data_processor = data_processor
        self.columns = self.data_processor.generate_column_names()
        self._is_running = True
        self.url_fetch = url_fetch
        self.max_fetch_attempts = max_fetch_attempts
        self.retry_delay_seconds = retry_delay_seconds
        self.counter = counter
        self.counter2 = counter

    def compute_smv(self, df):
        out_list = []
        in_array = df.to_numpy()
        
        for row in in_array:
            temp_list = []
            for i in range(0, len(row), 3):
                if i + 2 < len(row):
                    smv = np.sqrt(row[i]**2 + row[i+1]**2 + row[i+2]**2)
                    temp_list.append(smv)
            out_list.append(temp_list)
        
        return np.array(out_list)
    
    def classify_samples(self, smv_array, threshold=6.5):
        smv_classified = np.array([])
        for single_sample in smv_array:
            mean = np.mean(single_sample)
            classified = "Calcio" if mean > threshold else "Fermo"
            smv_classified = np.append(smv_classified, classified)

        return smv_classified
    
    def split(self, etichette, df_dati, dimensione_gruppo=20, min_calcio_len=3):
        # 1. Identifica tutti i blocchi (Fermo e Calcio)
        blocchi = []
        if len(etichette) == 0:
            return pd.DataFrame()
            
        label_corrente, start_index = etichette[0], 0
        for i in range(1, len(etichette)):
            if etichette[i] != label_corrente:
                blocchi.append({'label': label_corrente, 'start': start_index, 'end': i})
                label_corrente, start_index = etichette[i], i
        blocchi.append({'label': label_corrente, 'start': start_index, 'end': len(etichette)})
        
        # 2. Filtra per tenere solo i blocchi di Calcio che superano la lunghezza minima
        blocchi_calcio_validi = [
            b for b in blocchi 
            if b['label'] == 'Calcio' and (b['end'] - b['start']) >= min_calcio_len
        ]

        lista_gruppi_appiattiti = []
        ultimo_indice_usato = -1

        # 3. Itera sui blocchi validi per costruire i gruppi senza sovrapposizioni
        for calcio_block in blocchi_calcio_validi:
            calcio_start = calcio_block['start']
            
            if calcio_start <= ultimo_indice_usato:
                continue

            calcio_end = calcio_block['end']
            num_calcio = calcio_end - calcio_start

            # 4. Logica per garantire che il gruppo sia SEMPRE di dimensione fissa
            if num_calcio >= dimensione_gruppo:
                # Se il blocco è troppo grande, prendiamo solo i primi `dimensione_gruppo` elementi
                start_gruppo = calcio_start
                end_gruppo = calcio_start + dimensione_gruppo
            else:
                # Altrimenti, calcola il padding necessario
                padding_necessario = dimensione_gruppo - num_calcio
                pre_padding_target = padding_necessario // 2
                
                # Limiti da cui possiamo prelevare padding
                limite_pre = ultimo_indice_usato + 1
                limite_post = len(etichette)

                pre_padding_disponibile = calcio_start - limite_pre
                post_padding_disponibile = limite_post - calcio_end
                
                # Logica di compensazione per distribuire il padding
                pre_da_prendere = min(pre_padding_disponibile, pre_padding_target)
                # Calcola quanto serve dopo, tenendo conto di quanto abbiamo già preso prima
                post_da_prendere = min(post_padding_disponibile, padding_necessario - pre_da_prendere)
                
                # Se dopo non c'era abbastanza, prova a prendere il resto da prima
                mancanti = padding_necessario - (pre_da_prendere + post_da_prendere)
                if mancanti > 0:
                    pre_da_prendere += min(mancanti, pre_padding_disponibile - pre_da_prendere)
                
                # Se ancora non si raggiunge la dimensione, il gruppo non può essere formato
                if pre_da_prendere + post_da_prendere + num_calcio < dimensione_gruppo:
                    continue

                start_gruppo = calcio_start - pre_da_prendere
                end_gruppo = calcio_end + post_da_prendere

            # 5. Estrai la fetta dal DataFrame, appiattiscila e salvala
            gruppo_df = df_dati.iloc[start_gruppo:end_gruppo]
            riga_appiattita = gruppo_df.values.flatten()
            lista_gruppi_appiattiti.append(riga_appiattita)
            
            ultimo_indice_usato = end_gruppo - 1
            
        if not lista_gruppi_appiattiti:
            return pd.DataFrame()

        return pd.DataFrame(lista_gruppi_appiattiti)

        
    def predict(self, kick_df):
        try:
            x_numpy = kick_df.values
            y_pred = self.model.predict(x_numpy)
            prediction_result = y_pred[0]
        except Exception as e_df_pred:
            print(f"Prediction Error: {e_df_pred}")
            traceback.print_exc()

        return prediction_result
    
    def run(self):
        parsed = ""
        with open('kicks.txt') as f:
            with open('map.txt') as m:
                kicks_list = export_(f, m)

        lista = [kicks_list[i:i + 3] for i in range(0, len(kicks_list), 3)]

        

        while self._is_running:
    
            data = []
            try:
                response = requests.get(self.url_fetch, timeout=10)
                response.raise_for_status()
                raw = response.text
                if raw == "aspettaciola":
                    print("Waiting for the data...")
                    continue
                elif raw == "":
                    print("Nothing there! :(")
                else:
                    parsed = self.data_processor.parse(raw)
                    #data = self.data_processor.format(parsed)

            except requests.exceptions.Timeout:
                print(f"Timeout di requests.")
                break
            except requests.exceptions.RequestException as e:
                print(f"Errore connessione/HTTP: {e}")
                time.sleep(self.retry_delay_seconds)
                continue
            except UncompleteData as e:
                print(f"Error raised while formatting: {e}")
                time.sleep(self.retry_delay_seconds) 
                continue
            except UnsufficientSamples as e: 
                print(f"Error raised while formatting: {e}")
                time.sleep(self.retry_delay_seconds)
                continue
            except ValueError as e:
                print(e)
                time.sleep(self.retry_delay_seconds)
                continue

            except Exception as e:
                print(f"Unexpected error: {e}")
                traceback.print_exc()
                self._is_running = False
            
            if parsed:
                try:
                    kick_df = pd.DataFrame(data, columns=self.columns)
                    smv = self.compute_smv(kick_df)
                    
                    smv_classified = self.classify_samples(smv)
                    splitted_df = self.split(smv_classified, kick_df)
                    self.counter2 = self.counter
                    #print("Cleaning data buffer on Master ESP...")
                    if not self.data_processor.delete_data_on_master():
                        print("WARN: Failed cleaning buffer Master.")

                    for i in lista[self.counter]: 
                        print(i)
                        time.sleep(random.uniform(1, 2))
                    time.sleep(4)
                    
                    self.counter += 1

                except Exception as e_df:
                    print(f"\nERROR during prediction: {e_df}")
                    traceback.print_exc()
                    self._is_running = False

        if self._is_running:
            time.sleep(1)
            

    def stop(self):
        print("Request to stop the worker thread...")
        self._is_running = False

if __name__ == "__main__":
    print("Loading the model...")
    try:
        model = load(MODEL_PATH)
        print("Model loaded succesfully.")
    except FileNotFoundError:
        print(f"CRITICAL ERROR: Model's file not found in {MODEL_PATH}")
        sys.exit(1)
    except Exception as e:
        print(f"CRITICAL ERROR: Unable to upload the model: {e}")
        traceback.print_exc()
        sys.exit(1)

    data_processor = DataProcessor(
        url_delete=URL_DELETE,
        timesteps=TIMESTEPS,
        num_boards=NUM_BOARDS,
        sensors=SENSORS,
        axes=AXES,
        expected_esp_ids=EXPECTED_ESP_IDS
    )

    df_columns = data_processor.generate_column_names()
    if not df_columns or len(df_columns) != EXPECTED_DATA_COLUMNS:
        print("CRITICAL ERROR: Invalid column names or wrong number.")
        sys.exit(1)

    print("Starting the process of acquisition and prediction...")
    predictor = Predictor(
        model=model, 
        data_processor=data_processor,
        url_fetch=URL_FETCH,
        max_fetch_attempts=MAX_FETCH_ATTEMPTS,
        retry_delay_seconds=RETRY_DELAY_SECONDS,
        counter=counter,
        counter2=counter2
    )

    try:
        predictor.run()
    except KeyboardInterrupt:
        print("\nKeyboard interruption detected.")
        predictor.stop()
    except Exception as e:
        print(f"Critical error in the execution of the main worker: {e}")
        traceback.print_exc()
