import pandas as pd
import os
import json
import ast

from config import LOCATIE_REQUESTS, LOCATIE_RESPONSES, RUWE_DATA_PARQUET, RUWE_RESPONSES_PARQUET

def parse_filename(file_name) -> tuple:
    try:
        # rsplit verwijdert alleen de laatste .json extensie (voor mocht het bestand meerdere .json bevatten alhoewel dit onwaarschijnlijk lijkt)
        parts = file_name.rsplit('.json', 1)[0].split('-')
        if len(parts) >= 5:
            route_id = parts[0]
            date = parts[1]
            time = parts[2]
            number_of_tasks = parts[3]
            number_of_tasks_in_input_plan = parts[4]
            return route_id, date, time, number_of_tasks, number_of_tasks_in_input_plan
        else:
            print(f"Skipping file {file_name}: incorrect format")
            return None
    except Exception as e:
        print(f"Error parsing file name {file_name}: {e}")
        return None
    
    # check if parquet bestand bestaat
def check_parquet_bestand(bestandspadennaam) -> bool:
    if os.path.exists(bestandspadennaam):
        return True
    return False

# lees parquet bestand in als dataframe
def lees_dataframe_uit_parquet(bestandspadennaam) -> pd.DataFrame:
    df = pd.read_parquet(bestandspadennaam)  
    return df

# maak parquet bestand aan vanuit dataframe
def schrijf_dataframe_naar_parquet(df: pd.DataFrame, bestandspadennaam) -> None:
    # Maak de directory aan als deze niet bestaat
    os.makedirs(os.path.dirname(bestandspadennaam), exist_ok=True)
    # Schrijf het dataframe naar Parquet
    df.to_parquet(bestandspadennaam, index=False)
    return None

# de verschillende json-bestanden van de requests inladen en samenvoegen tot 1 grote dataframe 
# geeft een dataframe terug als resultaat
def lees_json_bestanden_en_maak_dataframe(locatie_requests) -> pd.DataFrame:
    df = pd.DataFrame()
    for folder_name in os.listdir(locatie_requests):
        folder_path = os.path.join(locatie_requests, folder_name)
        if os.path.isdir(folder_path):
            for file_name in os.listdir(folder_path):
                if file_name.endswith('.json'):
                    parsed_data = parse_filename(file_name)
                    if parsed_data == None:
                        break
                    else:
                        route_id, date, time, number_of_tasks, number_of_tasks_in_input_plan = parsed_data
                        # print(route_id, date, time, number_of_tasks, number_of_tasks_in_input_plan)
                        file_path = os.path.join(folder_path, file_name)
                        with open(file_path, 'r') as f:
                            data = json.load(f)
                            data['route_id'] = route_id
                            data['date'] = date
                            data['time'] = time
                            data['number_of_tasks'] = int(number_of_tasks)
                            data['number_of_tasks_in_input_plan'] = int(number_of_tasks_in_input_plan)
                            temp_df = pd.DataFrame([data])
                            # voeg tijdelijke dataframe toe aan de hoofddataframe
                            df = pd.concat([df, temp_df], ignore_index=True)
    
    return df


# de verschillende txt-bestanden van de responses inladen en samenvoegen tot 1 grote dataframe 
# geeft een dataframe terug als resultaat
def lees_txt_bestanden_en_maak_dataframe(locatie_responses) -> pd.DataFrame:
    df = pd.DataFrame()
    for folder_name in os.listdir(locatie_responses):
        folder_path = os.path.join(locatie_responses, folder_name)
        if os.path.isdir(folder_path):
            for file_name in os.listdir(folder_path):
                if file_name.endswith('.txt'):
                    parsed_data = parse_filename(file_name.replace('.txt', '.json'))
                    if parsed_data == None:
                        break
                    else:
                        route_id, date, time, number_of_tasks, number_of_tasks_in_input_plan = parsed_data
                        file_path = os.path.join(folder_path, file_name)
                        
                        # Lees task_ids uit txt bestand (één per regel)
                        with open(file_path, 'r') as f:
                            task_ids = [line.strip() for line in f.readlines() if line.strip()]
                        
                        # Maak een row met de metadata en de lijst van task_ids
                        data = {
                            'route_id': route_id,
                            'date': date,
                            'time': time,
                            'number_of_tasks': int(number_of_tasks),
                            'number_of_tasks_in_input_plan': int(number_of_tasks_in_input_plan),
                            'tasks_sequence': task_ids
                        }
                        temp_df = pd.DataFrame([data])
                        # voeg tijdelijke dataframe toe aan de hoofddataframe
                        df = pd.concat([df, temp_df], ignore_index=True)
    
    return df


def lees_data():
    if (check_parquet_bestand(RUWE_DATA_PARQUET) == False):
        # maak het bestand een eerste keer

        ingelezen_dataframe = lees_json_bestanden_en_maak_dataframe(LOCATIE_REQUESTS)
        # dit is dan dezelfde dataframe als in de gegeven excel file maar met tasks en fixedTasks eraan toegevoegd en zonder TriggerType
        # date en time zijn nog steeds strings, kan later nog omgezet worden naar datetime indien nodig
        schrijf_dataframe_naar_parquet(ingelezen_dataframe, RUWE_DATA_PARQUET)

        # print(f"Lengte van de dataframe: {len(ingelezen_dataframe)}")
        # # excel inlezen om de lengte te checken
        # df_excel = pd.read_excel(GEKREGEN_EXCEL_FILE)
        # print(f"Aantal rijen in ingelezen excel dataframe: {len(df_excel)}")
        # if (len(ingelezen_dataframe) != len(df_excel)):
        #     print("Waarschuwing: Aantal rijen in ingelezen dataframe komt niet overeen met aantal rijen in excel dataframe!")

    else:
        # lees het bestand en voeg toe aan dataframe om verder mee te werken
        ingelezen_dataframe = lees_dataframe_uit_parquet(RUWE_DATA_PARQUET)
        # Parquet behoudt de datatypes, dus geen conversie nodig
        # display(ingelezen_dataframe.tail())

    return ingelezen_dataframe


def lees_responses():
    if (check_parquet_bestand(RUWE_RESPONSES_PARQUET) == False):
        # maak het bestand een eerste keer

        ingelezen_dataframe = lees_txt_bestanden_en_maak_dataframe(LOCATIE_RESPONSES)
        # dataframe met route_id, date, time, number_of_tasks, number_of_tasks_in_input_plan en task_ids (als lijst)
        schrijf_dataframe_naar_parquet(ingelezen_dataframe, RUWE_RESPONSES_PARQUET)

    else:
        # lees het bestand en voeg toe aan dataframe om verder mee te werken
        ingelezen_dataframe = lees_dataframe_uit_parquet(RUWE_RESPONSES_PARQUET)
        # Parquet behoudt de datatypes, dus geen conversie nodig

    return ingelezen_dataframe
