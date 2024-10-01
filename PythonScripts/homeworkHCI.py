import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Carica il file CSV
emotions = [
    "anger",
    "disgust",
    "fear",
    "happiness",
    "sadness",
    "surprise",
    "amusement",
    "anxiety",
    "boredom",
    "interest"
]

activation_accumulation = dict((x, 0)for x in range(0,46))
print(activation_accumulation[0])
for emotion in emotions:
    csv_file_path = f"Processed/{emotion}.csv"
    df = pd.read_csv(csv_file_path)
    print("-------------------------------------------------")
    # Rimuovi spazi nei nomi delle colonne
    df.columns = df.columns.str.strip()

    # Supponiamo che il video rappresenti una delle emozioni di base


    # Definisci colonne per AU (attivazione e intensità)
    au_presence_columns = [col for col in df.columns if '_c' in col]  # Colonne di attivazione AU
    au_intensity_columns = [col for col in df.columns if '_r' in col]  # Colonne di intensità AU

    # Parte 3a: Istogramma per attivazione delle AU
    plt.figure(figsize=(10, 6))
    normalized_activations = []
    activated_aus = df[au_presence_columns].sum() > 0
    activated_aus_list = []

    for au, activated in activated_aus.items():
        if activated:
            au_number = int(au.split('_')[0][2:])
            activation_accumulation[au_number]=activation_accumulation[au_number]+1

    for au in au_presence_columns:
        normalized_activation = df[au].sum() / len(df)  # Normalizza per il numero di frame
        normalized_activations.append(normalized_activation)

    plt.bar(au_presence_columns, normalized_activations, alpha=0.7, color='b')
    plt.title('Normalized Histogram for AU activations')
    plt.xlabel('Action Units (AUs)')
    plt.ylabel('Normalized frequency activation')
    plt.xticks(rotation=90)
    plt.grid(axis='y', linestyle='--')
    plt.tight_layout()
    #plt.show()

    # Parte 3b: Istogramma per l'intensità delle AU
    plt.figure(figsize=(10, 6))
    for au in au_intensity_columns:
        au_data = df[au]
        bins = np.linspace(0, 1, 10)  # 10 intervalli uguali nell'intervallo [0, 1]
        plt.hist(au_data, bins=bins, density=True, alpha=0.5, label=au)

    plt.title("Normalized Histogram for AU Intensity activations")
    plt.xlabel('Intensity level')
    plt.ylabel('Normalized frequence')
    plt.legend(loc='upper right', bbox_to_anchor=(1.2, 1))
    plt.grid(axis='y', linestyle='--')
    plt.tight_layout()
    #plt.show()

    # Parte 4: Confronto delle AU attivate con la tabella delle emozioni

    # Riferimenti letteratura per le AU attese per ciascuna emozione:
    # 1. Amusement: AU 6 (cheek raiser) e AU 12 (lip corner puller) sono associati al divertimento. 
    #    - Ekman, P., & Friesen, W. V. (1980). The Facial Action Coding System. Consulting Psychologists Press.
    #    - Keltner, D. (1995). Journal of Personality and Social Psychology, 68(3), 441.
    # 2. Anxiety: AU 1 (inner brow raiser), AU 4 (brow lowerer), AU 5 (upper lid raiser), AU 20 (lip stretcher).
    #    - Cohn et al. (2002). Psychophysiology, 39(3), 322-328.
    #    - Ekman, P. (1999). Handbook of Cognition and Emotion. John Wiley & Sons.
    # 3. Boredom: AU 14 (dimpler), AU 15 (lip corner depressor), AU 43 (eye closure).
    #    - Hertenstein, M. J., & Keltner, D. (2009). Neuroscience & Biobehavioral Reviews, 34(1), 66-75.
    # 4. Interest: AU 1, 2, 5, 12 indicano interesse.
    #    - Tomasello et al. (2007). Behavioral and Brain Sciences, 28(5), 675-691.
    #    - Ekman, P., & Friesen, W. V. (1978). Facial Action Coding System: Investigator’s Guide.

    au_table = {
        "anger": [4, 5, 7, 10, 17, 22, 26],
        "disgust": [9, 10, 16, 17, 25, 26],
        "fear": [1, 2, 4, 5, 20, 25, 26, 27],
        "happiness": [6, 12, 25],
        "sadness": [1, 4, 6, 11, 15, 17],
        "surprise": [1, 2, 5, 26, 27],
        "pain": [4, 6, 7, 9, 10, 12, 20, 25, 26, 27, 43],
        "cluelessness": [1, 2, 5, 15, 17, 22],
        "speech": [10, 14, 16, 17, 18, 20, 22, 26, 28],
        "amusement": [6, 12, 25],
        "anxiety": [1, 4, 5, 20],
        "boredom": [14, 15, 43],
        "interest": [1, 2, 5, 12]
    }


    expected_aus = au_table[emotion]

    # Determina quali AU sono attivate
    activated_aus = df[au_presence_columns].sum() > 0
    activated_aus_list = []

    for au, activated in activated_aus.items():
        if activated:
            au_number = int(au.split('_')[0][2:])
            activated_aus_list.append(au_number)

    # Confronto delle AU attivate con quelle attese
    print(f"Emotion check: {emotion}")
    print(f"Expected AUs: {expected_aus}")
    print(f"Activated AUs in the video: {activated_aus_list}")

    matched_aus = [au for au in activated_aus_list if au in expected_aus]
    extra_aus = [au for au in activated_aus_list if au not in expected_aus]
    missing_aus = [au for au in expected_aus if au not in activated_aus_list]

    print(f"Corrisponding AU: {matched_aus}")
    print(f"Extra AUs (Unexpected but activated): {extra_aus}")
    print(f"Missing AUs (Expected but not activated): {missing_aus}")

    # Parte 5: Grafico dell'intensità delle AU nel tempo per l'emozione selezionata
    plt.figure(figsize=(10, 6))
    for au in au_intensity_columns:
        plt.plot(df.index, df[au], label=au)

    plt.title(f'AU intesity in time for {emotion}')
    plt.xlabel('Frame')
    plt.ylabel('Intensity')
    plt.legend(loc='upper right', bbox_to_anchor=(1.2, 1))
    plt.grid()
    plt.tight_layout()
    #plt.show()

    # Parte 6: Focus su amusement, anxiety, boredom, interest
    focus_emotions = ["amusement", "anxiety", "boredom", "interest"]
""" 
    for focus_emotion in focus_emotions:
        # Confronto per ogni emozione di interesse
        print(f"-----------------------------------------------------\nAnalisys for the expression: {focus_emotion}")
        expected_aus = au_table.get(focus_emotion, [])

        print(f"Expected AU in the video for  {focus_emotion}: {expected_aus}")
        print(f"Activated AU in the video for: {activated_aus_list}")

        matched_aus = [au for au in activated_aus_list if au in expected_aus]
        extra_aus = [au for au in activated_aus_list if au not in expected_aus]
        missing_aus = [au for au in expected_aus if au not in activated_aus_list]

        print(f"Expected AU: {matched_aus}")
        print(f"Extra AUs (Unexpected but activated): {extra_aus}")
        print(f"Missing AUs (Expected but not activated): {missing_aus}") """


print(activation_accumulation)