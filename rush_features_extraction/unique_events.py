import pandas as pd
import glob
with open('unique_events.txt', 'w') as f:
    event_files = glob.glob("C:\\Data Cup\\processed data\\*_Events_Cleaned.csv")
    if event_files:
        df = pd.read_csv(event_files[0])
        events = df['Event'].unique().tolist()
        for e in events:
            f.write(str(e) + "\n")
