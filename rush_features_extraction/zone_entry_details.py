import pandas as pd
import glob
with open('zone_entry_details.txt', 'w') as f:
    event_files = glob.glob("C:\\Data Cup\\processed data\\*_Events_Cleaned.csv")
    if event_files:
        df = pd.read_csv(event_files[0])
        entries = df[df['Event'] == 'Zone Entry']
        f.write("Detail_1 unique:\n")
        f.write(str(entries['Detail_1'].unique()) + "\n")
