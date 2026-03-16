import pandas as pd
import glob

print("EVENTS:")
event_files = glob.glob("C:\\Data Cup\\processed data\\*_Events_Cleaned.csv")
if event_files:
    try:
        df_e = pd.read_csv(event_files[0], nrows=5)
        print(df_e.columns.tolist())
    except Exception as e:
        print(e)
else:
    print("No events file found.")

print("\nTRACKING:")
tracking_files = glob.glob("C:\\Data Cup\\processed data\\*_Tracking_Cleaned.csv")
if tracking_files:
    try:
        df_t = pd.read_csv(tracking_files[0], nrows=5)
        print(df_t.columns.tolist())
    except Exception as e:
        print(e)
else:
    print("No tracking file found.")
