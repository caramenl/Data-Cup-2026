import pandas as pd
import glob

with open('C:\\Data Cup\\processed data\\schema_output.txt', 'w') as f:
    f.write("EVENTS:\n")
    event_files = glob.glob("C:\\Data Cup\\processed data\\*_Events_Cleaned.csv")
    if event_files:
        try:
            df_e = pd.read_csv(event_files[0], nrows=5)
            f.write(str(df_e.columns.tolist()) + "\n")
        except Exception as e:
            f.write(str(e) + "\n")

    f.write("\nTRACKING:\n")
    tracking_files = glob.glob("C:\\Data Cup\\processed data\\*_Tracking_Cleaned.csv")
    if tracking_files:
        try:
            df_t = pd.read_csv(tracking_files[0], nrows=5)
            f.write(str(df_t.columns.tolist()) + "\n")
        except Exception as e:
            f.write(str(e) + "\n")
