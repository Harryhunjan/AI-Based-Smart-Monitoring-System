import pandas as pd
import os
from datetime import datetime

class EventLogger:
    def __init__(self, log_dir="logs", filename="monitoring_events.csv"):
        self.log_dir = log_dir
        self.log_path = os.path.join(log_dir, filename)
        self.columns = ["Timestamp", "Event", "Person_Name", "Item_Class", "Confidence"]
        
        if not os.path.exists(self.log_dir):
            os.makedirs(self.log_dir)
            
        if not os.path.exists(self.log_path):
            df = pd.DataFrame(columns=self.columns)
            df.to_csv(self.log_path, index=False)
            
    def log_event(self, event_type, person_name="Unknown", item_class="None", confidence=""):
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        new_row = {
            "Timestamp": timestamp,
            "Event": event_type,
            "Person_Name": person_name,
            "Item_Class": item_class,
            "Confidence": confidence
        }
        
        # Open in append mode
        df = pd.DataFrame([new_row])
        df.to_csv(self.log_path, mode='a', header=False, index=False)
        print(f"[LOG] {timestamp} | {event_type} | Person: {person_name} | Item: {item_class}")
