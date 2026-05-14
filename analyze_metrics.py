import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

LOG_FILE = "logs/monitoring_events.csv"

def generate_graphs():
    if not os.path.exists(LOG_FILE):
        print(f"[!] Log file {LOG_FILE} not found. Run the main script first to log some events.")
        return

    # Read the CSV with correct headers since the CSV lacks them
    df = pd.read_csv(LOG_FILE, header=None, names=["timestamp", "event_type", "person_name", "item_class", "extra1", "extra2"])
    
    # Ensure there is data
    if df.empty:
        print("[!] No data in the log file.")
        return

    # Convert timestamp string to datetime
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    
    print("[INFO] Generating Event Distribution Graph...")
    plt.figure(figsize=(8, 6))
    
    # 1. Bar Chart of Event Types
    event_counts = df['event_type'].value_counts()
    sns.barplot(x=event_counts.index, y=event_counts.values, palette="viridis")
    plt.title("Distribution of System Events (Entry, Exit, Lost Item)")
    plt.xlabel("Event Type")
    plt.ylabel("Count")
    for i, count in enumerate(event_counts.values):
        plt.text(i, count + 0.1, str(count), ha='center')
    
    plt.tight_layout()
    plt.savefig("logs/event_distribution_chart.png")
    plt.show()

    # 2. Pie Chart of Persons Recognized
    print("[INFO] Generating Person Tracking Graph...")
    plt.figure(figsize=(8, 6))
    
    person_counts = df['person_name'].value_counts()
    plt.pie(person_counts.values, labels=person_counts.index, autopct='%1.1f%%', startangle=140, colors=sns.color_palette("pastel"))
    plt.title("Frequency of Identified Persons (Including Unknowns)")
    plt.tight_layout()
    plt.savefig("logs/person_frequency_pie.png")
    plt.show()

    # 3. Bar chart showing what kind of items were left behind
    left_behind_df = df[df['event_type'] == 'ITEM_LEFT_BEHIND']
    if not left_behind_df.empty:
        print("[INFO] Generating Lost Items Graph...")
        plt.figure(figsize=(8, 6))
        item_counts = left_behind_df['item_class'].value_counts()
        sns.barplot(x=item_counts.index, y=item_counts.values, palette="magma")
        plt.title("Types of Items Left Behind")
        plt.xlabel("Item Class")
        plt.ylabel("Count")
        for i, count in enumerate(item_counts.values):
            plt.text(i, count + 0.1, str(count), ha='center')
        
        plt.tight_layout()
        plt.savefig("logs/items_left_behind_chart.png")
        plt.show()
    else:
        print("[INFO] No 'ITEM_LEFT_BEHIND' events found. Skipping lost items graph.")

    print("\n[SUCCESS] Graphs generated and saved to the 'logs/' folder!")
    print("Add these PNG files to your project presentation/report.")

if __name__ == "__main__":
    generate_graphs()