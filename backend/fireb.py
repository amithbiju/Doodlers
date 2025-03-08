
import firebase_admin
from firebase_admin import credentials, firestore
import csv

# Initialize Firebase
def initialize_firebase():
    """
    Initializes Firebase with the provided service account credentials.
    """
    cred = credentials.Certificate("flight-maintenance-firebase-adminsdk-fbsvc-f087e37cb5.json")
    firebase_admin.initialize_app(cred)
    global db, collection_ref, collection_ref_bill
    db = firestore.client()
    collection_ref = db.collection("items")
    collection_ref_bill = db.collection("bills")


def update():
    """
    Fetches data from Firestore collections ('items' and 'bills') and writes it to CSV files.
    """
    # Fetch all documents from the "items" collection
    docs = collection_ref.stream()
    data_list = []

    # Process each document in "items"
    for doc in docs:
        doc_data = doc.to_dict()  # Convert Firestore document to dictionary

        # Remove 'description' field if it exists
        doc_data.pop("description", None)

        # Ensure 'part_id' is the first key (reorder dictionary)
        ordered_data = {
            "part_id": doc_data.get("part_id", ""),  # Ensure part_id exists
            "current_stock": doc_data.get("current_stock", ""),
            "lead_time": doc_data.get("lead_time", ""),
            "min_stock": doc_data.get("min_stock", ""),
        }

        data_list.append(ordered_data)

    # Define CSV file name for inventory data
    csv_filename = "component_inventory.csv"

    # Write inventory data to CSV
    with open(csv_filename, mode="w", newline="", encoding="utf-8") as file:
        # Define field names explicitly
        fieldnames = ["part_id", "current_stock", "lead_time", "min_stock"]

        # Create a CSV DictWriter
        writer = csv.DictWriter(file, fieldnames=fieldnames)

        # Write header row
        writer.writeheader()

        # Write data rows
        writer.writerows(data_list)

    print(f"CSV file '{csv_filename}' has been created successfully!")

    # Fetch all documents from the "bills" collection
    billdocs = collection_ref_bill.stream()
    data_list_bills = []

    # Process each document in "bills"
    for doc in billdocs:
        doc_data = doc.to_dict()  # Convert Firestore document to dictionary

        # Ensure 'ds' is the first key (reorder dictionary)
        ordered_data = {
            "ds": doc_data.get("ds", ""),  # Ensure ds exists
            "y": doc_data.get("y", ""),
            "part_id": doc_data.get("part_id", ""),
            "aircraft_type": doc_data.get("aircraft_type", ""),
        }

        data_list_bills.append(ordered_data)

    # Define CSV file name for demand data
    csv_filename = "aircraft_demand_dataset.csv"

    # Write demand data to CSV
    with open(csv_filename, mode="w", newline="", encoding="utf-8") as file:
        # Define field names explicitly
        fieldnames = ["ds", "y", "part_id", "aircraft_type"]

        # Create a CSV DictWriter
        writer = csv.DictWriter(file, fieldnames=fieldnames)

        # Write header row
        writer.writeheader()

        # Write data rows
        writer.writerows(data_list_bills)

    print(f"CSV file '{csv_filename}' has been created successfully!")
