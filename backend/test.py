import csv
import firebase_admin
from firebase_admin import credentials, firestore

# Initialize Firebase
cred = credentials.Certificate('flight-maintenance-firebase-adminsdk-fbsvc-f087e37cb5.json')
firebase_admin.initialize_app(cred)

db = firestore.client()

# Read the CSV file
with open('aircraft_demand_dataset.csv', mode='r') as file:
    csv_reader = csv.DictReader(file)
    for row in csv_reader:
        # Push each row to Firebase
        db.collection('bills').add({
            'ds': row['ds'],
            'y': int(row['y']),
            'part_id': row['part_id'],
            'aircraft_type': row['aircraft_type']
        })

print("Data pushed to Firebase successfully!")