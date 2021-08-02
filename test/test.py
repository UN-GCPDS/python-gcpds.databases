from gcpds.databases import EEG_data_raw


db = EEG_data_raw()
print(db)

db.load_subject(1)

data, cls = db.get_data()

data, cls = db.get_data(
    channels=['Fp1', 'AF7', 'AF3', 'F1', 'F3'], classes = [0])


db.test_integrity()
