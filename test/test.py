from gcpds.databases import GIGA_MI_ME


db = GIGA_MI_ME()
print(db)

db.load_subject(1)

data, cls = db.get_data()

data, cls = db.get_data(
    channels=['Fp1', 'AF7', 'AF3', 'F1', 'F3'], classes=[0])

