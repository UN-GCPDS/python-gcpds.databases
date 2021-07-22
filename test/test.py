from gcpds.databases import DUMMY


db = DUMMY()
print(db)

db.load_subject(1)

db.get_data(classes=['class 0', 'class 1', ],
            channels=['CH-1', 'CH-2', 'CH-4'])


db.get_data(classes=[0, 1],
            channels=[1, 2, 4])


db.test_integrity()
