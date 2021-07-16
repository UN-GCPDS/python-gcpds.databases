from gcpds.databases import GIGA_MI_ME, BCI_Competition_IV, HighGamma_ME, GIGA_BCI, HighGamma_ME, PhysioNet_MI_ME, DUMMY


db = DUMMY()
print(db)
db.load_subject(2)
