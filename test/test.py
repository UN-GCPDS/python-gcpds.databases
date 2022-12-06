<<<<<<< HEAD
import gcpds.databases as loaddb
=======
from gcpds.databases import GIGA_MI_ME
>>>>>>> b277fe7213d128de27677079eb96a24b7c238a66

# db = loaddb.GIGA_MI_ME()
# db.load_subject(1)
# db.get_data()
# db.non_task()
# epochs = db.get_epochs()
# epochs['left hand mi'].average().plot_topomap()
# epochs['right hand mi'].average().plot_topomap()

<<<<<<< HEAD
# db = loaddb.BCI_Competition_IV.Dataset_2a()
# db.load_subject(3)
# # db.get_data()
# epochs = db.get_epochs()
# epochs['left hand'].average().plot_topomap()
# epochs['right hand'].average().plot_topomap()

db = loaddb.GIGA_BCI.MI()
db.load_subject(2)
db.get_data()
epochs = db.get_epochs()
epochs['left'].average().plot_topomap()
epochs['right'].average().plot_topomap()
=======
db = GIGA_MI_ME()
print(db)

db.load_subject(1)

data, cls = db.get_data()

data, cls = db.get_data(
    channels=['Fp1', 'AF7', 'AF3', 'F1', 'F3'], classes=[0])

>>>>>>> b277fe7213d128de27677079eb96a24b7c238a66
