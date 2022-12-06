import gcpds.databases as loaddb

# db = loaddb.GIGA_MI_ME()
# db.load_subject(1)
# db.get_data()
# db.non_task()
# epochs = db.get_epochs()
# epochs['left hand mi'].average().plot_topomap()
# epochs['right hand mi'].average().plot_topomap()

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
