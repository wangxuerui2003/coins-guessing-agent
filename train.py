from model import Model

m = Model(epochs=200000)

m.train()
m.save()
