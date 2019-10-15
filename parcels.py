from pybrain3.tools.shortcuts import buildNetwork
from pybrain3.datasets import SupervisedDataSet
from pybrain3.supervised.trainers import BackpropTrainer


ds = SupervisedDataSet(2, 1)
#valor_inicial, numero_parcela, valor_final
ds.addSample((100, 1), (100))
ds.addSample((100, 2), (52.63))
ds.addSample((100, 3), (35.69))
ds.addSample((100, 4), (27.22))
ds.addSample((100, 5), (22.14))
ds.addSample((100, 6), (18.76))
ds.addSample((100, 7), (16.35))
ds.addSample((100, 8), (14.54))
ds.addSample((100, 9), (13.14))
ds.addSample((100, 10), (12.02))
ds.addSample((100, 11), (11.10))
ds.addSample((100, 12), (10.34))
'''
ds.addSample((200, 12), (20.68))
ds.addSample((6518.78, 1), (6518.78))
ds.addSample((6518.78, 3), (2326.33))
ds.addSample((6518.78, 5), (1443.38))
ds.addSample((6518.78, 8), (947.93))
ds.addSample((6518.78, 9), (856.47))
ds.addSample((6518.78, 10), (783.43))
ds.addSample((6518.78, 11), (723.79))
ds.addSample((6518.78, 12), (674.19))'''



nn = buildNetwork(2, 4, 1)

trainer = BackpropTrainer(nn, ds)

for i in range(20000):
    print(trainer.train())


while True:
    valor_inicial = float(input('Valor inicial: '))
    numero_parcela = int(input('Quantidade parcelas '))
    x = nn.activate((valor_inicial, numero_parcela))
    print(f'Corresponde ao valor {x[0]:.2f}')
