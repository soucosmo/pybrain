from pybrain3.tools.shortcuts import buildNetwork
from pybrain3.datasets import SupervisedDataSet
from pybrain3.supervised.trainers import BackpropTrainer


ds = SupervisedDataSet(2, 1)

#aqui vai a nossa base de aprendizado
base = (
    #8 horas dormidas, 2 horas estudadas, 7.1 de nota na prova
    ((8, 2), (7.1,)),
    ((10, 1), (2.3,)),
    ((7.5, 3), (8.0,)),
    ((3.5, 10), (2.5)),
)

#aqui aplicamos os numeros usados como base de aprendizado acima
for example in base:
    ds.addSample(example[0], example[1])


#aqui criamos a rede neural com a base de aprendizado anterior
nn = buildNetwork(2, 4, 1) # 2=neurónios, 4=camadas ocultas, 1 = uma saida


#aqui definimos o treinador, informamos a rede neural e a base de aprendizado
trainer = BackpropTrainer(nn, ds)

#aqui vamos treinar a rede neural, quanto maior de treinamentos, menores serão as chances de erros
for i in range(10000):#melhorar a base de aprendizado é infinitamente mais eficiente que deixar esse numero alto
    print(trainer.train())

#aqui vamos perguntar ao usuário o tempo dormindo e horas estudadas e vamos prever a nota dele
while True:
    horas_dormidas = float(input('O aluno vai dormir quantas horas? '))
    horas_estudadas = float(input('O aluno vai estudar quantas horas? '))
    nota = nn.activate((horas_dormidas, horas_estudadas))
    print(f'O aluno vai tirar aproxiamadamente {nota[0]:.2f} pontos na prova')

