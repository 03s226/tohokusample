#https://altema.is.tohoku.ac.jp/QA4U/

pip install dwave-ocean-sdk

pip install openjij

token = '**'
endpoint = 'https://cloud.dwavesys.com/sapi/'

import numpy as np
a = 0.5
x = np.linspace(0,10,11)
y = a*x**2

import matplotlib.pyplot as plt
plt.plot(x,y)
plt.show()

b = 3
y = a*(x-b)**2

plt.plot(x,y)
plt.show()

N = 10
QUBO = np.zeros(N**2).reshape(N,N)

for i in range(N):
  for j in range(N):
    QUBO[i][j] = a

for i in range(N):
  QUBO[i][i] = QUBO[i][i] - 2*a*b

from dwave.system import DWaveSampler, EmbeddingComposite
dw_sampler = DWaveSampler(solver='Advantage_system1.1', token=token)
sampler = EmbeddingComposite(dw_sampler)

sampleset = sampler.sample_qubo(QUBO,num_reads=10)
print(sampleset.record)

from pyqubo import Array
# バイナリ変数
x = Array.create(name='x', shape=(N), vartype='BINARY')
print(x)

y = a*(sum(x)-b)**2

model = y.compile()
qubo, offset = model.to_qubo()

print(qubo)

dw_sampler = DWaveSampler(solver='Advantage_system1.1', token=token)
sampler = EmbeddingComposite(dw_sampler)

from openjij import SQASampler
sampler = SQASampler()

sampleset = sampler.sample_qubo(qubo,num_reads=10)
print(sampleset.record)

print(offset)


