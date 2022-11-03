import numpy as np

a = np.array([[1, 1, 1]]) 
b = np.array([[2, 3, 5]]) 
c = np.array([[3, 3, 3],[2, 2, 2],[1, 1, 1]])



aux = [a[0] for i in range(3)]
aux = np.array(aux)

b = b.reshape(3, 1)

print(aux)
print(b)

multiplicar = np.multiply(aux, b)
print(multiplicar)
multiplicar = np.multiply(multiplicar, 2)
print(multiplicar)

suma= np.add(multiplicar, c)
print(suma)
#aux= a.shape
#print(aux)
#print(aux[1], aux[0])
