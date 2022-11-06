from RedNeuronal import RedNeuronal
from matplotlib.pyplot import plot, show

x = RedNeuronal([10, 10], 0.5, 0.5, 0.08, 100, 1000, [True, True, True]) 
x.Propagation()
x.error_patron_v_total.pop(0)
plot(x.error_patron_entrenamiento, marker='o', color='blue', label='Entrenamiento')
plot(x.error_patron_v_total, color='red', label='Validacion')
plot((len(x.error_patron_entrenamiento)-1), x.error_patron_t_total, marker='o', color='green', label='Test')
show()

print("error de test ", (x.error_patron_t_total - x.error_patron_entrenamiento[-1]))

b = [0,0,0,0,0,1,0,0,0,0,
0,0,1,0,0,1,0,0,0,0,
1,0,1,0,0,0,0,0,0,0,
0,0,1,0,0,0,0,0,0,0,
0,0,0,1,1,1,1,0,0,0,
1,1,1,1,0,1,0,1,0,0,
0,0,1,0,0,0,0,1,0,0,
0,0,1,0,1,1,0,1,1,0,
0,1,1,1,1,0,1,0,0,0,
0,0,0,0,0,0,0,0,1,0]

d = [0,1,0,0,0,0,0,1,0,0,
1,0,0,0,0,0,1,1,1,0,
0,0,0,0,0,1,0,1,0,0,
0,0,0,1,0,0,0,1,0,0,
1,0,0,1,1,1,0,1,0,0,
0,0,1,0,0,0,0,1,0,1,
0,1,1,0,1,0,0,1,0,1,
1,0,1,0,0,0,1,1,0,0,
1,0,0,1,1,0,1,1,1,1,
0,0,0,1,0,1,0,0,0,0]

f = [0,0,0,0,1,0,0,0,0,0,
0,0,0,0,0,1,1,0,0,0,
0,0,0,0,1,0,0,0,0,0,
1,0,0,0,1,0,0,0,1,0,
0,1,0,1,1,1,1,0,0,1,
0,0,1,1,1,0,0,0,0,0,
0,0,0,1,1,0,0,0,1,0,
0,0,0,1,1,0,0,1,0,0,
0,1,0,0,1,0,0,0,0,0,
0,0,0,0,0,0,1,0,1,0]

resultado = x.ForwardPropagation(b, True)
print(resultado)
if resultado[0] > resultado[1] and resultado[0] > resultado[2]:
    print("es una B")
elif resultado[1] > resultado[0] and resultado[1] > resultado[2]:
    print("es una D")
elif resultado[2] > resultado[0] and resultado[2] > resultado[1]:
    print("es una F")

resultado = x.ForwardPropagation(f, True)
print(resultado)
if resultado[0] > resultado[1] and resultado[0] > resultado[2]:
    print("es una B")
elif resultado[1] > resultado[0] and resultado[1] > resultado[2]:
    print("es una D")
elif resultado[2] > resultado[0] and resultado[2] > resultado[1]:
    print("es una F")

resultado = x.ForwardPropagation(d, True)
print(resultado)
if resultado[0] > resultado[1] and resultado[0] > resultado[2]:
    print("es una B")
elif resultado[1] > resultado[0] and resultado[1] > resultado[2]:
    print("es una D")
elif resultado[2] > resultado[0] and resultado[2] > resultado[1]:
    print("es una F")

resultado = x.ForwardPropagationValidacion(b, True)
print(resultado)
if resultado[0] > resultado[1] and resultado[0] > resultado[2]:
    print("es una B")
elif resultado[1] > resultado[0] and resultado[1] > resultado[2]:
    print("es una D")
elif resultado[2] > resultado[0] and resultado[2] > resultado[1]:
    print("es una F")

resultado = x.ForwardPropagationValidacion(f, True)
print(resultado)
if resultado[0] > resultado[1] and resultado[0] > resultado[2]:
    print("es una B")
elif resultado[1] > resultado[0] and resultado[1] > resultado[2]:
    print("es una D")
elif resultado[2] > resultado[0] and resultado[2] > resultado[1]:
    print("es una F")

resultado = x.ForwardPropagationValidacion(d, True)
print(resultado)
if resultado[0] > resultado[1] and resultado[0] > resultado[2]:
    print("es una B")
elif resultado[1] > resultado[0] and resultado[1] > resultado[2]:
    print("es una D")
elif resultado[2] > resultado[0] and resultado[2] > resultado[1]:
    print("es una F")