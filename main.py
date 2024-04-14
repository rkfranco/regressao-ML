import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import sklearn.metrics as skl

data = pd.read_csv('data/data_preg.csv', header=None)
col_x = data[0].values
col_y = data[1].values


def exibir_grafico(x_treino, y_treino, x_test, y_test, title):
    plt.title(title)
    plt.scatter(col_x, col_y)
    plt.xlabel('X')
    plt.ylabel('Y')

    coeficiente_um, coeficiente_dois, coeficiente_tres, coeficiente_oito = obter_coeficientes(x_treino, y_treino)

    (regressao_um,
     regressao_dois,
     regressao_tres,
     regressao_oito) = obter_regressao(x_test, coeficiente_um, coeficiente_dois, coeficiente_tres, coeficiente_oito)

    desenhar_grafico(y_test, regressao_um, regressao_dois, regressao_tres, regressao_oito)
    plt.legend()
    plt.show()


def desenhar_grafico(y, l1, l2, l3, l8):
    # TODO: VALIDAR SE O VALOR DO EQM E DO R2 ESTAO CORRETOS
    plt.plot(l1, label='Grau 1' + get_legenda_eqm_r2(y, l1), color='red')
    plt.plot(l2, label='Grau 2' + get_legenda_eqm_r2(y, l2), color='green')
    plt.plot(l3, label='Grau 3' + get_legenda_eqm_r2(y, l3), color='black')
    plt.plot(l8, label='Grau 8' + get_legenda_eqm_r2(y, l8), color='yellow')


def obter_regressao(x, b1, b2, b3, b8):
    # TODO: Os coeficientes estão na ordem contraria / VALIDAR SE PRECISA FAZER ISSO
    x_reverse = x[::-1]
    return np.polyval(b1, x_reverse), np.polyval(b2, x_reverse), np.polyval(b3, x_reverse), np.polyval(b8, x_reverse)


def obter_coeficientes(x, y):
    return np.polyfit(x, y, 1), np.polyfit(x, y, 2), np.polyfit(x, y, 3), np.polyfit(x, y, 8)


def get_legenda_eqm_r2(yi, yf):
    return ' - EQM: ' + str(round(calcular_eqm(yi, yf), 4)) + ' | R2: ' + str(round(skl.r2_score(yi, yf), 4))


def calcular_eqm(yi, yf):
    # eqm - erro quadratico medio
    soma = 0
    for i in range(len(yf)):
        soma += (yi[i] - yf[i]) ** 2 / (1 / (1 + i))
    return soma


def dividir_dados_treino(x, y):
    qtd_teste = round(x.size * 0.1)
    x_test = x[:qtd_teste]
    x_treinamento = x[qtd_teste:]
    y_test = y[:qtd_teste]
    y_treinamento = y[qtd_teste:]
    return x_test, y_test, x_treinamento, y_treinamento


def demo_regressaop():
    x_test, y_test, x_treinamento, y_treinamento = dividir_dados_treino(col_x, col_y)
    exibir_grafico(col_x, col_y, col_x, col_y, 'Gráfico de dispersão - Todos os dados')
    exibir_grafico(x_treinamento, y_treinamento, x_test, y_test, 'Gráfico de dispersão - Divisão dos dados (Treinamento/Teste)')


if __name__ == '__main__':
    demo_regressaop()
