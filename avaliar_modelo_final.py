"""Avalia o modelo selecionado pela acurácia de validação na época 20.

Execute: python avaliar_modelo_final.py
Mantenha treinar_referencia.py, digits_preparados.npz e os pesos finais de
Xavier com momentum 0,9 na mesma pasta. O resultado fica em avaliacao_final.csv.
"""

import csv
from pathlib import Path

import numpy as np

from treinar_referencia import contar_acertos


PASTA = Path(__file__).resolve().parent
PESOS = "pesos_xavier_momentum_0_9.npz"


def main():
    resultados = []
    with np.load(PASTA / PESOS) as pesos, np.load(PASTA / "digits_preparados.npz") as dados:
        for conjunto in ("validacao", "teste"):
            imagens = dados[f"X_{conjunto}"]
            rotulos = dados[f"y_{conjunto}"]
            acertos = contar_acertos(imagens, rotulos, pesos)
            total = len(rotulos)
            acuracia = acertos / total
            resultados.append((conjunto, total, acertos, total - acertos, acuracia))
            print(f"{conjunto}: {acertos}/{total} ({acuracia:.2%})")

    with (PASTA / "avaliacao_final.csv").open("w", newline="", encoding="utf-8") as arquivo:
        escritor = csv.writer(arquivo)
        escritor.writerow(("conjunto", "imagens", "acertos", "erros", "acuracia"))
        escritor.writerows(resultados)


if __name__ == "__main__":
    main()
