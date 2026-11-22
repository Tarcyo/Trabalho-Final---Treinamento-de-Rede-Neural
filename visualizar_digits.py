"""Carrega a base Digits e salva as dez primeiras imagens com seus rótulos.

Dependências: scikit-learn e matplotlib.
Execução: python visualizar_digits.py
"""

from pathlib import Path

import matplotlib.pyplot as plt
from sklearn.datasets import load_digits


def main():
    digits = load_digits()
    imagens = digits.images[:10]
    rotulos = digits.target[:10]

    print(f"Imagens na base: {len(digits.images)}")
    print(f"Dimensões de cada imagem: {digits.images.shape[1:]}")
    print(f"Dez primeiros rótulos: {rotulos.tolist()}")

    figura, paineis = plt.subplots(2, 5, figsize=(10, 4.9))
    figura.subplots_adjust(left=0.045, right=0.955, top=0.82, bottom=0.06,
                           wspace=0.18, hspace=0.44)
    figura.suptitle("As dez primeiras imagens da base Digits", fontsize=18, y=0.96)

    # O mesmo índice associa cada imagem ao rótulo fornecido pela base.
    for painel, imagem, rotulo in zip(paineis.flat, imagens, rotulos):
        painel.imshow(imagem, cmap="gray", vmin=0, vmax=16, interpolation="nearest")
        painel.set_title(f"Rótulo {rotulo}", fontsize=12, pad=6)
        painel.set_axis_off()

    destino = Path(__file__).with_name("digits_primeiras_10.png")
    figura.savefig(destino, dpi=160, facecolor="white")
    plt.close(figura)
    print(f"Figura salva em {destino.resolve()}")


if __name__ == "__main__":
    main()
