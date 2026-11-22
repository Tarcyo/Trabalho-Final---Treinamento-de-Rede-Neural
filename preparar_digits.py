"""Prepara treino, validação e teste da base Digits para os experimentos.

Execute com: python preparar_digits.py
O arquivo digits_preparados.npz é salvo ao lado deste script. Para ler o
treino, por exemplo, use np.load("digits_preparados.npz") e a chave "X_treino".
"""

from pathlib import Path

import numpy as np
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split


def main():
    digits = load_digits()
    rotulos = digits.target
    indices = np.arange(len(rotulos))

    # Cada imagem vira um vetor de 64 pixels. A escala original vai de 0 a 16.
    entradas = digits.data.astype(np.float32) / 16.0

    treino, reserva = train_test_split(
        indices, test_size=0.30, stratify=rotulos, random_state=42
    )
    validacao, teste = train_test_split(
        reserva, test_size=0.50, stratify=rotulos[reserva], random_state=42
    )

    dados = {}
    for nome, grupo in (("treino", treino), ("validacao", validacao), ("teste", teste)):
        dados[f"X_{nome}"] = entradas[grupo]
        dados[f"y_{nome}"] = rotulos[grupo]
        dados[f"indices_{nome}"] = grupo
        contagens = np.bincount(rotulos[grupo], minlength=10)
        print(f"{nome}: {len(grupo)} imagens | contagens de 0 a 9: {contagens.tolist()}")

    destino = Path(__file__).with_name("digits_preparados.npz")
    np.savez_compressed(destino, **dados)
    print(f"Dados salvos em {destino.resolve()}")


if __name__ == "__main__":
    main()
