"""Treina a MLP do projeto Digits com RMSprop ou Adam e compara configurações.

Com digits_preparados.npz na pasta deste script, execute:
    python treinar_referencia.py
    python treinar_referencia.py --inicializacao xavier
    python treinar_referencia.py --inicializacao he --taxa 0.0003
    python treinar_referencia.py --inicializacao he --taxa 0.001 --reduzir-apos 10
    python treinar_referencia.py --inicializacao he --taxa 0.001 --momentum 0.9
    python treinar_referencia.py --inicializacao xavier --taxa 0.001 --momentum 0.9
    python treinar_referencia.py --otimizador adam --inicializacao he --taxa 0.001
    python treinar_referencia.py --dropout 0.2

Dependências: NumPy e Matplotlib. O script grava os pesos finais, um CSV com
as métricas por época e um gráfico da acurácia de validação na mesma pasta.
"""

import argparse
import csv
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


PASTA = Path(__file__).resolve().parent
EPOCAS = 20
TAMANHO_LOTE = 32
TAXA = 0.001
L2 = 0.0001
DECAIMENTO_RMS = 0.99
BETA1_ADAM = 0.9
BETA2_ADAM = 0.999
EPSILON = 1e-8
SEMENTE = 42


def inicializar(gerador, metodo="he"):
    """He e Xavier usam distribuições normais com variâncias diferentes."""
    if metodo == "he":
        desvio_oculta, desvio_saida = np.sqrt(2 / 64), np.sqrt(2 / 32)
    elif metodo == "xavier":
        desvio_oculta, desvio_saida = np.sqrt(2 / (64 + 32)), np.sqrt(2 / (32 + 10))
    else:
        raise ValueError(f"Inicialização desconhecida: {metodo}")
    return {
        "W1": gerador.normal(0, desvio_oculta, (64, 32)),
        "b1": np.zeros(32),
        "W2": gerador.normal(0, desvio_saida, (32, 10)),
        "b2": np.zeros(10),
    }


def propagar(entradas, parametros, mascara_dropout=None):
    oculta = np.maximum(0, entradas @ parametros["W1"] + parametros["b1"])
    if mascara_dropout is not None:
        oculta *= mascara_dropout
    logits = oculta @ parametros["W2"] + parametros["b2"]
    return oculta, logits


def gradientes(entradas, rotulos, parametros, penalizacao_l2=L2, mascara_dropout=None):
    """Retropropagação da entropia cruzada média e da penalização L2 indicada."""
    oculta, logits = propagar(entradas, parametros, mascara_dropout)
    # Subtrair o maior logit mantém o cálculo do softmax estável.
    exponenciais = np.exp(logits - logits.max(axis=1, keepdims=True))
    erro_saida = exponenciais / exponenciais.sum(axis=1, keepdims=True)
    erro_saida[np.arange(len(rotulos)), rotulos] -= 1
    erro_saida /= len(rotulos)
    erro_oculta = (erro_saida @ parametros["W2"].T) * (oculta > 0)
    if mascara_dropout is not None:
        erro_oculta *= mascara_dropout
    return {
        "W1": entradas.T @ erro_oculta + penalizacao_l2 * parametros["W1"],
        "b1": erro_oculta.sum(axis=0),
        "W2": oculta.T @ erro_saida + penalizacao_l2 * parametros["W2"],
        "b2": erro_saida.sum(axis=0),
    }


def contar_acertos(entradas, rotulos, parametros):
    _, logits = propagar(entradas, parametros)
    return int(np.count_nonzero(logits.argmax(axis=1) == rotulos))


def atualizar_adam(parametros, derivadas, medias_gradientes, medias_quadrados, taxa, passo):
    """Aplica Adam com correção de viés. O passo conta minilotes desde o início do treino."""
    correcao_1 = 1 - BETA1_ADAM ** passo
    correcao_2 = 1 - BETA2_ADAM ** passo
    for nome, grad in derivadas.items():
        medias_gradientes[nome] *= BETA1_ADAM
        medias_gradientes[nome] += (1 - BETA1_ADAM) * grad
        medias_quadrados[nome] *= BETA2_ADAM
        medias_quadrados[nome] += (1 - BETA2_ADAM) * grad**2
        primeiro_momento = medias_gradientes[nome] / correcao_1
        segundo_momento = medias_quadrados[nome] / correcao_2
        parametros[nome] -= taxa * primeiro_momento / (np.sqrt(segundo_momento) + EPSILON)


def atualizar_rmsprop(parametros, derivadas, velocidades, medias_quadrados, taxa, momentum):
    """Normaliza o gradiente e conserva uma fração do deslocamento do lote anterior."""
    for nome, grad in derivadas.items():
        medias_quadrados[nome] *= DECAIMENTO_RMS
        medias_quadrados[nome] += (1 - DECAIMENTO_RMS) * grad**2
        passo = taxa * grad / (np.sqrt(medias_quadrados[nome]) + EPSILON)
        velocidades[nome] *= momentum
        velocidades[nome] += passo
        parametros[nome] -= velocidades[nome]


def salvar_grafico(historico, experimento="referencia", taxa=TAXA, reduzir_apos=None,
                   momentum=0.0, otimizador="rmsprop", dropout=0.0):
    epocas = [linha[0] for linha in historico]
    validacao = [100 * linha[3] for linha in historico]
    figura, eixo = plt.subplots(figsize=(7.2, 3.8), layout="constrained")
    titulo = "Treino da rede de referência · Digits"
    cor, legenda = "#185d77", "He"
    if experimento != "referencia":
        cor = "#ac441e"
        referencia = PASTA / "historico_referencia.csv"
        if dropout:
            titulo = "Regularização da rede · Digits"
            legenda, legenda_ref = f"Dropout de {dropout:.0%}", "L2 (referência)"
        elif otimizador == "adam":
            titulo = "Adam e RMSprop · Digits"
            legenda, legenda_ref = "Adam", "RMSprop (referência)"
        elif experimento.startswith("xavier_momentum_"):
            titulo = f"He e Xavier com RMSprop e momentum {momentum}".replace(".", ",")
            legenda, legenda_ref = "Xavier", "He"
            controle = experimento.replace("xavier_", "referencia_", 1)
            referencia = PASTA / f"historico_{controle}.csv"
        elif momentum:
            titulo = "Momentum no RMSprop · Digits"
            legenda = f"Momentum {momentum}".replace(".", ",")
            legenda_ref = "Momentum 0 (referência)"
        elif reduzir_apos is not None:
            titulo = "Taxa constante ou redução pela metade · Digits"
            legenda = f"{taxa} → {taxa / 2} na época {reduzir_apos + 1}".replace(".", ",")
            legenda_ref = f"{TAXA} constante (referência)".replace(".", ",")
            eixo.axvline(reduzir_apos + 0.5, color="#777777", linestyle=":", linewidth=1)
        elif taxa == TAXA:
            titulo = "Efeito da inicialização dos pesos · Digits"
            legenda, legenda_ref = "Xavier", "He (referência)"
        else:
            titulo = "Comparação das taxas de aprendizado · Digits"
            metodo = "Xavier" if experimento.startswith("xavier") else "He"
            legenda = f"{metodo} · taxa {taxa}".replace(".", ",")
            legenda_ref = f"He · taxa {TAXA} (referência)".replace(".", ",")
        if referencia.exists():
            with referencia.open(encoding="utf-8") as arquivo:
                linhas = list(csv.DictReader(arquivo))
            eixo.plot([int(r["epoca"]) for r in linhas],
                      [100 * float(r["acuracia_validacao"]) for r in linhas],
                      color="#185d77", linestyle="--", marker="o", markersize=4,
                      label=legenda_ref)
    eixo.plot(epocas, validacao, color=cor, marker=".", markersize=6, label=legenda)
    eixo.set(xlabel="Época", ylabel="Acurácia de validação (%)", title=titulo)
    eixo.set_xticks([1, 5, 10, 15, 20])
    if experimento != "referencia":
        eixo.set_ylim(top=100)
        eixo.legend()
    else:
        eixo.set_ylim(0, 100)
    eixo.grid(axis="y", alpha=0.25)
    figura.savefig(PASTA / f"validacao_{experimento}.png", dpi=160)
    plt.close(figura)


def main(inicializacao="he", taxa=TAXA, reduzir_apos=None, momentum=0.0, otimizador="rmsprop",
         dropout=0.0):
    if not np.isfinite(taxa) or taxa <= 0:
        raise ValueError("A taxa de aprendizado deve ser positiva e finita.")
    if reduzir_apos is not None and not 1 <= reduzir_apos < EPOCAS:
        raise ValueError(f"A redução precisa ocorrer após uma época entre 1 e {EPOCAS - 1}.")
    if not 0 <= momentum < 1:
        raise ValueError("O momentum deve estar no intervalo de 0 (incluso) a 1 (excluso).")
    if otimizador not in ("rmsprop", "adam"):
        raise ValueError(f"Otimizador desconhecido: {otimizador}")
    if otimizador == "adam" and momentum:
        raise ValueError("--momentum configura o RMSprop. O Adam usa BETA1_ADAM e BETA2_ADAM.")
    if not 0 <= dropout < 1:
        raise ValueError("A probabilidade de dropout deve ser maior ou igual a 0 e menor que 1.")
    experimento = "referencia" if inicializacao == "he" else inicializacao
    if taxa != TAXA:
        experimento = f"{inicializacao}_taxa_{taxa}".replace(".", "_")
    if reduzir_apos is not None:
        experimento = f"{experimento}_metade_apos_{reduzir_apos}"
    if momentum:
        experimento = f"{experimento}_momentum_{momentum}".replace(".", "_")
    if otimizador == "adam":
        experimento = "adam" if experimento == "referencia" else f"{experimento}_adam"
    if dropout:
        experimento = f"{experimento}_dropout_{dropout}".replace(".", "_")
    penalizacao_l2 = 0.0 if dropout else L2
    with np.load(PASTA / "digits_preparados.npz") as base:
        x_treino, y_treino = base["X_treino"], base["y_treino"]
        x_val, y_val = base["X_validacao"], base["y_validacao"]

    # Fluxos independentes permitem repetir a ordem dos lotes em outros experimentos.
    semente_pesos, semente_lotes, semente_dropout = np.random.SeedSequence(SEMENTE).spawn(3)
    parametros = inicializar(np.random.default_rng(semente_pesos), inicializacao)
    gerador_lotes = np.random.default_rng(semente_lotes)
    gerador_dropout = np.random.default_rng(semente_dropout)
    medias_quadrados = {nome: np.zeros_like(valor) for nome, valor in parametros.items()}
    # Este acumulador guarda a velocidade no RMSprop ou a média do gradiente no Adam.
    memoria_linear = {nome: np.zeros_like(valor) for nome, valor in parametros.items()}
    passo = 0
    historico = []

    for epoca in range(1, EPOCAS + 1):
        taxa_epoca = taxa / 2 if reduzir_apos is not None and epoca > reduzir_apos else taxa
        ordem = gerador_lotes.permutation(len(y_treino))
        for inicio in range(0, len(ordem), TAMANHO_LOTE):
            lote = ordem[inicio:inicio + TAMANHO_LOTE]
            mascara_dropout = None
            if dropout:
                # Uma máscara por imagem. A divisão pela retenção preserva a escala média.
                forma = (len(lote), parametros["b1"].size)
                mascara_dropout = (gerador_dropout.random(forma) >= dropout) / (1 - dropout)
            derivadas = gradientes(x_treino[lote], y_treino[lote], parametros,
                                   penalizacao_l2, mascara_dropout)
            passo += 1
            if otimizador == "adam":
                atualizar_adam(parametros, derivadas, memoria_linear, medias_quadrados, taxa_epoca, passo)
            else:
                atualizar_rmsprop(parametros, derivadas, memoria_linear, medias_quadrados, taxa_epoca, momentum)

        acertos_treino = contar_acertos(x_treino, y_treino, parametros)
        acertos_val = contar_acertos(x_val, y_val, parametros)
        acc_val = acertos_val / len(y_val)
        metricas = (epoca, acertos_treino / len(y_treino), acertos_val, acc_val)
        if reduzir_apos is not None:
            metricas += (taxa_epoca,)
        historico.append(metricas)
        print(f"Época {epoca:02d} | validação {acertos_val}/{len(y_val)} ({acc_val:.2%})")

    with (PASTA / f"historico_{experimento}.csv").open("w", newline="", encoding="utf-8") as arquivo:
        escritor = csv.writer(arquivo)
        colunas = ("epoca", "acuracia_treino", "acertos_validacao", "acuracia_validacao")
        if reduzir_apos is not None:
            colunas += ("taxa_aprendizado",)
        escritor.writerow(colunas)
        escritor.writerows(historico)
    np.savez_compressed(PASTA / f"pesos_{experimento}.npz", **parametros)
    salvar_grafico(historico, experimento, taxa, reduzir_apos, momentum, otimizador, dropout)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--inicializacao", choices=("he", "xavier"), default="he")
    parser.add_argument("--taxa", type=float, default=TAXA, help="Taxa de aprendizado inicial.")
    parser.add_argument("--reduzir-apos", type=int, metavar="EPOCA",
                        help="Reduz a taxa pela metade após a época indicada.")
    parser.add_argument("--momentum", type=float, default=0.0, help="Momentum clássico do RMSprop.")
    parser.add_argument("--otimizador", choices=("rmsprop", "adam"), default="rmsprop")
    parser.add_argument("--dropout", type=float, default=0.0,
                        help="Substitui L2 por dropout na camada oculta quando maior que zero.")
    opcoes = parser.parse_args()
    main(opcoes.inicializacao, opcoes.taxa, opcoes.reduzir_apos, opcoes.momentum,
         opcoes.otimizador, opcoes.dropout)
