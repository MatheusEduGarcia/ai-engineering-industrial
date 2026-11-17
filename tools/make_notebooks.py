"""
Builds and executes the study notebooks (Portuguese).

    python tools/make_notebooks.py          # all
    python tools/make_notebooks.py 01       # only those starting with 01

Text and code live here so every committed notebook is known to run top to bottom.
The notebooks follow the chapters of "AI Engineering" (Chip Huyen, O'Reilly, 2025)
as a study path. They contain original explanations, code and data; they do not
reproduce the book. Read the chapter, then run the notebook.
"""
from pathlib import Path
import sys

import nbformat as nbf
from nbclient import NotebookClient

ROOT = Path(__file__).resolve().parent.parent
SETUP = """import sys, pathlib, random
ROOT = pathlib.Path.cwd().parent if pathlib.Path.cwd().name == "notebooks" else pathlib.Path.cwd()
sys.path.insert(0, str(ROOT))
%matplotlib inline
import matplotlib.pyplot as plt"""


def build(name: str, cells: list[tuple[str, str]]):
    nb = nbf.v4.new_notebook()
    nb.metadata["kernelspec"] = {"name": "python3", "display_name": "Python 3", "language": "python"}
    for kind, src in cells:
        nb.cells.append(nbf.v4.new_markdown_cell(src.strip()) if kind == "md" else nbf.v4.new_code_cell(src.strip()))
    NotebookClient(nb, timeout=300, kernel_name="python3", resources={"metadata": {"path": str(ROOT / "notebooks")}}).execute()
    nbf.write(nb, ROOT / "notebooks" / name)
    print("ok", name, len(nb.cells), "células")


def quiz(items: list[tuple[str, str]]) -> str:
    return "## Confira o que ficou\n\n" + "\n\n".join(
        f"**{i}. {q}**\n\n<details><summary>Ver resposta</summary>\n\n{a}\n\n</details>" for i, (q, a) in enumerate(items, 1))


# ----------------------------------------------------------------------------- 01
nb01 = [
("md", """
# 01 · O que é engenharia de IA, construindo um modelo de linguagem do zero

> **Capítulo 1 do livro:** *Introduction to Building AI Applications with Foundation Models*.
> Leia o capítulo antes ou depois, este notebook não o substitui: aqui as ideias viram código e exemplos de fábrica.

**Objetivo:** entender, com as próprias mãos, o que um modelo de linguagem faz, por que ele aprende sem rótulos, o que é um token, e como decidir se um problema da fábrica merece uma aplicação de IA.

## Antes de começar

- **Você precisa saber:** Python básico (listas, dicionários, laços).
- **Você não precisa de:** GPU, chave de API, nem saber redes neurais. Tudo aqui roda em segundos num notebook comum.
- **Tempo:** 40 minutos.
- **Como estudar:** em cada "Pare e preveja", escreva sua resposta antes de rodar a célula. Errar a previsão é a parte que ensina.

## Mapa do notebook

1. Um modelo de linguagem é um preditor do próximo pedaço de texto
2. Por que ele aprende sem ninguém rotular nada
3. Gerar texto é sortear, de novo e de novo
4. Tokens: a unidade em que tudo é medido e cobrado
5. Do modelo de linguagem ao modelo de fundação
6. Quando a fábrica deve (e não deve) usar IA generativa
7. As camadas da engenharia de IA
"""),
("code", SETUP),
("md", """
## 1. Um modelo de linguagem prevê o próximo pedaço de texto

Vamos trabalhar com um tipo de texto que toda fábrica tem e quase ninguém usa: **as anotações do operador quando a máquina para**. São curtas, repetitivas e cheias de padrão. Este conjunto é sintético (gerado por `aiind/corpus.py`), mas imita o estilo real.
"""),
("code", """
from aiind.corpus import notes
treino = notes(2000, seed=42)     # o que o modelo vai ler
teste  = notes(300,  seed=7)      # texto novo, que ele nunca viu
for t in treino[:8]:
    print("-", t)
"""),
("md", """
A ideia central de qualquer modelo de linguagem, do mais simples ao maior que existe, cabe numa frase:

> **Dado o texto até aqui, qual é a probabilidade de cada próximo pedaço?**

O modelo mais simples que faz isso é o **bigrama**: ele conta, no texto de treino, qual palavra veio depois de qual. Nada de rede neural, só contagem. Vamos treinar um.
"""),
("code", """
from aiind.tinylm import NGram, word_tokens
bigrama = NGram(order=2).fit([word_tokens(t) for t in treino])

def top(model, contexto, n=5):
    p = model.next_probs(contexto)
    for tok, prob in sorted(p.items(), key=lambda kv: -kv[1])[:n]:
        print(f"   {tok:22s} {prob:6.1%}")

print("Depois de 'tampadora parada', o modelo espera:")
top(bigrama, ["tampadora", "parada"])
"""),
("md", """
### Pare e preveja

O bigrama só olha **uma** palavra para trás. Depois de `"cabeçote"`, o que ele vai prever? E depois de `"parada"`, a lista vai depender de qual máquina parou?
"""),
("code", """
print("Depois de 'cabeçote':"); top(bigrama, ["cabeçote"])
print("\\nDepois de 'enchedora parada':"); top(bigrama, ["enchedora", "parada"])
print("\\nDepois de 'prensa parada':");    top(bigrama, ["prensa", "parada"])
"""),
("md", """
As duas últimas listas são **idênticas**. O bigrama só enxerga `"parada"` e esquece a máquina. Por isso ele sugere "garra" depois de "prensa parada", uma peça que a prensa não tem.

Essa é a limitação que move toda a história dos modelos de linguagem: **quanto contexto o modelo consegue usar**. O bigrama usa uma palavra. Um trigrama usa duas. Os modelos atuais usam centenas de milhares de tokens, e é aí que mora boa parte do capítulo 2.

## 2. Aprender sem rótulos: autossupervisão

Para treinar um classificador de falhas, alguém precisa rotular milhares de exemplos ("isto é vazamento", "isto é travamento"). Rotular é caro e lento.

Um modelo de linguagem não precisa disso. **O próprio texto é o rótulo**: cada palavra é a resposta certa para o contexto que veio antes dela. Veja quantos exemplos de treino saem de uma única anotação:
"""),
("code", """
frase = "enchedora parada vedação vazando chamado manutenção"
for contexto, alvo in bigrama.training_pairs(word_tokens(frase)):
    print(f"contexto {str(contexto):28s} -> resposta certa: {alvo}")
"""),
("md", """
Uma frase de 6 palavras virou 7 exemplos rotulados, sem nenhum humano. Agora multiplique pela internet inteira: é por isso que modelos de linguagem puderam crescer tanto, enquanto modelos que dependem de rótulos humanos ficaram presos ao tamanho dos conjuntos rotulados.

`<s>` e `</s>` marcam início e fim. Aprender a prever `</s>` é o que ensina o modelo a **parar** de escrever.

### Medindo quão bom é o modelo: perplexidade

Como saber se um modelo aprendeu? Mostramos texto novo e medimos o quanto ele se **surpreende**. A perplexidade é, grosso modo, "entre quantas opções o modelo ficou em dúvida, em média, a cada palavra". Menor é melhor. (O capítulo 3 aprofunda.)
"""),
("code", """
from aiind.tinylm import NGram
seqs_tr = [word_tokens(t) for t in treino]
seqs_te = [word_tokens(t) for t in teste]
ordens, perp = [1, 2, 3], []
for o in ordens:
    m = NGram(order=o).fit(seqs_tr)
    perp.append(m.perplexity(seqs_te))
    print(f"ordem {o} (olha {o-1} palavra(s) para trás): perplexidade {perp[-1]:6.2f}")

fig, ax = plt.subplots(figsize=(5, 2.8))
ax.bar([f"{o-1} palavra(s)\\nde contexto" for o in ordens], perp, color=["#95a5a6", "#2980b9", "#27ae60"])
ax.set_ylabel("perplexidade (menor é melhor)"); ax.set_title("Mais contexto, menos surpresa")
ax.spines[["top", "right"]].set_visible(False); plt.tight_layout()
"""),
("md", """
Sem contexto nenhum (ordem 1), o modelo fica em dúvida entre ~40 palavras a cada passo. Com uma palavra de contexto, cai para ~4. Com duas, melhora mais um pouco. **Contexto é o que transforma estatística de palavras em algo que parece entendimento.**

## 3. Gerar texto é sortear o próximo token, repetidamente

Para escrever, o modelo faz o laço: calcula as probabilidades, sorteia um token, junta ao texto, repete até sortear `</s>`. A **temperatura** controla o sorteio:

- **0:** sempre o mais provável. Previsível e repetitivo.
- **1:** sorteia nas proporções aprendidas.
- **acima de 1:** dá chance demais ao improvável. Criativo e cheio de erro.

### Pare e preveja

Com temperatura 0, se gerarmos 5 anotações, quantas serão diferentes entre si?
"""),
("code", """
trigrama = NGram(order=3).fit(seqs_tr)
for temp in (0, 1.0, 3.0):
    rng = random.Random(3)
    print(f"temperatura {temp}:")
    for _ in range(5):
        print("   ", " ".join(trigrama.generate(rng, temperature=temp)))
"""),
("md", """
Com temperatura 0, todas iguais: o modelo é **determinístico**. Já com 1.0 aparecem frases que nenhum operador escreveria, com peças de outra máquina e ações sem sentido; com 3.0, piora.

Isso já mostra uma propriedade que vai acompanhar todo o livro: **a saída de um modelo de linguagem é probabilística**. A mesma pergunta pode ter respostas diferentes, e algumas podem ser bem escritas e erradas. Numa fábrica, isso define onde a IA generativa pode ser usada sem revisão humana e onde não pode.

## 4. Tokens: a unidade de tudo

Modelos não leem letras nem palavras: leem **tokens**. Limite de contexto, preço da API e velocidade de resposta são medidos em tokens. Há três formas de cortar texto:

| Tokenizador | Vocabulário | Problema |
|---|---|---|
| caractere | minúsculo (~50) | sequências enormes, cada token diz pouco |
| palavra | gigante, nunca completo | palavra nova vira "desconhecida" |
| subpalavra (BPE) | médio, escolhido | o meio-termo usado pelos modelos atuais |

O **BPE** (*byte-pair encoding*) começa com caracteres e, repetidamente, junta o par mais frequente num token novo. Vamos treinar um com 80 junções:
"""),
("code", """
from aiind.tinylm import BPE, char_tokens
bpe = BPE(merges=80).fit(treino)
print("primeiras junções aprendidas:", [a + b for a, b in bpe.merges[:12]])
print()
for frase in ["tampadora parada cabeçote travou", "robô colaborativo desalinhado"]:
    print(f"{frase!r}\\n   -> {bpe.tokens(frase)}")
"""),
("md", """
Palavras frequentes no texto de treino (`tampadora`, `parada`) viraram **um token só**. Uma palavra que o modelo nunca viu (`robô colaborativo`) não quebra nada: é decomposta em pedaços menores. É assim que um modelo real lida com nomes de peça e códigos de equipamento que nunca viu.

### Pare e preveja

Para os mesmos 300 textos de teste, ordene do maior para o menor o número de tokens: caracteres, palavras, BPE.
"""),
("code", """
texto = " ".join(teste)
contagem = {"caracteres": len(char_tokens(texto)), "BPE (80 junções)": len(bpe.tokens(texto)), "palavras": len(word_tokens(texto))}
for k, v in contagem.items():
    print(f"{k:18s} {v:6d} tokens")
"""),
("md", """
O BPE fica no meio. A consequência prática é direta: **um texto em português costuma gastar mais tokens que o mesmo texto em inglês**, porque o vocabulário dos grandes modelos foi aprendido com mais texto em inglês. Mais tokens significa mais custo, mais latência e menos espaço no contexto. Vale lembrar disso ao orçar uma aplicação.

## 5. Do modelo de linguagem ao modelo de fundação

O que fizemos aqui é, em essência, o que um LLM faz, com três diferenças de escala:

| | Nosso trigrama | Um LLM atual |
|---|---|---|
| Como prevê | tabela de contagens | rede neural (transformer) com bilhões de parâmetros |
| Contexto | 2 palavras | de milhares a centenas de milhares de tokens |
| Dados de treino | 2.000 anotações | uma fração enorme do texto público |

Quando o modelo passa a aceitar também imagens, áudio ou vídeo, e serve de base para muitas tarefas diferentes, ele é chamado de **modelo de fundação**. A mudança que o capítulo 1 descreve é esta: antes, cada tarefa exigia treinar um modelo do zero; agora, parte-se de um modelo de fundação pronto e o trabalho passa a ser **adaptá-lo** (com prompt, contexto ou ajuste fino) e **avaliá-lo**. Esse trabalho é a engenharia de IA.

## 6. Quando a fábrica deve usar IA generativa

Tecnologia nova não é motivo. A pergunta certa é: qual o custo de **não** fazer, e qual o custo de a IA errar?

Abaixo, uma matriz simples para comparar ideias. Os critérios e pesos são uma proposta deste repositório, ajuste aos da sua planta.
"""),
("code", """
criterios = {   # peso, e se nota alta é boa (+1) ou ruim (-1)
    "valor se der certo":            (3, +1),
    "custo de um erro":              (3, -1),
    "dados disponíveis":             (2, +1),
    "humano revisa a saída":         (2, +1),
    "dá para medir se funcionou":    (2, +1),
}
ideias = {  # notas de 1 a 5, na ordem dos critérios
    "Assistente que resume anotações de parada do turno": (4, 2, 5, 5, 4),
    "Busca em manuais de manutenção (perguntas e respostas)": (4, 2, 4, 5, 3),
    "Classificar texto livre de parada em código de falha": (5, 3, 5, 3, 5),
    "IA que muda setpoint do CLP sozinha":                 (5, 5, 2, 1, 3),
    "Gerar relatório de OEE em texto para a diretoria":   (2, 2, 5, 5, 3),
}
pesos = [p * s for p, s in criterios.values()]
nota = {k: sum(w * v for w, v in zip(pesos, notas)) for k, notas in ideias.items()}
for k, v in sorted(nota.items(), key=lambda kv: -kv[1]):
    print(f"{v:4d}  {k}")
"""),
("md", """
O resultado conversa com o que o capítulo discute sobre planejamento:

- **No topo:** usos em que um humano revisa e o erro é barato. A IA acelera, a pessoa decide.
- **Embaixo, por bons motivos:** a IA mexendo no processo sozinha. O valor é alto, mas o custo de um erro também é. E, como vimos na seção 3, a saída é probabilística.
- **Classificar parada em código de falha** fica no alto por outro motivo: dá para **medir**, comparando com o que a manutenção registrou. O que se mede, se melhora. Os capítulos 3 e 4 tratam disso.

Antes de começar um projeto, defina também: qual métrica de sucesso, qual o mínimo aceitável para ir para produção, e quem mantém o sistema quando o modelo mudar de versão (e ele vai mudar).

## 7. As camadas da engenharia de IA

Uma forma útil de enxergar o trabalho, em três camadas, de cima para baixo:

| Camada | O que se faz | Exemplo na fábrica |
|---|---|---|
| **Aplicação** | prompt, contexto (RAG), agentes, interface, avaliação | assistente que responde "por que a linha 2 parou ontem?" |
| **Modelo** | ajuste fino, engenharia de dados, otimização de inferência | ajustar um modelo pequeno ao vocabulário da planta |
| **Infraestrutura** | servir o modelo, GPUs, monitoramento, custos | rodar um modelo local num servidor da fábrica, sem dado saindo da rede OT |

Quem vem de **ML clássico** começava pela camada do meio: coletar dados, treinar. Na engenharia de IA, normalmente se **começa pela aplicação**, com um modelo pronto, e só se desce de camada quando a de cima não resolve. Os capítulos 5 e 6 são a camada de aplicação; o 7, o 8 e o 9 descem; o 10 junta tudo.

## Erros comuns

- **Achar que o modelo "sabe" coisas.** Ele estima probabilidades de texto. Uma frase bem escrita pode estar errada, como as peças trocadas entre máquinas da seção 3.
- **Contar palavras em vez de tokens** ao estimar custo e limite de contexto.
- **Escolher o caso de uso pela empolgação.** Comece pelo que um humano revisa e dá para medir.
- **Começar treinando um modelo.** Primeiro teste o que um modelo pronto faz com um bom prompt.
- **Esquecer a manutenção.** Modelos mudam de versão e de preço; o sistema precisa de avaliação contínua.
"""),
("md", quiz([
 ("Em uma frase, o que um modelo de linguagem faz?", "Estima a probabilidade de cada próximo token dado o texto anterior. Gerar texto é sortear dessas probabilidades repetidamente."),
 ("Por que modelos de linguagem puderam ser treinados com muito mais dados que classificadores tradicionais?", "Porque são autossupervisionados: o próprio texto fornece o rótulo (o próximo token). Não dependem de pessoas rotulando exemplos."),
 ("O bigrama sugeriu 'garra' depois de 'prensa parada'. Qual é a causa?", "Contexto curto demais: ele só enxerga 'parada' e esquece qual máquina parou."),
 ("O que acontece com a saída quando a temperatura é 0?", "O modelo sempre escolhe o token mais provável. A saída fica determinística e repetitiva."),
 ("Por que o mesmo texto em português pode custar mais na API que em inglês?", "O tokenizador foi aprendido com mais texto em inglês, então o português se quebra em mais tokens, e o preço é por token."),
 ("Qual ideia da matriz tem o maior risco, e por quê?", "A IA que muda setpoint do CLP sozinha: o custo de um erro é alto, não há revisão humana e a saída do modelo é probabilística."),
])),
("md", """
## Exercícios

1. **Contexto:** gere 10 anotações com o bigrama e 10 com o trigrama (temperatura 1). Conte quantas têm uma peça que não pertence à máquina. Qual erra menos?
2. **Dados fora do domínio:** calcule a perplexidade do trigrama em frases que nada têm a ver com fábrica (por exemplo, uma receita de bolo). O que o número diz?
3. **Tokens:** treine o BPE com 20, 80 e 300 junções e veja quantos tokens o conjunto de teste gasta em cada caso. Qual é a troca envolvida?
4. **Sua fábrica:** acrescente à matriz da seção 6 duas ideias de IA da sua realidade e dê as notas. Alguma ficou acima do "resumo de anotações"? Por quê?

## Glossário

- **Token:** pedaço de texto que o modelo lê e gera (pode ser caractere, palavra ou subpalavra).
- **Modelo de linguagem:** modelo que estima a probabilidade do próximo token.
- **Autossupervisão:** aprender com rótulos tirados dos próprios dados.
- **Perplexidade:** quão surpreso o modelo fica com texto novo; menor é melhor.
- **Temperatura:** controla quão arriscado é o sorteio do próximo token.
- **BPE:** tokenizador que junta os pares de símbolos mais frequentes.
- **Modelo de fundação:** modelo grande e geral, que serve de base para muitas tarefas e tipos de dado.
- **Engenharia de IA:** construir aplicações adaptando e avaliando modelos de fundação já treinados.

## Próximo passo

**Notebook 02, capítulo 2 (Understanding Foundation Models):** de onde vêm os dados de treino, o que é um transformer na prática, e como amostragem e saídas estruturadas transformam a anotação livre do operador num JSON que o sistema de OEE consegue usar.
"""),
]

# ----------------------------------------------------------------------------- 02
nb02 = [
("md", """
# 02 · Por dentro dos modelos de fundação: dados, tamanho, amostragem e saída estruturada

> **Capítulo 2 do livro:** *Understanding Foundation Models*.
> Este notebook segue a ordem do capítulo: dados de treino, arquitetura e tamanho, pós-treino, amostragem e saídas estruturadas. As ideias viram experimentos com texto de fábrica.

**Objetivo:** entender as escolhas por trás de um modelo de fundação que afetam quem constrói aplicações (idioma, tamanho, amostragem) e transformar a anotação livre do operador num JSON que o sistema de OEE aceita.

## Antes de começar

- **Você precisa de:** o notebook 01 (modelo de linguagem, token, temperatura).
- **Você não precisa de:** GPU nem chave de API.
- **Tempo:** 50 minutos.

## Mapa do notebook

1. Dados de treino: por que o português custa mais caro
2. Arquitetura: atenção em 15 linhas
3. Tamanho: parâmetros, memória e o que cabe na sua máquina
4. Pós-treino: de "completar texto" a "seguir instruções"
5. Amostragem: temperatura, top-k e top-p
6. Gastar mais na hora de responder: melhor de N com verificador
7. Saída estruturada: da anotação ao JSON
8. A natureza probabilística: mesma pergunta, respostas diferentes
"""),
("code", SETUP + "\nfrom aiind.corpus import notes\nfrom aiind.tinylm import NGram, BPE, word_tokens"),
("md", """
## 1. Dados de treino: o idioma tem preço

Um modelo aprende o que viu. Se ele viu muito mais inglês que português, o **tokenizador** aprende a cortar bem o inglês, e o português se quebra em pedaços menores.

Vamos simular isso: treinar um BPE com uma mistura de **90 % inglês e 10 % português** (a mesma anotação, traduzida palavra a palavra) e contar tokens das mesmas frases nos dois idiomas.

### Pare e preveja

O mesmo conteúdo em português vai gastar quantas vezes mais tokens que em inglês: 1×, 1,5×, 2× ou mais?
"""),
("code", """
from aiind.foundation import to_english
pt = notes(2000); en = [to_english(t) for t in pt]
bpe = BPE(merges=150).fit(en[:1800] + pt[:200])          # 90 % inglês, 10 % português

teste = notes(300, seed=7)
tok_pt = sum(len(bpe.tokens(t)) for t in teste)
tok_en = sum(len(bpe.tokens(to_english(t))) for t in teste)
print(f"português: {tok_pt} tokens | inglês: {tok_en} tokens | razão {tok_pt/tok_en:.1f}x\\n")
exemplo = teste[0]
print(exemplo, "->", len(bpe.tokens(exemplo)), "tokens"); print("  ", bpe.tokens(exemplo))
print(to_english(exemplo), "->", len(bpe.tokens(to_english(exemplo))), "tokens"); print("  ", bpe.tokens(to_english(exemplo)))
"""),
("md", """
O mesmo conteúdo custa cerca de **2,5 vezes mais tokens** em português. Palavras como `labeler` e `stopped` viraram um token só; `rotuladora` e `manutenção` ficaram picadas.

Na vida real a diferença entre idiomas é menor, porque os tokenizadores dos grandes modelos já veem bastante português, mas o efeito existe e é medido. Três consequências práticas:

- **Custo:** a API cobra por token.
- **Latência:** gerar cada token leva tempo.
- **Contexto:** o mesmo limite de tokens cabe menos texto em português.

O mesmo vale para domínio: vocabulário de fábrica (nomes de peça, códigos de falha, siglas) aparece pouco nos dados de treino. É por isso que existem **modelos de domínio**, treinados ou ajustados com texto de uma área específica.

## 2. Arquitetura: atenção em 15 linhas

No notebook 01, o trigrama só via duas palavras para trás. O **transformer**, a arquitetura dominante, resolve isso com **atenção**: para gerar cada posição, o modelo olha para **todas** as anteriores e decide quanto peso dar a cada uma.

O cálculo é curto. Cada token vira três vetores:
- **query (Q):** o que estou procurando;
- **key (K):** o que eu ofereço;
- **value (V):** o que eu entrego se for escolhido.

O peso de cada token anterior sai do produto Q·K, passado por um softmax. A máscara **causal** impede de olhar para o futuro, porque na hora de gerar o futuro ainda não existe.

Abaixo os vetores são escolhidos à mão, para o efeito ficar visível: o token `vazando` "procura" uma peça, e a `vedação` "oferece" ser uma peça.
"""),
("code", """
import numpy as np
from aiind.foundation import attention
frase = ["enchedora", "parada", "vedação", "vazando"]
# dimensões dos vetores: [é máquina, é estado, é peça, é sintoma]
K = np.array([[3,0,0,0], [0,3,0,0], [0,0,3,0], [0,0,0,3]], float)
Q = np.array([[0,0,0,0],     # enchedora: não procura nada em especial
              [3,0,0,0],     # parada: procura qual máquina
              [3,0,0,0],     # vedação: procura de qual máquina
              [0,0,3,0]], float)  # vazando: procura qual peça
V = np.eye(4)
saida, pesos = attention(Q, K, V, causal=True)

fig, ax = plt.subplots(figsize=(4.6, 3.6))
ax.imshow(pesos, cmap="Blues")
ax.set_xticks(range(4), frase, rotation=30); ax.set_yticks(range(4), frase)
for i in range(4):
    for j in range(4):
        ax.text(j, i, f"{pesos[i,j]:.2f}", ha="center", va="center", fontsize=8, color="white" if pesos[i,j] > .5 else "black")
ax.set_xlabel("olha para"); ax.set_ylabel("token gerando"); ax.set_title("Pesos de atenção (máscara causal)")
plt.tight_layout()
"""),
("md", """
Leia por linha:

- `vazando` põe quase todo o peso em `vedação`: achou a peça que vaza.
- `parada` olha para `enchedora`.
- A metade de cima à direita é zero: é a máscara causal.

Num modelo real, Q, K e V não são escritos à mão: saem de matrizes **aprendidas** no treino. O modelo também tem dezenas de "cabeças" de atenção em paralelo e dezenas de camadas empilhadas. Mas a conta de cada cabeça é exatamente esta.

O custo dessa conta cresce com o **quadrado** do comprimento do contexto: cada token olha para todos os anteriores. É por isso que contexto longo é caro, e o capítulo 9 volta a esse ponto.

## 3. Tamanho: quanto pesa um modelo

O tamanho de um modelo é dado em **parâmetros**, os números aprendidos. Uma conta aproximada para um transformer: cada camada tem cerca de 12·d² parâmetros (d é a largura do modelo), mais a tabela de embeddings, vocabulário × d.
"""),
("code", """
from aiind.foundation import transformer_params, weights_gb
configs = {   # camadas, largura, vocabulário: configurações típicas de três ordens de grandeza
    "pequeno": (24,  896, 150_000),
    "médio":   (32, 4096, 128_000),
    "grande":  (80, 8192, 128_000),
}
print(f"{'modelo':8s} {'parâmetros':>12s} {'FP32':>9s} {'FP16':>9s} {'INT8':>9s} {'INT4':>9s}   (GB só para os pesos)")
for nome, (L, d, V) in configs.items():
    p = transformer_params(L, d, V)
    print(f"{nome:8s} {p/1e9:10.2f} B " + " ".join(f"{weights_gb(p, b):8.1f}" for b in (32, 16, 8, 4)))
"""),
("md", """
### Pare e preveja

Um servidor de borda na fábrica tem 16 GB de memória para o modelo. Qual desses modelos cabe, e em qual precisão?

A tabela responde: o pequeno cabe em qualquer precisão; o médio só a partir de FP16 (justo) ou INT8; o grande não cabe nem em 4 bits. E isso é **só para os pesos**: rodar o modelo ainda gasta memória com o contexto (o "cache KV", capítulo 9).

Daí uma decisão comum na indústria: **modelo grande na nuvem** para tarefas difíceis, ou **modelo pequeno local** quando o dado não pode sair da rede OT. Reduzir de 16 para 4 bits por parâmetro (**quantização**) é o que torna o modelo local viável. O capítulo 7 faz essa conta em detalhe.

Mais parâmetros só ajudam se houver dados suficientes para treiná-los. A pesquisa sobre leis de escala mostrou que o treino eficiente usa muito mais tokens de treino do que parâmetros (dezenas de tokens por parâmetro). Por isso o preço de um modelo não está só no tamanho: está em **quanto dado e quanta computação** foram gastos nele.

## 4. Pós-treino: de completar texto a seguir instruções

Um modelo que só passou pelo pré-treino faz uma coisa: **continua texto**. Pergunte "Qual a causa da parada?" e ele pode responder com outra pergunta, porque em muitos textos perguntas vêm em lista.

O pós-treino tem duas etapas:

| Etapa | O que o modelo vê | O que ele aprende |
|---|---|---|
| **Ajuste supervisionado (SFT)** | exemplos de instrução e resposta boa, escritos por pessoas | o formato de "conversa": pergunta, resposta |
| **Ajuste por preferência** | duas respostas para a mesma pergunta e qual foi preferida | a responder do jeito que as pessoas preferem |

Na prática: o SFT ensina **o que** fazer; a preferência ensina **qual de duas respostas plausíveis** é melhor. Para a preferência, treina-se um **modelo de recompensa** que dá nota a respostas, e o modelo principal aprende a buscar notas altas.

O nosso trigrama não tem pós-treino, mas dá para mostrar a peça central dele, o **verificador**: uma função que dá nota a uma saída. Abaixo, um verificador para anotações geradas: a máquina existe, só cita peças dela e a frase não se perde.
"""),
("code", """
from aiind.structured import coherent_note
from aiind.sampling import generate
tri = NGram(order=3).fit([word_tokens(t) for t in notes(2000)])
rng = random.Random(5)
geradas = [" ".join(generate(tri, rng, temperature=1.0)[0]) for _ in range(500)]
print(f"anotações geradas aprovadas pelo verificador: {sum(map(coherent_note, geradas))/500:.0%}\\n")
for t in geradas[:6]:
    print("OK " if coherent_note(t) else "-- ", t)
"""),
("md", """
Só cerca de dois terços passam. Os erros mais comuns são peças de outra máquina e frases que não terminam, efeito direto do contexto curto.

## 5. Amostragem: o modelo dá as probabilidades, você decide o que fazer com elas

O modelo calcula uma distribuição para o próximo token. A **estratégia de amostragem** decide o que fazer com ela:

| Estratégia | O que faz | Efeito |
|---|---|---|
| gulosa (temperatura 0) | sempre o mais provável | repetitivo, determinístico |
| temperatura | achata (>1) ou afia (<1) a distribuição | mais ou menos variedade |
| top-k | só sorteia entre os k mais prováveis | corta a cauda longa de absurdos |
| top-p (núcleo) | só entre os mais prováveis que somam p | como top-k, mas adapta o corte |

Vejamos a distribuição depois de `"enchedora parada"`, e o que cada estratégia faz com ela:
"""),
("code", """
from aiind.sampling import apply_temperature, top_k, top_p
dist = {k: v for k, v in tri.next_probs(["enchedora", "parada"]).items() if v > 0.001}
versoes = {"original": dist, "temperatura 0.5": apply_temperature(dist, .5), "temperatura 2": apply_temperature(dist, 2),
           "top-k 3": top_k(dist, 3), "top-p 0.8": top_p(dist, .8)}
fig, axes = plt.subplots(1, 5, figsize=(15, 2.8), sharey=True)
ordem = sorted(dist, key=dist.get, reverse=True)
for ax, (nome, d) in zip(axes, versoes.items()):
    ax.bar(range(len(ordem)), [d.get(t, 0) for t in ordem], color="#2980b9")
    ax.set_title(f"{nome}\\n{sum(1 for t in ordem if d.get(t, 0) > 0.001)} tokens possíveis", fontsize=9)
    ax.set_xticks([]); ax.spines[["top", "right"]].set_visible(False)
axes[0].set_ylabel("probabilidade"); plt.tight_layout()
print("tokens em ordem:", ordem[:8], "...")
"""),
("md", """
### Pare e preveja

Qual estratégia gera **mais** anotações aprovadas pelo verificador: temperatura 1, temperatura 0.5, top-k 3 ou top-p 0.8? E qual gera **mais variedade** (anotações diferentes entre si)?
"""),
("code", """
estrategias = {"temperatura 1": dict(temperature=1.0), "temperatura 0.5": dict(temperature=0.5),
               "top-k 3": dict(temperature=1.0, k=3), "top-p 0.8": dict(temperature=1.0, p=0.8),
               "gulosa (temp. 0)": dict(temperature=0)}
for nome, kw in estrategias.items():
    rng = random.Random(1)
    g = [" ".join(generate(tri, rng, **kw)[0]) for _ in range(300)]
    print(f"{nome:18s} aprovadas {sum(map(coherent_note, g))/300:5.0%}   distintas {len(set(g)):3d} de 300")
"""),
("md", """
A troca aparece nos números: **cortar a cauda aumenta a qualidade e reduz a variedade**. A gulosa é perfeita e inútil para gerar dados, porque repete a mesma frase. Na prática:

- **Extrair dados, classificar, responder com fato:** temperatura baixa (0 a 0,3).
- **Gerar ideias, variações, dados sintéticos:** temperatura mais alta, com top-p para cortar absurdos.

## 6. Gastar mais na hora de responder: melhor de N

Outra forma de melhorar a saída sem trocar o modelo: **gerar várias respostas e escolher a melhor** com um verificador. Isso troca computação na hora da resposta por qualidade.

### Pare e preveja

Se 1 amostra é aprovada em ~67 % das vezes, com quantas amostras a chance de **ao menos uma** ser aprovada passa de 95 %?
"""),
("code", """
Ns, taxa = [1, 2, 3, 4, 8], []
rng = random.Random(9)
for N in Ns:
    ok = sum(any(coherent_note(" ".join(generate(tri, rng)[0])) for _ in range(N)) for _ in range(300))
    taxa.append(ok / 300)
    print(f"N = {N}: ao menos uma aprovada em {ok/300:.0%} das vezes")
fig, ax = plt.subplots(figsize=(5, 2.8))
ax.plot(Ns, [100 * t for t in taxa], "o-", color="#27ae60"); ax.set_xlabel("amostras por pergunta (N)")
ax.set_ylabel("% com saída aprovada"); ax.set_ylim(50, 101); ax.set_title("Melhor de N com verificador")
ax.spines[["top", "right"]].set_visible(False); plt.tight_layout()
"""),
("md", """
Com 3 amostras já passa de 95 %. Se cada tentativa é independente, a falha cai como 0,33 elevado a N.

O custo é óbvio: **N vezes mais tokens e mais tempo**. E tudo depende do verificador: se ele aprova saídas erradas, gerar mais só escolhe o erro com mais confiança. Na indústria, os melhores verificadores são **regras do domínio**: a peça pertence à máquina? O valor está dentro do limite? O código de falha existe no CMMS?

## 7. Saída estruturada: da anotação ao JSON

Agora a tarefa que interessa ao sistema de OEE:

```
"enchedora parada vedação vazando chamado manutenção"
  -> {"maquina": "enchedora", "peca": "vedação", "problema": "vazando", "categoria": "falha"}
```

Treinamos o trigrama com 3.000 linhas no formato `anotação => JSON` e comparamos três jeitos de obter o JSON:

1. **Pedir e torcer:** gerar livremente e tentar ler o JSON.
2. **Validar e repetir:** se não validar, gerar de novo (até 3 vezes).
3. **Decodificação restrita:** a cada passo, só os tokens que o esquema permite podem ser sorteados. A peça só pode ser uma peça **daquela** máquina.

### Pare e preveja

Com decodificação restrita, que porcentagem dos JSONs será válida? E que porcentagem estará **correta** (igual ao que uma pessoa extrairia)?
"""),
("code", """
from aiind.structured import labeled, training_sequences, schema_allowed, parse, valid, FIELDS, SEP, rule_extract
dados_tr, dados_te = labeled(3000), labeled(200, seed=7)
extrator = NGram(order=3).fit(training_sequences(dados_tr))

def avaliar(restrito=False, tentativas=1, temperatura=1.0):
    rng = random.Random(0); validos = corretos = campos = 0
    for nota, certo in dados_te:
        prompt = nota.split() + [SEP]
        for _ in range(tentativas):
            toks, _ = generate(extrator, rng, temperature=temperatura, prompt=prompt, max_len=len(prompt) + 16,
                               allowed=schema_allowed(len(prompt)) if restrito else None)
            rec = parse(toks[len(prompt):])
            if not valid(rec):
                break
        if not valid(rec):
            validos += 1; corretos += rec == certo; campos += sum(rec[f] == certo[f] for f in FIELDS)
    n = len(dados_te)
    return validos / n, corretos / n, campos / (4 * n)

res = {"pedir e torcer": avaliar(), "validar e repetir (3x)": avaliar(tentativas=3),
       "decodificação restrita": avaliar(restrito=True), "restrita, temperatura 0": avaliar(restrito=True, temperatura=0)}
print(f"{'estratégia':26s} {'JSON válido':>12s} {'registro correto':>17s} {'campos corretos':>16s}")
for k, (v, c, f) in res.items():
    print(f"{k:26s} {v:12.0%} {c:17.0%} {f:16.0%}")
rng = random.Random(3); nota, certo = dados_te[0]; prompt = nota.split() + [SEP]
toks, _ = generate(extrator, rng, prompt=prompt, max_len=len(prompt) + 16, allowed=schema_allowed(len(prompt)))
print("\\nanotação:", nota); print("gerado:  ", parse(toks[len(prompt):])); print("correto: ", certo)
"""),
("md", """
Este é o resultado mais importante do notebook:

- A decodificação restrita deu **100 % de JSON válido**, sempre, e nenhuma peça de máquina errada. O esquema virou garantia.
- Mas quase **nenhum registro está correto**. O trigrama vê só as duas últimas palavras (`... manutenção =>`) e não "lê" a anotação: inventa a máquina.

> **Estrutura não é correção.** A decodificação restrita garante o formato; quem garante o conteúdo é a capacidade do modelo de entender o texto, e é isso que um modelo grande traz.

Antes de culpar o modelo, vale um teste que todo projeto deveria fazer: **uma linha de base simples**. Um extrator por dicionário, sem IA nenhuma:
"""),
("code", """
acertos = [rule_extract(n) == c for n, c in dados_te]
print(f"extrator por regras: {sum(acertos)/len(acertos):.1%} dos registros corretos\\n")
for (n, c), ok in zip(dados_te, acertos):
    if not ok:
        print("errou em:", n); print("   regras: ", rule_extract(n)); print("   correto:", c)
"""),
("md", """
As regras acertam 99,5 %, e o único erro é instrutivo: em "falta de **garra**fa", a regra achou a peça "garra" dentro da palavra "garrafa". Regras quebram em casos assim, e anotação real é cheia de abreviação, erro de digitação e gíria da linha ("enchedora deu pau no bico").

A arquitetura que costuma funcionar na fábrica combina os dois:

1. **Regras** para o que é fixo e verificável (lista de máquinas, peças, códigos).
2. **Modelo grande** para ler texto bagunçado.
3. **Saída estruturada com esquema** para o modelo nunca devolver um formato quebrado.
4. **Validação de domínio** no final: a peça pertence à máquina? Se não, o registro volta para revisão.

O capítulo 6 (RAG e agentes) faz isso com um modelo de verdade.

## 8. A natureza probabilística

A mesma pergunta, feita várias vezes, pode ter respostas diferentes. Veja o extrator respondendo 8 vezes sobre a mesma anotação:
"""),
("code", """
nota, certo = dados_te[5]; prompt = nota.split() + [SEP]
print("anotação:", nota, "\\n")
for semente in range(8):
    toks, logp = generate(extrator, random.Random(semente), prompt=prompt, max_len=len(prompt) + 16, allowed=schema_allowed(len(prompt)))
    r = parse(toks[len(prompt):])
    print(f"log-prob {logp:7.2f}  {r['maquina']:12s} {r['peca']:20s} {r['problema']:12s} {r['categoria']}")
"""),
("md", """
Duas propriedades que acompanham todo modelo de linguagem, grande ou pequeno:

- **Inconsistência:** respostas diferentes para a mesma entrada. Mitigações: temperatura 0, semente fixa quando a API permite, e cache de respostas.
- **Alucinação:** resposta bem formada e sem base no que foi dado. Aqui, o extrator "vê" peças que não estão na anotação. Mitigações: dar o contexto certo (capítulo 6), validar contra a realidade e medir.

A coluna **log-prob** é a confiança do próprio modelo na sequência que gerou. Muitas APIs devolvem esse valor, e ele ajuda a decidir quando mandar uma resposta para revisão humana. Só não confunda confiança com acerto: um modelo pode estar muito confiante e errado.

## Erros comuns

- **Ignorar o idioma no orçamento.** O mesmo texto em português gasta mais tokens.
- **Escolher o maior modelo por padrão.** Primeiro veja onde ele vai rodar e quanto custa por chamada.
- **Usar temperatura alta para extrair dados.** Extração quer temperatura baixa.
- **Confiar que JSON válido é JSON certo.** Valide o conteúdo contra as regras do domínio.
- **Pular a linha de base.** Se regras acertam 99 %, a IA precisa justificar o custo no 1 % restante.
- **Melhor de N sem verificador bom.** Gerar mais só ajuda se a escolha for confiável.
"""),
("md", quiz([
 ("Por que um texto em português pode gastar mais tokens que o mesmo texto em inglês?", "Porque o tokenizador foi treinado com mais inglês: palavras inglesas viram um token, palavras em português se quebram em vários."),
 ("O que a máscara causal faz na atenção?", "Impede cada token de olhar para os tokens seguintes, porque na geração o futuro ainda não existe."),
 ("Um modelo de 7 bilhões de parâmetros em 4 bits ocupa cerca de quanto só nos pesos?", "Cerca de 3,5 GB (7e9 × 4 bits ÷ 8)."),
 ("Qual a diferença entre ajuste supervisionado e ajuste por preferência?", "O supervisionado aprende com exemplos de respostas boas; o de preferência aprende qual de duas respostas plausíveis as pessoas preferem."),
 ("Top-k e top-p fazem o quê com a distribuição?", "Descartam os tokens menos prováveis antes de sortear. O top-k mantém um número fixo; o top-p mantém os que somam a probabilidade p."),
 ("A decodificação restrita deu 100 % de JSON válido e quase 0 % de registros corretos. Por quê?", "Ela garante o formato, não o conteúdo. O trigrama não consegue ler a anotação inteira para escolher os valores certos."),
 ("Melhor de N depende de quê para funcionar?", "De um verificador confiável e de as tentativas serem independentes. E custa N vezes mais."),
])),
("md", """
## Exercícios

1. **Idioma:** mude a mistura do BPE para 50 % e 50 % e recalcule a razão de tokens. O que acontece? E com 100 % português?
2. **Atenção:** edite a matriz Q para que `vedação` procure um sintoma em vez de uma máquina. O que muda na linha dela? Por que ela não consegue olhar para `vazando`?
3. **Memória:** quantos parâmetros cabem em 8 GB em INT4? E em FP16?
4. **Amostragem:** encontre a temperatura que maximiza "aprovadas × distintas" para gerar dados sintéticos.
5. **Regras:** corrija o extrator por regras para não achar "garra" dentro de "garrafa" (dica: compare palavras inteiras) e rode de novo.
6. **Esquema:** acrescente ao esquema o campo `acao` (ajustado, trocado...) com o valor `nenhum` permitido, e atualize `schema_allowed`.

## Glossário

- **Modelo de domínio:** modelo treinado ou ajustado com texto de uma área específica.
- **Transformer:** arquitetura baseada em atenção, base dos modelos de linguagem atuais.
- **Atenção:** mecanismo em que cada token pondera todos os anteriores para decidir o que gerar.
- **Parâmetro:** número aprendido no treino; o tamanho do modelo.
- **Quantização:** guardar os parâmetros com menos bits para caber em menos memória.
- **SFT:** ajuste supervisionado com exemplos de instrução e resposta.
- **Modelo de recompensa:** modelo que dá nota a respostas, usado no ajuste por preferência.
- **Top-k, top-p:** estratégias que cortam os tokens improváveis antes do sorteio.
- **Decodificação restrita:** amostragem em que só os tokens permitidos pelo esquema podem sair.
- **Alucinação:** resposta bem formada e sem base nos dados fornecidos.

## Próximo passo

**Notebook 03, capítulo 3 (Evaluation Methodology):** como medir se um modelo é bom. Entropia, perplexidade, similaridade com embeddings e IA como juiz, aplicados a textos de manutenção.
"""),
]

NOTEBOOKS = [("01_o_que_e_engenharia_de_ia.ipynb", nb01), ("02_por_dentro_dos_modelos.ipynb", nb02)]

if __name__ == "__main__":
    only = sys.argv[1:]
    for name, cells in NOTEBOOKS:
        if not only or any(name.startswith(o) for o in only):
            build(name, cells)
