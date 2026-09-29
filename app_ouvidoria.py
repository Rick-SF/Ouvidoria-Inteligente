"""
Ouvidoria Inteligente — Triagem Semântica de Manifestações Cidadãs
Aluno: Ricardo Silva Flores

Como executar:
    pip install -r requirements_ouvidoria.txt
    streamlit run app_ouvidoria.py

O aplicativo tenta usar embeddings reais do modelo multilíngue escolhido.
Se o modelo não puder ser carregado, utiliza TF-IDF como modo de contingência,
identificado claramente na interface.
"""

import json
import re
import unicodedata
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st
from sklearn.decomposition import PCA
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.manifold import TSNE
from sklearn.metrics.pairwise import cosine_similarity

try:
    from sentence_transformers import SentenceTransformer
    TEM_SENTENCE_TRANSFORMERS = True
except ImportError:
    TEM_SENTENCE_TRANSFORMERS = False

try:
    from langchain_text_splitters import RecursiveCharacterTextSplitter
    TEM_LANGCHAIN = True
except ImportError:
    TEM_LANGCHAIN = False

st.set_page_config(page_title="Ouvidoria Inteligente", page_icon="🏛️", layout="wide")

ARQUIVO_DADOS = Path(__file__).parent / "manifestacoes.json"
CORES = {
    "infraestrutura": "#1f77b4",
    "saúde": "#d62728",
    "segurança": "#ff7f0e",
    "educação": "#2ca02c",
    "meio ambiente": "#9467bd",
}


def sem_acentos(texto):
    """Remove acentos para o fallback TF-IDF e para a divisão manual de texto."""
    normalizado = unicodedata.normalize("NFD", texto.lower())
    return "".join(c for c in normalizado if unicodedata.category(c) != "Mn")


@st.cache_data
def carregar_manifestacoes():
    with open(ARQUIVO_DADOS, "r", encoding="utf-8") as arquivo:
        return json.load(arquivo)


@st.cache_resource(show_spinner=False)
def carregar_modelo(nome_modelo):
    if not TEM_SENTENCE_TRANSFORMERS:
        return None
    try:
        return SentenceTransformer(nome_modelo)
    except Exception:
        return None


@st.cache_resource(show_spinner=False)
def criar_tfidf(textos):
    vetorizador = TfidfVectorizer(lowercase=True, strip_accents="unicode", ngram_range=(1, 2))
    matriz = vetorizador.fit_transform(textos)
    return vetorizador, matriz


def gerar_representacao(textos, nome_modelo):
    """Retorna matriz vetorial, modelo usado e nome do modo de representação."""
    modelo = carregar_modelo(nome_modelo)
    if modelo is not None:
        vetores = np.asarray(modelo.encode(textos, show_progress_bar=False), dtype=float)
        return vetores, modelo, "Embeddings reais"
    vetorizador, matriz = criar_tfidf(tuple(textos))
    return matriz.toarray(), vetorizador, "Fallback TF-IDF"


def vetor_consulta(consulta, recursos, modo):
    if modo == "Embeddings reais":
        return np.asarray(recursos.encode([consulta], show_progress_bar=False), dtype=float)
    return recursos.transform([consulta]).toarray()


def gerar_chunks(texto, tamanho, sobreposicao):
    """Usa RecursiveCharacterTextSplitter; realiza fallback simples se LangChain não estiver instalado."""
    if TEM_LANGCHAIN:
        divisor = RecursiveCharacterTextSplitter(
            chunk_size=tamanho,
            chunk_overlap=sobreposicao,
            separators=["\n\n", "\n", ". ", " ", ""],
        )
        return divisor.split_text(texto)

    chunks = []
    inicio = 0
    passo = max(1, tamanho - sobreposicao)
    while inicio < len(texto):
        chunks.append(texto[inicio : inicio + tamanho])
        inicio += passo
    return chunks


def reduzir_dimensoes(vetores, metodo):
    quantidade = len(vetores)
    if quantidade < 3:
        return np.column_stack([np.arange(quantidade), np.zeros(quantidade)])
    if metodo == "PCA":
        return PCA(n_components=2, random_state=42).fit_transform(vetores)
    perplexidade = min(10, max(2, quantidade // 3), quantidade - 1)
    return TSNE(n_components=2, perplexity=perplexidade, random_state=42, init="pca").fit_transform(vetores)


def cor_por_score(score):
    if score > 0.7:
        return "🟢"
    if score > 0.5:
        return "🟡"
    return "🔴"


def main():
    manifestacoes = carregar_manifestacoes()
    textos = [item["texto"] for item in manifestacoes]
    dados = pd.DataFrame(manifestacoes)

    st.title("🏛️ Ouvidoria Inteligente")
    st.caption("Triagem semântica de manifestações cidadãs | Ricardo Silva Flores")
    st.write(
        "Protótipo para localizar manifestações semelhantes, apoiar a detecção de duplicatas "
        "e visualizar relações semânticas entre problemas relatados pela população."
    )

    st.sidebar.header("Configurações")
    nome_modelo = st.sidebar.selectbox(
        "Modelo de embedding",
        [
            "paraphrase-multilingual-MiniLM-L12-v2",
            "BAAI/bge-small-pt-v1.5",
        ],
    )
    top_k = st.sidebar.slider("Quantidade de resultados (Top-k)", 1, 10, 5)
    metodo_visualizacao = st.sidebar.radio("Projeção 2D", ["PCA", "t-SNE"])

    with st.spinner("Preparando representações vetoriais da base..."):
        vetores, recursos, modo = gerar_representacao(textos, nome_modelo)

    if modo == "Embeddings reais":
        st.sidebar.success(f"Modo ativo: {modo}")
    else:
        st.sidebar.warning(
            "Modo de contingência: TF-IDF. Instale sentence-transformers e mantenha internet "
            "na primeira execução para utilizar embeddings reais."
        )

    aba_busca, aba_base, aba_espaco, aba_chunking = st.tabs(
        ["🔍 Busca Semântica", "📋 Base Completa", "🌐 Espaço Vetorial", "🧩 Chunking"]
    )

    with aba_busca:
        st.subheader("Buscar manifestações relacionadas")
        consulta = st.text_area(
            "Descreva o problema que deseja localizar",
            value="Rua com muitos buracos e asfalto danificado, causando acidentes.",
            height=100,
        )
        if st.button("Buscar manifestações", type="primary"):
            if not consulta.strip():
                st.warning("Digite uma descrição antes de buscar.")
            else:
                consulta_vetor = vetor_consulta(consulta, recursos, modo)
                scores = cosine_similarity(consulta_vetor, vetores)[0]
                melhores = np.argsort(scores)[::-1][:top_k]
                st.markdown(f"### {top_k} resultados mais semelhantes")
                for posicao, indice in enumerate(melhores, start=1):
                    item = manifestacoes[indice]
                    score = float(scores[indice])
                    icone = cor_por_score(score)
                    titulo = f"{icone} #{posicao} — {item['id']} | score: {score:.3f} | {item['categoria_oficial']}"
                    with st.expander(titulo, expanded=(posicao == 1)):
                        st.caption(f"Data: {item['data']} | Categoria oficial: {item['categoria_oficial']}")
                        st.write(item["texto"])
                st.caption("🟢 score > 0,7 | 🟡 score > 0,5 | 🔴 score ≤ 0,5")

    with aba_base:
        st.subheader("Base completa de manifestações")
        st.dataframe(dados, hide_index=True, use_container_width=True)
        st.markdown("### Matriz de similaridade")
        if st.button("Gerar matriz de similaridade"):
            matriz = cosine_similarity(vetores)
            figura, eixo = plt.subplots(figsize=(10, 8))
            imagem = eixo.imshow(matriz, cmap="YlOrRd", vmin=0, vmax=1)
            rotulos = dados["id"].tolist()
            eixo.set_xticks(range(len(rotulos)))
            eixo.set_yticks(range(len(rotulos)))
            eixo.set_xticklabels(rotulos, rotation=90, fontsize=7)
            eixo.set_yticklabels(rotulos, fontsize=7)
            eixo.set_title(f"Similaridade de cosseno — {modo}")
            figura.colorbar(imagem, ax=eixo, label="Similaridade")
            figura.tight_layout()
            st.pyplot(figura)

            pares = []
            for i in range(len(manifestacoes)):
                for j in range(i + 1, len(manifestacoes)):
                    if matriz[i, j] >= 0.70:
                        pares.append(
                            {
                                "Manifestação 1": manifestacoes[i]["id"],
                                "Manifestação 2": manifestacoes[j]["id"],
                                "Similaridade": round(float(matriz[i, j]), 3),
                            }
                        )
            if pares:
                st.markdown("#### Pares com similaridade igual ou superior a 0,70")
                st.dataframe(pd.DataFrame(pares).sort_values("Similaridade", ascending=False), hide_index=True)
            else:
                st.info("Nenhum par atingiu o limiar de 0,70 no modo atual.")

    with aba_espaco:
        st.subheader(f"Espaço vetorial em 2D — {metodo_visualizacao}")
        coordenadas = reduzir_dimensoes(vetores, metodo_visualizacao)
        figura, eixo = plt.subplots(figsize=(11, 7))
        for categoria in sorted(dados["categoria_oficial"].unique()):
            mascara = dados["categoria_oficial"].eq(categoria).to_numpy()
            eixo.scatter(
                coordenadas[mascara, 0],
                coordenadas[mascara, 1],
                s=90,
                c=CORES.get(categoria, "#777777"),
                label=categoria,
                edgecolors="black",
                alpha=0.82,
            )
        for indice, codigo in enumerate(dados["id"]):
            eixo.annotate(codigo, (coordenadas[indice, 0], coordenadas[indice, 1]), xytext=(4, 4), textcoords="offset points", fontsize=8)
        eixo.set_title(f"Projeção das manifestações — {modo}")
        eixo.set_xlabel("Componente 1")
        eixo.set_ylabel("Componente 2")
        eixo.grid(alpha=0.25)
        eixo.legend(title="Categoria")
        figura.tight_layout()
        st.pyplot(figura)
        st.info(
            "Interpretação: proximidade visual sugere textos semanticamente semelhantes. "
            "Os grupos podem não coincidir perfeitamente com a categoria oficial, pois uma manifestação "
            "pode citar mais de um problema, como iluminação e segurança."
        )

    with aba_chunking:
        st.subheader("Divisão de manifestações longas em chunks")
        texto_padrao = max(manifestacoes, key=lambda item: len(item["texto"]))["texto"]
        texto_longo = st.text_area("Cole ou edite uma manifestação longa", value=texto_padrao, height=210)
        coluna_tamanho, coluna_overlap = st.columns(2)
        with coluna_tamanho:
            tamanho = st.slider("chunk_size (caracteres)", 120, 800, 350, 10)
        with coluna_overlap:
            sobreposicao = st.slider("chunk_overlap (caracteres)", 0, 250, 60, 10)

        if sobreposicao >= tamanho:
            st.error("O overlap precisa ser menor que o chunk_size.")
        else:
            partes = gerar_chunks(texto_longo, tamanho, sobreposicao)
            st.metric("Chunks gerados", len(partes))
            st.caption(
                "O overlap repete parte do final de um chunk no início do próximo, preservando contexto "
                "quando uma ideia atravessa a fronteira entre blocos."
            )
            for numero, parte in enumerate(partes, start=1):
                with st.expander(f"Chunk {numero} — {len(parte)} caracteres"):
                    st.write(parte)

            if len(partes) >= 2:
                vetores_chunks, _, modo_chunks = gerar_representacao(partes, nome_modelo)
                semelhanca_chunks = cosine_similarity(vetores_chunks)
                tabela_chunks = pd.DataFrame(
                    {
                        "Par consecutivo": [f"Chunk {i} × Chunk {i + 1}" for i in range(1, len(partes))],
                        "Similaridade": [round(float(semelhanca_chunks[i - 1, i]), 3) for i in range(1, len(partes))],
                    }
                )
                st.markdown("#### Coesão entre chunks consecutivos")
                st.dataframe(tabela_chunks, hide_index=True, use_container_width=True)
                st.caption(f"Representação usada nos chunks: {modo_chunks}.")

                if len(partes) >= 3:
                    pontos_chunks = reduzir_dimensoes(vetores_chunks, "PCA")
                    figura, eixo = plt.subplots(figsize=(8, 5))
                    eixo.scatter(pontos_chunks[:, 0], pontos_chunks[:, 1], c=range(len(partes)), cmap="viridis", s=120, edgecolors="black")
                    for indice in range(len(partes)):
                        eixo.annotate(f"C{indice + 1}", (pontos_chunks[indice, 0], pontos_chunks[indice, 1]), xytext=(5, 5), textcoords="offset points")
                    eixo.set_title("PCA dos chunks da manifestação")
                    eixo.grid(alpha=0.25)
                    figura.tight_layout()
                    st.pyplot(figura)

    st.sidebar.markdown("---")
    st.sidebar.caption("Disciplina: Processamento de Linguagem Natural / Engenharia de IA")


if __name__ == "__main__":
    main()
