# app.py
import streamlit as st
import pandas as pd
import numpy as np
from scipy.stats import t as tdist
from io import BytesIO

st.set_page_config(page_title="FaceMesh RMS: Paired t + efeito + FDR", layout="wide")
st.title("😀 FaceMesh RMS (468 pontos) — t pareado + Tamanho de Efeito + FDR (BH)")

st.markdown(
    """
**Entrada esperada (2 CSVs):** cada arquivo com:
- 1ª coluna: **Marcador** (ex.: ponto_0, ponto_1, ...)
- demais colunas: **sujeitos** (mesmos nomes e mesma ordem nos 2 arquivos)
"""
)

col1, col2 = st.columns(2)
with col1:
    up_A = st.file_uploader("CSV — Condição A", type=["csv"], key="condA")
with col2:
    up_B = st.file_uploader("CSV — Condição B", type=["csv"], key="condB")

alpha = st.sidebar.slider("Alpha (FDR)", 0.001, 0.20, 0.05, 0.001)

name_A = st.sidebar.text_input("Nome Condição A (para colunas)", value="CondA")
name_B = st.sidebar.text_input("Nome Condição B (para colunas)", value="CondB")


def load_csv(uploaded_file) -> pd.DataFrame:
    df = pd.read_csv(uploaded_file)
    df = df.rename(columns={df.columns[0]: "Marcador"})
    return df


def bh_fdr(pvals: np.ndarray, alpha: float):
    """Retorna: significativos (bool), q-values (BH adjusted p)."""
    m = pvals.size
    order = np.argsort(pvals)
    p_sorted = pvals[order]
    ranks = np.arange(1, m + 1)

    thr = ranks / m * alpha
    mask = p_sorted <= thr

    if mask.any():
        kmax = ranks[mask].max()
        p_cut = p_sorted[kmax - 1]
        sig_sorted = p_sorted <= p_cut
    else:
        sig_sorted = np.zeros(m, dtype=bool)

    sig = np.zeros(m, dtype=bool)
    sig[order] = sig_sorted

    q_sorted = p_sorted * m / ranks
    q_sorted = np.minimum.accumulate(q_sorted[::-1])[::-1]
    q_sorted = np.clip(q_sorted, 0, 1)

    q = np.zeros(m, dtype=float)
    q[order] = q_sorted
    return sig, q


def run_paired(dfA: pd.DataFrame, dfB: pd.DataFrame, alpha: float, name_A: str, name_B: str) -> pd.DataFrame:
    # checks básicos
    markersA = dfA["Marcador"].astype(str).values
    markersB = dfB["Marcador"].astype(str).values
    if not np.array_equal(markersA, markersB):
        raise ValueError("Os marcadores (linhas) não estão alinhados entre os 2 CSVs.")

    subjectsA = list(dfA.columns[1:])
    subjectsB = list(dfB.columns[1:])
    if subjectsA != subjectsB:
        raise ValueError("As colunas de sujeitos não batem (nomes/ordem diferentes) entre os 2 CSVs.")

    A = dfA.iloc[:, 1:].astype(float).to_numpy()  # M x N
    B = dfB.iloc[:, 1:].astype(float).to_numpy()  # M x N
    M, N = A.shape

    # diferenças within
    D = A - B  # M x N
    mean_D = D.mean(axis=1)
    sd_D = D.std(axis=1, ddof=1)

    # t pareado vetorizado: t = mean(D)/(sd(D)/sqrt(N))
    with np.errstate(divide="ignore", invalid="ignore"):
        t = mean_D / (sd_D / np.sqrt(N))

    df = N - 1
    p = 2 * tdist.sf(np.abs(t), df)

    # tamanho de efeito: Cohen's dz = mean(D)/sd(D)
    with np.errstate(divide="ignore", invalid="ignore"):
        dz = mean_D / sd_D

    # eta_p² equivalente (a partir de t): eta_p² = t^2 / (t^2 + df)
    t2 = t * t
    eta_p2 = t2 / (t2 + df)

    # médias por condição e delta
    mean_A = A.mean(axis=1)
    mean_B = B.mean(axis=1)
    delta = np.abs(mean_A - mean_B)

    # FDR BH
    sig_fdr, q = bh_fdr(p, alpha=alpha)

    results = pd.DataFrame({
        "Marcador": markersA,
        "t": t,
        "df": df,
        "p": p,
        "q_BH": q,
        f"Significativo_FDR_{alpha:.3f}": sig_fdr,
        "dz": dz,
        "eta_p2": eta_p2,
        f"media_{name_A}": mean_A,
        f"media_{name_B}": mean_B,
        "delta_abs_medias": delta,
        "media_diff_A_minus_B": mean_A - mean_B,
    })

    # ordenar por número do ponto (ponto_123)
    def marker_key(s):
        try:
            return int(str(s).split("_")[-1])
        except:
            return 10**9

    results["_k"] = results["Marcador"].map(marker_key)
    results = results.sort_values("_k").drop(columns="_k").reset_index(drop=True)
    return results


def to_excel_bytes(df: pd.DataFrame, alpha: float) -> bytes:
    bio = BytesIO()
    sig_col = f"Significativo_FDR_{alpha:.3f}"

    with pd.ExcelWriter(bio, engine="openpyxl") as writer:
        df.to_excel(writer, index=False, sheet_name="Resultados")

        resumo = pd.DataFrame({
            "item": [
                "Modelo",
                "Alpha (FDR)",
                "N marcadores",
                "N significativos (p<0.05)",
                f"N significativos (FDR {alpha:.3f})",
                "Tamanho de efeito 1",
                "Tamanho de efeito 2",
            ],
            "valor": [
                "t pareado (within-subject): RMS ~ Condição (2 níveis)",
                alpha,
                len(df),
                int((df["p"] < 0.05).sum()),
                int(df[sig_col].sum()),
                "dz = mean(A-B) / sd(A-B)",
                "eta_p² = t² / (t² + df)",
            ]
        })
        resumo.to_excel(writer, index=False, sheet_name="Resumo")
    return bio.getvalue()


disabled_btn = not (up_A and up_B)

if st.button("Rodar análise", type="primary", disabled=disabled_btn):
    try:
        dfA = load_csv(up_A)
        dfB = load_csv(up_B)

        with st.spinner("Calculando t pareado + efeito + FDR..."):
            results = run_paired(dfA, dfB, alpha=alpha, name_A=name_A.strip(), name_B=name_B.strip())

        st.success("Concluído!")
        st.subheader("Prévia dos resultados")
        st.dataframe(results.head(30), use_container_width=True)

        st.subheader("Top 20 — maior tamanho de efeito (dz)")
        top20 = results.reindex(results["dz"].abs().sort_values(ascending=False).index).head(20)
        st.dataframe(top20, use_container_width=True)

        excel_bytes = to_excel_bytes(results, alpha=alpha)
        st.download_button(
            "⬇️ Baixar planilha Excel (.xlsx)",
            data=excel_bytes,
            file_name="Resultados_t_pareado_Efeito_FaceMesh.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        )

    except Exception as e:
        st.error(f"Erro: {e}")
        st.info("Dica: verifique se os 2 CSVs têm os mesmos marcadores (linhas) e os mesmos sujeitos (colunas, mesma ordem).")
