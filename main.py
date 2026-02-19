# app.py
import streamlit as st
import pandas as pd
import numpy as np
from scipy.stats import f as fdist
from io import BytesIO

st.set_page_config(page_title="FaceMesh RMS: ANOVA RM + eta² + FDR", layout="wide")
st.title("😀 FaceMesh RMS (468 pontos) — ANOVA de Medidas Repetidas + Tamanho de Efeito + FDR")

st.markdown(
    """
**Entrada esperada (3 CSVs):** cada arquivo com:
- 1ª coluna: **Marcador** (ex.: ponto_0, ponto_1, ...)
- demais colunas: **15 sujeitos** (mesmos nomes e mesma ordem nos 3 arquivos)
"""
)

col1, col2, col3 = st.columns(3)
with col1:
    up_sorrindo = st.file_uploader("CSV — Sorrindo", type=["csv"], key="sorrindo")
with col2:
    up_piscando = st.file_uploader("CSV — Piscando", type=["csv"], key="piscando")
with col3:
    up_neutro = st.file_uploader("CSV — Neutro", type=["csv"], key="neutro")

alpha = st.sidebar.slider("Alpha (FDR)", 0.001, 0.20, 0.05, 0.001)

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

    # q-values (BH): p*m/rank com monotonicidade
    q_sorted = p_sorted * m / ranks
    q_sorted = np.minimum.accumulate(q_sorted[::-1])[::-1]
    q_sorted = np.clip(q_sorted, 0, 1)

    q = np.zeros(m, dtype=float)
    q[order] = q_sorted
    return sig, q

def run_anova_rm(dfA: pd.DataFrame, dfB: pd.DataFrame, dfC: pd.DataFrame, alpha: float) -> pd.DataFrame:
    # checks básicos
    markersA = dfA["Marcador"].astype(str).values
    markersB = dfB["Marcador"].astype(str).values
    markersC = dfC["Marcador"].astype(str).values
    if not (np.array_equal(markersA, markersB) and np.array_equal(markersA, markersC)):
        raise ValueError("Os marcadores (linhas) não estão alinhados entre os 3 CSVs.")

    subjectsA = list(dfA.columns[1:])
    subjectsB = list(dfB.columns[1:])
    subjectsC = list(dfC.columns[1:])
    if not (subjectsA == subjectsB == subjectsC):
        raise ValueError("As colunas de sujeitos não batem (nomes/ordem diferentes) entre os 3 CSVs.")

    A = dfA.iloc[:, 1:].astype(float).to_numpy()
    B = dfB.iloc[:, 1:].astype(float).to_numpy()
    C = dfC.iloc[:, 1:].astype(float).to_numpy()

    M, N = A.shape
    K = 3
    df1 = K - 1
    df2 = (N - 1) * (K - 1)

    # ANOVA RM vetorizada (1 fator within-subject)
    Y = np.stack([A, B, C], axis=2)  # M x N x K

    grand = Y.mean(axis=(1, 2), keepdims=True)
    subj_mean = Y.mean(axis=2, keepdims=True)   # M x N x 1
    cond_mean = Y.mean(axis=1, keepdims=True)   # M x 1 x K

    SS_total = ((Y - grand) ** 2).sum(axis=(1, 2))
    SS_subject = K * ((subj_mean - grand) ** 2).sum(axis=(1, 2))
    SS_condition = N * ((cond_mean - grand) ** 2).sum(axis=(1, 2))
    SS_error = SS_total - SS_subject - SS_condition

    MS_condition = SS_condition / df1
    MS_error = SS_error / df2

    F = MS_condition / MS_error
    p = fdist.sf(F, df1, df2)

    # partial eta squared (exato)
    eta_p2 = SS_condition / (SS_condition + SS_error)

    # médias por condição
    mean_A = A.mean(axis=1)
    mean_B = B.mean(axis=1)
    mean_C = C.mean(axis=1)
    delta = np.maximum.reduce([mean_A, mean_B, mean_C]) - np.minimum.reduce([mean_A, mean_B, mean_C])

    # FDR BH
    sig_fdr, q = bh_fdr(p, alpha=alpha)

    results = pd.DataFrame({
        "Marcador": markersA,
        "F": F,
        "p": p,
        "q_BH": q,
        f"Significativo_FDR_{alpha:.3f}": sig_fdr,
        "df1": df1,
        "df2": df2,
        "eta_p2": eta_p2,
        "media_Sorrindo": mean_A,
        "media_Piscando": mean_B,
        "media_Neutro": mean_C,
        "delta_max_min_medias": delta,
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
    with pd.ExcelWriter(bio, engine="openpyxl") as writer:
        df.to_excel(writer, index=False, sheet_name="Resultados")
        resumo = pd.DataFrame({
            "item": [
                "Modelo",
                "Alpha (FDR)",
                "N marcadores",
                "N significativos (p<0.05)",
                f"N significativos (FDR {alpha:.3f})",
                "Tamanho de efeito",
            ],
            "valor": [
                "ANOVA de medidas repetidas (within-subject): RMS ~ Condicao",
                alpha,
                len(df),
                int((df["p"] < 0.05).sum()),
                int(df[f"Significativo_FDR_{alpha:.3f}"].sum()),
                "partial eta² = SS_cond / (SS_cond + SS_error)",
            ]
        })
        resumo.to_excel(writer, index=False, sheet_name="Resumo")
    return bio.getvalue()

if st.button("Rodar análise", type="primary", disabled=not (up_sorrindo and up_piscando and up_neutro)):
    try:
        dfA = load_csv(up_sorrindo)
        dfB = load_csv(up_piscando)
        dfC = load_csv(up_neutro)

        with st.spinner("Calculando ANOVA RM + eta² + FDR..."):
            results = run_anova_rm(dfA, dfB, dfC, alpha=alpha)

        st.success("Concluído!")
        st.subheader("Prévia dos resultados")
        st.dataframe(results.head(30), use_container_width=True)

        # ranking por tamanho de efeito
        st.subheader("Top 20 — maior tamanho de efeito (eta_p²)")
        top20 = results.sort_values("eta_p2", ascending=False).head(20)
        st.dataframe(top20, use_container_width=True)

        excel_bytes = to_excel_bytes(results, alpha=alpha)
        st.download_button(
            "⬇️ Baixar planilha Excel (.xlsx)",
            data=excel_bytes,
            file_name="Resultados_ANOVA_RM_Efeito_FaceMesh.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        )

    except Exception as e:
        st.error(f"Erro: {e}")
        st.info("Dica: verifique se os 3 CSVs têm os mesmos marcadores (linhas) e os mesmos sujeitos (colunas, mesma ordem).")
