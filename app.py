import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import streamlit as st
from matplotlib import gridspec

# ---------------------------------------------------------
# Streamlit 기본 설정
# ---------------------------------------------------------
st.set_page_config(
    page_title="TOA Energy Balance Explorer",
    layout="wide"
)

plt.rcParams["font.family"] = "DejaVu Sans"
plt.rcParams["axes.unicode_minus"] = False
sns.set_style("whitegrid")

DF0_U = "ΔF"

# ---------------------------------------------------------
# 물리 함수들
# ---------------------------------------------------------
DT_plot_max = 8.0
DT_grid = np.linspace(0, DT_plot_max, 200)

def alpha_net(aP, aWVLR, aAlb, aCl, aBio):
    return aP + aWVLR + aAlb + aCl + aBio

def N_line(DF0, a, DT):
    return DF0 + a * DT

def simulate_1box(DF0, a_net, C, t_end, dt, DT0=0.0):
    n = int(np.floor(t_end / dt)) + 1
    t = np.linspace(0, t_end, n)

    DT = np.zeros(n, dtype=float)
    N = np.zeros(n, dtype=float)

    DT[0] = DT0
    N[0] = DF0 + a_net * DT[0]

    for i in range(1, n):
        DT[i] = DT[i-1] + dt * (DF0 + a_net * DT[i-1]) / C
        N[i] = DF0 + a_net * DT[i]

    return t, DT, N

def compute_eq(DF0, a_net):
    if a_net == 0:
        return np.nan
    return -DF0 / a_net

# ---------------------------------------------------------
# 사이드바: 슬라이더 UI
# ---------------------------------------------------------
st.sidebar.title("설정값 조절")

st.sidebar.markdown("### 1. ERF (외부 복사강제, W/m²)")

erf_ghg = st.sidebar.slider("GHG ERF", 2.5, 5.0, 3.8, 0.05)
# min -1.4로 설정 → Total ΔF 최소 0.1 W/m² 보장
erf_aero = st.sidebar.slider("Aerosol ERF", -1.4, 0.0, -1.1, 0.05)
erf_surf = st.sidebar.slider("Surface ERF", -0.5, 0.5, -0.2, 0.05)
erf_contr = st.sidebar.slider("Contrails ERF", 0.0, 0.2, 0.06, 0.01)
erf_other = st.sidebar.slider("Other ERF", -0.5, 0.5, 0.0, 0.05)

st.sidebar.markdown("### 2. 피드백 계수 α (W/m²/°C)")

aP = st.sidebar.slider("α_P (Planck)", -3.4, -3.0, -3.22, 0.01)
aWVLR = st.sidebar.slider("α_WV+LR", 1.1, 1.5, 1.3, 0.01)
aAlb = st.sidebar.slider("α_Albedo", 0.1, 0.6, 0.35, 0.01)
aCl = st.sidebar.slider("α_Cloud", -0.1, 0.9, 0.42, 0.01)
aBio = st.sidebar.slider("α_Biogeo", -0.3, 0.3, -0.01, 0.01)

st.sidebar.markdown("### 3. 시간 / 열용량 설정")

C = st.sidebar.slider("열용량 C (W·yr/m²/°C)", 2.0, 50.0, 10.0, 1.0)
t_end = st.sidebar.slider("모델 적분 기간 t_end (years)", 50, 300, 200, 10)
t_now = st.sidebar.slider("표시할 시점 t (years)", 0.0, float(t_end), 0.0, 1.0)

# 👉 시간 단순화 경고 문구
st.sidebar.caption(
    "⚠️ **주의:** 이 모델에서의 시간은 단순화된 1-box 모형의 시간으로, "
    "실제 지구가 평형에 도달하는 시간과는 큰 오차가 있을 수 있습니다."
)

# ---------------------------------------------------------
# 파라미터 계산
# ---------------------------------------------------------
DF0 = erf_ghg + erf_aero + erf_surf + erf_contr + erf_other
a_net = alpha_net(aP, aWVLR, aAlb, aCl, aBio)

t_ts, DT_ts, N_ts = simulate_1box(DF0, a_net, C, t_end, dt=0.25, DT0=0.0)

# t_now에 가장 가까운 인덱스
idx_now = int(np.argmin(np.abs(t_ts - t_now)))
DT_now = DT_ts[idx_now]
N_now = N_ts[idx_now]

DT_eq = compute_eq(DF0, a_net)

# ---------------------------------------------------------
# 페이지 레이아웃
# ---------------------------------------------------------
st.title("TOA Energy Balance Explorer")
st.markdown(
    f"""
**TOA 에너지 수지 방정식**  
\\[
ΔN = {DF0_U} + α_{{net}} ΔT
\\]
"""
)

col_fig, col_info = st.columns([2.2, 1.1])

# ---------------------------------------------------------
# 왼쪽: 그림(ΔN–ΔT + 막대그래프들)
# ---------------------------------------------------------
with col_fig:
    fig = plt.figure(figsize=(10, 7))
    # 👉 레이아웃 변경: 1줄째 그래프 전체, 2줄째 막대그래프 2개
    gs = gridspec.GridSpec(
        2, 2,
        height_ratios=[2.2, 1.8],
        hspace=0.35,
        wspace=0.35
    )

    # 1) ΔN–ΔT (1줄 전체 차지)
    ax_NT = fig.add_subplot(gs[0, :])

    ax_NT.plot(
        DT_grid, N_line(DF0, aP, DT_grid),
        "--", label="Planck only (α_P)"
    )
    ax_NT.plot(
        DT_grid, N_line(DF0, a_net, DT_grid),
        "-", label="Net feedback (α_net)"
    )

    # 시간 진화 궤적
    ax_NT.plot(DT_ts, N_ts, linewidth=2, alpha=0.7, label="Time trajectory (net)")
    ax_NT.plot(
        [DT_now], [N_now],
        marker="o", markersize=7,
        label=f"Current state (t={t_now:.1f} yr)"
    )

    # 평형점 표시
    if not np.isnan(DT_eq):
        ax_NT.plot([DT_eq], [0.0], marker="X", markersize=8, label="Equilibrium (ΔN=0)")

    ax_NT.axhline(0, color="black", linewidth=1)
    ax_NT.set_xlabel("ΔT (°C)")
    ax_NT.set_ylabel("ΔN (W/m²)")
    ax_NT.set_title(f"TOA energy balance: ΔN = {DF0_U} + αΔT")

    ax_NT.set_xlim(0, DT_plot_max)
    ax_NT.set_ylim(-15, 15)
    ax_NT.set_xticks(np.arange(0, DT_plot_max + 0.1, 1.0))
    ax_NT.set_yticks(np.arange(-15, 16, 5))
    ax_NT.legend(fontsize=8)

    # 2) ERF 바 차트 (2줄째 왼쪽)
    ax_ERF = fig.add_subplot(gs[1, 0])
    erf_names = ["GHG", "Aerosol", "Surface", "Contrails", "Other"]
    erf_vals = [erf_ghg, erf_aero, erf_surf, erf_contr, erf_other]
    ax_ERF.bar(erf_names, erf_vals)
    ax_ERF.axhline(0, color="black", linewidth=1)
    ax_ERF.set_ylabel("ERF (W/m²)")
    ax_ERF.set_ylim(-2.5, 5.5)
    ax_ERF.set_title(f"ERF components (Total {DF0_U} = {DF0:.2f} W/m²)")

    # 3) Feedback 바 차트 (2줄째 오른쪽)
    ax_fb = fig.add_subplot(gs[1, 1])
    fb_names = ["Planck", "WV+LR", "Albedo", "Cloud", "Biogeo"]
    fb_vals = [aP, aWVLR, aAlb, aCl, aBio]
    ax_fb.bar(fb_names, fb_vals)
    ax_fb.axhline(0, color="black", linewidth=1)
    ax_fb.set_ylabel("αₓ (W/m²/°C)")
    ax_fb.set_ylim(-4.0, 2.5)
    ax_fb.set_title(f"Feedback coefficients (α_net = {a_net:.2f} W/m²/°C)")

    st.pyplot(fig)

# ---------------------------------------------------------
# 오른쪽: 숫자 요약 + 수업 아이디어
# ---------------------------------------------------------
with col_info:
    st.subheader("현재 설정 요약")

    st.markdown(f"- Total {DF0_U}: **{DF0:.2f} W/m²**")
    st.markdown(f"- α_net: **{a_net:.2f} W/m²/°C**")
    st.markdown(f"- 평형 온도 변화 ΔT_eq: **{DT_eq:.2f} °C** (ΔN = 0에서)")
    st.markdown(f"- 열용량 C: **{C:.1f} W·yr/m²/°C**")
    st.markdown(f"- 적분 기간 t_end: **{t_end} years**")
    st.markdown(f"- 현재 시점 t: **{t_now:.1f} years**")
    st.markdown(f"- 현재 상태 (ΔT, ΔN): **({DT_now:.2f} °C, {N_now:.2f} W/m²)**")

    st.markdown("---")
    st.markdown("**수업 아이디어**")
    st.markdown(
        """
- 그래프를 통해 여러 변수에 따른 **평형 온도 변화**를 비교할 수 있습니다.  
- **GHG ERF**를 조절하여 온난화 정도를 비교할 수 있습니다.  
- **α(피드백 계수)**를 조절하며 기후 시스템 내부 피드백의 역할을 탐구할 수 있습니다.  
- **Planck only (α_P)** 그래프는 기후 시스템에 의한 피드백이 없는 경우를 나타냅니다.  
- **열용량 C**를 바꾸어, 열용량에 따른 온도 변화 속도를 비교할 수 있습니다.
        """
    )
