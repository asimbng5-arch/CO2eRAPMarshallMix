import io
import pandas as pd
import streamlit as st
from PIL import Image
import pickle
import numpy as np
import matplotlib.pyplot as plt
import math

# Page setup
st.set_page_config(layout="centered")

st.markdown("""
<div style='text-align: center; font-size: 20px; color: black;'>An Innovative Design Automation Platform Enabling User-Defined Uncertainty Quantification of RAS-Asphalt Pavement Durability</div>
""", unsafe_allow_html=True)

st.markdown("""
<h1 style='color:#8805ed; font-size:30px; text-align:center;'>Probabilistic Predictions of RAS-Asphalt Pavement Durability through Natural Gradient Boosting</h1>
""", unsafe_allow_html=True)

st.write('---')

# === Load Visuals ===
st.image(Image.open('Flowchart-2.png'), use_container_width=True)

# === Load datasets (Real + Synthetic) ===
df_real = pd.read_excel("TSTR_MS.xlsx", sheet_name="real")
df_synth = pd.read_excel("TSTR_MS.xlsx", sheet_name="synthetic")
df_full = pd.concat([df_real, df_synth], ignore_index=True)

df_real_fm = pd.read_excel("TSTR_MF.xlsx", sheet_name="real")
df_synth_fm = pd.read_excel("TSTR_MF.xlsx", sheet_name="synthetic")
df_full_fm = pd.concat([df_real_fm, df_synth_fm], ignore_index=True)

df_real_CO2e = pd.read_excel("TSTR_CO2e.xlsx", sheet_name="real")
df_synth_CO2e = pd.read_excel("TSTR_CO2e.xlsx", sheet_name="synthetic")
df_full_CO2e = pd.concat([df_real_fm, df_synth_CO2e], ignore_index=True)

# Columns for MS prediction
target_col = 'MS (kN)'
categorical_cols = ['AgT', 'VAPG', 'RT', 'FT']
numerical_cols = [col for col in df_full.columns if col not in categorical_cols + [target_col]]

# Columns for MF Prediction
target_col_fm = 'MF (mm)'
categorical_cols_fm = ['AgT', 'VAPG', 'RT', 'FT']
numerical_cols_fm = [col for col in df_full_fm.columns if col not in categorical_cols_fm + [target_col_fm]]

# Columns for CO2e prediction
target_col_co2e = 'CO2'
categorical_cols_co2e = ['AgT', 'VAPG', 'RT', 'FT']
numerical_cols_co2e = [col for col in df_full_co2e.columns if col not in categorical_cols_co2e + [target_col_co2e]]

input_descriptions = {
    'Dmax': ("Maximum aggregate size", "mm", r"$D_{max}$"),
    'NMAS': ("Nominal maximum aggregate size", "mm", r"$NMAS$"),
    'n': ("Gradation power law exponent", "—", r"$n$"),
    'LAA': ("Los Angeles abrasion", "%", r"$LAA$"),
    'AT': ("Aggregate type", "—", r"$AT$"),
    'RAP': ("Reclaimed asphalt pavement content", "%", r"$RAP$"),
    'ABR': ("Asphalt binder replacement", "%", r"$ABR$"),
    'RT': ("Rejuvenator type", "—", r"$RT$"),
    'ARAP': ("RAP aggregate", "%", r"$ARAP$"),
    'AC': ("Asphalt content", "%", r"$AC$"),
    'VAPG': ("Virgin asphalt performance grade", "—", r"$VA_{PG}$"),
    'Gmb': ("Density of asphalt mixture", "kg/m3", r"$G_{mb}$"),
    'AV': ("Air voids", "%", r"$AV$"),
    'VMA': ("Void in mineral aggregate", "%", r"$VMA$"),
    'VFA': ("Voids filled with asphalt", "%", r"$VFA$"),
    'AMC': ("Aggregate moisture content", "%", r"$AMC$"),
    'FT': ("Fuel type", "—", r"$FT$"),
    'TM': ("Mixing temperature", "°C", r"$TM$"),
    'MS': ("Marshall stability", "kN", r"$MS$"),
    'MF': ("Marshall flow", "mm", r"$MF$"),
    'CO2e': ("Carbon emission", "kg", r"$CO_2e$")
}

default_input_row = df_synth.iloc[17]

def user_input_features(defaults=None):
    input_data = {}
    for col in numerical_cols:
        min_val = float(df_full[col].min())
        max_val = float(df_full[col].max())
        if defaults is not None and col in defaults:
            default_val = float(defaults[col])
            # Clip default_val to min and max to avoid Streamlit errors
            default_val = max(min_val, min(max_val, default_val))
        else:
            default_val = float(df_full[col].mean())

        if col in input_descriptions:
            _, unit, symbol = input_descriptions[col]
            unit_part = f" ({unit})" if unit != "—" else ""
            label = f"{symbol}{unit_part} [Min: {min_val:.2f}, Max: {max_val:.2f}]"
        else:
            label = f"{col} [Min: {min_val:.2f}, Max: {max_val:.2f}]"

        val = st.sidebar.number_input(
            label,
            min_value=min_val,
            max_value=max_val,
            value=default_val,
            step=(max_val - min_val) / 100 if (max_val - min_val) > 0 else 0.01,
            format="%.4f"
        )
        input_data[col] = val

    for col in categorical_cols:
        if col in input_descriptions:
            _, unit, symbol = input_descriptions[col]
            unit_part = f" ({unit})" if unit != "—" else ""
            label = f"{symbol}{unit_part}"
        else:
            label = col

        options = sorted(df_full[col].dropna().unique())
        if defaults is not None and col in defaults:
            default_val = defaults[col]
            # If default_val not in options, fallback to first option
            if default_val not in options:
                default_val = options[0]
            default_index = options.index(default_val)
        else:
            default_index = 0

        val = st.sidebar.selectbox(label, options, index=default_index)
        input_data[col] = val

    return pd.DataFrame([input_data])

df_input = user_input_features(defaults=default_input_row)

sections = {
    "(i) Aggregate and Gradation Properties": ['Dmax', 'NMAS', 'n', 'LAA', 'AT'],
    "(ii) Mixture Composition and Volumetric Properties": ['RAP', 'ABR', 'ARAP', 'AC', 'Gmb', 'AV', 'VMA', 'VFA'],
    "(iii) Binder and Production Properties": ['RT', 'VAPG', 'AMC', 'FT', 'TM']
}
input_values = df_input.iloc[0]

st.markdown(
    "<p style='text-align: center; font-weight: bold; font-size: 24px; color: green; margin-bottom: 25px;'>Specified Input Parameters</p>",
    unsafe_allow_html=True
)

section_colors = {
    "(i) Material Properties": "darkred",
    "(ii) Fabric Properties": "darkblue"
}

for section_name, cols in sections.items():
    color = section_colors.get(section_name, "black")
    st.markdown(f"<h5 style='color: {color}; font-weight: bold;'>{section_name}</h5>", unsafe_allow_html=True)
    for col in cols:
        if col in input_values.index and col in input_descriptions:
            desc, unit, symbol = input_descriptions[col]
            val = input_values[col]
            val_display = f"{val:.2f}" if isinstance(val, (int, float, np.floating, np.integer)) else val
            unit_part = f" ({unit})" if unit != "—" else ""
            st.markdown(f"**{desc} ({symbol}){unit_part}:** {val_display}", unsafe_allow_html=True)
    st.write('')

# Other parameters if any
all_section_cols = [col for cols in sections.values() for col in cols]
other_cols = [col for col in input_values.index if col not in all_section_cols]

if other_cols:
    st.subheader("Other Parameters")
    for col in other_cols:
        val = input_values[col]
        val_display = f"{val:.2f}" if isinstance(val, (int, float, np.floating, np.integer)) else val
        st.markdown(f"**{col}:** {val_display}")

# === MS ===
st.write('---')
ngb_model_MS = pickle.load(open('ngb_model_MS.pkl', 'rb'))
preprocessor = pickle.load(open('preprocessor_MS.pkl', 'rb'))
dist = ngb_model_MS.pred_dist(X_input_processed)
X_input_processed = preprocessor.transform(df_input)

dist = ngb_model1.pred_dist(X_input_processed)
mean_pred = float(dist.loc[0])
std_pred = float(dist.scale[0])
ci_lower = mean_pred - 1.96 * std_pred
ci_upper = mean_pred + 1.96 * std_pred

st.header('Predicted MS')
st.markdown(f"<b><font color='green'>μ (Mean Prediction):</font> {mean_pred:.2f} kPa</b>", unsafe_allow_html=True)
st.markdown(f"<b><font color='orange'>σ (Standard Deviation):</font> {std_pred:.2f} kPa</b>", unsafe_allow_html=True)
st.markdown(f"<b><font color='purple'>95% Confidence Interval:</font> [{ci_lower:.2f}, {ci_upper:.2f}] kPa</b>", unsafe_allow_html=True)

st.write('---')

st.markdown(
    "<p style='text-align: center; font-weight: bold; font-size: 24px; margin-bottom: 25px;'>MS Distribution Plot</p>",
    unsafe_allow_html=True
)

eps = 1e-9
std = max(std_pred, eps)

x_min = mean_pred - 3 * std
x_max = mean_pred + 3 * std
x = np.linspace(x_min, x_max, 1000)
pdf = dist.pdf(x)

fig, ax = plt.subplots(figsize=(9, 4.5), dpi=150)
ax.plot(x, pdf, color='red', linewidth=2, label='MS distribution')
ax.axvline(mean_pred, color='red', linestyle='--', linewidth=1.8, label='$MS_{mean}$')

sigma_colors = ['#1b9e77', '#66c2a5', '#a6dba0']
sigmas = [1, 2, 3]
for i, s in enumerate(sigmas):
    lower = mean_pred - s * std
    upper = mean_pred + s * std
    ax.fill_between(x, 0, pdf, where=(x >= lower) & (x <= upper),
                    color=sigma_colors[i], alpha=0.6 - i*0.15,
                    label=f'±{s}σ ({[68.3, 95.4, 99.7][i]}%)')

ax.set_xlabel(r'$MS$ (kN)')
ax.set_ylabel('PDF')
ax.ticklabel_format(style='plain', axis='x')
ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda val, _: f'{val:.1f}'))
ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda val, _: f'{val:.2f}'))
ax.legend(loc='upper right', fontsize='small')
ax.grid(alpha=0.15)
plt.tight_layout()
st.pyplot(fig)

buf = io.BytesIO()
fig.savefig(buf, format="pdf")
buf.seek(0)
st.download_button(
    label="Download Predictive Distribution Plot_MS (PDF)",
    data=buf,
    file_name="predictive_distribution.pdf",
    mime="application/pdf"
)

st.write('---')

# === MF ===

ngb_model2 = pickle.load(open('ngb_model_MF.pkl', 'rb'))
preprocessor = pickle.load(open('preprocessor_MF.pkl', 'rb'))
X_input_processed = preprocessor.transform(df_input)

dist = ngb_model_MF.pred_dist(X_input_processed)
mean_pred = float(dist.loc[0])
std_pred = float(dist.scale[0])
ci_lower = mean_pred - 1.96 * std_pred
ci_upper = mean_pred + 1.96 * std_pred

st.header('Predicted MF')
st.markdown(f"<b><font color='green'>μ (Mean Prediction):</font> {mean_pred:.4f} </b>", unsafe_allow_html=True)
st.markdown(f"<b><font color='orange'>σ (Standard Deviation):</font> {std_pred:.4f} </b>", unsafe_allow_html=True)
st.markdown(f"<b><font color='purple'>95% Confidence Interval:</font> [{ci_lower:.4f}, {ci_upper:.4f}] </b>", unsafe_allow_html=True)

st.write('---')

st.markdown(
    "<p style='text-align: center; font-weight: bold; font-size: 24px; margin-bottom: 25px;'>MF Distribution Plot</p>",
    unsafe_allow_html=True
)

eps = 1e-9
std = max(std_pred, eps)

x_min = mean_pred - 3 * std
x_max = mean_pred + 3 * std
x = np.linspace(x_min, x_max, 1000)
pdf = dist.pdf(x)

fig, ax = plt.subplots(figsize=(9, 4.5), dpi=150)
ax.plot(x, pdf, color='blue', linewidth=2, label='$MF$ distribution')
ax.axvline(mean_pred, color='red', linestyle='--', linewidth=1.8, label='$MF_{mean}$')

sigma_colors = ['#1b9e77', '#66c2a5', '#a6dba0']
sigmas = [1, 2, 3]
for i, s in enumerate(sigmas):
    lower = mean_pred - s * std
    upper = mean_pred + s * std
    ax.fill_between(x, 0, pdf, where=(x >= lower) & (x <= upper),
                    color=sigma_colors[i], alpha=0.6 - i*0.15,
                    label=f'±{s}σ ({[68.3, 95.4, 99.7][i]}%)')

ax.set_xlabel(r'$MF$')
ax.set_ylabel('PDF')
ax.ticklabel_format(style='plain', axis='x')
ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda val, _: f'{val:.1f}'))
ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda val, _: f'{val:.2f}'))
ax.legend(loc='upper right', fontsize='small')
ax.grid(alpha=0.15)
plt.tight_layout()
st.pyplot(fig)

buf = io.BytesIO()
fig.savefig(buf, format="pdf")
buf.seek(0)
st.download_button(
    label="Download Predictive Distribution Plot_MF (PDF)",
    data=buf,
    file_name="predictive_distribution.pdf",
    mime="application/pdf"
)

# === CO2e ===

ngb_model2 = pickle.load(open('ngb_model_CO2e.pkl', 'rb'))
preprocessor = pickle.load(open('preprocessor_CO2e.pkl', 'rb'))
X_input_processed = preprocessor.transform(df_input)

dist = ngb_mode_CO2e.pred_dist(X_input_processed)
mean_pred = float(dist.loc[0])
std_pred = float(dist.scale[0])
ci_lower = mean_pred - 1.96 * std_pred
ci_upper = mean_pred + 1.96 * std_pred

st.header('Predicted CO2e')
st.markdown(f"<b><font color='green'>μ (Mean Prediction):</font> {mean_pred:.4f} </b>", unsafe_allow_html=True)
st.markdown(f"<b><font color='orange'>σ (Standard Deviation):</font> {std_pred:.4f} </b>", unsafe_allow_html=True)
st.markdown(f"<b><font color='purple'>95% Confidence Interval:</font> [{ci_lower:.4f}, {ci_upper:.4f}] </b>", unsafe_allow_html=True)

st.write('---')

st.markdown(
    "<p style='text-align: center; font-weight: bold; font-size: 24px; margin-bottom: 25px;'>CO2e Distribution Plot</p>",
    unsafe_allow_html=True
)

eps = 1e-9
std = max(std_pred, eps)

x_min = mean_pred - 3 * std
x_max = mean_pred + 3 * std
x = np.linspace(x_min, x_max, 1000)
pdf = dist.pdf(x)

fig, ax = plt.subplots(figsize=(9, 4.5), dpi=150)
ax.plot(x, pdf, color='blue', linewidth=2, label='$CO{2}e$ distribution')
ax.axvline(mean_pred, color='red', linestyle='--', linewidth=1.8, label='$CO2e_{mean}$')

sigma_colors = ['#1b9e77', '#66c2a5', '#a6dba0']
sigmas = [1, 2, 3]
for i, s in enumerate(sigmas):
    lower = mean_pred - s * std
    upper = mean_pred + s * std
    ax.fill_between(x, 0, pdf, where=(x >= lower) & (x <= upper),
                    color=sigma_colors[i], alpha=0.6 - i*0.15,
                    label=f'±{s}σ ({[68.3, 95.4, 99.7][i]}%)')

ax.set_xlabel(r'$CO2e$')
ax.set_ylabel('PDF')
ax.ticklabel_format(style='plain', axis='x')
ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda val, _: f'{val:.1f}'))
ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda val, _: f'{val:.2f}'))
ax.legend(loc='upper right', fontsize='small')
ax.grid(alpha=0.15)
plt.tight_layout()
st.pyplot(fig)

buf = io.BytesIO()
fig.savefig(buf, format="pdf")
buf.seek(0)
st.download_button(
    label="Download Predictive Distribution Plot_CO2e (PDF)",
    data=buf,
    file_name="predictive_distribution.pdf",
    mime="application/pdf"
)

st.write('---')

# About Authors section
st.markdown(
    "<p style='text-align: center; font-weight: bold; font-size: 24px; margin-bottom: 25px;'>Authors</p>",
    unsafe_allow_html=True
)

st.image("Author's_Photograph1.png", use_container_width=True)
