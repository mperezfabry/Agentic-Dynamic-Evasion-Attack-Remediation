import streamlit as st
import time
import json
import pandas as pd
import re
import altair as alt
from collections import Counter
import os

# Set page configuration
st.set_page_config(layout="wide", page_title="Agentic SOC - Mimicry Attack Mitigation")

X_DOMAIN = [0, 1000000]
Y_FNR_DOMAIN = [0, 25]
Y_FPR_DOMAIN = [0, 60]

def load_baseline_metrics():
    """Loads baseline cross-validation metrics."""
    path = "baseline_metrics_v2.json"
    if os.path.exists(path):
        try:
            with open(path, "r") as f: return json.load(f)
        except: pass
    return None

def parse_attack_data(raw_data):
    """Helper to parse miss counts from logs."""
    misses = Counter()
    try:
        raw_data = raw_data.strip()
        if raw_data.startswith('{'):
            clean_json = raw_data.replace("'", '"')
            data_dict = json.loads(clean_json)
            for s, count in data_dict.items():
                name = "DarkNexus" if s in ["Zero-Day", "DarkNexus"] else s
                misses[name] += count
        elif raw_data.startswith('['):
            strains = [a.strip().strip("'").strip('"') for a in raw_data.strip('[]').split(",") if a.strip()]
            for s in strains:
                if s != "nan":
                    name = "DarkNexus" if s in ["Zero-Day", "DarkNexus"] else s
                    misses[name] += 1
    except: pass
    return misses

@st.cache_data
def load_static_baseline_data():
    """Calculates metrics and history for the static baseline model."""
    try:
        log_path = "static_logs_v2.json"
        if not os.path.exists(log_path): return None, pd.DataFrame(), {}
        with open(log_path, "r") as f: data = json.load(f)
        rows, tps, fps, fns, tns, missed_history = [], [], [], [], [], []
        current_misses = Counter()
        cum_tp, cum_fp, cum_fn, cum_tn = 0, 0, 0, 0
        for e in data:
            if e['type'] == 'soc':
                m = re.search(r'\[(\d+)\] .* TP: (\d+) \| FP: (\d+) \| FN: (\d+) \| TN: (\d+)', e['text'])
                if m:
                    r, tp, fp, fn, tn = map(int, [m.group(1), m.group(2), m.group(3), m.group(4), m.group(5)])
                    rows.append(r); tps.append(tp); fps.append(fp); fns.append(fn); tns.append(tn)
                    cum_tp += tp; cum_fp += fp; cum_fn += fn; cum_tn += tn
                    at_m = re.search(r'Attacks: (\{.*?\}|\[.*?\])', e["text"])
                    if at_m: current_misses.update(parse_attack_data(at_m.group(1)))
                    for s, total in current_misses.items():
                        missed_history.append({'Rows': r, 'Strain': s, 'Missed Samples': total})
        df = pd.DataFrame({'Rows': rows, 'TP': tps, 'FP': fps, 'FN': fns, 'TN': tns})
        df['Rolling FNR'] = (df['FN'].rolling(20).sum() / (df['TP'].rolling(20).sum() + df['FN'].rolling(20).sum()) * 100).fillna(0)
        df['Rolling FPR'] = (df['FP'].rolling(20).sum() / (df['TN'].rolling(20).sum() + df['FP'].rolling(20).sum()) * 100).fillna(0)
        total = cum_tp + cum_fp + cum_fn + cum_tn
        metrics = {
            "accuracy": (cum_tp + cum_tn) / total if total > 0 else 0,
            "precision": cum_tp / (cum_tp + cum_fp) if (cum_tp + cum_fp) > 0 else 0,
            "recall": cum_tp / (cum_tp + cum_fn) if (cum_tp + cum_fn) > 0 else 0,
            "f1": 2 * (cum_tp / (cum_tp + cum_fp) * cum_tp / (cum_tp + cum_fn)) / (cum_tp / (cum_tp + cum_fp) + cum_tp / (cum_tp + cum_fn)) if (cum_tp + cum_fp) > 0 and (cum_tp + cum_fn) > 0 else 0
        }
        return df, pd.DataFrame(missed_history), metrics
    except: return None, pd.DataFrame(), {}

def sync_model_roster():
    """Scans the models directory and cross-references with logs."""
    roster = {}
    models_dir = "models"
    if not os.path.exists(models_dir): return roster
    files = sorted([f for f in os.listdir(models_dir) if f.endswith('.joblib')], reverse=True)
    log_meta = {}
    if os.path.exists("logs.json"):
        try:
            with open("logs.json", "r") as f:
                logs = json.load(f)
                for e in logs:
                    if e['type'] == 'model_metadata':
                        try:
                            m_data = json.loads(e['text'])
                            log_meta[m_data['model_name']] = m_data
                        except: pass
        except: pass
    for f in files:
        if f in log_meta: roster[f] = log_meta[f]
        else: roster[f] = {"model_name": f, "type": "Active Specialist", "params": {"Status": "Active"}, "deployed_at": "N/A"}
    return roster

# UI Setup
baseline_metrics_data = load_baseline_metrics()
baseline_df, df_static_missed, static_stream_metrics = load_static_baseline_data()

st.title("Autonomous SOC: Agility and Mitigation of Mimicry Attacks")
with st.expander("Agentic Architecture Info"):
    st.markdown("### Agentic Self-Healing vs. Mimicry Evasion\nDemonstrating detection of malicious patterns mutated to mimic benign behavior.")

st.subheader("Simulation Controls")
start_sim = st.button("Start Ensemble Simulation", use_container_width=True)
st.divider()

# Static Model View
st.subheader("Baseline Performance (Static Model)")
if baseline_metrics_data:
    cols = st.columns(4)
    for i, k in enumerate(["accuracy", "precision", "recall", "f1"]):
        val = baseline_metrics_data.get(k, {}).get("mean", 0.0)
        cols[i].metric(f"Baseline {k.capitalize()} (CV)", f"{val*100:.2f}%")

with st.expander("View Static Model Performance Metrics"):
    if static_stream_metrics:
        scols = st.columns(4)
        for i, k in enumerate(["accuracy", "precision", "recall", "f1"]):
            scols[i].write(f"**{k.capitalize()}:** {static_stream_metrics[k]*100:.2f}%")
    if baseline_df is not None:
        fnr_c = alt.Chart(baseline_df).mark_area(line={'color':'red'}, color=alt.Gradient(gradient='linear', stops=[alt.GradientStop(color='red', offset=0), alt.GradientStop(color='transparent', offset=1)], x1=1, x2=1, y1=1, y2=0)).encode(x=alt.X('Rows:Q', scale=alt.Scale(domain=X_DOMAIN)), y=alt.Y('Rolling FNR:Q', scale=alt.Scale(domain=Y_FNR_DOMAIN))).properties(height=200, title="Static Model Rolling FNR")
        fpr_c = alt.Chart(baseline_df).mark_area(line={'color':'orange'}, color=alt.Gradient(gradient='linear', stops=[alt.GradientStop(color='orange', offset=0), alt.GradientStop(color='transparent', offset=1)], x1=1, x2=1, y1=1, y2=0)).encode(x=alt.X('Rows:Q', scale=alt.Scale(domain=X_DOMAIN)), y=alt.Y('Rolling FPR:Q', scale=alt.Scale(domain=Y_FPR_DOMAIN))).properties(height=200, title="Static Model Rolling FPR")
        st.altair_chart(fnr_c.interactive(), use_container_width=True); st.altair_chart(fpr_c.interactive(), use_container_width=True)
        if not df_static_missed.empty:
            line = alt.Chart(df_static_missed).mark_line(strokeWidth=3).encode(x=alt.X('Rows:Q', scale=alt.Scale(domain=X_DOMAIN)), y='Missed Samples:Q', color='Strain:N').properties(height=300, title="Cumulative Missed Samples")
            st.altair_chart(line.interactive(), use_container_width=True)
st.divider()

# Live Ensemble View
st.subheader("Live Ensemble Performance")
i_cols = st.columns(3)
stream_light, agent_light, ensemble_size = i_cols[0].empty(), i_cols[1].empty(), i_cols[2].empty()
stream_light.markdown("**STREAM:** IDLE"); agent_light.markdown("**AGENT:** STANDBY"); ensemble_size.metric("Ensemble Size", "0")

l_cols = st.columns(4)
m_acc, m_prec, m_rec, m_f1 = [c.empty() for c in l_cols]
st.divider()
fnr_cont, fpr_cont = st.empty(), st.empty()
attack_cont = st.empty()
st.divider()

with st.expander("Specialist Agent Roster", expanded=True):
    roster_placeholder = st.empty()
st.divider()

left_c, right_c = st.columns(2)
soc_c, agent_c = left_c.empty(), right_c.empty()

if start_sim:
    c_tp, c_fn, c_fp, c_tn, p_idx = 0, 0, 0, 0, 0
    rolling_history, chart_data, retrain_pts = [], [], []
    missed_total, missed_history = Counter(), []
    soc_t, agent_t = "", ""
    
    while True:
        try:
            if not os.path.exists("logs.json"): time.sleep(1); continue
            with open("logs.json", "r") as f: log_data = json.load(f)
        except: time.sleep(0.5); continue

        if p_idx >= len(log_data): time.sleep(0.1); continue

        for i in range(p_idx, len(log_data)):
            entry = log_data[i]; p_idx += 1
            if entry["type"] == "soc":
                stream_light.markdown("**STREAM:** ACTIVE"); agent_light.markdown("**AGENT:** MONITORING")
                r_m = re.search(r'\[(\d+)\]', entry["text"])
                tp_m, fp_m, fn_m, tn_m = [re.search(f'{k}: (\\d+)', entry["text"]) for k in ['TP', 'FP', 'FN', 'TN']]
                ens_m = re.search(r'Ensemble: (\d+)', entry["text"])
                
                if all([r_m, tp_m, fp_m, fn_m, tn_m]):
                    rows, tp, fp, fn, tn = [int(m.group(1)) for m in [r_m, tp_m, fp_m, fn_m, tn_m]]
                    c_tp += tp; c_fp += fp; c_fn += fn; c_tn += tn
                    rolling_history.append({'tp': tp, 'fp': fp, 'fn': fn, 'tn': tn})
                    if len(rolling_history) > 20: rolling_history.pop(0)
                    r_tp, r_fn, r_fp, r_tn = [sum(c[k] for c in rolling_history) for k in ['tp', 'fn', 'fp', 'tn']]
                    r_fnr = (r_fn / max(1, (r_tp + r_fn)) * 100); r_fpr = (r_fp / max(1, (r_tn + r_fp)) * 100)
                    
                    total = c_tp + c_fp + c_fn + c_tn
                    acc = (c_tp + c_tn) / total if total > 0 else 0
                    prec = c_tp / (c_tp + c_fp) if (c_tp + c_fp) > 0 else 0
                    rec = c_tp / (c_tp + c_fn) if (c_tp + c_fn) > 0 else 0
                    f1 = 2 * (prec * rec) / (prec + rec) if (prec + rec) > 0 else 0
                    
                    m_acc.metric("Accuracy", f"{acc*100:.2f}%"); m_prec.metric("Precision", f"{prec*100:.2f}%"); m_rec.metric("Recall", f"{rec*100:.2f}%"); m_f1.metric("F1-Score", f"{f1*100:.2f}%")

                    chart_data.append({"Rows": rows, "Rolling FNR": r_fnr, "Rolling FPR": r_fpr})
                    df_c = pd.DataFrame(chart_data)
                    f_c = alt.Chart(df_c).mark_area(line={'color':'red'}, color=alt.Gradient(gradient='linear', stops=[alt.GradientStop(color='red', offset=0), alt.GradientStop(color='transparent', offset=1)], x1=1, x2=1, y1=1, y2=0)).encode(x=alt.X('Rows:Q', scale=alt.Scale(domain=X_DOMAIN)), y=alt.Y('Rolling FNR:Q', scale=alt.Scale(domain=Y_FNR_DOMAIN))).properties(height=200)
                    p_c = alt.Chart(df_c).mark_area(line={'color':'orange'}, color=alt.Gradient(gradient='linear', stops=[alt.GradientStop(color='orange', offset=0), alt.GradientStop(color='transparent', offset=1)], x1=1, x2=1, y1=1, y2=0)).encode(x=alt.X('Rows:Q', scale=alt.Scale(domain=X_DOMAIN)), y=alt.Y('Rolling FPR:Q', scale=alt.Scale(domain=Y_FPR_DOMAIN))).properties(height=200)
                    if retrain_pts:
                        vl = alt.Chart(pd.DataFrame({'x': [p for p in retrain_pts if p >= df_c['Rows'].min()]})).mark_rule(strokeDash=[5, 5], color='white', opacity=0.3).encode(x='x:Q')
                        f_c += vl; p_c += vl
                    fnr_cont.altair_chart(f_c.interactive(), use_container_width=True); fpr_cont.altair_chart(p_c.interactive(), use_container_width=True)
                    
                    at_m = re.search(r'Attacks: (\{.*?\}|\[.*?\])', entry["text"])
                    if at_m:
                        missed_total.update(parse_attack_data(at_m.group(1)))
                        for s, val in missed_total.items(): missed_history.append({'Rows': rows, 'Strain': s, 'Missed Samples': val})
                        if i % 5 == 0 and missed_history:
                            line = alt.Chart(pd.DataFrame(missed_history)).mark_line(strokeWidth=3).encode(x=alt.X('Rows:Q', scale=alt.Scale(domain=X_DOMAIN)), y='Missed Samples:Q', color='Strain:N').properties(height=300)
                            attack_cont.altair_chart(line.interactive(), use_container_width=True)
                    if ens_m: ensemble_size.metric("Ensemble Size", ens_m.group(1))

                soc_t = (entry["text"].replace("Zero-Day", "DarkNexus") + "\n" + soc_t)[:2000]
                soc_c.code(soc_t, language="bash")
            elif entry["type"] == "agent":
                agent_light.markdown("**AGENT:** TRAINING")
                txt = entry["text"]
                if "Warning" in txt or "Error" in txt: txt = txt.split('\n')[0] + "\n[SYSTEM] Tuning in progress..."
                agent_t = (txt + "\n\n" + agent_t)[:5000]; agent_c.markdown(f"```text\n{agent_t}\n```")
            elif entry["type"] == "critical":
                agent_light.markdown("**AGENT:** BREACH")
                r_m = re.search(r'\[(\d+)\]', entry["text"])
                if r_m: retrain_pts.append(int(r_m.group(1)))
                st.toast(entry["text"]); time.sleep(15)
            elif entry["type"] == "model_metadata" or (i % 20 == 0):
                roster = sync_model_roster()
                with roster_placeholder.container():
                    sorted_m = sorted(roster.values(), key=lambda x: x.get('model_name', ''), reverse=True)
                    active, retired = sorted_m[:5], sorted_m[5:]
                    a_col, r_col = st.columns([3, 1])
                    with a_col:
                        st.markdown("#### Active Ensemble")
                        m_cols = st.columns(len(active) if active else 1)
                        for idx, m_data in enumerate(active):
                            with m_cols[idx]:
                                st.success(f"**{m_data.get('type', 'Specialist')}**")
                                st.caption(f"Deployed: {m_data.get('deployed_at', 'N/A')}")
                                st.json(m_data.get('params', {}))
                    with r_col:
                        st.markdown("#### Retirement Log")
                        for m_data in retired: st.write(f"~~{m_data['model_name']}~~")
    stream_light.markdown("**STREAM:** COMPLETE"); agent_light.markdown("**AGENT:** STANDBY")
