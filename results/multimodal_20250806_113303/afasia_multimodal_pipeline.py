# afasia_multimodal_pipeline.py
"""
**Resumen rápido**
===================
Este script construye **un vector de 1 792 números por paciente**
(1 024 audio + 768 texto) y entrena un pequeño MLP para predecir la
puntuación **WAB-AQ / QA** (0-100) que mide la severidad de la afasia.

1. **Datos de entrada**
   * `df_aphbank_pos_metrics.csv`  → inglés (`en`) y castellano (`es`).
   * `df_catalan_pos_metrics.csv`  → catalán (`ca`).
   * Cada fila = **chunk** de audio con su transcripción (`Marca`) y
     ~160 métricas léxicas / POS.  Solo usamos la ruta `name_chunk_audio_path`
     y la transcripción cruda.

2. **Embeddings por chunk**
   * **Audio**   → wav2vec2-XLS-R (1 024 dim).
   * **Texto**   → LaBSE (768 dim).

3. **Agregación paciente (`CIP`)**
   * Para cada paciente promediamos **todas** las representaciones de
     sus chunks → 1 vector audio + 1 vector texto.
   * Concatenamos y añadimos la etiqueta `QA`.

4. **Splits**
   * **Train + Val**: solo pacientes `en`  (80 % / 20 %, estratificado
     a nivel CIP).
   * **Test-int**  : todos los pacientes `es` (transferencia idiomática).
   * **Test-ext**  : todos los pacientes `ca` (dominio externo).

5. **Modelo y métrica**
   * MLP (1792 → 512 → 128 → 1) con ReLU+Dropout.
   * Optimiza **MAE**; se reportan además R² y ρ de Spearman.

Resultado: en la última tirada obtuvimos ≈ **MAE 12.4** en Test-int
(castellano) y **MAE 29.5** en Test-ext (catalán).

---

Debajo se muestra el código completo del pipeline.
"""
from __future__ import annotations
import os, datetime, pathlib, warnings; warnings.filterwarnings("ignore")
from typing import Optional

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import librosa, torch
from tqdm import tqdm
from sklearn.model_selection import GroupShuffleSplit
from sklearn.metrics import r2_score
from scipy.stats import spearmanr
from torch import nn
from torch.utils.data import Dataset, DataLoader

# --------------------------------------------------
# 1 · RUTAS Y CONFIG
# --------------------------------------------------
DATA_BASE     = "/lhome/ext/upc150/upc1503/afasia_cat/codigos_julio2025/data"
PATH_APHBANK  = os.path.join(DATA_BASE, "df_aphbank_pos_metrics.csv")   # en + es
PATH_CAT      = os.path.join(DATA_BASE, "df_catalan_pos_metrics.csv")   # ca

AUDIO_SUBDIRS = ["aphasiabank_en", "aphasiabank_es", "audios_chunks"]
PATH_PREFIX_MAP = [
    ("/content/drive/MyDrive/tesis_monica/afasia/data/", DATA_BASE + "/"),
    (DATA_BASE + "/", DATA_BASE + "/")        
]

HF_WAV2VEC = "/lhome/ext/upc150/upc1503/hf_models/wav2vec2-xls-r-1b"
HF_LABSE   = "/lhome/ext/upc150/upc1503/hf_models/LaBSE"

SR   = 16_000
DEV  = "cuda" if torch.cuda.is_available() else "cpu"

# --------------------------------------------------
# 2 · CARGA LAZY DE MODELOS HF
# --------------------------------------------------
feat_ext = aud_model = tok = txt_model = None
def _load_models():
    global feat_ext, aud_model, tok, txt_model
    if feat_ext is None:
        from transformers import AutoModel, AutoTokenizer, Wav2Vec2FeatureExtractor
        print("[INFO] Loading wav2vec2-XLS-R …")
        feat_ext  = Wav2Vec2FeatureExtractor.from_pretrained(HF_WAV2VEC, local_files_only=True)
        aud_model = AutoModel.from_pretrained(HF_WAV2VEC,     local_files_only=True).to(DEV).eval()
        print("[INFO] Loading LaBSE …")
        tok       = AutoTokenizer.from_pretrained(HF_LABSE,   local_files_only=True)
        txt_model = AutoModel.from_pretrained(HF_LABSE,       local_files_only=True).to(DEV).eval()

# --------------------------------------------------
# 3 · EMBEDDINGS
# --------------------------------------------------
def emb_audio(path: str) -> np.ndarray:
    _load_models()
    wav,_ = librosa.load(path, sr=SR)
    inp   = feat_ext(wav, sampling_rate=SR, return_tensors="pt")
    with torch.no_grad():
        out = aud_model(**{k:v.to(DEV) for k,v in inp.items()})
    return out.last_hidden_state.mean(1).cpu().numpy().squeeze()

def emb_text(txt: str) -> np.ndarray:
    _load_models()
    inp = tok(txt, return_tensors="pt", truncation=True, padding=True)
    with torch.no_grad():
        out = txt_model(**{k:v.to(DEV) for k,v in inp.items()})
    return out.pooler_output.cpu().numpy().squeeze()

# --------------------------------------------------
# 4 · RESOLVER RUTA WAV
# --------------------------------------------------
def resolve(rel_path: str) -> Optional[str]:
    p = rel_path or ""
    for old,new in PATH_PREFIX_MAP:
        if p.startswith(old):
            p = p.replace(old, new, 1)
            break
    if os.path.isabs(p) and os.path.isfile(p):
        return p
    rel = p.lstrip("/")
    cand = os.path.join(DATA_BASE, rel)
    if os.path.isfile(cand):
        return cand
    base = os.path.basename(rel)
    for sd in AUDIO_SUBDIRS:
        cand = os.path.join(DATA_BASE, sd, base)
        if os.path.isfile(cand):
            return cand
    return None

# --------------------------------------------------
# 5 · AGRUPAR A NIVEL PACIENTE
# --------------------------------------------------
def build_patient(df_chunks: pd.DataFrame, out: pathlib.Path) -> pd.DataFrame:
    rows, miss = [], []
    for pid, grp in tqdm(df_chunks.groupby("CIP"), desc="Pacientes"):
        a_vecs, t_vecs, qa = [], [], None
        for _,r in grp.iterrows():
            wav = resolve(str(r["name_chunk_audio_path"]))
            if wav is None:
                miss.append({"CIP":pid,"path":r["name_chunk_audio_path"],"reason":"path_not_found"}); continue
            try:
                a_vecs.append(emb_audio(wav))
            except Exception as e:
                miss.append({"CIP":pid,"path":wav,"reason":f"load_error:{e}"}); continue
            t_vecs.append(emb_text(str(r["Marca"]))); qa = r["QA"]
        if not a_vecs:
            miss.append({"CIP":pid,"path":"<no_valid>","reason":"no_audio"}); continue
        rows.append({"CIP":pid,"QA":float(qa),
                     **{f"a{i}":v for i,v in enumerate(np.mean(a_vecs,0))},
                     **{f"t{i}":v for i,v in enumerate(np.mean(t_vecs,0))}})
    pd.DataFrame(miss).to_csv(out/"missing_audios_log.csv", index=False)
    return pd.DataFrame(rows)

# --------------------------------------------------
# 6 · DATASET PyTorch
# --------------------------------------------------
class PDataset(Dataset):
    def __init__(self, df: pd.DataFrame):
        drop = [c for c in ("QA","CIP","lang") if c in df.columns]
        self.X = df.drop(columns=drop).select_dtypes(include=[np.number]).values.astype(np.float32)
        self.y = df["QA"].values.astype(np.float32)
    def __len__(self): return len(self.y)
    def __getitem__(self,i): return torch.from_numpy(self.X[i]), torch.tensor(self.y[i])

# --------------------------------------------------
# 7 · MODELO
# --------------------------------------------------
class MLP(nn.Module):
    def __init__(self, d_in:int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_in,512), nn.ReLU(), nn.Dropout(0.3),
            nn.Linear(512,128),  nn.ReLU(),
            nn.Linear(128,1))
    def forward(self,x): return self.net(x).squeeze(-1)

# --------------------------------------------------
# 8 · ENTRENAMIENTO
# --------------------------------------------------
def mae(model: nn.Module, loader: DataLoader) -> float:
    model.eval(); s=n=0
    with torch.no_grad():
        for xb,yb in loader:
            xb,yb = xb.to(DEV), yb.to(DEV)
            s += torch.sum(torch.abs(model(xb)-yb)).item(); n += len(yb)
    return s/n

def fit(model, dl_tr, dl_val, lr=1e-4, epochs=100, patience=10):
    model = model.to(DEV); opt = torch.optim.Adam(model.parameters(), lr=lr)
    best,wait,ht,hv = float("inf"),0,[],[]
    crit = nn.L1Loss()
    for e in range(epochs):
        model.train(); s=n=0
        for xb,yb in dl_tr:
            xb,yb = xb.to(DEV), yb.to(DEV)
            opt.zero_grad(); loss = crit(model(xb),yb); loss.backward(); opt.step()
            s += torch.sum(torch.abs(model(xb)-yb)).item(); n += len(yb)
        tr, val = s/n, mae(model, dl_val)
        ht.append(tr); hv.append(val)
        print(f"Ep {e:03d} | tr {tr:.2f} | val {val:.2f}")
        if val + 0.05 < best:
            best,wait = val,0; torch.save(model.state_dict(),"best.pt")
        else:
            wait += 1
            if wait >= patience:
                print("Early-stopping"); break
    model.load_state_dict(torch.load("best.pt"))
    return model,ht,hv

# --------------------------------------------------
# 9 · EVALUACIÓN + GRÁFICOS
# --------------------------------------------------
def eval_and_plot(model, df, out_dir: pathlib.Path, tag:str):
    out_dir.mkdir(parents=True, exist_ok=True)
    dl = DataLoader(PDataset(df), batch_size=64)
    model = model.to(DEV).eval()
    preds,trues = [],[]
    with torch.no_grad():
        for xb,yb in dl:
            xb = xb.to(DEV)
            preds.extend(model(xb).cpu().numpy()); trues.extend(yb.numpy())
    preds,trues = np.array(preds), np.array(trues)
    mae_v = np.mean(np.abs(preds-trues)); r2_v = r2_score(trues,preds); rho_v,_ = spearmanr(preds,trues)

    pd.DataFrame({"CIP":df.CIP,"QA":trues,"pred":preds,"err":preds-trues}).to_csv(out_dir/"preds.csv",index=False)

    plt.figure(figsize=(4,4)); plt.scatter(trues,preds,alpha=.7)
    mn,mx = trues.min(),trues.max(); plt.plot([mn,mx],[mn,mx],'r--')
    plt.title(f"{tag}  MAE={mae_v:.2f}  R2={r2_v:.2f}  ρ={rho_v:.2f}")
    plt.tight_layout(); plt.savefig(out_dir/"scatter.png",dpi=150); plt.close()

    plt.figure(); plt.hist(preds-trues, bins=15)
    plt.title(f"Histograma errores {tag}"); plt.xlabel("pred-real")
    plt.tight_layout(); plt.savefig(out_dir/"err_hist.png",dpi=150); plt.close()
    return mae_v,r2_v,rho_v

# --------------------------------------------------
# 10 · MAIN
# --------------------------------------------------
if __name__=="__main__":
    # 10.1 Cargar CSVs
    dfs=[]
    if os.path.isfile(CSV_EN_ES):
        df_aph=pd.read_csv(CSV_EN_ES,encoding="utf-8")
        m_en=df_aph["name_chunk_audio_path"].str.contains("aphasiabank_en",na=False)
        df_aph.loc[m_en,"LLengWAB"]=3
        dfs.append(df_aph)
    if os.path.isfile(CSV_CA):
        dfs.append(pd.read_csv(CSV_CA,encoding="utf-8"))
    if not dfs: raise SystemExit("Sin CSV de entrada")
    df=pd.concat(dfs,ignore_index=True)\
         .dropna(subset=["QA","CIP","name_chunk_audio_path","Marca"])

    # 10.2 Idioma
    def lang_row(r):
        p=r["name_chunk_audio_path"]; l=r.get("LLengWAB",0)
        if "aphasiabank_en" in p or l==3: return "en"
        if "aphasiabank_es" in p or l==2: return "es"
        return "ca"
    df["lang"]=df.apply(lang_row,axis=1)

    # 10.3 Resultados
    run_dir = pathlib.Path("results")/f"multimodal_{timestamp}"
    run_dir.mkdir(parents=True)

    # 10.4 Embeddings + agregación paciente
    df_pat=build_patient(df,run_dir)
    df_pat["lang"]=df_pat["CIP"].map(df.groupby("CIP")["lang"].first())

    n_en,n_es,n_ca=(df_pat.lang=="en").sum(),(df_pat.lang=="es").sum(),(df_pat.lang=="ca").sum()
    print(f"Pacientes  en:{n_en}  es:{n_es}  ca:{n_ca}")
    if n_en==0: raise SystemExit("Sin pacientes en inglés")

    # 10.5 Splits
    df_en=df_pat[df_pat.lang=="en"].reset_index(drop=True)
    df_es=df_pat[df_pat.lang=="es"].reset_index(drop=True)
    df_ca=df_pat[df_pat.lang=="ca"].reset_index(drop=True)

    if len(df_en)<3:
        df_tr,df_val=df_en,pd.DataFrame(columns=df_en.columns)
    else:
        gss=GroupShuffleSplit(test_size=0.20,random_state=42)
        tr,val=next(gss.split(df_en,groups=df_en.CIP))
        df_tr,df_val=df_en.iloc[tr],df_en.iloc[val]

    df_int,df_ext=df_es,df_ca

    pd.concat([
        pd.DataFrame({"CIP":df_tr.CIP,"split":"train"}),
        pd.DataFrame({"CIP":df_val.CIP,"split":"val"}),
        pd.DataFrame({"CIP":df_int.CIP,"split":"test_int"}),
        pd.DataFrame({"CIP":df_ext.CIP,"split":"test_ext"}),
    ]).to_csv(run_dir/"cip_split.csv",index=False)

    print(f"Train {len(df_tr)} | Val {len(df_val)} | Test-int {len(df_int)} | Test-ext {len(df_ext)}")

    # 10.6 Entrenamiento
    dl_tr=DataLoader(PDataset(df_tr),batch_size=32,shuffle=True)
    dl_val=DataLoader(PDataset(df_val),batch_size=64) if len(df_val) else dl_tr
    model=MLP(dl_tr.dataset[0][0].numel())
    model,h_tr,h_val=fit(model,dl_tr,dl_val)

    # curva aprendizaje
    plt.figure(figsize=(6,4)); plt.plot(h_tr,label="train"); plt.plot(h_val,label="val")
    plt.xlabel("Época"); plt.ylabel("MAE"); plt.grid(); plt.legend(); plt.title("Curva aprendizaje")
    plt.tight_layout(); plt.savefig(run_dir/"learning_curve.png",dpi=150); plt.close()

    # 10.7 Evaluaciones (cuatro splits)
    mae_tr,r2_tr,rho_tr = eval_and_plot(model,df_tr,  run_dir/"train","TRAIN")
    if len(df_val):
        mae_v,r2_v,rho_v = eval_and_plot(model,df_val,run_dir/"val",  "VAL")
    if len(df_int):
        mae_i,r2_i,rho_i = eval_and_plot(model,df_int,run_dir/"int",  "INT")
    if len(df_ext):
        mae_e,r2_e,rho_e = eval_and_plot(model,df_ext,run_dir/"ext",  "EXT")

    print("Fin del pipeline – archivos en",run_dir)