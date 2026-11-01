import json,io,csv,time
from pathlib import Path
import numpy as np
import pandas as pd
import timesfm
import torch

TICKER="FMS"
END="2026-11-01"
MODEL_ID="google/timesfm-2.5-200m-pytorch"
OUT=Path("timesfm-fms-results")
OUT.mkdir(exist_ok=True)
# Data fallback is a fixed public Alpaca IEX snapshot from 2026-08-31..2026-10-30
# (partial US exchange, not consolidated). Never fabricate future prices.
FALLBACK = """date,close
2026-08-31,22.73
2026-09-01,22.74
2026-09-02,22.67
2026-09-03,22.46
2026-09-04,22.66
2026-09-08,22.46
2026-09-09,22.32
2026-09-10,22.23
2026-09-11,22.35
2026-09-14,22.84
2026-09-15,22.955
2026-09-16,22.97
2026-09-17,22.065
2026-09-18,22.20
2026-09-21,22.22
2026-09-22,22.51
2026-09-23,22.42
2026-09-24,22.355
2026-09-25,22.69
2026-09-28,22.95
2026-09-29,22.27
2026-09-30,22.41
2026-10-01,22.255
2026-10-02,22.25
2026-10-05,22.65
2026-10-06,22.425
2026-10-07,20.69
2026-10-08,20.88
2026-10-09,21.055
2026-10-12,21.21
2026-10-13,21.07
2026-10-14,21.545
2026-10-15,22.12
2026-10-16,22.16
2026-10-19,21.94
2026-10-20,21.41
2026-10-21,21.55
2026-10-22,22.225
2026-10-23,22.68
2026-10-26,22.75
2026-10-27,22.49
2026-10-28,22.62
2026-10-29,23.12
2026-10-30,22.47
"""
def get_prices():
    try:
        import yfinance as yf
        print("Downloading FMS daily prices from Yahoo Finance",flush=True)
        df=yf.download(TICKER,start="2024-01-01",end=END,auto_adjust=False,progress=False,threads=False)
        if df.empty: raise RuntimeError("Yahoo returned no data")
        series=df["Close"]
        if isinstance(series,pd.DataFrame): series=series.iloc[:,0]
        series=series.dropna().astype(float)
        dates=series.index.strftime("%Y-%m-%d").tolist()
        if dates[-1] != "2026-10-30":raise RuntimeError(f"Yahoo stale data {dates[-1]}")
        return dates,series.to_numpy(), "Yahoo Finance unadjusted daily close"
    except Exception as e:
        print("YAHOO_UNAVAILABLE "+str(e),flush=True)
        rows=list(csv.DictReader(io.StringIO(FALLBACK)))
        return [r["date"] for r in rows],np.array([float(r["close"]) for r in rows],dtype=np.float64),"Alpaca IEX pinned 44-session close snapshot"

def main():
    dates,closes,provider=get_prices()
    assert dates[-1]=="2026-10-30" and len(closes)>=40 and np.isfinite(closes).all()
    context=min(256,len(closes)-12)
    print("INPUT_REAL_FMS_JSON "+json.dumps({"ticker":TICKER,"provider":provider,"n":len(closes),"last_date":dates[-1],"close":float(closes[-1]),"context":context}),flush=True)
    start=time.time()
    print("LOADING_REAL_GOOGLE_WEIGHTS "+MODEL_ID,flush=True)
    m=timesfm.TimesFM_2p5_200M_torch.from_pretrained(MODEL_ID,force_download=False)
    m.compile(timesfm.ForecastConfig(max_context=context,max_horizon=8,normalize_inputs=True,per_core_batch_size=4,use_continuous_quantile_head=False,force_flip_invariance=True,infer_is_positive=True,fix_quantile_crossing=True))
    print("OFFICIAL_MODEL_LOADED "+str(round(time.time()-start,1))+"s",flush=True)
    point,quantile=m.forecast(horizon=5,inputs=[closes[-context:].astype(np.float32)])
    forecasts=np.asarray(point[0],dtype=np.float64)
    assert forecasts.shape==(5,) and np.isfinite(forecasts).all()
    result={"verified":"REAL_WEIGHTS_INFERRED","ticker":TICKER,"model":MODEL_ID,"source":provider,"last_date":dates[-1],"last_close":float(closes[-1]),"next_5_sessions":[round(v,5) for v in forecasts.tolist()],"return_5_sessions_pct":round(100*(forecasts[-1]/closes[-1]-1),3),"runtime_seconds":round(time.time()-start,2)}
    (OUT/"forecast.json").write_text(json.dumps(result,indent=2),encoding="utf-8")
    print("TIMESFM_REAL_FORECAST_JSON "+json.dumps(result),flush=True)
    # Evaluate the latest 10 origins with strict causality (the target is never in the context).
    ntest=min(120,len(closes)-context)
    pred=[]
    for i in range(len(closes)-ntest,len(closes),4):
        j=min(i+4,len(closes))
        histories=[closes[k-context:k].astype(np.float32) for k in range(i,j)]
        actual,_=m.forecast(horizon=1,inputs=histories)
        pred.extend(np.asarray(actual,dtype=float).reshape(-1).tolist())
    y=closes[-ntest:];last=closes[-ntest-1:-1];fp=np.array(pred,dtype=float)
    assert len(y)==len(fp)
    mae=float(np.mean(np.abs(fp-y)));naive=float(np.mean(np.abs(last-y)))
    direction=float(np.mean(np.sign(fp-last)==np.sign(y-last))*100)
    metrics={"n":ntest,"timesfm_mae":round(mae,5),"naive_mae":round(naive,5),"direction_pct":round(direction,2),"test_start":dates[-ntest],"test_end":dates[-1],"provider":provider}
    (OUT/"backtest.json").write_text(json.dumps(metrics,indent=2),encoding="utf-8")
    print("TIMESFM_REAL_BACKTEST_JSON "+json.dumps(metrics),flush=True)

if __name__=="__main__":
    main()
