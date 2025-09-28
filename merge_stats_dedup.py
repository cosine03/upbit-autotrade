import sys, pandas as pd, numpy as np
from pathlib import Path

def pick_existing(df, cols):
    return [c for c in cols if c in df.columns]

def load(path):
    df = pd.read_csv(path)
    # 표준화(있는 경우만)
    for c in ['event','side','symbol']:
        if c in df.columns and df[c].dtype == object:
            df[c] = df[c].str.strip()
    # 타임스탬프 표준화(있는 경우만)
    for c in ['ts_sig','entry_ts','exit_ts','ts_bar','ts_entry','ts_exit']:
        if c in df.columns:
            df[c] = pd.to_datetime(df[c], errors='coerce', utc=True)
    return df

def main(trades1, trades2, out_summary, out_trades=None):
    d1, d2 = load(trades1), load(trades2)
    # 둘 다에 'strategy' 라벨 부여(출처 표식)
    d1 = d1.copy()
    d2 = d2.copy()
    d1['strategy'] = d1.get('strategy', 'breakout_only')
    d2['strategy'] = d2.get('strategy', 'boxin_linebreak')

    all_tr = pd.concat([d1, d2], ignore_index=True)

    # 중복 키 정의 (존재하는 컬럼만 사용)
    # 핵심: 동일 신호/동일 심볼/동일 만기/동일 방향이면 동일 트레이드로 간주
    # 가능하면 시간 컬럼도 포함
    candidate_keys = [
        ['symbol','event','side','expiry_h','ts_sig','entry_ts'],
        ['symbol','event','side','expiry_h','ts_sig'],
        ['symbol','event','side','expiry_h','ts_entry'],
        ['symbol','event','side','expiry_h']
    ]
    for key in candidate_keys:
        key_exist = pick_existing(all_tr, key)
        if key_exist and len(key_exist) >= 3:  # 최소 몇 개는 맞춰야 의미 있음
            dedup_key = key_exist
            break
    else:
        # 최후의 수단: 거의-전체 열로 중복 제거(계산비용↑)
        dedup_key = [c for c in all_tr.columns if c not in ['strategy']]

    before = len(all_tr)
    all_tr_dedup = all_tr.drop_duplicates(subset=dedup_key).reset_index(drop=True)
    removed = before - len(all_tr_dedup)

    # 요약 재계산
    # 수익 컬럼 후보(있는 것 우선 사용)
    profit_cols = ['net', 'net_pct', 'pnl', 'ret', 'ret_pct']
    pnl_col = None
    for c in profit_cols:
        if c in all_tr_dedup.columns:
            pnl_col = c
            break
    if pnl_col is None:
        raise SystemExit("수익(손익) 컬럼(net/net_pct/pnl/ret/ret_pct)이 없습니다.")

    # 승패 판단(>0 승, ==0 무승부는 패로 간주)
    win = (all_tr_dedup[pnl_col] > 0).astype(int)

    group_cols = pick_existing(all_tr_dedup, ['event','expiry_h'])
    if not group_cols:
        group_cols = ['event'] if 'event' in all_tr_dedup.columns else []
    if not group_cols:
        group_cols = ['_all']
        all_tr_dedup['_all'] = 'all'

    def agg_fn(g):
        trades = len(g)
        win_rate = (g[pnl_col] > 0).mean() if trades else np.nan
        avg_net = g[pnl_col].mean() if trades else np.nan
        med_net = g[pnl_col].median() if trades else np.nan
        total_net = g[pnl_col].sum() if trades else np.nan
        return pd.Series(dict(
            trades=trades,
            win_rate=win_rate,
            avg_net=avg_net,
            median_net=med_net,
            total_net=total_net
        ))

    summary = all_tr_dedup.groupby(group_cols, dropna=False).apply(agg_fn).reset_index()

    # 저장
    Path(out_summary).parent.mkdir(parents=True, exist_ok=True)
    summary.to_csv(out_summary, index=False, encoding='utf-8')
    if out_trades:
        all_tr_dedup.to_csv(out_trades, index=False, encoding='utf-8')

    # 콘솔 리포트
    print(f"[DE-DUP] input rows={before} -> dedup={len(all_tr_dedup)} (removed {removed})")
    print(f"[OUT] summary -> {out_summary}")
    if out_trades:
        print(f"[OUT] trades  -> {out_trades}")

if __name__ == "__main__":
    if len(sys.argv) < 4:
        print("usage: python merge_stats_dedup.py <trades1.csv> <trades2.csv> <out_summary.csv> [out_trades.csv]")
        sys.exit(1)
    trades1, trades2, out_summary = sys.argv[1], sys.argv[2], sys.argv[3]
    out_trades = sys.argv[4] if len(sys.argv) >= 5 else None
    main(trades1, trades2, out_summary, out_trades)