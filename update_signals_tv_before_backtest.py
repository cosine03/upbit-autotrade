import argparse, csv, sys, re
from pathlib import Path
from datetime import datetime, timezone

# -------- settings --------
REQUIRED_COLS = ["ts","event","side","level","touches","symbol","timeframe","extra","host","message","distance_pct"]
KEY_COLS      = ["ts","symbol","event","side","level","touches"]

MSG_PIPE_NOISE   = re.compile(r"\s*\|\s*[_?]+\s*")     # " | _ " / " | ?? " -> " | "
MSG_LEAD_NOISE   = re.compile(r"^\s*[_?]+\s*")         # 앞쪽 "_" "?" 제거
MSG_MULTI_Q      = re.compile(r"\?\?+")                # "??" -> ""
MSG_SQUEEZE_WS   = re.compile(r"\s{2,}")               # 다중 공백 -> 하나
INTERNAL_HOST_RX = re.compile(r"System\.Management\.Automation\.Internal\.Host", re.I)

def clean_message(m: str) -> str:
    if not m: return ""
    m = MSG_PIPE_NOISE.sub(" | ", m)
    m = MSG_LEAD_NOISE.sub("", m)
    m = MSG_MULTI_Q.sub("", m)
    m = MSG_SQUEEZE_WS.sub(" ", m.strip())
    return m

def parse_ts_utc(s: str) -> str:
    # 다양한 ISO8601 변형을 UTC로 normalize -> ISO8601(Z 없음, +00:00 오프로 저장)
    # ex) 2025-09-30T18:15:25.436917+00:00 / 2025-09-30T18:15:25+00:00
    # 실패 시 예외 -> 상위에서 스킵 또는 보고
    dt = datetime.fromisoformat(s.replace("Z","+00:00"))
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    else:
        dt = dt.astimezone(timezone.utc)
    # 문자열로 환원 (백테스트 파이프와 맞춤: 마이크로초 보존)
    if dt.microsecond:
        return dt.strftime("%Y-%m-%dT%H:%M:%S.%f+00:00")
    return dt.strftime("%Y-%m-%dT%H:%M:%S+00:00")

def row_key(r: dict) -> str:
    return "|".join(r.get(k,"") for k in KEY_COLS)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input",  default=r"D:\upbit_autotrade_starter\logs\signals_tv_full.csv")
    ap.add_argument("--output", default=r"D:\upbit_autotrade_starter\logs\signals_tv_ready.csv")
    ap.add_argument("--symbol-prefix", default="", help="예: KRW- 로 제한하려면 KRW- 지정")
    ap.add_argument("--timeframe", default="", help="예: 15m 로 제한. 빈값이면 제한하지 않음")
    ap.add_argument("--exclude-internalhost", action="store_true", help="InternalHost 문자열 호스트 행 제외")
    ap.add_argument("--exclude-localhost",  action="store_true", help="127.0.0.1 행 제외(기본 미제외)")
    ap.add_argument("--start", default="", help="UTC 시작(포함), 예: 2025-09-17T00:00:00+00:00")
    ap.add_argument("--end",   default="", help="UTC 끝(포함),   예: 2025-10-04T23:59:59+00:00")
    args = ap.parse_args()

    src = Path(args.input)
    if not src.exists():
        print(f"[ERR] input not found: {src}", file=sys.stderr); sys.exit(1)

    with src.open("r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        # 헤더 보강(누락시 빈칸)
        cols = list(reader.fieldnames or [])
        for c in REQUIRED_COLS:
            if c not in cols:
                cols.append(c)

        rows_ok, dropped = [], {"missing_fields":0,"ts_parse_fail":0,"filters":0}
        for raw in reader:
            r = {c: raw.get(c, "") for c in cols}

            # 필수 필드
            if not r["ts"] or not r["event"] or not r["symbol"]:
                dropped["missing_fields"] += 1; continue

            # ts normalize(UTC)
            try:
                r["ts"] = parse_ts_utc(r["ts"])
            except Exception:
                dropped["ts_parse_fail"] += 1; continue

            # 심볼/타임프레임 필터
            if args.symbol_prefix and not r["symbol"].startswith(args.symbol_prefix):
                dropped["filters"] += 1; continue
            if args.timeframe:
                tf = r.get("timeframe","")
                if tf and tf != args.timeframe:
                    dropped["filters"] += 1; continue

            # 호스트 처리
            h = r.get("host","") or ""
            if INTERNAL_HOST_RX.search(h):
                # 내부호스트는 “제외”가 아니라 기본은 “정상화(빈칸)” — 제외 원하면 플래그 사용
                if args.exclude_internalhost:
                    dropped["filters"] += 1; continue
                r["host"] = ""  # normalize
            if args.exclude_localhost and h == "127.0.0.1":
                dropped["filters"] += 1; continue

            # 메시지 클린업
            r["message"] = clean_message(r.get("message",""))

            rows_ok.append(r)

    # 중복 제거
    seen = set()
    dedup = []
    for r in rows_ok:
        k = row_key(r)
        if k not in seen:
            seen.add(k)
            dedup.append(r)

    # ts 정렬
    dedup.sort(key=lambda x: x["ts"])

    # 저장
    dst = Path(args.output)
    with dst.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=REQUIRED_COLS, quoting=csv.QUOTE_MINIMAL)
        writer.writeheader()
        for r in dedup:
            writer.writerow({c: r.get(c,"") for c in REQUIRED_COLS})

    # 요약
    from collections import Counter
    by_event = Counter(r["event"] for r in dedup)
    by_host  = Counter(r.get("host","") for r in dedup)
    by_day   = Counter(r["ts"][:10] for r in dedup)

    print("✅ prepare_signals_tv_before_backtest: DONE")
    print(f"- input : {src}")
    print(f"- output: {dst}")
    print(f"- kept  : {len(dedup)} rows")
    print(f"- drop  : {dropped}")
    print("- events:", dict(by_event.most_common()))
    print("- hosts :", dict(by_host.most_common()))
    print("- range :", dedup[0]['ts'] if dedup else "-", "→", dedup[-1]['ts'] if dedup else "-")
    print("- days  :", len(by_day))