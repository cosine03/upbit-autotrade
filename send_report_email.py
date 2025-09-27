# send_report_email.py
# Usage:
#   python send_report_email.py --daily-dir ".\logs\daily\2025-09-27_AM" --subject "[Autotrade] Daily Report" --attach-glob "daily_report_*.png;daily_report_upbit_main.xlsx"
import os, glob, argparse, mimetypes, smtplib
from email.message import EmailMessage
from datetime import datetime, timezone
from dotenv import load_dotenv

def gather_attachments(daily_dir:str, patterns:str):
    files = []
    for pat in patterns.split(";"):
        pat = pat.strip()
        if not pat: continue
        files += glob.glob(os.path.join(daily_dir, pat))
    # 중복 제거 & 존재하는 파일만
    uniq = []
    seen = set()
    for f in files:
        f = os.path.normpath(f)
        if os.path.exists(f) and f not in seen:
            uniq.append(f); seen.add(f)
    return uniq

def attach_file(msg:EmailMessage, path:str):
    ctype, encoding = mimetypes.guess_type(path)
    if ctype is None:
        ctype = "application/octet-stream"
    maintype, subtype = ctype.split("/", 1)
    with open(path, "rb") as fh:
        msg.add_attachment(fh.read(), maintype=maintype, subtype=subtype,
                           filename=os.path.basename(path))

def main():
    load_dotenv()
    ap = argparse.ArgumentParser()
    ap.add_argument("--daily-dir", required=True)
    ap.add_argument("--subject", default="[Autotrade] Daily Report")
    ap.add_argument("--body", default="자동 생성 리포트입니다.")
    ap.add_argument("--attach-glob", default="daily_report_upbit_main.xlsx;daily_report_*.png")
    args = ap.parse_args()

    smtp_host = os.getenv("SMTP_HOST")
    smtp_port = int(os.getenv("SMTP_PORT", "587"))
    smtp_user = os.getenv("SMTP_USER")
    smtp_pass = os.getenv("SMTP_PASS")
    mail_from = os.getenv("MAIL_FROM", smtp_user)
    mail_to   = os.getenv("MAIL_TO")  # "a@x.com,b@y.com"

    if not (smtp_host and smtp_user and smtp_pass and mail_to):
        raise SystemExit("[ERR] SMTP_* / MAIL_* 환경변수(.env) 부족")

    # 제목에 날짜/폴더명 태그 포함
    ts = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    subject = f"{args.subject} | {os.path.basename(args.daily_dir)} | {ts}"

    msg = EmailMessage()
    msg["From"] = mail_from
    msg["To"]   = mail_to
    msg["Subject"] = subject
    msg.set_content(args.body)

    atts = gather_attachments(args.daily_dir, args.attach_glob)
    if not atts:
        print("[WARN] 첨부파일 없음. 본문만 보냄.")
    for p in atts:
        attach_file(msg, p)
        print("[ATTACH]", p)

    with smtplib.SMTP(smtp_host, smtp_port, timeout=30) as s:
        s.ehlo()
        s.starttls()
        s.login(smtp_user, smtp_pass)
        s.send_message(msg)
    print("[MAIL] sent to", mail_to)

if __name__ == "__main__":
    main()