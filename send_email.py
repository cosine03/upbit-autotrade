# send_email.py
import os, argparse, smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from email.mime.base import MIMEBase
from email import encoders

try:
    from dotenv import load_dotenv
    load_dotenv()  # .env 로드
except Exception:
    pass  # python-dotenv 미설치면 건너뜀 (환경변수 직접 지정한 경우 OK)

def add_attachment(msg, path):
    if not path or not os.path.exists(path):
        print(f"[WARN] attachment not found: {path}")
        return
    with open(path, "rb") as f:
        part = MIMEBase("application", "octet-stream")
        part.set_payload(f.read())
    encoders.encode_base64(part)
    part.add_header("Content-Disposition", f'attachment; filename="{os.path.basename(path)}"')
    msg.attach(part)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--subject", required=True)
    ap.add_argument("--body", default="")
    ap.add_argument("--attach", action="append", default=[], help="파일 첨부. 여러 번 지정 가능")
    args = ap.parse_args()

    smtp_host = os.getenv("SMTP_HOST")
    smtp_port = int(os.getenv("SMTP_PORT", "587"))
    smtp_user = os.getenv("SMTP_USER")
    smtp_pass = os.getenv("SMTP_PASS")
    mail_from = os.getenv("MAIL_FROM", smtp_user or "")
    mail_to   = [x.strip() for x in os.getenv("MAIL_TO","").split(",") if x.strip()]

    if not (smtp_host and smtp_user and smtp_pass and mail_to):
        raise SystemExit("[ERR] SMTP or MAIL_* env not set. Check .env")

    msg = MIMEMultipart()
    msg["From"] = mail_from
    msg["To"] = ", ".join(mail_to)
    msg["Subject"] = args.subject
    msg.attach(MIMEText(args.body, "plain", "utf-8"))

    for p in args.attach:
        add_attachment(msg, p)

    with smtplib.SMTP(smtp_host, smtp_port, timeout=30) as server:
        server.starttls()
        server.login(smtp_user, smtp_pass)
        server.sendmail(mail_from, mail_to, msg.as_string())
    print("[MAIL] sent")

if __name__ == "__main__":
    main()