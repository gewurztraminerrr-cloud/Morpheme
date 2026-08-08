import pexpect

def main():
    password = "CT4n2S#sQ918"
    ip = "132.148.72.249"
    images = [
        "static/images/boggle_classic.jpg",
        "static/images/boggle_big.jpg",
        "static/images/boggle_super_big.jpg",
    ]
    for f in images:
        print(f"Uploading {f}...")
        cmd = f"scp {f} morpheme@{ip}:/home/morpheme/morpheme/{f}"
        child = pexpect.spawn(cmd, encoding='utf-8', timeout=60)
        idx = child.expect([r"Are you sure you want to continue connecting", r"[Pp]assword:"])
        if idx == 0:
            child.sendline("yes")
            child.expect([r"[Pp]assword:"])
        child.sendline(password)
        child.expect(pexpect.EOF)
        print(f"Uploaded {f} successfully.")
    print("All Boggle images uploaded!")

if __name__ == "__main__":
    main()
