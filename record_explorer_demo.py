#!/usr/bin/env python3
"""Record the scripted DeepTrace Explorer walkthrough with Playwright."""

import argparse
import shutil
import subprocess
import sys
import time
from pathlib import Path

from playwright.sync_api import sync_playwright


ROOT = Path(__file__).resolve().parent


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "assets/slides/DeepTrace_Explorer_Demo.webm",
    )
    parser.add_argument("--browser", help="Chrome/Chromium executable")
    args = parser.parse_args()

    browser_path = args.browser or shutil.which("google-chrome") or shutil.which("chromium")
    if not browser_path:
        raise SystemExit("Chrome/Chromium was not found; pass --browser /path/to/browser")

    capture_dir = ROOT / "assets/slides/explorer_demo"
    capture_dir.mkdir(parents=True, exist_ok=True)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    server = subprocess.Popen(
        [sys.executable, str(ROOT / "app.py"), "--port", str(args.port)],
        cwd=ROOT,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )

    try:
        time.sleep(2)
        with sync_playwright() as playwright:
            browser = playwright.chromium.launch(
                executable_path=browser_path,
                headless=True,
                args=["--no-sandbox", "--disable-gpu"],
            )
            context = browser.new_context(
                viewport={"width": 1280, "height": 720},
                record_video_dir=str(capture_dir),
                record_video_size={"width": 1280, "height": 720},
            )
            page = context.new_page()
            page.goto(f"http://127.0.0.1:{args.port}", wait_until="networkidle")
            page.evaluate("""()=>{const d=document.createElement('div');d.id='demo-caption';Object.assign(d.style,{position:'fixed',left:'50%',bottom:'18px',transform:'translateX(-50%)',zIndex:9999,background:'#07111ddd',color:'white',padding:'10px 18px',border:'1px solid #6ee7b7',borderRadius:'8px',font:'600 16px system-ui',boxShadow:'0 4px 24px #0008'});document.body.appendChild(d)}""")

            def caption(value):
                page.evaluate("v=>document.querySelector('#demo-caption').textContent=v", value)

            def show_case(label, model, video_id):
                page.get_by_role("button", name="Explore").click()
                page.locator("#predModel").select_option(model)
                page.locator("#q").fill(video_id)
                caption(f"{label} — loading test sample")
                page.wait_for_timeout(1800)
                page.locator(".card").first.click()
                page.wait_for_timeout(1200)
                caption(f"{label} — red: expert box · blue: model box · IoU at right")
                page.locator("#detailVideo").evaluate("v=>v.play().catch(()=>{})")
                page.wait_for_timeout(5200)
                page.locator(".close").click()
                page.wait_for_timeout(800)

            caption("DeepTrace Explorer — local dataset and benchmark walkthrough")
            page.wait_for_timeout(3500)
            show_case("Track C · Qwen2.5-VL-3B · SAME", "qwen2.5-vl-3b", "cm49k6zcn06710759k0g9aq4v")
            show_case("Track C · Qwen2.5-VL-3B · DIFFERENT", "qwen2.5-vl-3b", "cm2p0z6r111at0709f1l5ubdb")
            show_case("Track B · TruFor · SAME", "trufor", "cm3e9rzrb1zq20705fhr8zg58")
            show_case("Track B · TruFor · DIFFERENT", "trufor", "cm2u5s9m400hu0736tnep1q3q")

            page.locator("#q").fill("")
            page.get_by_role("button", name="Dataset analytics").click()
            caption("Dataset analytics — sources, splits, trace categories and coverage")
            page.wait_for_timeout(6500)
            page.get_by_role("button", name="Model performance").click()
            page.locator("#performanceModel").select_option("qwen2.5-vl-3b")
            caption("Best Track C model — Qwen2.5-VL-3B")
            page.wait_for_timeout(6500)
            page.mouse.wheel(0, 500)
            page.wait_for_timeout(3500)
            page.locator("#performanceModel").select_option("trufor")
            caption("Best Track B model — TruFor")
            page.wait_for_timeout(6500)
            page.mouse.wheel(0, 550)
            page.wait_for_timeout(3500)
            caption("End of walkthrough")
            page.wait_for_timeout(2500)

            video = page.video
            context.close()
            browser.close()
            Path(video.path()).replace(args.output)
            print(args.output)
    finally:
        if server.poll() is None:
            server.terminate()
            server.wait(timeout=5)


if __name__ == "__main__":
    main()
