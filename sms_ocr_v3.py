#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import argparse
from datetime import datetime
from typing import List, Dict, Tuple

import csv
import cv2
import numpy as np
from PIL import Image
import pytesseract

# ---- regex fallback
try:
    import regex as re2
    _HANGUL_CLASS = r'\p{Hangul}'
    def hangul_findall(pat, s): return re2.findall(pat, s)
    def hangul_sub(pat, rep, s): return re2.sub(pat, rep, s)
    def hangul_search(pat, s): return re2.search(pat, s)
except Exception:
    import re as re2
    _HANGUL_CLASS = r'[가-힣]'
    def hangul_findall(pat, s): return re2.findall(pat, s)
    def hangul_sub(pat, rep, s): return re2.sub(pat, rep, s)
    def hangul_search(pat, s): return re2.search(pat, s)

TIME_RX = re2.compile(r'(오전|오후)\s*\d{1,2}\s*[:：]\s*\d{2}')
HANGUL = _HANGUL_CLASS

# =========================
# 유틸/전처리
# =========================
def auto_params(img_h: int, img_w: int) -> Tuple[float, Tuple[int,int]]:
    base = max(img_w, img_h)
    scale = 2.0 if base < 1100 else (1.8 if base < 1700 else 1.4)
    kx = max(11, int(img_w * 0.012))
    ky = max(5, int(img_h * 0.006))
    return scale, (kx | 1, ky | 1)

def upscale_and_gray(color_bgr: np.ndarray, scale: float) -> Tuple[np.ndarray, np.ndarray]:
    if scale != 1.0:
        color_bgr = cv2.resize(color_bgr, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)
    color_bgr = cv2.bilateralFilter(color_bgr, 7, 40, 40)
    gray = cv2.cvtColor(color_bgr, cv2.COLOR_BGR2GRAY)
    clahe = cv2.createCLAHE(clipLimit=2.2, tileGridSize=(8,8))
    gray = clahe.apply(gray)
    return color_bgr, gray

def find_bubbles(gray: np.ndarray, kernel_wh: Tuple[int,int]) -> List[Tuple[int,int,int,int]]:
    k = cv2.getStructuringElement(cv2.MORPH_RECT, kernel_wh)
    closed = cv2.morphologyEx(gray, cv2.MORPH_CLOSE, k, iterations=2)
    closed = cv2.medianBlur(closed, 3)
    cnts, _ = cv2.findContours(closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    H, W = gray.shape[:2]
    boxes = []
    min_area = (W * H) * 0.00015
    for c in cnts:
        x,y,w,h = cv2.boundingRect(c)
        if w*h < min_area or h < H*0.015 or w < W*0.06:
            continue
        if w > W*0.98 and h > H*0.05:
            continue
        boxes.append((x,y,w,h))
    boxes.sort(key=lambda b: (b[1], b[0]))
    return nms_merge(boxes)

def nms_merge(boxes: List[Tuple[int,int,int,int]], iou_th: float=0.3) -> List[Tuple[int,int,int,int]]:
    out = []
    for x,y,w,h in boxes:
        merged = False
        for i,(X,Y,W,H) in enumerate(out):
            xx1, yy1 = max(x,X), max(y,Y)
            xx2, yy2 = min(x+w, X+W), min(y+h, Y+H)
            inter = max(0,xx2-xx1) * max(0,yy2-yy1)
            area = w*h + W*H - inter
            iou = inter/area if area>0 else 0
            if iou > iou_th:
                nx, ny = min(x,X), min(y,Y)
                nw, nh = max(x+w, X+W)-nx, max(y+h, Y+H)-ny
                out[i] = (nx, ny, nw, nh)
                merged = True
                break
        if not merged:
            out.append((x,y,w,h))
    return sorted(out, key=lambda b: (b[1], b[0]))

# =========================
# 색상 기반 발화자 판별
# =========================
def sender_by_color(crop_bgr: np.ndarray, me_side_right: bool) -> str:
    """
    Kakao 기본 테마 가정:
      - 내 말풍선: 노란색(높은 S, H≈20~40@OpenCV)
      - 상대방: 회청/회색(낮은 S) 배경
    색 기준이 애매하면 위치 규칙으로 폴백.
    """
    if crop_bgr.size == 0:
        return None
    hsv = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2HSV)
    H, S, V = cv2.split(hsv)
    h_mean = float(np.mean(H))
    s_mean = float(np.mean(S))
    v_mean = float(np.mean(V))

    # 노란색 범위 (OpenCV H: 0~179)
    is_yellow = (18 <= h_mean <= 40) and (s_mean >= 80) and (v_mean >= 120)
    if is_yellow:
        return "ME"
    # 매우 낮은 채도는 회/회청으로 간주 → 상대방일 가능성 높음
    is_grayish = s_mean <= 45 and v_mean >= 60
    if is_grayish:
        return "상대방"
    return None  # 애매 → 위치로 결정

# =========================
# OCR (멀티 전략)
# =========================
def hangul_ratio(s: str) -> float:
    h = hangul_findall(HANGUL, s)
    return 0 if len(s)==0 else len(h) / max(1, len(s))

def ocr_try(bin_img: np.ndarray, psm: int) -> Tuple[str, float, float]:
    pil = Image.fromarray(bin_img)
    cfg = f"--oem 1 --psm {psm} -c user_defined_dpi=300"
    data = pytesseract.image_to_data(pil, lang="kor+eng", config=cfg, output_type=pytesseract.Output.DICT)
    words, confs = [], []
    for t, c in zip(data["text"], data["conf"]):
        t = (t or "").strip()
        if not t or c in ("-1",""):
            continue
        words.append(t)
        try:
            cf = float(c)
            if cf >= 0: confs.append(cf)
        except: pass
    txt = " ".join(words).strip()
    conf = float(np.mean(confs)) if confs else -1.0
    score = 0.7 * max(conf,0) / 100.0 + 0.3 * hangul_ratio(txt)
    return txt, conf, score

def ocr_best_auto(gray_crop: np.ndarray) -> Tuple[str, float]:
    trials = []
    for inv in (False, True):
        _, bin_img = cv2.threshold(
            gray_crop, 0, 255,
            (cv2.THRESH_BINARY_INV if inv else cv2.THRESH_BINARY) + cv2.THRESH_OTSU
        )
        for psm in (6,7):
            trials.append(ocr_try(bin_img, psm))
    for method in (cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.ADAPTIVE_THRESH_MEAN_C):
        for inv in (False, True):
            bin_img = cv2.adaptiveThreshold(
                gray_crop, 255, method,
                cv2.THRESH_BINARY_INV if inv else cv2.THRESH_BINARY, 31, 7
            )
            for psm in (6,7):
                trials.append(ocr_try(bin_img, psm))
    trials.sort(key=lambda x: (x[2], len(x[0])), reverse=True)
    best_txt, best_conf, _ = trials[0]
    return best_txt, best_conf

# =========================
# 시간 추출 & 후처리
# =========================
def extract_time_and_message(text: str) -> Tuple[str, str]:
    m = TIME_RX.search(text.replace('：', ':'))
    tm = m.group() if m else None
    msg = (text[:m.start()] + text[m.end():]).strip() if m else text.strip()
    msg = hangul_sub(r'\[?\s*(오전|오후)\s*\d{1,2}\s*[:：]\s*\d{2}\s*\]?', '', msg).strip()
    return msg, tm

def ocr_time_near_right(crop_color_bgr: np.ndarray) -> str:
    h, w = crop_color_bgr.shape[:2]
    if w < 160: return None
    roi = crop_color_bgr[:, max(0, w-160):w]
    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    _, bin_img = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    t = pytesseract.image_to_string(Image.fromarray(bin_img), lang="kor+eng",
                                    config="--oem 1 --psm 7 -c user_defined_dpi=300")
    t = t.replace(' ', '').replace('：', ':')
    m = TIME_RX.search(t)
    return m.group() if m else None

PROFILE_PATTERNS = [
    r'컴\s*공\s*\d{2}',
    r'^\s*[ㅎ\.·\s]*컴\s*공.*$',
    r'\d{4}\.\d{1,2}\.\d{1,2}',
    r'^\s*오전\s*\d{1,2}:\d{2}\s*$',
    r'^\s*오후\s*\d{1,2}:\d{2}\s*$',
    r'^\s*\(\d+\)\s*[A-Z]{2,}\b.*$',   # (2) RACE / (2) BAC 류
]

def looks_spaced_korean(s: str) -> bool:
    pairs = hangul_findall(fr'({_HANGUL_CLASS})\s+({_HANGUL_CLASS})', s)
    han = hangul_findall(fr'{_HANGUL_CLASS}', s)
    return (len(han) >= 6) and (len(pairs) >= 2)

def glue_spaced_korean(s: str) -> str:
    return hangul_sub(fr'({_HANGUL_CLASS})\s+({_HANGUL_CLASS})', r'\1\2', s)

def normalize_common_errors(s: str) -> str:
    return hangul_sub(r'\s{2,}', ' ', s).strip()

def is_profile_or_meta(msg: str) -> bool:
    if len(msg.strip()) <= 1: return True
    only_han = ''.join(hangul_findall(HANGUL, msg))
    if len(only_han) == 0 and len(msg.strip()) <= 4: return True
    for p in PROFILE_PATTERNS:
        if hangul_search(p, msg): return True
    return False

def postprocess_message(raw_text: str) -> str:
    s = raw_text.strip()
    if looks_spaced_korean(s): s = glue_spaced_korean(s)
    s = normalize_common_errors(s)
    return s

# =========================
# 메시지 추출 + 줄바꿈 병합
# =========================
def extract_messages(img_path: str, me_name="ME", other_name="상대방", me_side="right") -> Tuple[List[Dict], Dict]:
    color = cv2.imread(img_path, cv2.IMREAD_COLOR)
    if color is None: raise FileNotFoundError(img_path)

    H0, W0 = color.shape[:2]
    scale, kernel_wh = auto_params(H0, W0)
    color, gray = upscale_and_gray(color, scale)

    H, W = color.shape[:2]
    center_x = W / 2.0
    me_is_right = (me_side.lower() == "right")

    boxes = find_bubbles(gray, kernel_wh)
    messages = []

    if boxes:
        for (x,y,w,h) in boxes:
            pad = 8
            x0, y0 = max(0, x-pad), max(0, y-pad)
            x1, y1 = min(W, x+w+pad), min(H, y+h+pad)

            # 왼쪽 아바타/닉네임 가드 (5%)
            left_guard = int(0.05 * W)
            if x < left_guard:
                x0 = min(W, x0 + left_guard)

            crop_c = color[y0:y1, x0:x1]
            crop_g = gray[y0:y1, x0:x1]

            text, conf = ocr_best_auto(crop_g)
            msg, tm = extract_time_and_message(text)
            if not tm:
                tm = ocr_time_near_right(crop_c)

            msg = postprocess_message(msg)
            if is_profile_or_meta(msg) or not msg:
                continue

            # sender_by_color 결과만 사용 (폴백 제거)
            s_color = sender_by_color(crop_c, me_is_right)
            if s_color:
                sender = s_color
            else:
                sender = "상대방"  # 기본값 (혹은 그냥 continue)

            messages.append({"sender": sender, "text": msg, "time": tm, "y": y, "x": x, "w": w, "conf": conf})
    else:
        # 폴백: 라인 OCR
        cfg = "--oem 1 --psm 6 -c user_defined_dpi=300"
        data = pytesseract.image_to_data(Image.fromarray(gray), lang="kor+eng",
                                         config=cfg, output_type=pytesseract.Output.DICT)
        lines = {}
        n = len(data["text"])
        for i in range(n):
            t = (data["text"][i] or "").strip()
            if not t: continue
            ln = (data["block_num"][i], data["par_num"][i], data["line_num"][i])
            lines.setdefault(ln, {"x": [], "y": [], "texts": [], "confs": []})
            lines[ln]["x"].append(data["left"][i])
            lines[ln]["y"].append(data["top"][i])
            lines[ln]["texts"].append(t)
            try:
                cf = float(data["conf"][i])
                if cf >= 0: lines[ln]["confs"].append(cf)
            except: pass

        for (_, v) in lines.items():
            if not v["texts"]: continue
            line_text = " ".join(v["texts"])
            y = int(np.median(v["y"])); x = int(np.median(v["x"]))
            msg, tm = extract_time_and_message(line_text)
            msg = postprocess_message(msg)
            if is_profile_or_meta(msg) or not msg: continue
            sender = "ME" if ((x >= center_x) == me_is_right) else "상대방"
            conf = float(np.mean(v["confs"])) if v["confs"] else -1.0
            messages.append({"sender": sender, "text": msg, "time": tm, "y": y, "x": x, "w": 0, "conf": conf})

    # 위→아래 정렬
    messages.sort(key=lambda m: (m["y"], m["x"]))

    # 같은 발화(줄바꿈) 병합: 같은 sender이고 x가 비슷하고 y가 가깝고 시간 동일/없음
    merged = []
    for m in messages:
        if merged:
            prev = merged[-1]
            same_sender = (prev["sender"] == m["sender"])
            dy = m["y"] - prev["y"]
            # y 간격만 느슨하게: 화면 높이의 16% 또는 최소 18px 이내면 같은 말풍선으로 간주
            close_y = 0 < dy <= max(18, int(0.16 * H))

            if same_sender and close_y:
                # 줄바꿈 병합
                prev["text"] = (prev["text"].rstrip(" .") + " " + m["text"].lstrip()).strip()
                prev["conf"] = max(prev["conf"], m["conf"])
                prev["y"] = m["y"]
                prev["w"] = max(prev.get("w", 0), m.get("w", 0))
                continue

        merged.append(m)
    messages = merged

    meta = {
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "source_image": os.path.basename(img_path),
        "participants": (other_name, me_name)
    }
    return messages, meta

# =========================
# 저장 (CSV)
# =========================
def save_csv(messages: List[Dict], out_csv: str, me_name: str, other_name: str):
    os.makedirs(os.path.dirname(out_csv) or ".", exist_ok=True)
    with open(out_csv, "w", encoding="utf-8-sig", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["idx", "sender", "text", "time", "confidence"])
        for i, m in enumerate(messages, 1):
            sender = me_name if m["sender"] == "ME" else other_name
            conf_str = "" if m.get("conf") is None else f"{m.get('conf', -1):.2f}"
            writer.writerow([i, sender, m["text"], m.get("time") or "", conf_str])
    print(f"[OK] CSV 저장 완료 → {out_csv}")

# =========================
# CLI
# =========================
def main():
    ap = argparse.ArgumentParser(description="카카오톡/문자 스크린샷 OCR → CSV (색상우선 발화자 판별)")
    ap.add_argument("input", help="이미지 경로(.png/.jpg/.jpeg)")
    ap.add_argument("-o", "--output", default="conversation.csv", help="저장 경로(.csv)")
    ap.add_argument("--me-name", default="ME", help="내 표시 이름")
    ap.add_argument("--other-name", default="상대방", help="상대방 표시 이름")
    ap.add_argument("--me-side", choices=["left","right"], default="right",
                    help="내 말풍선 위치(기본 right)")
    args = ap.parse_args()

    messages, _meta = extract_messages(
        img_path=args.input,
        me_name=args.me_name,
        other_name=args.other_name,
        me_side=args.me_side
    )
    save_csv(messages, args.output, args.me_name, args.other_name)

if __name__ == "__main__":
    main()
