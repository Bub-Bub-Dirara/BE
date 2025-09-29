#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
카카오톡/문자 스크린샷 OCR → Markdown(.md)
- 범용 전처리: 해상도 비례 업스케일, 자동 커널, 다중 이진화/극성/PSM 시도 후 최고결과 선택
- 말풍선(버블) 검출 실패 시 라인 OCR 폴백
- 중앙선 기준 좌/우 → 상대방/ME 분류
- 시간([오전/오후 HH:MM]) 자동 추출(본문·우측영역 둘 다 시도)
- 한글 과다 띄어쓰기 결합 + 흔한 오인식 치환 + 프로필/메타 라인 필터
필수: Tesseract 설치 + kor 언어(traineddata)
  - Windows: tesseract --list-langs 에 kor가 보여야 함
  - tessdata_best의 kor.traineddata를 쓰면 인식률 ↑
"""

import os
import argparse
from datetime import datetime
from typing import List, Dict, Tuple

import cv2
import numpy as np
from PIL import Image
import pytesseract

# ---- regex: 'regex' 모듈이 있으면 사용(유니코드 클래스 지원), 없으면 내장 re로 폴백
try:
    import regex as re2  # pip install regex
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

# 필요 시 Tesseract 경로 지정 (Windows에서 PATH에 없으면 주석 해제)
# pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"

# =========================
# 유틸/전처리
# =========================
def auto_params(img_h: int, img_w: int) -> Tuple[float, Tuple[int,int]]:
    """해상도에 따른 업스케일 배수 및 모폴로지 커널 크기 자동 산출"""
    base = max(img_w, img_h)
    scale = 2.0 if base < 1100 else (1.8 if base < 1700 else 1.4)
    kx = max(11, int(img_w * 0.012))  # 가로 닫힘 커널
    ky = max(5, int(img_h * 0.006))   # 세로 닫힘 커널
    # 홀수 보장(bit OR 1)
    return scale, (kx | 1, ky | 1)

def upscale_and_gray(color_bgr: np.ndarray, scale: float) -> Tuple[np.ndarray, np.ndarray]:
    """업스케일 + 노이즈 완화 + 그레이 + CLAHE"""
    if scale != 1.0:
        color_bgr = cv2.resize(color_bgr, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)
    # 가장자리 노이즈 줄이기
    color_bgr = cv2.bilateralFilter(color_bgr, 7, 40, 40)
    gray = cv2.cvtColor(color_bgr, cv2.COLOR_BGR2GRAY)
    clahe = cv2.createCLAHE(clipLimit=2.2, tileGridSize=(8,8))
    gray = clahe.apply(gray)
    return color_bgr, gray

def find_bubbles(gray: np.ndarray, kernel_wh: Tuple[int,int]) -> List[Tuple[int,int,int,int]]:
    """형태학적 닫힘으로 말풍선/텍스트 덩어리 후보 추출"""
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
        # 화면 전체선 비슷한 긴 라인은 제외
        if w > W*0.98 and h > H*0.05:
            continue
        boxes.append((x,y,w,h))
    boxes.sort(key=lambda b: (b[1], b[0]))
    return nms_merge(boxes)

def nms_merge(boxes: List[Tuple[int,int,int,int]], iou_th: float=0.3) -> List[Tuple[int,int,int,int]]:
    """간단 NMS 병합으로 중복 박스 정리"""
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
# OCR 시도(멀티 전략)
# =========================
def hangul_ratio(s: str) -> float:
    h = hangul_findall(HANGUL, s)
    return 0 if len(s)==0 else len(h) / max(1, len(s))

def ocr_try(bin_img: np.ndarray, psm: int) -> Tuple[str, float, float]:
    pil = Image.fromarray(bin_img)
    cfg = f"--oem 1 --psm {psm} -c preserve_interword_spaces=1 -c user_defined_dpi=300"
    data = pytesseract.image_to_data(pil, lang="kor+eng", config=cfg, output_type=pytesseract.Output.DICT)
    words, confs = [], []
    for t, c in zip(data["text"], data["conf"]):
        t = (t or "").strip()
        if not t or c in ("-1",""): 
            continue
        words.append(t)
        try: confs.append(float(c))
        except: pass
    txt = " ".join(words).strip()
    conf = float(np.mean(confs)) if confs else -1.0
    score = 0.7 * max(conf,0) / 100.0 + 0.3 * hangul_ratio(txt)
    return txt, conf, score

def ocr_best_auto(gray_crop: np.ndarray) -> Tuple[str, float]:
    """정/역 + 전역/적응형 + psm 6/7 전부 시도 → 스코어 최고 선택"""
    trials = []
    # Otsu 정/역
    for inv in (False, True):
        _, bin_img = cv2.threshold(
            gray_crop, 0, 255,
            (cv2.THRESH_BINARY_INV if inv else cv2.THRESH_BINARY) + cv2.THRESH_OTSU
        )
        for psm in (6,7):
            txt, conf, score = ocr_try(bin_img, psm)
            trials.append((txt, conf, score))
    # Adaptive (Gaussian/Mean) 정/역
    for method in (cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.ADAPTIVE_THRESH_MEAN_C):
        for inv in (False, True):
            bin_img = cv2.adaptiveThreshold(
                gray_crop, 255, method,
                cv2.THRESH_BINARY_INV if inv else cv2.THRESH_BINARY, 31, 7
            )
            for psm in (6,7):
                txt, conf, score = ocr_try(bin_img, psm)
                trials.append((txt, conf, score))
    trials.sort(key=lambda x: (x[2], len(x[0])), reverse=True)
    best_txt, best_conf, _ = trials[0]
    return best_txt, best_conf

# =========================
# 시간 추출 & 후처리
# =========================
def extract_time_and_message(text: str) -> Tuple[str, str]:
    m = TIME_RX.search(text.replace('：', ':'))
    tm = m.group() if m else None
    if m:
        msg = (text[:m.start()] + text[m.end():]).strip()
    else:
        msg = text.strip()
    # 대괄호 포함 표기 제거
    msg = hangul_sub(r'\[?\s*(오전|오후)\s*\d{1,2}\s*[:：]\s*\d{2}\s*\]?', '', msg).strip()
    return msg, tm

def ocr_time_near_right(crop_color_bgr: np.ndarray) -> str:
    """버블 우측 140px 영역에서 시간만 다시 OCR(작게 찍힌 시간 보완)"""
    h, w = crop_color_bgr.shape[:2]
    if w < 160:
        return None
    roi = crop_color_bgr[:, max(0, w-160):w]
    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    _, bin_img = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    txt, _ = ocr_best_auto(bin_img)
    t = txt.replace(' ', '').replace('：', ':')
    m = TIME_RX.search(t)
    return m.group() if m else None

# ---- 후처리(메시지 정리)
PROFILE_PATTERNS = [
    r'컴\s*공\s*\d{2}',                 # 학과+학번류
    r'^\s*[ㅎ\.·\s]*컴\s*공.*$',        # 닉/이름 라인 변형
    r'\d{4}\.\d{1,2}\.\d{1,2}',         # 날짜줄
    r'^\s*오전\s*\d{1,2}:\d{2}\s*$',
    r'^\s*오후\s*\d{1,2}:\d{2}\s*$',
]

SUBS = [
    (r'브\s*브', 'ㅂㅂ'),
    (r'즈\s*즈', 'ㅈㅈ'),
    (r'앗\s*어', '았어'),
    (r'앗\s*서', '았어'),
]

def looks_spaced_korean(s: str) -> bool:
    pairs = hangul_findall(fr'({_HANGUL_CLASS})\s+({_HANGUL_CLASS})', s)
    hanguls = hangul_findall(fr'{_HANGUL_CLASS}', s)
    return (len(pairs) >= 1) and (len(pairs) >= max(3, int(0.6 * max(1, len(hanguls)))))

def glue_spaced_korean(s: str) -> str:
    return hangul_sub(fr'({_HANGUL_CLASS})\s+({_HANGUL_CLASS})', r'\1\2', s)

def normalize_common_errors(s: str) -> str:
    t = s
    for pat, rep in SUBS:
        t = hangul_sub(pat, rep, t)
    # 비한글 잡문자 축소
    t = hangul_sub(r'[^\w\s\[\]\(\)\:\.\,\~\!\?가-힣]', '', t)
    # 다중 공백 축소
    t = hangul_sub(r'\s{2,}', ' ', t)
    return t.strip()

def is_profile_or_meta(msg: str) -> bool:
    if len(msg.strip()) <= 1:
        return True
    # 순수 영문/기호가 대부분이고 너무 짧으면 제거
    only_han = ''.join(hangul_findall(HANGUL, msg))
    if len(only_han) == 0 and len(msg.strip()) <= 4:
        return True
    for p in PROFILE_PATTERNS:
        if hangul_search(p, msg):
            return True
    return False

def postprocess_message(raw_text: str) -> str:
    s = raw_text.strip()
    if looks_spaced_korean(s):
        s = glue_spaced_korean(s)
    s = normalize_common_errors(s)
    return s

# =========================
# 파이프라인
# =========================
def process_image_to_markdown(
    img_path: str,
    out_md: str,
    me_name: str = "ME",
    other_name: str = "상대방",
    me_side: str = "right"
) -> None:
    color = cv2.imread(img_path, cv2.IMREAD_COLOR)
    if color is None:
        raise FileNotFoundError(img_path)

    H0, W0 = color.shape[:2]
    scale, kernel_wh = auto_params(H0, W0)
    color, gray = upscale_and_gray(color, scale)

    H, W = color.shape[:2]
    center_x = W / 2.0
    me_is_right = (me_side.lower() == "right")

    # 1) 버블 검출
    boxes = find_bubbles(gray, kernel_wh)

    messages = []
    if boxes:
        # 버블 단위 OCR
        for (x,y,w,h) in boxes:
            pad = 8
            x0, y0 = max(0, x-pad), max(0, y-pad)
            x1, y1 = min(W, x+w+pad), min(H, y+h+pad)
            crop_c = color[y0:y1, x0:x1]
            crop_g = gray[y0:y1, x0:x1]

            text, conf = ocr_best_auto(crop_g)
            msg, tm = extract_time_and_message(text)
            if not tm:
                tm = ocr_time_near_right(crop_c)

            msg = postprocess_message(msg)
            if is_profile_or_meta(msg) or not msg:
                continue

            cx = x + w / 2.0
            if me_is_right:
                sender = "ME" if cx >= center_x else "상대방"
            else:
                sender = "ME" if cx < center_x else "상대방"

            messages.append({"sender": sender, "text": msg, "time": tm, "y": y, "conf": conf})
    else:
        # 2) 폴백: 라인 OCR
        cfg = "--oem 1 --psm 6 -c preserve_interword_spaces=1 -c user_defined_dpi=300"
        data = pytesseract.image_to_data(Image.fromarray(gray), lang="kor+eng",
                                         config=cfg, output_type=pytesseract.Output.DICT)
        # line_num 기준 그룹화
        lines = {}
        n = len(data["text"])
        for i in range(n):
            t = (data["text"][i] or "").strip()
            if not t:
                continue
            ln = (data["block_num"][i], data["par_num"][i], data["line_num"][i])
            lines.setdefault(ln, {"x": [], "y": [], "texts": []})
            lines[ln]["x"].append(data["left"][i])
            lines[ln]["y"].append(data["top"][i])
            lines[ln]["texts"].append(t)
        for (_, v) in lines.items():
            if not v["texts"]:
                continue
            line_text = " ".join(v["texts"])
            y = int(np.median(v["y"]))
            x = int(np.median(v["x"]))
            msg, tm = extract_time_and_message(line_text)
            msg = postprocess_message(msg)
            if is_profile_or_meta(msg) or not msg:
                continue
            sender = "ME" if ((x >= center_x) == me_is_right) else "상대방"
            messages.append({"sender": sender, "text": msg, "time": tm, "y": y, "conf": -1})

    # 시간 오름차순(화면 상단→하단)으로 정렬
    messages.sort(key=lambda m: m["y"])

    # 3) Markdown 저장
    ts = datetime.now().isoformat(timespec="seconds")
    os.makedirs(os.path.dirname(out_md) or ".", exist_ok=True)
    with open(out_md, "w", encoding="utf-8") as f:
        f.write("# 대화 OCR 추출 결과\n\n")
        f.write(f"- 생성 시각: {ts}\n")
        f.write(f"- 원본 이미지: `{os.path.basename(img_path)}`\n")
        f.write(f"- 참여자: {other_name}, {me_name}\n\n")
        f.write("## Messages\n\n")
        for i, m in enumerate(messages, 1):
            sender = me_name if m["sender"] == "ME" else other_name
            if m["time"]:
                f.write(f"{i}. **{sender}**: {m['text']} [{m['time']}]\n")
            else:
                f.write(f"{i}. **{sender}**: {m['text']}\n")

    print(f"[OK] Markdown 저장 완료 → {out_md}")

# =========================
# CLI
# =========================
def main():
    ap = argparse.ArgumentParser(description="카카오톡/문자 스크린샷 OCR → Markdown")
    ap.add_argument("input", help="이미지 경로(.png/.jpg/.jpeg)")
    ap.add_argument("-o", "--output", default="conversation.md", help=".md 저장 경로")
    ap.add_argument("--me-name", default="ME", help="내 표시 이름")
    ap.add_argument("--other-name", default="상대방", help="상대방 표시 이름")
    ap.add_argument("--me-side", choices=["left","right"], default="right",
                    help="내 말풍선 위치(기본 right)")
    args = ap.parse_args()

    process_image_to_markdown(
        img_path=args.input,
        out_md=args.output,
        me_name=args.me_name,
        other_name=args.other_name,
        me_side=args.me_side
    )

if __name__ == "__main__":
    main()
