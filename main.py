import cv2
import numpy as np
import os
import json
import tempfile
import uuid
import base64
import requests
from pathlib import Path
from flask import Flask, request, jsonify
import cloudinary
import cloudinary.uploader
from openai import OpenAI
from urllib.parse import urlparse
from concurrent.futures import ThreadPoolExecutor
import logging
from dotenv import load_dotenv

# ---------------- Logging ----------------
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ---------------- Load Environment ----------------
load_dotenv()

# ---------------- Cloudinary Config ----------------
cloudinary.config(
    cloud_name=os.getenv("CLOUDINARY_CLOUD_NAME"),
    api_key=os.getenv("CLOUDINARY_API_KEY"),
    api_secret=os.getenv("CLOUDINARY_API_SECRET")
)

# ---------------- OpenAI Config ----------------
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
if not OPENAI_API_KEY:
    raise ValueError("OPENAI_API_KEY not set in environment variables")

client = OpenAI(api_key=OPENAI_API_KEY)

# ---------------- Product Extractor Class ----------------
class BorderBoxProductExtractor:
    def __init__(self, flyer_path, out_dir="extracted_products", target_count=12):
        self.flyer_path = flyer_path
        self.out_dir = Path(out_dir)
        self.target_count = target_count

        # Load image
        self.img = cv2.imread(flyer_path)
        if self.img is None:
            raise FileNotFoundError(f"Flyer not found: {flyer_path}")
        self.h, self.w = self.img.shape[:2]
        self.hsv = cv2.cvtColor(self.img, cv2.COLOR_BGR2HSV)

        # Color thresholds
        self.blue_lower = np.array([90, 60, 40], np.uint8)
        self.blue_upper = np.array([140, 255, 255], np.uint8)
        self.white_s_thr, self.white_v_thr = 95, 160
        self.white_min_frac, self.ry_min_frac = 0.40, 0.01

    def _cluster_positions(self, vals, tol):
        if not vals: return []
        vals.sort()
        groups = [[vals[0]]]
        for v in vals[1:]:
            if abs(v - groups[-1][-1]) <= tol: 
                groups[-1].append(v)
            else: 
                groups.append([v])
        return [int(np.mean(g)) for g in groups]

    def _white_fraction(self, roi_hsv):
        S, V = roi_hsv[..., 1], roi_hsv[..., 2]
        return float(np.mean((S < self.white_s_thr) & (V > self.white_v_thr)))

    @staticmethod
    def _red_yellow_fraction(roi_hsv):
        H, S, V = cv2.split(roi_hsv)
        mask = ((H <= 10) | (H >= 160) | ((H >= 15) & (H <= 45))) & (S > 100) & (V > 120)
        return float(np.mean(mask))

    def _detect_grid_lines(self):
        blue = cv2.inRange(self.hsv, self.blue_lower, self.blue_upper)
        edges = cv2.Canny(cv2.GaussianBlur(blue, (5, 5), 0), 50, 150)
        lines = cv2.HoughLinesP(edges, 1, np.pi/180, 120, 
                                minLineLength=min(self.h, self.w)//6, maxLineGap=20)
        xs, ys = [], []
        if lines is not None:
            for x1, y1, x2, y2 in lines[:,0]:
                dx, dy = abs(x2-x1), abs(y2-y1)
                if dx < dy*0.2: xs.append((x1+x2)//2)
                elif dy < dx*0.2: ys.append((y1+y2)//2)
        xl = self._cluster_positions(xs, int(0.015*self.w))
        yl = self._cluster_positions(ys, int(0.015*self.h))
        if not xl or xl[0] > 10: xl = [0]+xl
        if xl[-1] < self.w-10: xl += [self.w-1]
        if not yl or yl[0] > 10: yl = [0]+yl
        if yl[-1] < self.h-10: yl += [self.h-1]
        return sorted(set(xl)), sorted(set(yl))

    def _grid_rects(self, xs, ys):
        rects = []
        for yi in range(len(ys)-1):
            for xi in range(len(xs)-1):
                x0, x1 = xs[xi], xs[xi+1]
                y0, y1 = ys[yi], ys[yi+1]
                if (x1-x0) < 50 or (y1-y0) < 50: continue
                rects.append((x0, y0, x1-x0, y1-y0))
        return rects

    def _looks_like_product(self, x, y, w, h):
        area = w*h
        img_area = self.h*self.w
        if not (img_area*0.02 <= area <= img_area*0.6): return False
        ar = w/float(h)
        if not (0.45 <= ar <= 2.2): return False
        inset = max(int(0.006*max(w,h)), 8)
        xi0, yi0 = max(x+inset,0), max(y+inset,0)
        xi1, yi1 = min(x+w-inset,self.w), min(y+h-inset,self.h)
        if xi1 <= xi0 or yi1 <= yi0: return False
        roi = self.hsv[yi0:yi1, xi0:xi1]
        white_frac = self._white_fraction(roi)
        ry_frac = self._red_yellow_fraction(roi)
        return white_frac >= self.white_min_frac and ry_frac >= self.ry_min_frac

    def _enhance_image_for_gpt(self, bgr_img):
        rgb_img = cv2.cvtColor(bgr_img, cv2.COLOR_BGR2RGB)
        h, w = rgb_img.shape[:2]
        if w < 200 or h < 200:
            scale = max(200/w, 200/h, 1.5)
            rgb_img = cv2.resize(rgb_img, (int(w*scale), int(h*scale)), interpolation=cv2.INTER_CUBIC)
        lab = cv2.cvtColor(rgb_img, cv2.COLOR_RGB2LAB)
        l, a, b = cv2.split(lab)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
        l = clahe.apply(l)
        enhanced = cv2.merge([l, a, b])
        return cv2.cvtColor(enhanced, cv2.COLOR_LAB2RGB)

    def _extract_product_info_with_gpt(self, image_path):
        try:
            with open(image_path, "rb") as f:
                image_data = base64.b64encode(f.read()).decode('utf-8')

            prompt = """
            Analyze this product image from a grocery flyer and extract JSON:
            {
                "product_name": "...",
                "description": "...",
                "price": "...",
                "original_price": "...",
                "category": "..."
            }
            """
            response = client.chat.completions.create(
                model="gpt-4o",
                messages=[{
                    "role": "user",
                    "content": [
                        {"type": "text", "text": prompt},
                        {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{image_data}", "detail":"high"}}
                    ]
                }],
                temperature=0.1
            )

            gpt_response = response.choices[0].message.content.strip()
            if "```" in gpt_response:
                gpt_response = gpt_response.split("```")[-2].strip()
            result = json.loads(gpt_response)

            return {
                "product_name": result.get("product_name"),
                "description": result.get("description"),
                "price": result.get("price"),
                "original_Price": result.get("original_price"),
                "category": result.get("category")
            }
        except Exception as e:
            logger.error(f"GPT Vision API error: {e}")
            return {
                "product_name": None,
                "description": None,
                "price": None,
                "original_Price": None,
                "category": None
            }

    def extract(self):
        self.out_dir.mkdir(parents=True, exist_ok=True)
        (self.out_dir / "product_images").mkdir(exist_ok=True)

        xs, ys = self._detect_grid_lines()
        rects = self._grid_rects(xs, ys)
        products = [r for r in rects if self._looks_like_product(*r)]
        products = sorted(products, key=lambda b: (b[1], b[0]))

        if len(products) > self.target_count:
            areas = np.array([w*h for (_, _, w, h) in products], dtype=np.float32)
            med = float(np.median(areas))
            products = sorted(products, key=lambda r: abs((r[2]*r[3])-med))[:self.target_count]
            products = sorted(products, key=lambda b: (b[1], b[0]))

        results = []

        for i, (x, y, w, h) in enumerate(products, 1):
            logger.info(f"Processing product {i}/{len(products)}")
            inset = max(int(0.006*max(w,h)), 8)
            x0, y0 = max(x+inset,0), max(y+inset,0)
            x1, y1 = min(x+w-inset,self.w), min(y+h-inset,self.h)
            crop = self.img[y0:y1, x0:x1]
            enhanced_crop = self._enhance_image_for_gpt(crop)

            local_path = str(self.out_dir / "product_images" / f"product_{i:02d}.png")
            cv2.imwrite(local_path, crop)
            enhanced_path = str(self.out_dir / "product_images" / f"product_{i:02d}_enhanced.png")
            cv2.imwrite(enhanced_path, cv2.cvtColor(enhanced_crop, cv2.COLOR_RGB2BGR))

            try:
                upload_result = cloudinary.uploader.upload(local_path, folder="flyer_products")
                url = upload_result.get("secure_url")
            except Exception as e:
                logger.error(f"Cloudinary upload failed: {e}")
                url = None

            parsed = self._extract_product_info_with_gpt(enhanced_path)
            extracted_text_parts = []
            for key in ["product_name", "category", "description", "price", "original_Price"]:
                if parsed.get(key):
                    val = parsed[key]
                    if key in ["price", "original_Price"]:
                        val = f"{key.replace('_',' ').title()}: {val}"
                    extracted_text_parts.append(val)
            extracted_text = " | ".join(extracted_text_parts)

            results.append({
                "id": str(uuid.uuid4()),
                "image_url": url,
                **parsed,
                "extracted_text": extracted_text
            })

        with open(self.out_dir / "products_master_data.json", "w", encoding="utf-8") as f:
            json.dump(results, f, indent=2, ensure_ascii=False)

        logger.info(f"Successfully extracted {len(results)} products")
        return results

# ---------------- Flask App ----------------
app = Flask(__name__)
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024
executor = ThreadPoolExecutor(max_workers=4)

def download_image_sync(image_url):
    try:
        response = requests.get(image_url, timeout=30)
        response.raise_for_status()
        if not response.headers.get('content-type','').startswith('image/'):
            raise ValueError("URL does not point to an image")
        return response.content
    except requests.RequestException as e:
        raise Exception(f"Failed to download image: {str(e)}")

@app.route("/", methods=["GET"])
def root():
    return jsonify({
        "message": "Product Extractor API with GPT Vision",
        "features": [
            "Advanced product detection using computer vision",
            "GPT-4 Vision for intelligent text extraction",
            "Structured product data extraction",
            "Cloudinary integration for image hosting"
        ],
        "endpoints": {
            "/": "API info",
            "/health": "Health check",
            "/extract-products": "Extract products from flyer (POST)"
        },
        "version": "1.0.0"
    })

@app.route("/health", methods=["GET"])
def health_check():
    return jsonify({
        "status": "healthy",
        "ai_engine": "GPT-4 Vision",
        "services": {
            "opencv": "available",
            "openai": "configured",
            "cloudinary": "configured"
        }
    })

@app.route("/extract-products", methods=["POST"])
def extract_products():
    image_url = request.form.get('image_url')
    if not image_url:
        return jsonify({"success": False, "error": "image_url is required"}), 400

    try:
        parsed_url = urlparse(image_url)
        if not parsed_url.scheme or not parsed_url.netloc:
            raise ValueError("Invalid URL format")
    except Exception:
        return jsonify({"success": False, "error": "Invalid URL format"}), 400

    try:
        image_data = download_image_sync(image_url)
    except Exception as e:
        return jsonify({"success": False, "error": f"Failed to download image: {str(e)}"}), 400

    file_ext = os.path.splitext(parsed_url.path)[1] or ".png"
    with tempfile.NamedTemporaryFile(delete=False, suffix=file_ext) as tmp:
        tmp.write(image_data)
        tmp_path = tmp.name

    try:
        with tempfile.TemporaryDirectory() as tmp_dir:
            extractor = BorderBoxProductExtractor(tmp_path, tmp_dir, 12)
            products = extractor.extract()
            return jsonify({
                "success": True,
                "count": len(products),
                "products": products,
                "message": f"Successfully extracted {len(products)} products using GPT Vision"
            })
    except Exception as e:
        logger.error(f"Extraction failed: {e}")
        return jsonify({"success": False, "error": f"Extraction failed: {str(e)}"}), 500
    finally:
        try:
            os.unlink(tmp_path)
        except OSError:
            pass

@app.errorhandler(413)
def too_large(e):
    return jsonify({"success": False, "error": "File too large. Max 16MB"}), 413

@app.errorhandler(400)
def bad_request(e):
    return jsonify({"success": False, "error": "Bad request"}), 400

@app.errorhandler(500)
def internal_error(e):
    return jsonify({"success": False, "error": "Internal server error"}), 500

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8000, debug=True, threaded=True)
