import cv2
import numpy as np
import os, json, tempfile, uuid,   base64, requests
from pathlib import Path
from flask import Flask, request, jsonify
import cloudinary
import cloudinary.uploader
from openai import OpenAI
from urllib.parse import urlparse
from concurrent.futures import ThreadPoolExecutor
import logging
from dotenv import load_dotenv

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

load_dotenv()

# ---- Cloudinary Config ----
cloudinary.config(
    cloud_name=os.getenv("CLOUDINARY_CLOUD_NAME"),
    api_key=os.getenv("CLOUDINARY_API_KEY"),
    api_secret=os.getenv("CLOUDINARY_API_SECRET")
)

# ---- OpenAI Config ----
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
if not OPENAI_API_KEY:
    raise ValueError("OPENAI_API_KEY not set in environment variables")

client = OpenAI(api_key=OPENAI_API_KEY)

class BorderBoxProductExtractor:
    def __init__(self, flyer_path, out_dir="extracted_products", target_count=12):
        """
        Initialize the BorderBoxProductExtractor
        
        Args:
            flyer_path (str): Path to the flyer image
            out_dir (str): Output directory for extracted products
            target_count (int): Target number of products to extract
        """
        self.flyer_path = flyer_path
        self.out_dir = Path(out_dir)
        self.target_count = target_count
        
        # Load and validate image
        self.img = cv2.imread(flyer_path)
        if self.img is None:
            raise FileNotFoundError(f"Flyer not found: {flyer_path}")
            
        self.h, self.w = self.img.shape[:2]
        self.hsv = cv2.cvtColor(self.img, cv2.COLOR_BGR2HSV)
        
        # Color detection parameters
        self.blue_lower = np.array([90, 60, 40], np.uint8)
        self.blue_upper = np.array([140, 255, 255], np.uint8)
        self.white_s_thr, self.white_v_thr = 95, 160
        self.white_min_frac, self.ry_min_frac = 0.40, 0.01

    def _cluster_positions(self, vals, tol):
        """
        Cluster similar position values together
        
        Args:
            vals (list): List of position values
            tol (int): Tolerance for clustering
            
        Returns:
            list: List of clustered average positions
        """
        if not vals: 
            return []
        
        vals.sort()
        groups = [[vals[0]]]
        
        for v in vals[1:]:
            if abs(v - groups[-1][-1]) <= tol: 
                groups[-1].append(v)
            else: 
                groups.append([v])
                
        return [int(np.mean(g)) for g in groups]

    def _white_fraction(self, roi_hsv):
        """
        Calculate the fraction of white pixels in a region
        
        Args:
            roi_hsv (numpy.ndarray): HSV image region
            
        Returns:
            float: Fraction of white pixels
        """
        S, V = roi_hsv[..., 1], roi_hsv[..., 2]
        return float(np.mean((S < self.white_s_thr) & (V > self.white_v_thr)))

    @staticmethod
    def _red_yellow_fraction(roi_hsv):
        """
        Calculate the fraction of red/yellow pixels in a region
        
        Args:
            roi_hsv (numpy.ndarray): HSV image region
            
        Returns:
            float: Fraction of red/yellow pixels
        """
        H, S, V = cv2.split(roi_hsv)
        mask = ((H <= 10) | (H >= 160) | ((H >= 15) & (H <= 45))) & (S > 100) & (V > 120)
        return float(np.mean(mask))

    def _detect_grid_lines(self):
        """
        Detect grid lines in the flyer image
        
        Returns:
            tuple: Lists of x and y coordinates of grid lines
        """
        # Create blue mask and detect edges
        blue = cv2.inRange(self.hsv, self.blue_lower, self.blue_upper)
        edges = cv2.Canny(cv2.GaussianBlur(blue, (5, 5), 0), 50, 150)
        
        # Detect lines using Hough transform
        lines = cv2.HoughLinesP(
            edges, 1, np.pi / 180, 120, 
            minLineLength=min(self.h, self.w) // 6, 
            maxLineGap=20
        )
        
        xs, ys = [], []
        if lines is not None:
            for x1, y1, x2, y2 in lines[:, 0]:
                dx, dy = abs(x2 - x1), abs(y2 - y1)
                # Vertical lines
                if dx < dy * 0.2: 
                    xs.append((x1 + x2) // 2)
                # Horizontal lines
                elif dy < dx * 0.2: 
                    ys.append((y1 + y2) // 2)
        
        # Cluster similar positions
        xl = self._cluster_positions(xs, int(0.015 * self.w))
        yl = self._cluster_positions(ys, int(0.015 * self.h))
        
        # Add image boundaries if not present
        if not xl or xl[0] > 10: 
            xl = [0] + xl
        if xl[-1] < self.w - 10: 
            xl += [self.w - 1]
        if not yl or yl[0] > 10: 
            yl = [0] + yl
        if yl[-1] < self.h - 10: 
            yl += [self.h - 1]
            
        return sorted(set(xl)), sorted(set(yl))

    def _grid_rects(self, xs, ys):
        """
        Generate rectangles from grid lines
        
        Args:
            xs (list): X coordinates of vertical lines
            ys (list): Y coordinates of horizontal lines
            
        Returns:
            list: List of rectangles (x, y, width, height)
        """
        rects = []
        for yi in range(len(ys) - 1):
            for xi in range(len(xs) - 1):
                x0, x1 = xs[xi], xs[xi + 1]
                y0, y1 = ys[yi], ys[yi + 1]
                
                # Skip rectangles that are too small
                if (x1 - x0) < 50 or (y1 - y0) < 50: 
                    continue
                    
                rects.append((x0, y0, x1 - x0, y1 - y0))
        return rects

    def _looks_like_product(self, x, y, w, h):
        """
        Determine if a rectangle looks like a product based on various criteria
        
        Args:
            x, y, w, h (int): Rectangle coordinates and dimensions
            
        Returns:
            bool: True if the rectangle looks like a product
        """
        area = w * h
        img_area = self.h * self.w
        
        # Check area constraints
        if not (img_area * 0.02 <= area <= img_area * 0.6): 
            return False
            
        # Check aspect ratio
        ar = w / float(h)
        if not (0.45 <= ar <= 2.2): 
            return False
        
        # Analyze interior region
        inset = max(int(0.006 * max(w, h)), 8)
        xi0, yi0 = max(x + inset, 0), max(y + inset, 0)
        xi1, yi1 = min(x + w - inset, self.w), min(y + h - inset, self.h)
        
        if xi1 <= xi0 or yi1 <= yi0: 
            return False
            
        roi = self.hsv[yi0:yi1, xi0:xi1]
        
        # Check color composition
        white_frac = self._white_fraction(roi)
        ry_frac = self._red_yellow_fraction(roi)
        
        return white_frac >= self.white_min_frac and ry_frac >= self.ry_min_frac

    def _enhance_image_for_gpt(self, bgr_img):
        """
        Enhanced image preprocessing for better GPT Vision results
        
        Args:
            bgr_img (numpy.ndarray): Input BGR image
            
        Returns:
            numpy.ndarray: Enhanced RGB image
        """
        # Convert to RGB for better color representation
        rgb_img = cv2.cvtColor(bgr_img, cv2.COLOR_BGR2RGB)
        
        # Resize if too small (GPT works better with clearer images)
        h, w = rgb_img.shape[:2]
        if w < 200 or h < 200:
            scale_factor = max(200/w, 200/h, 1.5)
            new_w, new_h = int(w * scale_factor), int(h * scale_factor)
            rgb_img = cv2.resize(rgb_img, (new_w, new_h), interpolation=cv2.INTER_CUBIC)
        
        # Enhance contrast and brightness using CLAHE
        lab = cv2.cvtColor(rgb_img, cv2.COLOR_RGB2LAB)
        l, a, b = cv2.split(lab)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
        l = clahe.apply(l)
        enhanced = cv2.merge([l, a, b])
        rgb_enhanced = cv2.cvtColor(enhanced, cv2.COLOR_LAB2RGB)
        
        return rgb_enhanced

    def _extract_product_info_with_gpt(self, image_path: str) -> dict:
        """
        Extract product information using GPT-4 Vision
        
        Args:
            image_path (str): Path to the product image
            
        Returns:
            dict: Extracted product information
        """
        try:
            # Read and encode image
            with open(image_path, "rb") as image_file:
                image_data = base64.b64encode(image_file.read()).decode('utf-8')

            # Craft a detailed prompt for product extraction
            prompt = """
            Analyze this product image from a grocery flyer and extract the following information in JSON format:

            1. **product_name**: The main product name (clean, concise, max 120 characters)
            2. **description**: Additional details like brand, size, weight, flavor, etc. (max 400 characters)
            3. **price**: Current/sale price (extract exact number, include currency if visible)
            4. **original_price**: Original/regular price if there's a discount (extract exact number, include currency if visible)
            5. **category**: Suggested category for this product (examples: Electronics, Grocery, Dairy, Beverages, Health & Beauty, Bakery, Meat & Fish, Frozen Foods, Confectionery & Snacks, Paper & Disposables, Baby & Mom, Home & Furniture, Pets, etc.)

            Instructions:
            - Focus on the most prominent text
            - Ignore promotional words like "SAVE", "OFF", "DEAL", "SPECIAL", "LIMITED"
            - For prices, look for numbers with currency symbols (PKR, Rs, $, etc.) or standalone numbers that appear to be prices
            - If multiple prices exist, the smaller one is usually the current price, larger one is original price
            - If text is unclear or blurry, make your best interpretation
            - If information is not available, use null for that field

            Return ONLY a valid JSON object with these exact keys:
            {
                "product_name": "...",
                "description": "...", 
                "price": "...",
                "original_price": "...",
                "category": "..."
            }
            """

            response = client.chat.completions.create(
                model="gpt-4o",  # Use GPT-4 with vision capabilities
                messages=[
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": prompt},
                            {
                                "type": "image_url",
                                "image_url": {
                                    "url": f"data:image/png;base64,{image_data}",
                                    "detail": "high"
                                }
                            }
                        ]
                    }
                ],
                temperature=0.1  # Low temperature for more consistent results
            )

            # Parse the response
            gpt_response = response.choices[0].message.content.strip()
            
            # Try to extract JSON from the response
            try:
                # Remove any markdown formatting or extra text
                if "```json" in gpt_response:
                    json_start = gpt_response.find("```json") + 7
                    json_end = gpt_response.find("```", json_start)
                    gpt_response = gpt_response[json_start:json_end].strip()
                elif "```" in gpt_response:
                    json_start = gpt_response.find("```") + 3
                    json_end = gpt_response.find("```", json_start)
                    gpt_response = gpt_response[json_start:json_end].strip()
                
                # Parse JSON
                result = json.loads(gpt_response)
                
                # Validate and clean the result
                cleaned_result = {
                    "product_name": result.get("product_name"),
                    "description": result.get("description"),
                    "price": result.get("price"),
                    "original_Price": result.get("original_price"),  # Note: keeping original key name for consistency
                    "category": result.get("category")
                }
                
                return cleaned_result
                
            except json.JSONDecodeError as e:
                logger.error(f"JSON parsing error: {e}")
                logger.error(f"GPT Response: {gpt_response}")
                # Return a fallback structure
                return {
                    "product_name": None,
                    "description": None,
                    "price": None,
                    "original_Price": None,
                    "category": None
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
        """
        Main extraction method that processes the flyer and extracts products
        
        Returns:
            list: List of extracted product information
        """
        # Create output directories
        self.out_dir.mkdir(parents=True, exist_ok=True)
        (self.out_dir / "product_images").mkdir(exist_ok=True)
        
        # Detect grid and extract product regions
        xs, ys = self._detect_grid_lines()
        rects = self._grid_rects(xs, ys)
        products = [r for r in rects if self._looks_like_product(*r)]
        products = sorted(products, key=lambda b: (b[1], b[0]))
        
        # Limit to target count if we have too many products
        if len(products) > self.target_count:
            areas = np.array([w * h for (_, _, w, h) in products], dtype=np.float32)
            med = float(np.median(areas))
            products = sorted(products, key=lambda r: abs((r[2] * r[3]) - med))[:self.target_count]
            products = sorted(products, key=lambda b: (b[1], b[0]))
        
        results = []
        
        # Process each product
        for i, (x, y, w, h) in enumerate(products, 1):
            logger.info(f"Processing product {i}/{len(products)}")
            
            # Calculate crop boundaries with inset
            inset = max(int(0.006 * max(w, h)), 8)
            x0, y0 = max(x + inset, 0), max(y + inset, 0)
            x1, y1 = min(x + w - inset, self.w), min(y + h - inset, self.h)
            crop = self.img[y0:y1, x0:x1]

            # Enhance image for better GPT Vision results
            enhanced_crop = self._enhance_image_for_gpt(crop)
            
            # Save original crop
            local_path = str(self.out_dir / "product_images" / f"product_{i:02d}.png")
            cv2.imwrite(local_path, crop)
            
            # Save enhanced version for GPT processing
            enhanced_path = str(self.out_dir / "product_images" / f"product_{i:02d}_enhanced.png")
            cv2.imwrite(enhanced_path, cv2.cvtColor(enhanced_crop, cv2.COLOR_RGB2BGR))

            # Upload to Cloudinary
            try:
                upload_result = cloudinary.uploader.upload(local_path, folder="flyer_products")
                url = upload_result.get("secure_url")
                logger.info(f"Successfully uploaded product {i} to Cloudinary")
            except Exception as e:
                logger.error(f"Cloudinary upload failed for product {i}: {e}")
                url = None

            parsed = self._extract_product_info_with_gpt(enhanced_path)

     

            extracted_text_parts = []
            if parsed.get("product_name"):
                extracted_text_parts.append(parsed["product_name"])
            if parsed.get("category"):
                extracted_text_parts.append(parsed["category"])
            if parsed.get("description"):
                extracted_text_parts.append(parsed["description"])
            if parsed.get("price"):
                extracted_text_parts.append(f"Price: {parsed['price']}")
            if parsed.get("original_Price"):
                extracted_text_parts.append(f"Original: {parsed['original_Price']}")
            
            extracted_text = " | ".join(extracted_text_parts)

            results.append({
                "id": str(uuid.uuid4()),
                "image_url": url,
                "product_name": parsed["product_name"],
                "description": parsed["description"],
                "price": parsed["price"],
                "category": parsed["category"],
                "original_Price": parsed["original_Price"],
                "extracted_text": extracted_text
            })

        # Save results to JSON
        with open(self.out_dir / "products_master_data.json", "w", encoding="utf-8") as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        
        logger.info(f"Successfully extracted {len(results)} products")
        return results


# --- Flask App Initialization ---
app = Flask(__name__)
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max file size

# Thread pool for handling async operations
executor = ThreadPoolExecutor(max_workers=4)


def download_image_sync(image_url):
    """
    Synchronous function to download image from URL
    
    Args:
        image_url (str): URL of the image to download
        
    Returns:
        bytes: Image data
        
    Raises:
        Exception: If download fails
    """
    try:
        response = requests.get(image_url, timeout=30)
        response.raise_for_status()
        
        # Check if it's an image
        content_type = response.headers.get('content-type', '')
        if not content_type.startswith('image/'):
            raise ValueError("URL does not point to an image")
            
        return response.content
    except requests.RequestException as e:
        raise Exception(f"Failed to download image: {str(e)}")


@app.route("/", methods=["GET"])
def root():
    """
    Root endpoint providing API information
    """
    return jsonify({
        "message": "Product Extractor API with GPT Vision", 
        "features": [
            "Advanced product detection using computer vision",
            "GPT-4 Vision for intelligent text extraction",
            "Structured product data extraction",
            "Cloudinary integration for image hosting"
        ],
        "endpoints": {
            "/": "API information",
            "/health": "Health check",
            "/extract-products": "Extract products from flyer (POST)"
        },
        "version": "1.0.0",
        "framework": "Flask"
    })


@app.route("/health", methods=["GET"])
def health_check():
    """
    Health check endpoint
    """
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
    """
    Extract products from flyer image using GPT Vision
    
    Expected form data:
        image_url (str): URL of the flyer image
        
    Returns:
        JSON response with extracted products
    """
    try:
        # Get image URL from form data
        if not request.form.get('image_url'):
            return jsonify({
                "success": False,
                "error": "image_url is required in form data"
            }), 400
            
        image_url = request.form.get('image_url')
        
        # Validate URL format
        try:
            parsed_url = urlparse(image_url)
            if not parsed_url.scheme or not parsed_url.netloc:
                return jsonify({
                    "success": False,
                    "error": "Invalid URL format"
                }), 400
        except Exception:
            return jsonify({
                "success": False,
                "error": "Invalid URL format"
            }), 400
        
        logger.info(f"Processing flyer from URL: {image_url}")
        
        # Download image from URL
        try:
            image_data = download_image_sync(image_url)
            logger.info("Successfully downloaded image from URL")
        except Exception as e:
            logger.error(f"Failed to download image: {str(e)}")
            return jsonify({
                "success": False,
                "error": f"Failed to download image: {str(e)}"
            }), 400
        
        # Save downloaded image temporarily
        file_extension = os.path.splitext(urlparse(image_url).path)[1] or '.png'
        
        with tempfile.NamedTemporaryFile(delete=False, suffix=file_extension) as tmp:
            tmp.write(image_data)
            tmp_path = tmp.name
        
        try:
            # Create temporary directory for processing
            with tempfile.TemporaryDirectory() as tmp_dir:
                logger.info("Starting product extraction...")
                
                # Initialize extractor and process image
                extractor = BorderBoxProductExtractor(tmp_path, tmp_dir, 12)
                products = extractor.extract()
                
                logger.info(f"Successfully extracted {len(products)} products")
                
                return jsonify({
                    "success": True, 
                    "count": len(products), 
                    "products": products,
                    "message": f"Successfully extracted {len(products)} products using GPT Vision",
                    "processing_details": {
                        "ai_engine": "GPT-4 Vision",
                        "target_count": 12,
                        "extraction_method": "Grid-based detection + AI analysis"
                    }
                })
                
        except Exception as e:
            logger.error(f"Extraction failed: {str(e)}")
            return jsonify({
                "success": False,
                "error": f"Extraction failed: {str(e)}"
            }), 500
            
        finally:
            # Clean up temporary file
            try:
                os.unlink(tmp_path)
            except OSError:
                pass
    
    except Exception as e:
        logger.error(f"Unexpected error in extract_products: {str(e)}")
        return jsonify({
            "success": False,
            "error": f"Internal server error: {str(e)}"
        }), 500


@app.errorhandler(413)
def too_large(e):
    """
    Handle file too large error
    """
    return jsonify({
        "success": False,
        "error": "File too large. Maximum size is 16MB."
    }), 413


@app.errorhandler(400)
def bad_request(e):
    """
    Handle bad request errors
    """
    return jsonify({
        "success": False,
        "error": "Bad request. Please check your input parameters."
    }), 400


@app.errorhandler(500)
def internal_error(e):
    """
    Handle internal server errors
    """
    return jsonify({
        "success": False,
        "error": "Internal server error. Please try again later."
    }), 500


# Development server runner
if __name__ == "__main__":
    # Set up logging for development
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # Run the Flask development server
    app.run(
        host="localhost", 
        port=8000, 
        debug=True,  # Enable debug mode for development
        threaded=True  # Enable threading for better performance
    )