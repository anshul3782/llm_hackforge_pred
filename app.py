import pymysql
import json
import requests
import numpy as np
from sentence_transformers import SentenceTransformer
from flask import Flask, request, jsonify
from flask_cors import CORS

app = Flask(__name__)

# ✅ Enable CORS for all routes
CORS(app, resources={r"/*": {"origins": "*"}})

# Constants
DB_HOST = "db-mysql-nyc3-54076-do-user-19716193-0.k.db.ondigitalocean.com"
DB_USERNAME = "doadmin"
DB_PASSWORD = "AVNS_oAN9S2VKGNizJx9BtBA"
DB_NAME = "hackforge"
DB_PORT = 25060

# LLM API endpoint - Using Groq
API_URL = "https://api.groq.com/openai/v1/chat/completions"
API_KEY = "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"
MODEL = "llama3-70b-8192"

# Load the embedding model
model = SentenceTransformer('sentence-transformers/all-mpnet-base-v2')

def get_db_connection():
    """Establish a connection to the MySQL database"""
    return pymysql.connect(
        host=DB_HOST,
        user=DB_USERNAME,
        password=DB_PASSWORD,
        database=DB_NAME,
        port=DB_PORT,
        ssl={'ssl': {}},
        cursorclass=pymysql.cursors.DictCursor
    )

def get_product_description(product_name):
    """Retrieve product description from RAG_product table"""
    conn = get_db_connection()
    cursor = conn.cursor()
    
    try:
        query = "SELECT description FROM RAG_product WHERE product_name = %s LIMIT 1"
        cursor.execute(query, (product_name,))
        result = cursor.fetchone()
        
        if result:
            return result['description']
        else:
            return None
    finally:
        cursor.close()
        conn.close()

def cosine_similarity(a, b):
    """Calculate cosine similarity between two vectors"""
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

def get_related_feedback(product_name, query_embedding, top_n=5):
    """Retrieve related feedback from RAG_product_fb table using vector similarity"""
    conn = get_db_connection()
    cursor = conn.cursor()
    
    try:
        # Get all feedback for the product
        query = "SELECT description, embedding FROM RAG_product_fb WHERE product_name = %s"
        cursor.execute(query, (product_name,))
        results = cursor.fetchall()
        
        if not results:
            return []
        
        # Calculate similarity for each feedback
        similarities = []
        for result in results:
            embedding = json.loads(result['embedding'])
            similarity = cosine_similarity(query_embedding, embedding)
            similarities.append((result['description'], similarity))
        
        # Sort by similarity and return top N
        similarities.sort(key=lambda x: x[1], reverse=True)
        return [item[0] for item in similarities[:top_n]]
    
    finally:
        cursor.close()
        conn.close()

def analyze_product(product_name):
    """Main function to analyze a product"""
    # Get product description
    description = get_product_description(product_name)
    if not description:
        return {
            "error": f"Product '{product_name}' not found in the database."
        }
    
    # Generate embedding for the product description
    query_embedding = model.encode(description).tolist()
    
    # Get related feedback from RAG_product_fb
    feedback_items = get_related_feedback(product_name, query_embedding)
    
    if not feedback_items:
        return {
            "error": f"No feedback found for product '{product_name}'."
        }
    
    # Format feedback for prompt
    feedback_text = "\n".join([f"- {feedback}" for feedback in feedback_items])
    
    # Create prompt for LLM
    prompt = f"""
Analyze this product data and provide ONLY a prediction with confidence rating.

Product: {product_name}
Description: {description}
Customer Feedback: {feedback_text}
Target Audience: Tech-savvy consumers looking for convenience and reliability in their daily life.

Structure your response EXACTLY as follows:
MARKET SENTIMENT: [Positive/Negative/Mixed] - one word only
TARGET AUDIENCE FIT: [High/Medium/Low] - one word only
OVERALL PREDICTION: [Will succeed/May succeed/Will not succeed] - brief phrase only
CONFIDENCE: [High/Medium/Low] - one word only
EVIDENCE: 2-3 bullet points from the data only
"""
    
    # Prepare payload for LLM
    payload = {
        "model": MODEL,
        "messages": [
            {"role": "system", "content": "You are a product analyst providing structured predictions."},
            {"role": "user", "content": prompt}
        ],
        "temperature": 0.7
    }
    
    # Headers including authorization
    headers = {
        "Authorization": f"Bearer {API_KEY}",
        "Content-Type": "application/json"
    }
    
    # Send request to LLM
    try:
        response = requests.post(API_URL, headers=headers, json=payload)
        response.raise_for_status()  # Raise exception for HTTP errors
        
        result = response.json()
        analysis = result['choices'][0]['message']['content']
        
        # Return success response with all the details
        return {
            "product_name": product_name,
            "description": description,
            "feedback_count": len(feedback_items),
            "analysis": analysis
        }
        
    except requests.RequestException as e:
        return {"error": f"LLM API request failed: {str(e)}"}
    except (KeyError, IndexError) as e:
        return {"error": f"Error parsing LLM response: {str(e)}"}

@app.route('/api/analyze', methods=['POST'])
def api_analyze_product():
    data = request.json
    
    # Check if product_name is provided
    if not data or 'product_name' not in data:
        return jsonify({"error": "Please provide a product_name in the request body"}), 400
    
    # Get the product name from the request
    product_name = data['product_name']
    
    # Process the analysis
    result = analyze_product(product_name)
    
    # Return appropriate response
    if 'error' in result:
        return jsonify(result), 404 if 'not found' in result['error'] else 500
    else:
        return jsonify(result), 200

@app.route('/api/test/<path:product_name>', methods=['GET'])
def api_test_product(product_name):
    """Test endpoint that accepts product name as URL parameter"""
    # Process the analysis
    result = analyze_product(product_name)
    
    # Return appropriate response
    if 'error' in result:
        return jsonify(result), 404 if 'not found' in result['error'] else 500
    else:
        return jsonify(result), 200

@app.route('/api/products', methods=['GET'])
def api_get_products():
    """Get a list of available products"""
    try:
        conn = get_db_connection()
        cursor = conn.cursor()
        
        # Query to get distinct product names
        query = "SELECT DISTINCT product_name FROM RAG_product"
        cursor.execute(query)
        
        # Fetch all product names
        products = [row['product_name'] for row in cursor.fetchall()]
        
        cursor.close()
        conn.close()
        
        return jsonify({"products": products}), 200
        
    except Exception as e:
        return jsonify({"error": f"Failed to fetch products: {str(e)}"}), 500

@app.route('/', methods=['GET'])
def home():
    """Simple home route to verify the API is running"""
    return jsonify({
        "status": "online",
        "message": "Product Analysis API is running",
        "endpoints": {
            "/api/products": "GET - List all available products",
            "/api/analyze": "POST - Analyze a product (requires product_name in request body)",
            "/api/test/<product_name>": "GET - Test endpoint for product analysis"
        }
    })

if __name__ == "__main__":
    # Run the Flask app
    app.run(host='0.0.0.0', port=5000, debug=True)
