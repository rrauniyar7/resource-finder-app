from flask import Flask, render_template, request, jsonify
import requests
from bs4 import BeautifulSoup
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from googlesearch import search

app = Flask(__name__)

# Function to fetch and parse webpage content
def fetch_and_parse(url):
    try:
        response = requests.get(url, timeout=10)
        soup = BeautifulSoup(response.text, 'html.parser')
        for script in soup(["script", "style", "nav", "footer"]):
            script.decompose()
        return " ".join(soup.stripped_strings)
    except Exception as e:
        print(f"Error fetching {url}: {e}")
        return ""

# Function to rank documents based on relevance to the query
def rank_documents(query, documents, keywords):
    vectorizer = TfidfVectorizer()
    tfidf_matrix = vectorizer.fit_transform(documents)
    query_vec = vectorizer.transform([query])
    similarities = cosine_similarity(query_vec, tfidf_matrix).flatten()

    keyword_scores = []
    for doc in documents:
        score = sum(1 for keyword in keywords if keyword.lower() in doc.lower())
        keyword_scores.append(score)

    combined_scores = similarities + keyword_scores
    ranked_indices = combined_scores.argsort()[::-1]
    return ranked_indices

# Function to search for articles based on keywords
def search_articles(query, num_results=10):
    search_results = []
    try:
        for result in search(query, num_results=num_results):
            search_results.append(result)
            if len(search_results) >= num_results:
                break
    except Exception as e:
        print(f"Error during search: {e}")
    return search_results

@app.route("/", methods=["GET", "POST"])
def index():
    if request.method == "POST":
        title = request.form.get("title")
        description = request.form.get("description")
        query = title + " " + description if description else title
        keywords = description.split(", ") if description else []

        search_results = search_articles(query)
        documents = []
        for url in search_results:
            content = fetch_and_parse(url)
            if content:
                documents.append((url, content))

        if documents:
            ranked_indices = rank_documents(query, [doc[1] for doc in documents], keywords)
            ranked_results = [documents[idx] for idx in ranked_indices[:5]]
            return jsonify({"results": ranked_results})
        else:
            return jsonify({"error": "No relevant documents found."})

    return render_template("index.html")

if __name__ == "__main__":
    app.run(debug=True)