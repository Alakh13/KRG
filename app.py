from flask import Flask, render_template, request
import spacy
import networkx as nx
import matplotlib.pyplot as plt
import tempfile
import os
from werkzeug.utils import secure_filename
import fitz  # PyMuPDF
import requests
from bs4 import BeautifulSoup

app = Flask(__name__)
nlp = spacy.load("en_core_web_sm")

def extract_text(source, source_type):
    if source_type == 'pdf':
        text = ""
        with fitz.open(source) as doc:
            for page in doc:
                text += page.get_text()
        return text
    elif source_type == 'url':
        res = requests.get(source)
        soup = BeautifulSoup(res.text, 'html.parser')
        return soup.get_text()
    else:
        return source

def build_knowledge_graph(text):
    doc = nlp(text)
    G = nx.Graph()
    for sent in doc.sents:
        tokens = [token.text for token in sent if not token.is_stop and not token.is_punct]
        for i in range(len(tokens)-1):
            G.add_edge(tokens[i], tokens[i+1])
    return G

def save_graph(G):
    plt.figure(figsize=(12, 8))
    nx.draw(G, with_labels=True, node_color='skyblue', node_size=2000, font_size=10, font_weight='bold')
    tmp_file = os.path.join(tempfile.gettempdir(), "graph.png")
    plt.savefig(tmp_file)
    return tmp_file

@app.route("/", methods=["GET", "POST"])
def index():
    graph_url = None
    if request.method == "POST":
        content_type = request.form["input_type"]
        if content_type == "pdf":
            f = request.files["file"]
            filename = secure_filename(f.filename)
            filepath = os.path.join(tempfile.gettempdir(), filename)
            f.save(filepath)
            text = extract_text(filepath, "pdf")
        elif content_type == "url":
            text = extract_text(request.form["url_input"], "url")
        else:
            text = extract_text(request.form["text_input"], "text")

        G = build_knowledge_graph(text)
        graph_path = save_graph(G)
        graph_url = "/static/graph.png"
        os.system(f"cp {graph_path} {os.path.join('static', 'graph.png')}")

    return render_template("index.html", graph_url=graph_url)

if __name__ == "__main__":
    app.run(debug=True)
