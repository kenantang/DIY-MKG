import json
import random
import os
import glob
from datetime import datetime
import networkx as nx
from flask import Flask, render_template, jsonify, request
from markdown import markdown

# --- LLM Imports ---
try:
    import openai
except ImportError:
    openai = None

try:
    import torch
    from transformers import pipeline
except ImportError:
    torch = None
    pipeline = None

# --- Configuration & Initialization ---
try:
    with open('config.json', 'r') as config_file:
        config = json.load(config_file)
except FileNotFoundError:
    print("WARNING: config.json not found. Using default values.")
    config = {
        "QUIZ_MCQ_COUNT": 3, "QUIZ_FILL_IN_BLANK_COUNT": 2,
        "llm": {"provider": "placeholder", "api_key_env_var": "OPENAI_API_KEY", "local_model_name": "Qwen/Qwen2.5-0.5B", "local_model_directory": "/data2/kenantang/.cache"}
    }

llm_config = config.get("llm", {})
local_model_directory = llm_config.get("local_model_directory", "")
if local_model_directory:
    os.environ['HF_HOME'] = local_model_directory
    os.environ['HF_DATASETS_CACHE'] = local_model_directory
    os.environ['TORCH_HOME'] = local_model_directory
    os.environ['VLLM_CACHE_ROOT'] = local_model_directory

GRAPH_SNAPSHOT_DIR = 'graph_snapshots'
QUIZ_HISTORY_DIR = 'quiz_history'
app = Flask(__name__)
os.makedirs(GRAPH_SNAPSHOT_DIR, exist_ok=True)
os.makedirs(QUIZ_HISTORY_DIR, exist_ok=True)
G = nx.Graph()

# --- LLM Wrapper Classes ---
class LLMWrapper:
    def query(self, prompt):
        raise NotImplementedError

class PlaceholderWrapper(LLMWrapper):
    def query(self, prompt):
        print("--- Using Placeholder LLM ---")
        response_text = f"This is a placeholder response for the following prompt:\n\n---\n{prompt}\n---\n\nTo use a real LLM, update your config.json and install the required libraries (e.g., 'openai' or 'transformers')."
        return response_text

class OpenAIWrapper(LLMWrapper):
    def __init__(self, api_key, model="gpt-4o-mini"):
        if not openai:
            raise ImportError("OpenAI library not installed. Please run 'pip install openai'.")
        self.client = openai.OpenAI(api_key=api_key)
        self.model = model
        print(f"--- Initialized OpenAI LLM with model: {self.model} ---")

    def query(self, prompt):
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.7,
            )
            return response.choices[0].message.content
        except Exception as e:
            return f"Error querying OpenAI: {e}"

class TransformersWrapper(LLMWrapper):
    def __init__(self, model_name="Qwen/Qwen2.5-0.5B"):
        if not torch or not pipeline:
            raise ImportError("PyTorch or Transformers not installed. Please run 'pip install torch transformers accelerate'.")

        print(f"--- Loading local model: {model_name}. This may take some time... ---")
        device = "cuda" if torch.cuda.is_available() else "cpu"
        try:

            self.pipe = pipeline("text-generation", model=model_name, torch_dtype=torch.bfloat16, device=device)
        except:
            print("!!!!")
            exit()
        print(f"--- Local model loaded successfully on device: {device} ---")

    def query(self, prompt):
        try:
            outputs = self.pipe(prompt, max_new_tokens=512, do_sample=True, temperature=0.7, top_k=50, top_p=0.95)
            return outputs[0]['generated_text']
        except Exception as e:
            return f"Error querying local model: {e}"

llm_client = PlaceholderWrapper() # Default

def initialize_llm():
    global llm_client
    llm_config = config.get("llm", {})
    provider = llm_config.get("provider", "placeholder")

    if provider == "openai":
        api_key = os.getenv(llm_config.get("api_key_env_var"))
        if api_key:
            llm_client = OpenAIWrapper(api_key=api_key, model=llm_config.get("openai_model", "gpt-4o-mini"))
        else:
            print(f"Warning: OpenAI provider selected but environment variable '{llm_config.get('api_key_env_var')}' not set. Using placeholder.")
    elif provider == "local":
        try:
            llm_client = TransformersWrapper(model_name=llm_config.get("local_model_name"))
        except (ImportError, Exception) as e:
            print(f"Warning: Could not initialize local model. {e}. Using placeholder.")
    else:
        print("--- Using Placeholder LLM ---")

# --- Graph Management & Helpers ---

def load_latest_snapshot():
    global G
    snapshot_files = sorted(glob.glob(os.path.join(GRAPH_SNAPSHOT_DIR, 'graph_*.json')))
    if snapshot_files:
        latest_snapshot = snapshot_files[-1]
        try:
            with open(latest_snapshot, 'r') as f:
                G = nx.node_link_graph(json.load(f))
            print(f"Loaded latest snapshot: {os.path.basename(latest_snapshot)}")
        except Exception as e:
            print(f"Error loading snapshot {latest_snapshot}: {e}. Starting fresh.")
            create_default_graph()
    else:
        print("No snapshots found. Creating a new default graph.")
        create_default_graph()
        save_snapshot()

def create_default_graph():
    global G
    G = nx.Graph()
    G.add_node("Pan", title={'tags': ['Food'], 'description': 'The Spanish word for **bread**.'}, click_count=0)
    G.add_node("빵", title={'tags': ['Food'], 'description': 'The Korean word for **bread**.'}, click_count=0)
    G.add_node("パン", title={'tags': ['Food'], 'description': 'The Japanese word for **bread**.'}, click_count=0)
    G.add_edge("Pan", "빵", label="Cognates", details={'tags': ['Same Root'], 'description': 'Both have the same Latin root.'})
    G.add_edge("빵", "パン", label="Cognates", details={'tags': ['Same Root'], 'description': 'Both have the same Portuguese root.'})
    G.add_edge("パン", "Pan", label="Cognates", details={'tags': ['Same Root'], 'description': 'Both have the same Latin root.'})

def save_snapshot():
    timestamp = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
    filename = f"graph_{timestamp}.json"
    filepath = os.path.join(GRAPH_SNAPSHOT_DIR, filename)
    with open(filepath, 'w') as f:
        json.dump(nx.node_link_data(G), f, indent=4)
    print(f"Snapshot saved: {filename}")
    return filename

def format_details_to_html(details_data):
    if not isinstance(details_data, dict): return str(details_data)
    tags = details_data.get('tags', [])
    description_md = details_data.get('description', '')
    tags_html = ''.join(f'<span class="tag">{tag}</span>' for tag in tags)
    description_html = markdown(description_md) if description_md else '<p><em>No description.</em></p>'
    return f'<div class="tooltip-tags">{tags_html}</div><div class="tooltip-description">{description_html}</div>'

def to_visjs_format(graph):
    nodes = [{'id': n, 'label': n, 'title': format_details_to_html(d.get('title'))} for n, d in graph.nodes(data=True)]
    edges = [{'from': u, 'to': v, 'label': d.get('label', ''), 'title': format_details_to_html(d.get('details', {}))} for u, v, d in graph.edges(data=True)]
    return {'nodes': nodes, 'edges': edges}

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/api/graph/full')
def get_full_graph():
    return jsonify({'raw_data': nx.node_link_data(G), 'vis_data': to_visjs_format(G)})

@app.route('/api/graph/snapshots/list')
def list_snapshots():
    snapshot_files = sorted(glob.glob(os.path.join(GRAPH_SNAPSHOT_DIR, 'graph_*.json')), reverse=True)
    return jsonify([os.path.basename(f) for f in snapshot_files])

@app.route('/api/graph/save_snapshot', methods=['POST'])
def save_snapshot_endpoint():
    filename = save_snapshot()
    return jsonify({'status': 'success', 'filename': filename})

@app.route('/api/graph/load_snapshot', methods=['POST'])
def load_snapshot_endpoint():
    global G
    filename = request.json.get('filename')
    if not filename or '..' in filename or os.path.sep in filename:
        return jsonify({'error': 'Invalid filename'}), 400
    filepath = os.path.join(GRAPH_SNAPSHOT_DIR, filename)
    try:
        with open(filepath, 'r') as f:
            G = nx.node_link_graph(json.load(f))
        print(f"Loaded snapshot: {filename}")
        return jsonify({'status': 'success', 'filename': filename})
    except Exception as e:
        return jsonify({'error': f'Could not load snapshot: {e}'}), 500

@app.route('/api/graph/nodes/list')
def get_node_list(): return jsonify(list(G.nodes()))

@app.route('/api/graph/tags/nodes')
def get_all_node_tags():
    all_tags = set()
    for _, data in G.nodes(data=True): all_tags.update(data.get('title', {}).get('tags', []))
    return jsonify(sorted(list(all_tags)))

@app.route('/api/graph/tags/edges')
def get_all_edge_tags():
    all_tags = set()
    for _, _, data in G.edges(data=True): all_tags.update(data.get('details', {}).get('tags', []))
    return jsonify(sorted(list(all_tags)))

@app.route('/api/graph/node/<node_id>')
def get_node_details(node_id):
    if node_id not in G: return jsonify({'error': 'Node not found'}), 404
    subgraph = G.subgraph(list(G.neighbors(node_id)) + [node_id])
    vis_data = to_visjs_format(subgraph)
    raw_title = G.nodes[node_id].get('title', {})
    if not isinstance(raw_title, dict): raw_title = {'tags': [], 'description': str(raw_title)}
    return jsonify({'vis_data': vis_data, 'focus_node_details': raw_title})

@app.route('/api/graph/node/<node_id>/increment_click', methods=['POST'])
def increment_node_click(node_id):
    if node_id not in G: return jsonify({'error': 'Node not found'}), 404
    G.nodes[node_id]['click_count'] = G.nodes[node_id].get('click_count', 0) + 1
    return jsonify({'status': 'success'})

@app.route('/api/graph/edge')
def get_edge_details():
    node1, node2 = request.args.get('node1'), request.args.get('node2')
    if not G.has_edge(node1, node2): return jsonify({'error': 'Edge not found'}), 404
    edge_data = G.edges[node1, node2]
    details = edge_data.get('details', {})
    if not isinstance(details, dict): details = {'tags': [], 'description': str(details)}
    return jsonify({'label': edge_data.get('label', ''), 'details': details})

# --- LLM-Powered Endpoints ---
@app.route('/api/llm/query', methods=['POST'])
def handle_llm_query():
    query_text = request.json.get('query')
    response = llm_client.query(query_text)
    return jsonify({'response': response})

@app.route('/api/llm/suggest_nodes', methods=['POST'])
def suggest_llm_nodes():
    node_id = request.json.get('node_id')
    if not node_id or node_id not in G:
        return jsonify({'error': 'Valid node ID is required'}), 400
    
    node_details = G.nodes[node_id]
    description = node_details["title"]["description"]
    prompt = f"Given a word \"{node_id}\" and its description \"{description}\", suggest 10 new, related words or phrases in the same language that could extend the vocabulary of a language learner. Ensure the list contains words with different parts of speech, such as nouns, verbs, adjectives, adverbs, etc. Your answer must be a single JSON array of strings. Do not include any explanation. For example: [\"apple\", \"pear\", \"orange\"]"
    
    response_text = llm_client.query(prompt)

    # Safety Guardrail

    prompt = f"""You are an expert language educator reviewing a list of words to ensure they are appropriate for learners aged 6 to 18.

Your task is to **remove any words that are clearly inappropriate** for this age range. This includes:
- Explicitly sexual or erotic terms
- Profanity, vulgar language, or offensive slurs
- Extremely graphic or gory terms (e.g., related to torture or mutilation)
- Drug abuse or addiction terminology (e.g., heroin, meth)
- Hate speech or discriminatory terms
- Strongly adult-only slang or references (e.g., stripper, porn)

**Mild or general terms related to death, war, or alcohol (e.g., kill, battle, wine) may be allowed** if they are commonly encountered in literature or academic contexts, but use judgment and err on the side of caution for younger teens.

Please return only the **filtered list** of appropriate words. Your answer must be a single JSON array of strings. Do not list or explain the removed words.

If uncertain, remove the word.

Here is the list of words:
{response_text}"""
    
    response_text = llm_client.query(prompt)

    # Clean the response to find the JSON array
    start = response_text.find('[')
    end = response_text.rfind(']') + 1
    if start != -1 and end != -1:
        suggestions = json.loads(response_text[start:end])
        return jsonify(suggestions)
    else:
        raise ValueError("No JSON array found in LLM response.")
    
    # KT: Can add error handling here.

# --- Quiz Endpoints ---
def generate_quiz_simple():
    # This is the old, non-LLM logic to be used as a fallback
    questions = []
    mcq_count, fib_count = config["QUIZ_MCQ_COUNT"], config["QUIZ_FILL_IN_BLANK_COUNT"]
    
    nodes_with_neighbors = [n for n in G.nodes() if list(G.neighbors(n))]
    if len(nodes_with_neighbors) >= 2:
        for _ in range(mcq_count):
            subject_node = random.choice(nodes_with_neighbors)
            correct_answer = random.choice(list(G.neighbors(subject_node)))
            distractors = random.sample(list(set(G.nodes()) - set(G.neighbors(subject_node)) - {subject_node}), 3)
            options = distractors + [correct_answer]
            random.shuffle(options)
            questions.append({'type': 'mcq', 'question': f"Which node is connected to '{subject_node}'?", 'options': options, 'answer': correct_answer})

    edges_with_labels = [(u, v, d) for u, v, d in G.edges(data=True) if d.get('label')]
    for _ in range(fib_count):
        if not edges_with_labels: break
        u, v, data = random.choice(edges_with_labels)
        questions.append({'type': 'fill_in_blank', 'question': f"The relationship between '{u}' and '{v}' is...", 'answer': data['label']})
    
    random.shuffle(questions)
    return questions

@app.route('/api/quiz/generate')
def generate_quiz():
    """
    Generates a quiz by finding the 5 least-clicked nodes and creating
    a question for each one individually using the LLM.
    """

    mcq_count = config.get("QUIZ_MCQ_COUNT", 3)
    fib_count = config.get("QUIZ_FILL_IN_BLANK_COUNT", 2)

    # 1. Obtain the 5 nodes with the lowest click count
    if G.number_of_nodes() < 5:
        return jsonify({'error': 'Not enough nodes in the graph to generate a quiz.'}), 400

    sorted_nodes = sorted(G.nodes(data=True), key=lambda x: x[1].get('click_count', 0))
    target_nodes = sorted_nodes[:5]

    all_questions = []

    # 2. Generate questions separately for each node
    for question_id, (node_id, node_data) in enumerate(target_nodes):

        word = node_id
        description = node_data.get('title', {}).get('description', 'No description available.')
        
        if question_id < mcq_count:
            # Generate MCQ
            prompt = f"""
You are a quiz master creating a quiz based on a knowledge graph of words. Your questions test the ability of a language learner.
Your task is to generate a multiple-choice question (mcq) based on a word and its description.
Your response MUST be a single, valid JSON object. Do not include any other text or explanation.

The JSON format for the question object is the following:
{{"type": "mcq", "question": "Which of the following is a fruit?", "options": ["Apple", "Desk", "Novel", "Creativity"], "answer": "Apple"}}

The questions need to be generated in the same language as the given word. For example, if a given word is a Japanese word, the question itself should be in Japanese.

The answer should not appear in the question.

Here is the word and its description:
Word: {word}
Description: {description}
""".strip()
        else:
            # Generate FIB
            prompt = f"""
You are a quiz master creating a quiz based on a knowledge graph of words. Your questions test the ability of a language learner.
Your task is to generate a fill-in-the-blank question based on a word and its description.
Your response MUST be a single, valid JSON object. Do not include any other text or explanation.

The JSON format for the question object is the following:
{{"type": "fill_in_blank", "question": "An _____ is a red and round fruit.", "answer": "apple"}}

The questions need to be generated in the same language as the given word. For example, if a given word is a Japanese word, the question itself should be in Japanese.

The correct answer should be the provided word. The answer should not appear in the question.

Here is the word and its description:
Word: {word}
Description: {description}
""".strip()
            pass
        
        response_text = llm_client.query(prompt)

        print(word)
        print(description)
        print(response_text)

        try:
            # Clean the response to find the JSON object
            start = response_text.find('{')
            end = response_text.rfind('}') + 1
            if start != -1 and end != -1:
                question = json.loads(response_text[start:end])
                # Basic validation
                if 'type' in question and 'question' in question and 'answer' in question:
                    all_questions.append(question)
                else:
                    print(f"LLM response for node '{node_id}' was malformed. Skipping.")
            else:
                raise ValueError("No JSON object found in LLM response.")
        except (json.JSONDecodeError, ValueError) as e:
            print(f"LLM quiz generation for node '{node_id}' failed: {e}. Skipping.")
            continue
    
    # 3. Combine the questions and return the quiz
    if not all_questions:
        print("LLM failed to generate any questions. Using simple generator as fallback.")
        return jsonify(generate_quiz_simple())

    random.shuffle(all_questions)
    return jsonify(all_questions)

@app.route('/api/quiz/save_results', methods=['POST'])
def save_quiz_results():
    results = request.json
    timestamp = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
    filename = f"quiz_{timestamp}.json"
    filepath = os.path.join(QUIZ_HISTORY_DIR, filename)
    with open(filepath, 'w') as f: json.dump(results, f, indent=4)
    return jsonify({'status': 'success', 'message': f'Results saved to {filepath}'})

@app.route('/api/graph/node/<node_id>/update', methods=['POST'])
def update_node_details(node_id):
    if node_id not in G: return jsonify({'error': 'Node not found'}), 404
    G.nodes[node_id]['title'] = request.json
    return jsonify({'status': 'success'})

@app.route('/api/graph/edge/update', methods=['POST'])
def update_edge_details():
    node1, node2 = request.args.get('node1'), request.args.get('node2')
    if not G.has_edge(node1, node2): return jsonify({'error': 'Edge not found'}), 404
    G.edges[node1, node2]['details'] = request.json.get('details')
    G.edges[node1, node2]['label'] = request.json.get('label')
    return jsonify({'status': 'success'})

@app.route('/api/graph/add_node', methods=['POST'])
def add_node():
    node_id = request.json.get('id')
    if not node_id: return jsonify({'error': 'Node ID is required'}), 400
    if G.has_node(node_id): return jsonify({'error': 'Node already exists'}), 409
    G.add_node(node_id, title={'tags': [], 'description': ''}, click_count=0)
    return jsonify({'status': 'success', 'node_id': node_id})

@app.route('/api/graph/remove_node', methods=['POST'])
def remove_node():
    node_id = request.json.get('id')
    if not node_id: return jsonify({'error': 'Node ID is required'}), 400
    if not G.has_node(node_id): return jsonify({'error': 'Node not found'}), 404
    G.remove_node(node_id)
    return jsonify({'status': 'success', 'node_id': node_id})

@app.route('/api/graph/add_edge', methods=['POST'])
def add_edge():
    node1, node2 = request.json.get('node1'), request.json.get('node2')
    if not all([node1, node2]): return jsonify({'error': 'Both node IDs are required'}), 400
    if not G.has_node(node1) or not G.has_node(node2): return jsonify({'error': 'One or both nodes do not exist'}), 404
    G.add_edge(node1, node2, label="", details={'tags': [], 'description': ''})
    return jsonify({'status': 'success', 'edge': [node1, node2]})

@app.route('/api/graph/remove_edge', methods=['POST'])
def remove_edge():
    node1, node2 = request.json.get('node1'), request.json.get('node2')
    if not all([node1, node2]): return jsonify({'error': 'Both node IDs are required'}), 400
    if not G.has_edge(node1, node2): return jsonify({'error': 'Edge does not exist'}), 404
    G.remove_edge(node1, node2)
    return jsonify({'status': 'success', 'edge': [node1, node2]})

@app.route('/api/graph/batch_add', methods=['POST'])
def batch_add_nodes():
    source_node, nodes_to_add = request.json.get('source_node'), request.json.get('nodes_to_add')
    if not all([source_node, nodes_to_add]): return jsonify({'error': 'Source node and nodes to add are required'}), 400
    if not G.has_node(source_node): return jsonify({'error': 'Source node does not exist'}), 404
    added_count = 0
    for node_name in nodes_to_add:
        if not G.has_node(node_name):
            G.add_node(node_name, title={'tags': [], 'description': f'Connected to {source_node}'}, click_count=0)
            G.add_edge(source_node, node_name, label="", details={'tags': [], 'description': 'Auto-generated connection'})
            added_count += 1
    return jsonify({'status': 'success', 'message': f'Added {added_count} new nodes and connected them to "{source_node}".'})

initialize_llm()
load_latest_snapshot()

if __name__ == '__main__':
    app.run(debug=True)