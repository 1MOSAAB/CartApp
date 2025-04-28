import firebase_admin
from firebase_admin import credentials, db
import os
import networkx as nx
import plotly.graph_objects as go
import groq
import queue
import threading
import time
import streamlit as st



st.title("🛒Go CART ")


# Define the supermarket nodes and products
nodes = {
    "A": ["Milk", "Bread", "Water"],
    "B": ["Jam", "Banana", "Peanut Butter"],
    "C": ["Apple", "Fish", "Cheese"],
    "D": ["Egg", "Chicken", "Rice"],
    "E": ["Pasta", "Tomato Sauce", "Olive Oil"],
    "F": ["Cereal", "Yogurt", "Honey"],
    "G": ["Carrot", "Potato", "Onion"],
    "H": ["Orange", "Lettuce", "Cucumber"],
    "I": ["Beef", "Shrimp", "Mushrooms"],
    "J": ["Flour", "Sugar", "Baking Powder"],
    "K": ["Chocolate", "Cookies", "Ice Cream"],
    "L": ["Tea", "Coffee", "Juice"],
    "M": ["Soap", "Shampoo", "Toothpaste"]
}

product_to_node = {product.lower(): node for node, products in nodes.items() for product in products}



#--------------------------------------------------------------------------------------- Map Creation Edges Nodes ---------------------------------------------------------------------------------------#

# Create graph with weighted edges
G = nx.Graph()

# Add edges with weights as defined
G.add_edge("A", "B", weight=4)
G.add_edge("A", "F", weight=4)
G.add_edge("B", "C", weight=4)
G.add_edge("C", "D", weight=4)
G.add_edge("C", "H", weight=4)
G.add_edge("D", "E", weight=4)
G.add_edge("E", "J", weight=4)
G.add_edge("F", "G", weight=4)
G.add_edge("F", "K", weight=4)
G.add_edge("G", "H", weight=4)
G.add_edge("H", "I", weight=4)
G.add_edge("H", "L", weight=4)
G.add_edge("H", "C", weight=4)
G.add_edge("I", "J", weight=4)
G.add_edge("J", "E", weight=4)
G.add_edge("J", "M", weight=4)
G.add_edge("K", "L", weight=8)
G.add_edge("L", "M", weight=8)
G.add_edge("L", "H", weight=4)
G.add_edge("M", "J", weight=4)

# Fixed node positions (scaled)
pos = {
    'A': (2, 6), 'B': (2, 5), 'C': (2, 4), 'D': (2, 3), 'E': (2, 2),
    'F': (1, 6), 'G': (1, 5), 'H': (1, 4), 'I': (1, 3), 'J': (1, 2),
    'K': (0, 6), 'L': (0, 4), 'M': (0, 2)
}

#--------------------------------------------------------------------------------------- Firebase Listener ---------------------------------------------------------------------------------------#

service_account_info = {
    "type": "service_account",
    "project_id": st.secrets["type"]["project_id"],
    "private_key_id": st.secrets["type"]["private_key_id"],
    "private_key": st.secrets["type"]["private_key"],
    "client_email": st.secrets["type"]["client_email"],
    "client_id": st.secrets["type"]["client_id"],
    "auth_uri": st.secrets["type"]["auth_uri"],
    "token_uri": st.secrets["type"]["token_uri"],
    "auth_provider_x509_cert_url": st.secrets["type"]["auth_provider_x509_cert_url"],
    "client_x509_cert_url": st.secrets["type"]["client_x509_cert_url"],
    "universe_domain": st.secrets["type"]["universe_domain"],
}


# Initialize Firebase (only once)
if not firebase_admin._apps:
    cred = credentials.Certificate(service_account_info)
    firebase_admin.initialize_app(cred, {
        "databaseURL": "https://read-write-e65d9-default-rtdb.firebaseio.com/"
    })

# Firebase reference
ref = db.reference("/rfid_tags/latest")

# Queue for handling RFID updates in the main thread
rfid_queue = queue.Queue()

# Firebase listener
def stream_handler(event):
    if event.data:  # Ensure data is not None
        rfid_queue.put(event.data)  # Send RFID update to main thread


def start_firebase_listener():
    ref.listen(stream_handler)

listener_thread = threading.Thread(target=start_firebase_listener)
listener_thread.daemon = True  # This ensures the thread will exit when the program exits
listener_thread.start()

#--------------------------------------------------------------------------------------- Groq AI ---------------------------------------------------------------------------------------#


# Set your Groq API key
GROQ_API_KEY = st.secrets["groq"]["api_key"]

# Initialize the Groq client
client = groq.Client(api_key=GROQ_API_KEY)

# Extract products function (AI prompt)
def extract_products(sentence):
    """Extracts product names from the input sentence using Groq AI."""
    
    prompt = f"""
    Extract only the product names mentioned in this sentence and return them as a comma-separated list and singular and correct the spelling.
    If no products are found, return an empty string. if only there is no product in the sentence respond with "product list is empty" and i dont want any feed back
    
    Sentence: "{sentence}"
    
    Response:
    """

    # Call Groq API
    response = client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.2,
    )
    
    # Extract AI response and convert to a list
    extracted_products = response.choices[0].message.content.strip()
    return [p.strip().lower() for p in extracted_products.split(',') if p.strip()]

# Function to find nodes for given product names
def find_nodes_by_products(product_names):
    return {product_to_node[product] for product in product_names if product in product_to_node}


#--------------------------------------------------------------------------------------- Dijkstra Shortest path ---------------------------------------------------------------------------------------#

# Function to find the shortest path
def find_shortest_path(start, targets):
    full_path = [start]
    current = start
    unvisited = set(targets)

    while unvisited:
        nearest = min(unvisited, key=lambda node: nx.shortest_path_length(G, current, node, weight="weight"))
        shortest_path = nx.shortest_path(G, current, nearest, weight="weight")
        full_path.extend(shortest_path[1:])  
        unvisited.remove(nearest)
        current = nearest

    return full_path

#--------------------------------------------------------------------------------------- Map update ---------------------------------------------------------------------------------------#

# Update the map with Plotly
def update_map(current, path, targets, fig):
    # Clear the previous data from the figure
    fig.data = []

    # Add edges (gray color)
    edge_x = []
    edge_y = []
    for edge in G.edges:
        x0, y0 = pos[edge[0]]
        x1, y1 = pos[edge[1]]
        edge_x.append(x0)
        edge_y.append(y0)
        edge_x.append(x1)
        edge_y.append(y1)
        edge_x.append(None)  # Line break between edges
        edge_y.append(None)
    
    fig.add_trace(go.Scatter(x=edge_x, y=edge_y, mode='lines', line=dict(color='gray', width=1), name="Edges"))

    # Add nodes (light gray color)
    node_x = [pos[node][0] for node in G.nodes]
    node_y = [pos[node][1] for node in G.nodes]

    fig.add_trace(go.Scatter(x=node_x, y=node_y, mode='markers+text', text=list(G.nodes),
                             marker=dict(size=20, color='lightgray', line=dict(width=2, color='black')),
                             textposition="top right", textfont=dict(size=12, color='black', family='Arial'), name="Nodes"))

    # Update the path (highlighted in blue)
    if path:
        path_x = []
        path_y = []
        for i in range(len(path) - 1):
            x0, y0 = pos[path[i]]
            x1, y1 = pos[path[i + 1]]
            path_x.append(x0)
            path_y.append(y0)
            path_x.append(x1)
            path_y.append(y1)
            path_x.append(None)
            path_y.append(None)
        
        fig.add_trace(go.Scatter(x=path_x, y=path_y, mode='lines', line=dict(color='blue', width=3), name="Path"))

    # Update the target nodes (blue color)
    if targets:
        target_x = [pos[node][0] for node in targets]
        target_y = [pos[node][1] for node in targets]
        fig.add_trace(go.Scatter(x=target_x, y=target_y, mode='markers', marker=dict(size=20, color='blue', line=dict(width=2, color='black')), name="Targets"))

    # Update the current node (green color)
    if current:
        current_x = [pos[current][0]]
        current_y = [pos[current][1]]
        fig.add_trace(go.Scatter(x=current_x, y=current_y, mode='markers', marker=dict(size=30, color='green', line=dict(width=3, color='black')), name="Current"))

    # Update the layout (center title and show legend)
    fig.update_layout(title="Supermarket Map", showlegend=True)

    fig.update_layout(
        title="Supermarket Map",
        title_x=0.35,  # Center the title
        showlegend=True,
        width=600,  # Width of the figure (4x6 ratio -> 4 * 150)
        height=900,  # Height of the figure (6 * 150)
    )

    # Display the updated figure in Streamlit
    map_placeholder.plotly_chart(fig)


#--------------------------------------------------------------------------------------- RFID Tags Process ---------------------------------------------------------------------------------------#

# Function to process RFID tag updates
def process_rfid_tag(rfid_tag, fig):
    global current_position, targets

    if rfid_tag in G.nodes:  # Ensure valid node
        print(f"📥 New RFID Tag Detected: {rfid_tag}")
        current_position = rfid_tag  # Update current position

        # Find shortest path
        if targets:
            path = find_shortest_path(current_position, targets)
            shortest_path_placeholder.markdown(f"🔄 **Shortest Path:** {' → '.join(path)}")
            update_map(current_position, path, targets, fig)

            # If the scanned RFID tag is in targets, mark it as visited
            if current_position in targets:
                targets.remove(current_position)
                st.write(f"✅ Visited {current_position}")

        if not targets:
            print("🎯 All nodes visited!")


#/////////////////////////////////////////////////////////////////////////////////////// Start of the Interactions \\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\#

# --- Initialize session state ---
for key, default in {
    "product_input": [],
    "available": [],
    "unavailable": [],
    "target": set(),
    "targets": set(),
    "input_text": ""
}.items():
    if key not in st.session_state:
        st.session_state[key] = default

# --- UI input ---
user_input = st.text_input("**grocery list** (enter what you want to buy)", value=st.session_state.input_text)

if user_input and user_input != st.session_state.input_text:
    st.session_state.input_text = user_input  # update session
    product_input = extract_products(user_input)

    for product in product_input:
        if product not in st.session_state.product_input:
            st.session_state.product_input.append(product)

            if product.lower() in product_to_node:
                st.session_state.available.append(product)
                node = product_to_node.get(product.lower())
                if node:
                    st.session_state.target.add(node)
            else:
                st.session_state.unavailable.append(product)

# Assign session state to local variables (keep your original names)
product_input = st.session_state.product_input
available = st.session_state.available
unavailable = st.session_state.unavailable
target = st.session_state.target

# --- Display results ---
if available:
    st.write("**🛒 Available Products List:**")
    for item in available:
        st.write(f"- {item}")
else:
    st.write("No available products found.")

if unavailable:
    if len(unavailable) == 1:
        st.write("❌", f"{unavailable[0]} is not available")
    else:
        st.write("❌", ", ".join(unavailable) + " are not available")

# --- Reset button ---
if st.button("🧹 Clear Grocery List"):
    st.session_state.product_input = []
    st.session_state.available = []
    st.session_state.unavailable = []
    st.session_state.target = set()
    st.session_state.input_text = ""  # clear input field
    st.rerun()

if target:
    st.write("📍 Available products found at nodes: " + ", ".join(sorted(target)))



#--------------------------------------------------------------------------------------- Map Drawing ---------------------------------------------------------------------------------------#

if st.button("Navigate"):
    targets = target   


shortest_path_placeholder = st.empty()
map_placeholder = st.empty()
fig = go.Figure()

while True:
    try:
        # Process RFID tags as they come in
        rfid_tag = rfid_queue.get(timeout=2)  # Adjust timeout as necessary
        process_rfid_tag(rfid_tag, fig)
        time.sleep(1)  # Adjust sleep time if necessary to prevent blocking

    except queue.Empty:
        # If no new RFID tags, continue the loop
        continue

