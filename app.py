import heapq
import itertools
from collections import deque

import matplotlib.pyplot as plt
import pydeck as pdk
import streamlit as st

st.set_page_config(page_title="Bipartite Graphs for Job Assignment", page_icon="🔗", layout="wide")

# =====================================================================
# STYLE: geometric background (3 colors: navy, blue, crimson)
# =====================================================================
st.markdown(
    """
<style>
.stApp {
    background-color: #0b1437;
    background-image:
        linear-gradient(135deg, rgba(59,106,168,.65) 25%, transparent 25%),
        linear-gradient(225deg, rgba(59,106,168,.65) 25%, transparent 25%),
        linear-gradient(45deg,  rgba(200,32,60,.55) 25%, transparent 25%),
        linear-gradient(315deg, rgba(200,32,60,.55) 25%, #0b1437 25%);
    background-size: 64px 64px;
    background-position: 32px 0, 32px 0, 0 0, 0 0;
    background-attachment: fixed;
}
[data-testid="stHeader"] { background: transparent; }
[data-testid="stSidebar"] {
    background-color: #0b1437;
    background-image:
        linear-gradient(160deg, rgba(200,32,60,.35) 0%, transparent 45%),
        linear-gradient(20deg, rgba(59,106,168,.55) 0%, transparent 55%);
    border-right: 3px solid #c8203c;
}
.block-container {
    background: rgba(8, 14, 40, .84);
    backdrop-filter: blur(4px);
    border: 1px solid rgba(59,106,168,.7);
    border-radius: 18px;
    padding: 2rem 2.5rem !important;
    margin-top: 1.2rem;
    box-shadow: 0 8px 30px rgba(0,0,0,.45);
}
.block-container h1, .block-container h2, .block-container h3, .block-container h4,
.block-container label, .block-container p, .block-container li,
.block-container [data-testid="stMarkdownContainer"],
.block-container [data-testid="stCaptionContainer"],
[data-testid="stSidebar"] * { color: #f4f1ea; }
.block-container h1, .block-container h2 { border-bottom: 3px solid #c8203c; padding-bottom: .3rem; }
</style>
""",
    unsafe_allow_html=True,
)

# =====================================================================
# JAVA MAP DATA (city coordinates & approximate road distances in km)
# =====================================================================
CITIES = {
    "Jakarta": (-6.2088, 106.8456),
    "Bandung": (-6.9175, 107.6191),
    "Cirebon": (-6.7063, 108.5570),
    "Tegal": (-6.8694, 109.1402),
    "Purwokerto": (-7.4242, 109.2302),
    "Semarang": (-6.9667, 110.4167),
    "Yogyakarta": (-7.7956, 110.3695),
    "Solo": (-7.5755, 110.8243),
    "Madiun": (-7.6298, 111.5239),
    "Surabaya": (-7.2575, 112.7521),
    "Malang": (-7.9666, 112.6326),
}
ROADS = [
    ("Jakarta", "Bandung", 150), ("Jakarta", "Cirebon", 250), ("Bandung", "Cirebon", 130),
    ("Cirebon", "Tegal", 70), ("Cirebon", "Purwokerto", 150), ("Tegal", "Purwokerto", 90),
    ("Tegal", "Semarang", 160), ("Purwokerto", "Yogyakarta", 190), ("Semarang", "Yogyakarta", 120),
    ("Semarang", "Solo", 100), ("Solo", "Yogyakarta", 65), ("Solo", "Madiun", 100),
    ("Madiun", "Surabaya", 170), ("Madiun", "Malang", 190), ("Surabaya", "Malang", 95),
    ("Semarang", "Surabaya", 350),
]


# =====================================================================
# HELPERS
# =====================================================================
def draw_bipartite(left, right, edges, highlight=None, left_title="", right_title=""):
    """Draw a bipartite graph: left set on the left, right set on the right. Highlighted edges are red."""
    highlight = highlight or set()
    fig, ax = plt.subplots(figsize=(5, 3.5))
    pos = {}
    for i, v in enumerate(left):
        pos[v] = (0, -i)
    for i, v in enumerate(right):
        pos[v] = (1, -i)
    for u, v in edges:
        x = [pos[u][0], pos[v][0]]
        y = [pos[u][1], pos[v][1]]
        hl = (u, v) in highlight
        ax.plot(x, y, color="crimson" if hl else "#3b6aa8", lw=3 if hl else 1.5, zorder=1)
    for v, (x, y) in pos.items():
        ax.scatter([x], [y], s=1400, color="#76c7c0", edgecolor="#a3237a", linewidth=2, zorder=2)
        ax.text(x, y, v, ha="center", va="center", fontsize=9, zorder=3)
    ax.text(0, 0.7, left_title, ha="center", fontsize=11, weight="bold")
    ax.text(1, 0.7, right_title, ha="center", fontsize=11, weight="bold")
    ax.set_xlim(-0.3, 1.3)
    ax.set_ylim(-max(len(left), len(right)) + 0.4, 1.1)
    ax.axis("off")
    return fig


def is_bipartite(vertices, edges):
    """Check bipartiteness with 2-coloring (BFS)."""
    adj = {v: [] for v in vertices}
    for u, v in edges:
        adj[u].append(v)
        adj[v].append(u)
    color = {}
    for s in vertices:
        if s in color:
            continue
        color[s] = 0
        q = deque([s])
        while q:
            u = q.popleft()
            for w in adj[u]:
                if w == u:
                    return False, {}
                if w not in color:
                    color[w] = 1 - color[u]
                    q.append(w)
                elif color[w] == color[u]:
                    return False, {}
    return True, color


def dijkstra(src, dst):
    """Shortest route between two cities. Returns (distance_km, list_of_cities)."""
    adj = {c: [] for c in CITIES}
    for u, v, w in ROADS:
        adj[u].append((v, w))
        adj[v].append((u, w))
    dist = {c: float("inf") for c in CITIES}
    prev = {}
    dist[src] = 0
    pq = [(0, src)]
    while pq:
        d, u = heapq.heappop(pq)
        if d > dist[u]:
            continue
        for v, w in adj[u]:
            if d + w < dist[v]:
                dist[v] = d + w
                prev[v] = u
                heapq.heappush(pq, (dist[v], v))
    if dist[dst] == float("inf"):
        return float("inf"), []
    path = [dst]
    while path[-1] != src:
        path.append(prev[path[-1]])
    return dist[dst], path[::-1]


def show_map(paths=None):
    """Map of Java: all roads (grey) + selected routes (red)."""
    paths = paths or []
    grey = [{"s": [CITIES[u][1], CITIES[u][0]], "t": [CITIES[v][1], CITIES[v][0]]} for u, v, _ in ROADS]
    red = []
    for p in paths:
        for a, b in zip(p, p[1:]):
            red.append({"s": [CITIES[a][1], CITIES[a][0]], "t": [CITIES[b][1], CITIES[b][0]]})
    pts = [{"name": n, "pos": [lon, lat]} for n, (lat, lon) in CITIES.items()]
    layers = [
        pdk.Layer("LineLayer", grey, get_source_position="s", get_target_position="t",
                  get_color=[130, 130, 130], get_width=3, width_min_pixels=2),
        pdk.Layer("LineLayer", red, get_source_position="s", get_target_position="t",
                  get_color=[220, 20, 60], get_width=8, width_min_pixels=5),
        pdk.Layer("ScatterplotLayer", pts, get_position="pos", get_radius=9000,
                  get_fill_color=[59, 106, 168], pickable=True),
        pdk.Layer("TextLayer", pts, get_position="pos", get_text="name", get_size=15,
                  get_color=[20, 20, 20], get_pixel_offset=[0, -18]),
    ]
    view = pdk.ViewState(latitude=-7.2, longitude=109.8, zoom=5.6, pitch=0)
    st.pydeck_chart(pdk.Deck(layers=layers, initial_view_state=view, map_style="light",
                             tooltip={"text": "{name}"}))


# =====================================================================
# SIDEBAR
# =====================================================================
st.sidebar.title("Group 3 – Discrete Math")
st.sidebar.caption("Bipartite Graphs for Job Assignment")
page = st.sidebar.radio(
    "Menu",
    [
        "Home",
        "1. Introduction to the Graph",
        "2. Types of Graphs",
        "3. Bipartite Graph",
        "4. Complete Bipartite Graph",
        "5. Job Assignment",
        "6. Real-World Example (Map)",
        "7. Conclusion",
    ],
)
st.sidebar.markdown("---")
st.sidebar.markdown(
    "**Presented by:**  \n"
    "Muhammad Dzaky Herwandi (021202600035)  \n"
    "Ucok Aulian Sihombing (021202600021)  \n"
    "Wianda Nandita Zebua (021202600006)"
)

# =====================================================================
# PAGES
# =====================================================================
if page == "Home":
    st.title("Bipartite Graphs for Job Assignment")
    st.subheader("Discrete Math – Group 3")
    st.write("Presented by:")
    st.markdown(
        "- Muhammad Dzaky Herwandi (021202600035)\n"
        "- Ucok Aulian Sihombing (021202600021)\n"
        "- Wianda Nandita Zebua (021202600006)"
    )
    st.markdown("### Table of Contents")
    st.markdown(
        "1. Introduction to the Graph\n"
        "2. Types of Graphs\n"
        "3. Bipartite Graph\n"
        "4. Complete Bipartite Graph\n"
        "5. Job Assignment\n"
        "6. Real-World Example (Map)\n"
        "7. Conclusion"
    )
    st.info("Choose a topic from the menu on the left.")

elif page == "1. Introduction to the Graph":
    st.header("Introduction to the Graph")
    st.markdown(
        "- A graph is written as **G = (V, E)**, where **V** is a set of vertices (nodes) "
        "and **E** is a set of edges connecting them.\n"
        "- **Undirected graph:** the edges have no direction.\n"
        "- **Directed graph (digraph):** the edges have a direction."
    )
    c1, c2 = st.columns(2)
    pts = {"1": (0, 1), "2": (1, 1.2), "3": (0.5, 0.5), "4": (-0.2, 0), "5": (1.1, 0)}
    with c1:
        st.markdown("**(a) Undirected graph**")
        fig, ax = plt.subplots(figsize=(4, 3))
        for u, v in [("1", "2"), ("1", "3"), ("2", "3"), ("3", "4"), ("3", "5"), ("4", "5")]:
            ax.plot([pts[u][0], pts[v][0]], [pts[u][1], pts[v][1]], color="#3b6aa8")
        for k, (x, y) in pts.items():
            ax.scatter([x], [y], s=500, color="white", edgecolor="#3b6aa8", zorder=2)
            ax.text(x, y, k, ha="center", va="center")
        ax.axis("off")
        st.pyplot(fig)
    with c2:
        st.markdown("**(b) Directed graph**")
        fig, ax = plt.subplots(figsize=(4, 3))
        for u, v in [("1", "2"), ("1", "3"), ("3", "2"), ("3", "4"), ("5", "3"), ("4", "5")]:
            ax.annotate("", xy=pts[v], xytext=pts[u],
                        arrowprops=dict(arrowstyle="->", color="#3b6aa8", shrinkA=14, shrinkB=14))
        for k, (x, y) in pts.items():
            ax.scatter([x], [y], s=500, color="white", edgecolor="#3b6aa8", zorder=2)
            ax.text(x, y, k, ha="center", va="center")
        ax.axis("off")
        st.pyplot(fig)
    st.caption("See menu 6 (Real-World Example) for a graph drawn on the map of Java.")

elif page == "2. Types of Graphs":
    st.header("Types of Graphs")
    c1, c2, c3 = st.columns(3)
    c1.markdown("#### Simple Graph")
    c1.write("There are no loops (no edge from a vertex to itself) and no multiple edges "
             "(no duplicate connections).")
    c2.markdown("#### Multigraph")
    c2.write("Two vertices can be connected by more than one edge.")
    c3.markdown("#### Pseudograph")
    c3.write("Contains loops and multiple edges.")

elif page == "3. Bipartite Graph":
    st.header("Bipartite Graph")
    st.write(
        "A graph G = (V, E) is said to be bipartite if the vertex set V can be divided into two "
        "disjoint subsets V1 and V2 such that every edge in E connects a vertex in V1 to a vertex "
        "in V2. No edge in the graph connects two vertices within V1 or two vertices within V2."
    )
    edges = [("A", "X"), ("A", "Y"), ("B", "X"), ("B", "Y"), ("C", "Y")]
    st.pyplot(draw_bipartite(["A", "B", "C"], ["X", "Y", "Z"], edges, left_title="V1", right_title="V2"))
    st.caption("Example: V1 = {A, B, C}, V2 = {X, Y, Z}. Z has no edge, but the graph is still bipartite.")

    st.markdown("#### Try it yourself: is this graph bipartite?")
    txt = st.text_input("Enter edges separated by commas (example: A-B, B-C, C-A)", "A-X, A-Y, B-X, B-Y, C-Y")
    try:
        es = [tuple(p.strip().split("-")) for p in txt.split(",") if p.strip()]
        vs = sorted({v for e in es for v in e})
        ok, col = is_bipartite(vs, es)
        if ok:
            st.success("This graph IS bipartite.")
            st.write("V1 =", [v for v in vs if col[v] == 0], "| V2 =", [v for v in vs if col[v] == 1])
        else:
            st.error("This graph is NOT bipartite (it contains an odd cycle or a loop).")
    except Exception:
        st.warning("Invalid format. Use the format A-B, B-C.")

elif page == "4. Complete Bipartite Graph":
    st.header("Complete Bipartite Graph")
    st.write(
        "A complete bipartite graph or **biclique** is a special type of bipartite graph in which "
        "every vertex in the first set is connected to every vertex in the second set."
    )
    m = st.slider("Number of vertices in the first set (m)", 1, 5, 4)
    n = st.slider("Number of vertices in the second set (n)", 1, 5, 3)
    left = [chr(65 + i) for i in range(m)]
    right = [chr(80 + i) for i in range(n)]
    edges = [(u, v) for u in left for v in right]
    st.pyplot(draw_bipartite(left, right, edges))
    st.success(f"K({m},{n}) has {m} × {n} = {m * n} edges.")

elif page == "5. Job Assignment":
    st.header("Job Assignment")
    st.markdown(
        "- Three jobs are to be distributed among 3 persons (A, B, C).\n"
        "- Each job can only be done by persons who are qualified for it.\n"
        "- The table is drawn as a bipartite graph: jobs on the left, persons on the right, "
        "and an edge means \"qualified\"."
    )
    jobs = ["Admin", "Programmer", "Professor"]
    persons = ["A", "B", "C"]
    default = {"Admin": ["A", "C"], "Programmer": ["B", "C"], "Professor": ["A", "B"]}

    tab1, tab2 = st.tabs(["Basic case (qualification table)", "Complete Bipartite Case K(3,3)"])

    with tab1:
        st.markdown("Tick the boxes to change who is qualified:")
        cols = st.columns(4)
        cols[0].markdown("**Job**")
        for i, p in enumerate(persons):
            cols[i + 1].markdown(f"**{p}**")
        edges = []
        for j in jobs:
            cols = st.columns(4)
            cols[0].write(j)
            for i, p in enumerate(persons):
                if cols[i + 1].checkbox("qualified", value=p in default[j], key=f"q_{j}_{p}",
                                        label_visibility="collapsed"):
                    edges.append((j, p))
        sols = [perm for perm in itertools.permutations(persons)
                if all((j, p) in edges for j, p in zip(jobs, perm))]
        c1, c2 = st.columns(2)
        with c1:
            hl = set(zip(jobs, sols[0])) if sols else set()
            st.pyplot(draw_bipartite(jobs, persons, edges, highlight=hl, left_title="Jobs", right_title="Persons"))
        with c2:
            if sols:
                st.success(f"There are {len(sols)} valid ways to assign the jobs.")
                for s in sols:
                    st.write(" | ".join(f"{j} → {p}" for j, p in zip(jobs, s)))
                st.caption("Red edges = one example of an assignment (a perfect matching).")
            else:
                st.error("No assignment can fill every job.")

    with tab2:
        st.write("Case: every person must work part-time in every department. "
                 "Every job is connected to every person.")
        edges = [(j, p) for j in jobs for p in persons]
        st.pyplot(draw_bipartite(jobs, persons, edges, left_title="Jobs", right_title="Persons"))
        st.success("The graph becomes a complete bipartite graph K(3,3) with 3 × 3 = 9 edges.")

elif page == "6. Real-World Example (Map)":
    st.header("Real-World Example (Map)")
    st.write(
        "A road network is a graph **G = (V, E)**: **V** = cities, **E** = roads between cities, and each "
        "edge has a weight (distance in km). Below we first find the shortest route between two cities, "
        "then use the same map for a bipartite job-assignment problem."
    )

    # ---- Part 1: shortest route ----
    st.subheader("Part 1: Shortest route on the map of Java")
    names = list(CITIES)
    c1, c2 = st.columns(2)
    a = c1.selectbox("Origin city (A)", names, index=0)
    b = c2.selectbox("Destination city (B)", names, index=names.index("Surabaya"))
    d, path = dijkstra(a, b)
    if a == b:
        st.info("The origin and the destination are the same city.")
        show_map()
    else:
        st.success(f"Shortest route: {' → '.join(path)}  |  Distance ≈ {d} km")
        show_map([path])
    st.caption("Computed with Dijkstra's algorithm. Road distances are approximate and for illustration only.")

    ok, _ = is_bipartite(list(CITIES), [(u, v) for u, v, _ in ROADS])
    if ok:
        st.info("This road graph is bipartite.")
    else:
        st.info(
            "**Is this road graph bipartite? No.** It contains an odd cycle, for example "
            "Jakarta–Bandung–Cirebon–Jakarta (three cities that are all connected to each other), "
            "so the cities cannot be split into two groups with no road inside a group. "
            "To get a bipartite graph, we separate the cities from the couriers in Part 2."
        )

    st.markdown("---")

    # ---- Part 2: bipartite courier assignment ----
    st.subheader("Part 2: Job assignment on the map (couriers → cities)")
    st.write(
        "Same idea as the job-assignment case, but on the map of Java. The **left set** is the couriers, "
        "the **right set** is the destination cities, and an **edge** means the courier can deliver to "
        "that city. Every courier gets exactly one city and every city is served by one courier "
        "(a perfect matching). Among all valid assignments, we choose the one with the **shortest total distance**."
    )
    couriers = ["Courier A", "Courier B", "Courier C"]
    dests = ["Bandung", "Semarang", "Malang"]
    default_bases = ["Jakarta", "Yogyakarta", "Surabaya"]
    default_q = {
        "Courier A": ["Bandung", "Semarang"],
        "Courier B": ["Semarang", "Malang"],
        "Courier C": ["Bandung", "Malang"],
    }
    st.markdown("**Home base of each courier:**")
    bcols = st.columns(3)
    bases = {}
    for i, c in enumerate(couriers):
        bases[c] = bcols[i].selectbox(c, list(CITIES), index=list(CITIES).index(default_bases[i]),
                                      key=f"base_{c}")

    st.markdown("**Which courier can deliver to which city (qualified):**")
    cols = st.columns(4)
    cols[0].markdown("**Courier**")
    for i, dname in enumerate(dests):
        cols[i + 1].markdown(f"**{dname}**")
    l_edges = []
    for c in couriers:
        cols = st.columns(4)
        cols[0].write(c)
        for i, dname in enumerate(dests):
            if cols[i + 1].checkbox("can", value=dname in default_q[c], key=f"l_{c}_{dname}",
                                    label_visibility="collapsed"):
                l_edges.append((c, dname))

    options = []
    for perm in itertools.permutations(dests):
        if all((c, dname) in l_edges for c, dname in zip(couriers, perm)):
            legs = [dijkstra(bases[c], dname) for c, dname in zip(couriers, perm)]
            options.append((sum(x[0] for x in legs), perm, legs))
    options.sort(key=lambda x: x[0])

    c1, c2 = st.columns([1, 2])
    with c1:
        hl = set(zip(couriers, options[0][1])) if options else set()
        st.pyplot(draw_bipartite(couriers, dests, l_edges, highlight=hl,
                                 left_title="Couriers", right_title="Destination cities"))
    with c2:
        if options:
            total, perm, legs = options[0]
            st.success(f"{len(options)} valid assignment(s). Best total distance ≈ {total} km")
            for c, dname, (km, p) in zip(couriers, perm, legs):
                st.write(f"**{c}** ({bases[c]}) → **{dname}**: {' → '.join(p)} ({km} km)")
        else:
            st.error("No assignment can serve every city.")
    if options:
        show_map([leg[1] for leg in options[0][2]])
        st.caption("Red lines = the shortest route of each courier in the best assignment. "
                   "Distances are approximate.")
    else:
        show_map()

elif page == "7. Conclusion":
    st.header("Conclusion")
    st.write(
        "In job assignment problems, bipartite graphs help represent the relationship between two "
        "distinct sets — workers and jobs — where edges show possible assignments. When every worker "
        "is connected to every job, the graph becomes a complete bipartite graph, showing full "
        "connectivity. This concept allows tasks to be distributed efficiently and visually "
        "demonstrates how assignments can be optimized."
    )
    st.balloons()
    st.markdown("## Thank you!")
