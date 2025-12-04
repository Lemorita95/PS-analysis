from helpers import nx, plt

def plot_5bus_network(V, I, pos=None, graph_labels=None, type='current'):
    """
    Plot a 5-bus network (with optional Load node 6) with voltages and currents/power.

    Arguments:
    - V: dict, bus voltages {bus: (magnitude, angle)}
    - I: dict, values i->j {(i,j): (mag, ang)} OR {(i,j): mag}
    - pos: dict, bus positions
    - graph_labels: custom current/power labels {(i,j): str}
    - type: 'current' or 'power' (controls text prefix and units)
    """
    
    # ---- TYPE LOGIC ----
    if type == 'Ppower':
        label_type = 'P'
        label_unit = 'W'
    elif type == 'current':
        label_type = 'I'
        label_unit = 'A'
    elif type == 'Qpower':
        label_type = 'Q'
        label_unit = 'VAr'
    elif type == 'Spower':
        label_type = 'S'
        label_unit = 'VA'
    else:
        raise ValueError("not supported type.")
    # ---------------------

    if graph_labels is None:
        graph_labels = {}

    # Add node 6 if not present in V
    if 6 not in V:
        V[6] = (0,0)
    
    # Create graph
    G = nx.Graph()
    buses = list(V.keys())
    G.add_nodes_from(buses)
    
    # Lines from I keys (undirected)
    lines = set([tuple(sorted(edge)) for edge in I.keys()])
    G.add_edges_from(lines)
    
    # Default positions
    if pos is None:
        pos = {
            1: (0,0), 2: (1,0), 3: (2,0), 4: (3,0),
            5: (1.5, -0.25), 6: (1.5, -0.5)
        }
    
    fig, ax = plt.subplots(figsize=(12,5.5))
    
    # Draw edges and nodes
    nx.draw_networkx_edges(G, pos, width=2, ax=ax)
    nx.draw_networkx_nodes(G, pos, node_color='lightblue', node_size=800, ax=ax)
    
    # Node numbers (Load for node 6)
    for i in G.nodes():
        x, y = pos[i]
        label = "Load" if i == 6 else str(i)
        ax.text(x, y, label, fontsize=12, ha='center', va='center', fontweight='bold')
    
    # Voltage labels
    v_offset = 0.05
    for i in G.nodes():
        if i == 6:
            continue
        x, y = pos[i]
        text = f'{V[i][0]:.1f}∠{V[i][1]:.1f}°'
        if i == 5:
            ax.text(x, y - v_offset, text, fontsize=10, ha='center', va='top')
        else:
            ax.text(x, y + v_offset, text, fontsize=10, ha='center', va='bottom')

    # ---- Annotate currents/powers AND draw arrows ----
    edge_offset = 0.25
    for (i,j), value in I.items():

        # --- Accept multiple formats ---
        if isinstance(value, tuple):
            if len(value) == 2:
                mag, ang = value
                value_text = f'{mag:.2f}∠{ang:.1f}° [{label_unit}]'
            else:
                mag = value[0]
                value_text = f'{mag:.2f} [{label_unit}]'
        else:
            mag = value
            value_text = f'{mag:.2f} [{label_unit}]'
        # -------------------------------

        # Vector
        dx = pos[j][0] - pos[i][0]
        dy = pos[j][1] - pos[i][1]
        length = (dx**2 + dy**2)**0.5
        ux, uy = dx/length, dy/length
        
        # Label position
        x_i = pos[i][0] + ux*edge_offset
        y_i = pos[i][1] + uy*edge_offset + 0.025

        # special case (5,6)
        if (i, j) == (5, 6):
            dx_56 = pos[6][0] - pos[5][0]
            dy_56 = pos[6][1] - pos[5][1]
            dist_56 = (dx_56**2 + dy_56**2)**0.5

            x_i = pos[5][0]
            y_i = pos[5][1] - 2 * edge_offset * dist_56

        # Label key, e.g. I12 or P12
        default_label = f'{label_type}{i}{j}'
        label = graph_labels.get((i,j), default_label)

        ax.text(
            x_i, y_i, f'{label}={value_text}',
            color='red', fontsize=9, ha='center', va='center',
            bbox=dict(facecolor='white', edgecolor='none', boxstyle='round,pad=0.2')
        )

        # ---- ARROW: from node i to 25% of edge length ----
        arrow_fraction = 0.25          # 25% of the edge
        margin_fraction = 0.10         # start 10% away from node i

        start_d = margin_fraction * length
        end_d   = arrow_fraction * length

        start_x = pos[i][0] + ux * start_d
        start_y = pos[i][1] + uy * start_d

        end_x   = pos[i][0] + ux * end_d
        end_y   = pos[i][1] + uy * end_d

        ax.annotate(
            "",
            xy=(end_x, end_y),
            xytext=(start_x, start_y),
            arrowprops=dict(
                arrowstyle="-|>",
                color="black",
                linewidth=1.3,
                mutation_scale=15,   # <--- controls head size
            )
        )
        # ----------------------------------------------------


    ax.axis('off')
    
    return fig


def plot_5bus_PQ(V, P, Q, pos=None, graph_labels=None):

    if graph_labels is None:
        graph_labels = {}

    # Override helper (symmetric)
    def get_label_config(i, j):
        if (i, j) in graph_labels:
            return graph_labels[(i, j)]
        if (j, i) in graph_labels:
            return graph_labels[(j, i)]
        return {}

    if 6 not in V:
        V[6] = (0,0)

    G = nx.Graph()
    buses = list(V.keys())
    G.add_nodes_from(buses)

    lines = set([tuple(sorted(edge)) for edge in P.keys()])
    G.add_edges_from(lines)

    if pos is None:
        pos = {
            1: (0,0), 2: (1,0), 3: (2,0), 4: (3,0),
            5: (1.5, -0.25), 6: (1.5, -0.4)
        }

    fig, ax = plt.subplots(figsize=(12,6))
    nx.draw_networkx_edges(G, pos, width=2, ax=ax)
    nx.draw_networkx_nodes(G, pos, node_color='lightblue', node_size=800, ax=ax)

    # Node labels
    for i in G.nodes():
        x, y = pos[i]
        label = "Load" if i == 6 else str(i)
        ax.text(x, y, label, fontsize=10, ha='center', va='center', fontweight='bold')

    # Voltage labels
    v_offset = 0.025
    for i in G.nodes():
        if i == 6:
            continue
        x, y = pos[i]
        text = f'{V[i][0]:.1f}∠{V[i][1]:.1f}°'
        if i == 5:
            ax.text(x, y - v_offset, text, fontsize=8, ha='center', va='top')
        else:
            ax.text(x, y + v_offset, text, fontsize=8, ha='center', va='bottom')

    # Base offset
    edge_offset = 0.225

    # P/Q labels
    for (i, j) in P.keys():

        if (i, j) == (6, 5):
            continue

        dx = pos[j][0] - pos[i][0]
        dy = pos[j][1] - pos[i][1]
        length = (dx**2 + dy**2)**0.5
        ux, uy = dx / length, dy / length

        # ========= APPLY OVERRIDES =============
        cfg = get_label_config(i, j)

        # rotation: default 0
        rot = cfg.get("rotation", 0)

        # offset override: replaces base offset
        local_offset = cfg.get("offset", edge_offset)
        # =======================================

        x_i = pos[i][0] + ux * local_offset
        y_i = pos[i][1] + uy * local_offset

        # special case (5,6)
        if (i, j) == (5, 6):
            dx_56 = pos[6][0] - pos[5][0]
            dy_56 = pos[6][1] - pos[5][1]
            dist_56 = (dx_56**2 + dy_56**2)**0.5

            x_i = pos[5][0]
            y_i = pos[5][1] - 2 * edge_offset * dist_56

        padding = 0.01

        # P label
        ax.text(
            x_i, y_i + padding, f'P{i}{j}={P[i,j]:.1f} W',
            color='red', fontsize=8, ha='center', va='bottom',
            rotation=rot, rotation_mode='anchor',
            bbox=dict(facecolor='white', edgecolor='none', boxstyle='round,pad=0.2')
        )

        # Q label
        ax.text(
            x_i, y_i - padding, f'Q{i}{j}={Q[i,j]:.1f} VAr',
            color='blue', fontsize=8, ha='center', va='top',
            rotation=rot, rotation_mode='anchor',
            bbox=dict(facecolor='white', edgecolor='none', boxstyle='round,pad=0.2')
        )

        # Arrow drawing
        arrow_fraction = 0.20
        margin_fraction = 0.05
        start_d = margin_fraction * length
        end_d   = arrow_fraction * length

        start_x = pos[i][0] + ux * start_d
        start_y = pos[i][1] + uy * start_d
        end_x   = pos[i][0] + ux * end_d
        end_y   = pos[i][1] + uy * end_d

        ax.annotate(
            "",
            xy=(end_x, end_y),
            xytext=(start_x, start_y),
            arrowprops=dict(
                arrowstyle="-|>",
                color="black",
                linewidth=1.3,
                mutation_scale=15,
            )
        )

    ax.axis('off')
    plt.tight_layout()
    return fig
