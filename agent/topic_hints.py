"""
Topic visual hints — static fallback + dynamic Gemini-powered Animation Advisor.

The static TOPIC_VISUAL_HINTS acts as a fast fallback. For any topic,
`get_topic_hints()` first tries a dynamic Gemini call to produce topic-specific
animation recommendations, then falls back to the static dictionary.

Usage:
    from agent.topic_hints import get_topic_hints, format_hints_for_prompt
    hints = get_topic_hints("attention mechanism in transformers")
    # Returns a dict with visual_style, layout_tips, animation_suggestions, etc.
"""

import json
import os
from functools import lru_cache
from typing import Optional

from agent.llm import call_llm

ADVISOR_MODEL = os.getenv("GEMINI_ADVISOR_MODEL", "gemini-2.5-flash")

# ═══════════════════════════════════════════════════════════════════════
# DYNAMIC ANIMATION ADVISOR — asks Gemini what animations fit the topic
# ═══════════════════════════════════════════════════════════════════════

_ADVISOR_PROMPT = """You are an expert educational animation advisor who recommends the BEST visual approach for teaching a given topic using Manim.

Topic: {query}

Analyze this topic and recommend the ideal animation approach. Think about:
1. What visual style would best explain this concept?
2. What specific Manim animations and objects should be used?
3. What visual metaphors make this concept intuitive?
4. What are common mistakes animators make when visualizing this topic?
5. What should be shown vs. what should be kept as text/voiceover only?

Return ONLY a JSON object in this exact format:
{{
    "visual_style": "one of: array_boxes, bar_chart, diagram, graph_plot, physics_motion, timeline, flowchart, neural_network, matrix_grid",
    "layout_tips": [
        "Specific tip 1 for how to lay out visuals for THIS topic",
        "Specific tip 2...",
        "Specific tip 3..."
    ],
    "animation_suggestions": [
        "Concrete animation idea 1 — what Manim objects to use and how to animate them",
        "Concrete animation idea 2...",
        "Concrete animation idea 3..."
    ],
    "common_elements": ["element1", "element2", "element3"],
    "visual_metaphor": "A concrete visual metaphor that makes this topic intuitive",
    "key_formulas": ["formula1_if_relevant"],
    "avoid": [
        "What NOT to do when animating this topic"
    ],
    "max_simultaneous_objects": 8,
    "recommended_segments": {{
        "intro_hook": "What question or problem should open the video",
        "theory_focus": "What theory text should appear on screen",
        "core_animation": "What the main visual demo should show",
        "takeaway": "What the viewer should remember"
    }}
}}

RULES:
- Be SPECIFIC to this exact topic — not generic advice.
- animation_suggestions should reference actual Manim classes (Axes, Circle, Arrow, Line, Text, MathTex, etc.)
- For ML/DL topics: prefer graph_plot or diagram over bar_chart.
- For algorithms: prefer array_boxes or bar_chart.
- For math: prefer graph_plot.
- For processes: prefer flowchart or timeline.
- visual_metaphor must be a CONCRETE image, not an abstract description.
- Return ONLY the JSON. No explanation."""

_ADVISOR_SCHEMA = {
    "type": "OBJECT",
    "properties": {
        "visual_style": {"type": "STRING"},
        "layout_tips": {"type": "ARRAY", "items": {"type": "STRING"}},
        "animation_suggestions": {"type": "ARRAY", "items": {"type": "STRING"}},
        "common_elements": {"type": "ARRAY", "items": {"type": "STRING"}},
        "visual_metaphor": {"type": "STRING"},
        "key_formulas": {"type": "ARRAY", "items": {"type": "STRING"}},
        "avoid": {"type": "ARRAY", "items": {"type": "STRING"}},
        "max_simultaneous_objects": {"type": "INTEGER"},
        "recommended_segments": {
            "type": "OBJECT",
            "properties": {
                "intro_hook": {"type": "STRING"},
                "theory_focus": {"type": "STRING"},
                "core_animation": {"type": "STRING"},
                "takeaway": {"type": "STRING"},
            },
            "required": ["intro_hook", "theory_focus", "core_animation", "takeaway"],
        },
    },
    "required": [
        "visual_style", "layout_tips", "animation_suggestions",
        "common_elements", "visual_metaphor", "avoid",
        "max_simultaneous_objects", "recommended_segments",
    ],
}


@lru_cache(maxsize=64)
def generate_dynamic_hints(query: str) -> Optional[dict]:
    """Ask Gemini what animations would best teach this topic.

    Returns a dict of animation recommendations, or None on failure.
    Results are cached in-memory so repeated queries don't re-hit the API.
    """
    try:
        print(f"[AnimAdvisor] Generating dynamic hints for: {query[:60]}")
        response = call_llm(
            _ADVISOR_PROMPT.format(query=query),
            max_tokens=2048,
            response_mime_type="application/json",
            response_schema=_ADVISOR_SCHEMA,
            preferred_model=ADVISOR_MODEL,
        )
        hints = json.loads(response.strip())
        print(f"[AnimAdvisor] Dynamic hints: style={hints.get('visual_style')}, "
              f"suggestions={len(hints.get('animation_suggestions', []))}")
        return hints
    except Exception as e:
        print(f"[AnimAdvisor] Dynamic hints failed: {e} — falling back to static")
        return None


# ═══════════════════════════════════════════════════════════════════════
# STATIC FALLBACK HINTS (expanded with comprehensive ML/DL coverage)
# ═══════════════════════════════════════════════════════════════════════

TOPIC_VISUAL_HINTS = {
    # ── Sorting Algorithms ──────────────────────────────────────────
    "sorting": {
        "keywords": ["sort", "sorting", "bubble sort", "merge sort", "quick sort",
                      "insertion sort", "selection sort", "heap sort", "radix sort"],
        "visual_style": "bar_chart",
        "layout_tips": [
            "Use vertical bars whose height represents value — this is the standard for sorting visualizations.",
            "Highlight the two elements being compared with a bright color (e.g., yellow).",
            "Show swaps with smooth Transform animations, not teleportation.",
            "Keep array indices visible below each bar.",
            "For arrays >10 elements, split into 2 rows or use narrower bars.",
            "Add a 'sorted' color (green) to elements that reach their final position.",
        ],
        "common_elements": ["bars", "index_labels", "comparison_arrows", "sorted_markers"],
        "max_simultaneous_objects": 12,
        "avoid": ["Do not use circles or nodes for sorting — bars are universally expected."],
    },

    # ── Search Algorithms ───────────────────────────────────────────
    "searching": {
        "keywords": ["search", "binary search", "linear search", "searching"],
        "visual_style": "array_boxes",
        "layout_tips": [
            "Use a horizontal row of boxes with values inside — standard for search visualization.",
            "Highlight the current search window with a colored border.",
            "For binary search: show low/high pointers as arrows below the array.",
            "Gray out eliminated elements to show progress.",
            "Show the target value prominently above or beside the array.",
        ],
        "common_elements": ["value_boxes", "pointer_arrows", "target_label", "eliminated_overlay"],
        "max_simultaneous_objects": 10,
        "avoid": ["Do not show all comparisons simultaneously — reveal them one at a time."],
    },

    # ── Trees & Graphs ──────────────────────────────────────────────
    "tree": {
        "keywords": ["tree", "binary tree", "bst", "binary search tree", "avl",
                      "red-black tree", "heap", "trie", "traversal", "inorder",
                      "preorder", "postorder", "bfs", "dfs", "breadth first", "depth first"],
        "visual_style": "diagram",
        "layout_tips": [
            "Use circles for nodes with values inside, and lines/arrows for edges.",
            "Layout: root at top, children below. Use Manim's Tree/Graph layout or manual positioning.",
            "Scale the ENTIRE tree group to fit within frame after construction — this is critical.",
            "For traversal: highlight the current node with a bright color, and dim visited nodes.",
            "Show the traversal order as a growing list at the bottom of the screen.",
            "For trees with >7 nodes: use smaller node radius (0.3) and font_size=20.",
            "CRITICAL: After building the full tree, call .scale_to_fit_width(config.frame_width - 3) to prevent overflow.",
        ],
        "common_elements": ["node_circles", "edge_lines", "value_labels", "traversal_list"],
        "max_simultaneous_objects": 15,
        "avoid": ["Do not try to animate a tree with >15 nodes — it will overflow the frame."],
    },

    # ── Linked Lists ────────────────────────────────────────────────
    "linked_list": {
        "keywords": ["linked list", "singly linked", "doubly linked", "linkedlist"],
        "visual_style": "array_boxes",
        "layout_tips": [
            "Use boxes with arrows pointing to the next node — the classic linked list visualization.",
            "Arrange horizontally with arrows between boxes.",
            "For >5 nodes: wrap to a second row or use smaller boxes.",
            "Highlight the current pointer with a different color during traversal.",
            "Show null/None at the end with a special symbol.",
        ],
        "common_elements": ["node_boxes", "next_arrows", "pointer_label", "null_marker"],
        "max_simultaneous_objects": 8,
        "avoid": [],
    },

    # ── Stacks & Queues ─────────────────────────────────────────────
    "stack_queue": {
        "keywords": ["stack", "queue", "push", "pop", "enqueue", "dequeue", "fifo", "lifo"],
        "visual_style": "array_boxes",
        "layout_tips": [
            "For stacks: show vertical column of boxes, new elements added on top.",
            "For queues: show horizontal row, elements enter from right, exit from left.",
            "Animate push/enqueue with FadeIn + shift, pop/dequeue with FadeOut + shift.",
            "Show a pointer/label for 'top' (stack) or 'front'/'rear' (queue).",
        ],
        "common_elements": ["element_boxes", "top_pointer", "operation_label"],
        "max_simultaneous_objects": 8,
        "avoid": [],
    },

    # ── Math Functions & Equations ──────────────────────────────────
    "math": {
        "keywords": ["equation", "function", "graph", "parabola", "quadratic",
                      "linear", "polynomial", "derivative", "integral", "calculus",
                      "sine", "cosine", "trigonometry", "algebra", "slope",
                      "tangent line", "limit", "differentiation", "integration"],
        "visual_style": "graph_plot",
        "layout_tips": [
            "Use Axes() or NumberPlane() for coordinate systems.",
            "Plot functions with smooth curves using plot() or ParametricFunction().",
            "Show the equation in MathTex at the top or corner of the screen.",
            "For derivatives: show the tangent line moving along the curve.",
            "For integrals: shade the area under the curve progressively.",
            "Use color-coding: function in one color, derivative in another.",
        ],
        "common_elements": ["axes", "function_curve", "equation_tex", "labels", "shaded_area"],
        "max_simultaneous_objects": 8,
        "avoid": ["Do not use bar charts for math functions — use proper coordinate plots."],
    },

    # ── Physics ─────────────────────────────────────────────────────
    "physics": {
        "keywords": ["physics", "force", "velocity", "acceleration", "momentum",
                      "gravity", "projectile", "newton", "motion", "wave",
                      "pendulum", "spring", "friction", "energy", "work",
                      "electric", "magnetic", "circuit", "optics", "lens",
                      "reflection", "refraction"],
        "visual_style": "physics_motion",
        "layout_tips": [
            "Use arrows for force vectors — length proportional to magnitude.",
            "Animate motion with smooth path animations (MoveAlongPath or shift).",
            "Show formulas alongside the visual demonstration.",
            "For projectile motion: trace the parabolic path as the object moves.",
            "For waves: use sine curves with progressive animation.",
            "Label all forces, velocities, and angles clearly.",
        ],
        "common_elements": ["force_arrows", "moving_object", "path_trace", "formula_tex", "labels"],
        "max_simultaneous_objects": 10,
        "avoid": ["Do not use static diagrams for physics — motion must be animated."],
    },

    # ── Biology ─────────────────────────────────────────────────────
    "biology": {
        "keywords": ["biology", "cell", "dna", "rna", "protein", "mitosis",
                      "meiosis", "photosynthesis", "respiration", "evolution",
                      "genetics", "chromosome", "enzyme", "organism", "ecosystem",
                      "nucleus", "membrane", "organ"],
        "visual_style": "diagram",
        "layout_tips": [
            "Use labeled diagrams with clear component identification.",
            "For processes (photosynthesis, respiration): show flowchart-style step progression.",
            "For cell structure: show cross-section with labeled parts.",
            "For DNA: use the classic double-helix representation or simplified ladder.",
            "Use arrows to show process flow (input → process → output).",
            "Color-code different components for easy identification.",
        ],
        "common_elements": ["labeled_shapes", "process_arrows", "component_labels", "legend"],
        "max_simultaneous_objects": 10,
        "avoid": ["Do not overcrowd — show one process phase at a time, then transition."],
    },

    # ── Chemistry ───────────────────────────────────────────────────
    "chemistry": {
        "keywords": ["chemistry", "atom", "molecule", "reaction", "bond",
                      "element", "compound", "periodic table", "electron",
                      "proton", "neutron", "orbital", "ionic", "covalent",
                      "acid", "base", "pH", "oxidation", "reduction"],
        "visual_style": "diagram",
        "layout_tips": [
            "For atomic structure: show nucleus with orbiting electrons.",
            "For reactions: show reactants on left, arrow in middle, products on right.",
            "Use color-coded circles for different atoms.",
            "For bonds: show shared electron pairs with lines between atoms.",
            "Animate reaction progression from reactants to products.",
        ],
        "common_elements": ["atom_circles", "bond_lines", "reaction_arrow", "element_labels"],
        "max_simultaneous_objects": 10,
        "avoid": [],
    },

    # ── History & Timelines ─────────────────────────────────────────
    "history": {
        "keywords": ["history", "timeline", "war", "revolution", "civilization",
                      "empire", "dynasty", "century", "era", "period",
                      "independence", "constitution"],
        "visual_style": "timeline",
        "layout_tips": [
            "Use a horizontal timeline with labeled points for key events.",
            "Reveal events progressively from left to right.",
            "Use icons or small images at each timeline point.",
            "Show dates clearly below or above each event.",
            "Group related events with colored segments.",
        ],
        "common_elements": ["timeline_line", "event_markers", "date_labels", "event_descriptions"],
        "max_simultaneous_objects": 8,
        "avoid": ["Do not show all events at once — reveal them sequentially."],
    },

    # ── Flowcharts & Processes ──────────────────────────────────────
    "process": {
        "keywords": ["process", "flowchart", "algorithm", "workflow", "decision",
                      "if else", "conditional", "loop", "iteration", "recursion",
                      "step by step", "procedure"],
        "visual_style": "flowchart",
        "layout_tips": [
            "Use standard flowchart shapes: rectangles for process, diamonds for decisions, ovals for start/end.",
            "Arrange top-to-bottom or left-to-right with clear arrows.",
            "Highlight the current step during walkthrough.",
            "Use color to distinguish different types of steps.",
            "For >5 steps: show portions at a time, scrolling or transitioning between groups.",
        ],
        "common_elements": ["process_boxes", "decision_diamonds", "flow_arrows", "labels"],
        "max_simultaneous_objects": 8,
        "avoid": ["Do not create overly complex flowcharts — simplify to 5-7 key steps."],
    },

    # ── Networking & Systems ────────────────────────────────────────
    "networking": {
        "keywords": ["network", "internet", "tcp", "ip", "http", "dns",
                      "server", "client", "protocol", "router", "packet",
                      "osi model", "api", "request", "response", "database"],
        "visual_style": "diagram",
        "layout_tips": [
            "Use icons/boxes for servers, clients, routers, databases.",
            "Show data flow with animated arrows between components.",
            "For layered models (OSI, TCP/IP): show horizontal layers stacked vertically.",
            "Animate packet/request movement from source to destination.",
            "Label each component and connection clearly.",
        ],
        "common_elements": ["component_boxes", "connection_arrows", "data_packets", "labels"],
        "max_simultaneous_objects": 8,
        "avoid": [],
    },

    # ══════════════════════════════════════════════════════════════════
    # ML/DL TOPICS — COMPREHENSIVE COVERAGE (Phase 7)
    # ══════════════════════════════════════════════════════════════════

    # ── Gradient Descent & Optimization ────────────────────────────
    "gradient_descent": {
        "keywords": ["gradient descent", "optimization", "learning rate",
                      "stochastic gradient descent", "sgd", "adam optimizer",
                      "momentum", "convergence", "cost function", "loss function",
                      "objective function", "minima", "maxima", "saddle point"],
        "visual_style": "graph_plot",
        "layout_tips": [
            "Use a smooth parabolic cost curve with a dot/ball rolling down — THE visual metaphor.",
            "Show parameter updates as dot moving along curve, not just numbers changing.",
            "Before/after comparison: bad predictions → good predictions after optimization.",
            "Show learning rate effect: too big = overshooting, too small = barely moving, just right = smooth.",
            "Use split-screen: left 60% for cost curve, right 40% for key text/formulas.",
            "Layer concepts ONE at a time: don't show gradient + learning rate + cost simultaneously.",
            "Use Axes() with continuous curve via .plot(), then animate a Dot along it.",
        ],
        "common_elements": ["cost_curve", "parameter_dot", "gradient_arrow", "learning_rate_label"],
        "max_simultaneous_objects": 8,
        "avoid": [
            "Do not show too many mathematical formulas — focus on visual intuition.",
            "Do not introduce gradient, learning rate, and cost function simultaneously.",
            "Do not use bar charts for cost functions — use smooth curves on Axes.",
        ],
    },

    # ── Neural Networks (General) ──────────────────────────────────
    "neural_network": {
        "keywords": ["neural network", "perceptron", "deep learning",
                      "feedforward", "dense layer", "fully connected",
                      "weight", "bias", "neuron", "node", "layer",
                      "hidden layer", "deep neural network", "dnn",
                      "multilayer perceptron", "mlp"],
        "visual_style": "diagram",
        "layout_tips": [
            "Show layers as columns of circles (nodes) with lines connecting them.",
            "Use 3-4 layers max: input (blue), hidden (yellow/orange), output (green).",
            "Animate data flowing left → right through the network with color changes.",
            "Show weights as line thickness or opacity variations.",
            "Use small font_size (18-20) for node labels to avoid overflow.",
            "Scale entire network to fit width: .scale_to_fit_width(config.frame_width - 3).",
            "For forward pass: animate signals traveling through connections sequentially.",
        ],
        "common_elements": ["node_circles", "connection_lines", "layer_labels", "weight_labels"],
        "max_simultaneous_objects": 10,
        "avoid": [
            "Do not draw more than 5 nodes per layer — it becomes unreadable.",
            "Do not show all connections at full opacity — use faded lines and highlight active ones.",
        ],
    },

    # ── Backpropagation ────────────────────────────────────────────
    "backpropagation": {
        "keywords": ["backpropagation", "backprop", "backward pass",
                      "chain rule", "gradient flow", "error propagation"],
        "visual_style": "diagram",
        "layout_tips": [
            "Start with a forward pass through the network (left → right).",
            "Then show error signal flowing BACKWARD (right → left) with different color (red/orange).",
            "Use arrows that reverse direction to show the backward flow.",
            "Highlight how each weight gets updated based on the error it caused.",
            "Show the chain rule visually: multiply gradients at each node.",
            "Use a simple 3-layer network to keep it clear — don't overcomplicate.",
        ],
        "common_elements": ["network_layers", "forward_arrows", "backward_arrows", "gradient_labels", "error_signal"],
        "max_simultaneous_objects": 10,
        "avoid": [
            "Do not show the full math of chain rule — show it visually as signals flowing back.",
            "Do not animate forward and backward pass simultaneously — show them sequentially.",
        ],
    },

    # ── Activation Functions ───────────────────────────────────────
    "activation_function": {
        "keywords": ["activation function", "relu", "sigmoid", "tanh",
                      "softmax", "leaky relu", "elu", "swish", "gelu"],
        "visual_style": "graph_plot",
        "layout_tips": [
            "Use Axes() to plot each activation function as a smooth curve.",
            "Compare 2-3 functions on the same axes with different colors.",
            "Show input values on x-axis and output on y-axis.",
            "Highlight the key property: where the function is zero, where it saturates.",
            "For ReLU: emphasize the 'dead zone' (negative inputs → zero output).",
            "For Sigmoid: show saturation at extremes and the S-curve shape.",
        ],
        "common_elements": ["axes", "function_curves", "labels", "region_highlights"],
        "max_simultaneous_objects": 6,
        "avoid": ["Do not plot more than 3 functions on one set of axes."],
    },

    # ── CNNs (Convolutional Neural Networks) ────────────────────────
    "cnn": {
        "keywords": ["cnn", "convolutional neural network", "convolution",
                      "filter", "kernel", "feature map", "pooling",
                      "max pooling", "stride", "padding", "image recognition",
                      "computer vision"],
        "visual_style": "diagram",
        "layout_tips": [
            "Show the kernel/filter as a small colored grid sliding over a larger input grid.",
            "Animate the sliding: move the filter one step at a time across the input.",
            "Show the output feature map being built cell by cell as the filter slides.",
            "Use color intensity in cells to represent pixel/activation values.",
            "For pooling: show a 2x2 window selecting the max value from each region.",
            "Use a pipeline diagram: Input → Conv → Pool → Conv → Pool → Flatten → Dense → Output.",
            "Build each stage one at a time — don't show the full pipeline at once.",
        ],
        "common_elements": ["input_grid", "filter_grid", "feature_map", "pooling_window", "pipeline_stages"],
        "max_simultaneous_objects": 10,
        "avoid": [
            "Do not use real images — use colored grids with numbers to represent pixels.",
            "Do not show more than 3 filters at once.",
        ],
    },

    # ── RNNs & LSTMs ──────────────────────────────────────────────
    "rnn": {
        "keywords": [
            "rnn", "recurrent neural network", "lstm",
            "long short term memory", "gru", "sequence model",
            "time series", "hidden state", "cell state",
            "vanishing gradient",
        ],
        "visual_style": "diagram",
        "layout_tips": [
            "LSTM CELL LAYOUT: Draw the cell as a large rounded rectangle (width=9, height=5). "
            "Place the Cell State (C_t) as a thick horizontal line (color=#4FACFE, stroke_width=5) "
            "running ACROSS THE TOP of the cell — this is the memory highway.",
            "GATE POSITIONS (inside cell, bottom half): "
            "Forget Gate (RED_E) at LEFT inside cell, "
            "Input Gate (GREEN_E) at CENTER-LEFT, "
            "Output Gate (PURPLE_E) at CENTER-RIGHT. "
            "Each gate is a RoundedRectangle (width=1.6, height=1.0, corner_radius=0.15) with the gate name in white text.",
            "SIGMA/TANH SYMBOLS: Add small circles with 'σ' (MathTex) below each gate label. "
            "Add a separate small 'tanh' box near the input gate for the candidate cell state. "
            "These symbols are CRITICAL for correctness — do not omit them.",
            "FORMULA DISPLAY: Show the cell state update formula C_t = f_t ⊙ C_{t-1} + i_t ⊙ C̃_t "
            "using MathTex on the RIGHT KEY TEXT PANEL. Show each gate formula as the gate is introduced.",
            "DATA FLOW ARROWS: Use color-coded arrows — RED for forget path, GREEN for input/add path, "
            "BLUE (#4FACFE) for cell state highway, PURPLE for output path. "
            "Inputs x_t and h_{t-1} enter from the LEFT.",
            "ANIMATION SEQUENCE: Introduce elements ONE AT A TIME — first show empty cell + cell state line, "
            "then add forget gate + red arrows, then input gate + green arrows, then output gate + purple arrows. "
            "DO NOT show all gates simultaneously at first.",
            "DATA PACKETS: Use Circle(radius=0.15) packets of matching gate color to animate data flow. "
            "IMPORTANT: Create ONE packet variable and animate it along Arrow paths using MoveAlongPath. "
            "DO NOT use .copy() in a loop — it creates ghost objects that pile up. "
            "Clean up each packet with FadeOut before creating the next one.",
            "SPLIT SCREEN: Main LSTM cell diagram LEFT 60%, key text + formulas RIGHT 40%.",
            "SEGMENT 2 ANALOGY: Show a horizontal conveyor belt (series of dots moving right) "
            "to represent the cell state. Show a red X stamp for forget, green plus for input, "
            "blue arrow for output. This is more convincing than fading 'Msg' text.",
        ],
        "common_elements": [
            "lstm_cell_rect", "cell_state_line", "forget_gate_red",
            "input_gate_green", "output_gate_purple", "sigma_circles",
            "tanh_box", "gate_formulas_mathtex", "colored_arrows",
            "data_packets", "h_t_output", "x_t_input",
        ],
        "key_formulas": [
            r"f_t = \sigma(W_f \cdot [h_{t-1}, x_t] + b_f)",
            r"i_t = \sigma(W_i \cdot [h_{t-1}, x_t] + b_i)",
            r"C_t = f_t \odot C_{t-1} + i_t \odot \tilde{C}_t",
            r"h_t = o_t \odot \tanh(C_t)",
        ],
        "max_simultaneous_objects": 8,
        "avoid": [
            "NEVER use Rectangle(corner_radius=X) — Rectangle does not accept corner_radius. "
            "Use RoundedRectangle(corner_radius=X, width=W, height=H) for all gate and cell boxes.",
            "NEVER chain more than 4 animations of run_time=tracker.duration*0.2 inside one voiceover block. "
            "7 × 0.2 = 1.4 > 1.0 and will crash with 'wait() duration <= 0'. "
            "Keep the sum of all tracker.duration fractions in one block BELOW 0.85.",
            "NEVER use .copy() to create data packets in a loop — ghost copies pile up and never get cleaned. "
            "Create one packet, animate it, FadeOut it, then create the next.",
            "Do not show all 3 gates simultaneously from the start — introduce Forget, then Input, then Output.",
            "Do not show more than 5 time steps in an unrolled RNN — it overflows the frame.",
            "Do not omit sigma (σ) and tanh symbols — they are the core of what makes LSTM different from RNN.",
            "Do not label gates 'Gate 1', 'Gate 2' — always use the real names: Forget Gate, Input Gate, Output Gate.",
        ],
    },


    # ── Transformers & Attention ────────────────────────────────────
    "transformer": {
        "keywords": ["transformer", "attention", "self attention",
                      "multi head attention", "query key value",
                      "positional encoding", "encoder decoder",
                      "bert", "gpt", "large language model", "llm"],
        "visual_style": "diagram",
        "layout_tips": [
            "For self-attention: show tokens as boxes, with colored lines between them showing attention weights.",
            "Use line thickness or opacity to represent attention strength.",
            "Show Q, K, V (Query, Key, Value) as three parallel transformed versions of input.",
            "For multi-head attention: show 2-3 attention heads side by side with different colors.",
            "Animate the attention score computation: Q × K → scores → softmax → multiply with V.",
            "Use a simple sentence with 4-5 tokens to demonstrate — not a long sequence.",
            "For the encoder-decoder architecture: show two stacked columns with arrows between them.",
        ],
        "common_elements": ["token_boxes", "attention_lines", "qkv_matrices", "softmax_output", "head_colors"],
        "max_simultaneous_objects": 10,
        "avoid": [
            "Do not show the full transformer architecture at once — build it layer by layer.",
            "Do not use more than 5 tokens — it becomes visually overwhelming.",
            "Do not show matrix multiplication formulas — show them visually as colored connections.",
        ],
    },

    # ── Loss Functions ─────────────────────────────────────────────
    "loss_function": {
        "keywords": ["loss function", "mse", "mean squared error",
                      "cross entropy", "binary cross entropy",
                      "categorical cross entropy", "hinge loss",
                      "log loss", "mae", "mean absolute error"],
        "visual_style": "graph_plot",
        "layout_tips": [
            "Plot the loss function as a curve on Axes — x-axis is prediction, y-axis is loss.",
            "For MSE: show the parabolic curve and how error grows quadratically.",
            "For cross-entropy: show the logarithmic curve and how it penalizes wrong confident predictions.",
            "Show data points with predicted vs actual values, and highlight the 'error' gap.",
            "Animate loss decreasing over training epochs as a line plot.",
            "Use before/after: high loss (bad predictions) → low loss (good predictions).",
        ],
        "common_elements": ["loss_curve", "data_points", "error_lines", "epoch_plot"],
        "max_simultaneous_objects": 8,
        "avoid": ["Do not show multiple loss functions simultaneously — compare them sequentially."],
    },

    # ── Regularization ─────────────────────────────────────────────
    "regularization": {
        "keywords": ["regularization", "l1", "l2", "lasso", "ridge",
                      "dropout", "batch normalization", "data augmentation",
                      "early stopping", "overfitting", "underfitting"],
        "visual_style": "graph_plot",
        "layout_tips": [
            "For overfitting/underfitting: show three curves on same axes — underfit, good fit, overfit.",
            "Use training vs validation loss curves to show overfitting divergence.",
            "For L1/L2: show constraint regions (diamond for L1, circle for L2) on a 2D parameter space.",
            "For dropout: show a network with nodes randomly grayed out / crossed out.",
            "For batch norm: show distribution histograms before → after normalization.",
            "Animate the transition from overfitting to good generalization.",
        ],
        "common_elements": ["fit_curves", "train_val_loss", "constraint_region", "dropout_nodes"],
        "max_simultaneous_objects": 8,
        "avoid": ["Do not show L1 and L2 at the same time — compare sequentially."],
    },

    # ── Regression & Classification ────────────────────────────────
    "regression_classification": {
        "keywords": ["regression", "classification", "logistic regression",
                      "linear regression", "decision boundary", "feature",
                      "prediction", "label", "supervised learning",
                      "k-nearest neighbors", "knn", "svm", "support vector machine"],
        "visual_style": "graph_plot",
        "layout_tips": [
            "For linear regression: show scattered data points with a best-fit line on Axes.",
            "Animate the line rotating/shifting to find the best fit.",
            "For classification: show two clusters of colored dots with a decision boundary.",
            "For logistic regression: show the sigmoid curve mapping inputs to probabilities.",
            "Use 2D scatter plots with clear class colors (blue dots vs red dots).",
            "Show model improvement: bad boundary → good boundary with animation.",
        ],
        "common_elements": ["scatter_dots", "regression_line", "decision_boundary", "axes"],
        "max_simultaneous_objects": 10,
        "avoid": [
            "Do not show more than 15-20 data points — it becomes cluttered.",
            "Do not use 3D plots — keep it 2D for clarity.",
        ],
    },

    # ── Statistics & Probability ───────────────────────────────────
    "statistics": {
        "keywords": ["statistics", "probability", "mean", "median", "mode",
                      "standard deviation", "variance", "distribution", "normal distribution",
                      "histogram", "correlation", "regression analysis", "hypothesis",
                      "p-value", "confidence interval", "bayes", "bayesian"],
        "visual_style": "bar_chart",
        "layout_tips": [
            "For distributions: use smooth bell curves on Axes with shaded regions.",
            "For mean/median: show them as labeled vertical lines on the distribution.",
            "For probability: use area-based visuals (shaded regions under curves).",
            "For hypothesis testing: show two overlapping distributions with decision boundary.",
            "Use progressive reveal: show data first, then overlay the statistic.",
        ],
        "common_elements": ["distribution_curve", "labeled_lines", "shaded_regions", "data_points", "formula_tex"],
        "max_simultaneous_objects": 8,
        "avoid": ["Do not show raw data tables — use visual representations."],
    },

    # ── Embeddings & Representation Learning ───────────────────────
    "embeddings": {
        "keywords": ["embedding", "word2vec", "word embedding",
                      "vector representation", "latent space",
                      "dimensionality reduction", "pca", "t-sne",
                      "feature extraction"],
        "visual_style": "graph_plot",
        "layout_tips": [
            "Show words as points in a 2D space, positioned by semantic similarity.",
            "Animate king - man + woman = queen as vector arithmetic with arrows.",
            "Use color clusters for semantically related words.",
            "Show the transformation: high-dimensional → low-dimensional with PCA/t-SNE.",
            "Use arrows between related words to show relationships.",
        ],
        "common_elements": ["word_dots", "vector_arrows", "cluster_colors", "axes_2d"],
        "max_simultaneous_objects": 10,
        "avoid": ["Do not plot more than 10-12 words — keep it focused on the key relationships."],
    },

    # ── GANs ───────────────────────────────────────────────────────
    "gan": {
        "keywords": ["gan", "generative adversarial network", "generator",
                      "discriminator", "adversarial training", "fake",
                      "real data", "generative model"],
        "visual_style": "diagram",
        "layout_tips": [
            "Show Generator and Discriminator as two opposing boxes/blocks.",
            "Animate: Generator produces 'fake' data → Discriminator judges real vs fake.",
            "Use a feedback loop: Discriminator's judgment flows back to Generator.",
            "Show improvement over iterations: Generator output gets better (closer to real).",
            "Color code: real data = green, fake data = red/orange, improving fake = yellow → green.",
        ],
        "common_elements": ["generator_box", "discriminator_box", "real_data", "fake_data", "feedback_arrow"],
        "max_simultaneous_objects": 8,
        "avoid": ["Do not show internal architecture of G and D — treat them as black boxes first."],
    },

    # ── Reinforcement Learning ─────────────────────────────────────
    "reinforcement_learning": {
        "keywords": ["reinforcement learning", "reward", "penalty",
                      "agent", "environment", "policy", "q-learning",
                      "exploration", "exploitation", "markov decision process"],
        "visual_style": "diagram",
        "layout_tips": [
            "Show the Agent-Environment loop: Agent → Action → Environment → Reward → Agent.",
            "Use a grid world or simple game board for concrete demonstration.",
            "Animate the agent moving through the grid, receiving rewards/penalties.",
            "Show the policy improving: random movement → directed movement over episodes.",
            "Color code: rewards = green, penalties = red, neutral = gray.",
        ],
        "common_elements": ["agent_icon", "grid_world", "reward_labels", "action_arrows", "policy_table"],
        "max_simultaneous_objects": 10,
        "avoid": ["Do not show Q-tables with more than 4x4 cells — keep it simple."],
    },
}


# ═══════════════════════════════════════════════════════════════════════
# PUBLIC API
# ═══════════════════════════════════════════════════════════════════════

def get_topic_hints(query: str) -> Optional[dict]:
    """Get animation hints for a topic.

    Tries dynamic Gemini-powered analysis first, then falls back to
    static keyword matching.

    Returns None if no matching topic is found.
    """
    # 1. Try dynamic hints from Gemini first
    dynamic = generate_dynamic_hints(query)
    if dynamic:
        return dynamic

    # 2. Fall back to static keyword matching
    return _static_topic_match(query)


def _static_topic_match(query: str) -> Optional[dict]:
    """Match against the static TOPIC_VISUAL_HINTS dictionary."""
    query_lower = query.lower()

    best_match = None
    best_score = 0

    for category, hints in TOPIC_VISUAL_HINTS.items():
        score = 0
        for keyword in hints["keywords"]:
            if keyword in query_lower:
                score += len(keyword)

        if score > best_score:
            best_score = score
            best_match = hints

    if best_match and best_score > 0:
        result = {k: v for k, v in best_match.items() if k != "keywords"}
        return result

    return None


def format_hints_for_prompt(hints: Optional[dict]) -> str:
    """Format topic hints into a readable string for injection into prompts.

    Handles both static (dict with layout_tips) and dynamic (dict with
    animation_suggestions and recommended_segments) hint formats.
    """
    if not hints:
        return ""

    parts = []
    parts.append(f"Recommended visual style: {hints.get('visual_style', 'diagram')}")
    parts.append(f"Max simultaneous on-screen objects: {hints.get('max_simultaneous_objects', 8)}")

    # Visual metaphor (from dynamic hints)
    metaphor = hints.get("visual_metaphor", "")
    if metaphor:
        parts.append(f"Visual metaphor: {metaphor}")

    # Layout tips
    layout_tips = hints.get("layout_tips", [])
    if layout_tips:
        tips_str = "\n".join(f"  - {tip}" for tip in layout_tips)
        parts.append(f"Layout tips:\n{tips_str}")

    # Animation suggestions (from dynamic hints)
    suggestions = hints.get("animation_suggestions", [])
    if suggestions:
        sug_str = "\n".join(f"  - {s}" for s in suggestions)
        parts.append(f"Animation suggestions:\n{sug_str}")

    # Common elements
    common_elements = hints.get("common_elements", [])
    if common_elements:
        parts.append(f"Common elements: {', '.join(common_elements)}")

    # Avoid
    avoid = hints.get("avoid", [])
    if avoid:
        avoid_str = "\n".join(f"  - {a}" for a in avoid)
        parts.append(f"Avoid:\n{avoid_str}")

    # Recommended segments (from dynamic hints)
    segments = hints.get("recommended_segments", {})
    if segments:
        parts.append("Recommended video segments:")
        parts.append(f"  - Intro hook: {segments.get('intro_hook', '')}")
        parts.append(f"  - Theory focus: {segments.get('theory_focus', '')}")
        parts.append(f"  - Core animation: {segments.get('core_animation', '')}")
        parts.append(f"  - Takeaway: {segments.get('takeaway', '')}")

    # Key formulas (from dynamic hints)
    formulas = hints.get("key_formulas", [])
    if formulas:
        parts.append(f"Key formulas: {', '.join(formulas)}")

    return "\n".join(parts)
