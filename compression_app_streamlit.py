import streamlit as st
import math
import heapq

# ===================== HUFFMAN IMPLEMENTATION =====================

class Node:
    def __init__(self, char=None, freq=0, left=None, right=None):
        self.char = char
        self.freq = freq
        self.left = left
        self.right = right

    def __lt__(self, other):
        return self.freq < other.freq


def build_frequency(text):
    freq = {}
    for ch in text:
        freq[ch] = freq.get(ch, 0) + 1
    return freq


def huffman_height(node):
    if node is None:
        return 0
    return 1 + max(huffman_height(node.left), huffman_height(node.right))


def orient_huffman_tree(node):
    """
    Recursively orient the tree so that the deeper subtree
    is on the right (to push longer codes towards the right).
    """
    if node is None:
        return 0
    hl = orient_huffman_tree(node.left)
    hr = orient_huffman_tree(node.right)
    if hl > hr:
        node.left, node.right = node.right, node.left
        hl, hr = hr, hl
    return 1 + max(hl, hr)


def build_huffman_tree(freq_dict):
    heap = []
    for ch, fr in freq_dict.items():
        heapq.heappush(heap, Node(ch, fr))

    if len(heap) == 1:
        only = heapq.heappop(heap)
        root = Node(freq=only.freq, left=only, right=None)
        orient_huffman_tree(root)
        return root

    while len(heap) > 1:
        left = heapq.heappop(heap)
        right = heapq.heappop(heap)
        merged = Node(freq=left.freq + right.freq, left=left, right=right)
        heapq.heappush(heap, merged)

    root = heapq.heappop(heap)
    orient_huffman_tree(root)
    return root


def build_codes(root):
    codes = {}

    def traverse(node, current_code):
        if node is None:
            return
        if node.char is not None:
            codes[node.char] = current_code or "0"
            return
        traverse(node.left, current_code + "0")
        traverse(node.right, current_code + "1")

    traverse(root, "")
    return codes


def huffman_encode(text):
    if not text:
        return "", None, None, None

    freq = build_frequency(text)
    root = build_huffman_tree(freq)
    codes = build_codes(root)
    encoded_bits = "".join(codes[ch] for ch in text)
    return encoded_bits, root, freq, codes


def huffman_decode(encoded_bits, root):
    if not encoded_bits or root is None:
        return ""

    decoded_chars = []
    node = root
    for bit in encoded_bits:
        node = node.left if bit == "0" else node.right
        if node.char is not None:
            decoded_chars.append(node.char)
            node = root
    return "".join(decoded_chars)


def display_symbol(ch: str) -> str:
    """Pretty label for a symbol in the tree/table."""
    if ch == " ":
        return "sp"
    if ch == "\n":
        return "\\n"
    if ch == "\t":
        return "\\t"
    return ch


def huffman_tree_to_dot(root):
    """
    Build a Graphviz DOT string for the Huffman tree.
    Internal nodes: label = frequency
    Leaves: label = "<symbol>\\n<freq>"
    Left edges labeled 0, right edges labeled 1.
    """
    if root is None:
        return "digraph HuffmanTree {}"

    lines = [
        "digraph HuffmanTree {",
        "rankdir=TB;",                     # top to bottom
        "nodesep=0.4;",                    # horizontal spacing
        "ranksep=0.5;",                    # vertical spacing
        'node [shape=circle, style=filled, fillcolor="#00c0a0", '
        'fontname="Arial", fontsize=12];',
        'edge [fontname="Arial", fontsize=11];',
    ]
    counter = {"id": 0}

    def visit(node):
        my_id = f"n{counter['id']}"
        counter["id"] += 1

        if node.char is None:
            label = f"{node.freq}"
        else:
            sym = display_symbol(node.char)
            label = f"{sym}\\n{node.freq}"

        lines.append(f'{my_id} [label="{label}"];')

        if node.left:
            left_id = visit(node.left)
            lines.append(f'{my_id} -> {left_id} [label="0"];')

        if node.right:
            right_id = visit(node.right)
            lines.append(f'{my_id} -> {right_id} [label="1"];')

        return my_id

    visit(root)
    lines.append("}")
    return "\n".join(lines)


# ===================== LZW IMPLEMENTATION (CONFIGURABLE) =====================

def lzw_encode_alphabet(text, start_index=1):
    """
    LZW where the initial dictionary is built from
    sorted unique characters of the input text.

    start_index: 0 or 1 (e.g. a->0,b->1 or a->1,b->2)
    """
    if not text:
        return [], [], []

    alphabet = sorted(set(text))
    dictionary = {ch: start_index + i for i, ch in enumerate(alphabet)}
    next_code = start_index + len(alphabet)

    w = ""
    codes = []

    for c in text:
        wc = w + c
        if wc in dictionary:
            w = wc
        else:
            codes.append(dictionary[w])
            dictionary[wc] = next_code
            next_code += 1
            w = c

    if w:
        codes.append(dictionary[w])

    dict_code_to_str = sorted(
        ((code, s) for s, code in dictionary.items()),
        key=lambda x: x[0]
    )

    return codes, alphabet, dict_code_to_str


def lzw_decode_alphabet(codes, alphabet, start_index=1):
    if not codes:
        return ""

    dictionary = {start_index + i: ch for i, ch in enumerate(alphabet)}
    next_code = start_index + len(alphabet)

    w = dictionary[codes[0]]
    result = [w]

    for k in codes[1:]:
        if k in dictionary:
            entry = dictionary[k]
        elif k == next_code:
            entry = w + w[0]
        else:
            raise ValueError(f"Bad LZW code: {k}")

        result.append(entry)
        dictionary[next_code] = w + entry[0]
        next_code += 1
        w = entry

    return "".join(result)


def lzw_encode_ascii(text):
    """
    Classic LZW with an ASCII dictionary (0–255).
    """
    if not text:
        return [], []

    dictionary = {chr(i): i for i in range(256)}
    next_code = 256

    w = ""
    codes = []

    for c in text:
        wc = w + c
        if wc in dictionary:
            w = wc
        else:
            codes.append(dictionary[w])
            dictionary[wc] = next_code
            next_code += 1
            w = c

    if w:
        codes.append(dictionary[w])

    dict_code_to_str = sorted(
        ((code, s) for s, code in dictionary.items()),
        key=lambda x: x[0]
    )

    return codes, dict_code_to_str


def lzw_decode_ascii(codes):
    if not codes:
        return ""

    dictionary = {i: chr(i) for i in range(256)}
    next_code = 256

    w = chr(codes[0])
    result = [w]

    for k in codes[1:]:
        if k in dictionary:
            entry = dictionary[k]
        elif k == next_code:
            entry = w + w[0]
        else:
            raise ValueError(f"Bad LZW code: {k}")

        result.append(entry)
        dictionary[next_code] = w + entry[0]
        next_code += 1
        w = entry

    return "".join(result)


# ===================== STREAMLIT APP =====================

DEFAULT_KEYS = [
    "original_text",
    "method",
    "encoded_str",
    "decoded_str",
    "original_size",
    "compressed_size",
    "ratio",
    "huf_bits",
    "huf_tree",
    "huf_freq",
    "huf_codes",
    "lzw_codes",
    "lzw_variant",      # "alphabet" or "ascii"
    "lzw_alphabet",     # only for "alphabet"
    "lzw_start_index",  # 0 or 1 for "alphabet"
    "lzw_dict_entries", # list of (code, entry)
    "last_method",
]

for key in DEFAULT_KEYS:
    if key not in st.session_state:
        st.session_state[key] = None

st.set_page_config(
    page_title="Text Compression Studio",
    page_icon="📦",
    layout="wide"
)

st.title("📦 Text Compression Studio – Huffman & LZW")

st.markdown(
    """
Experiment with **Huffman** and **LZW** text compression.

1. Type some text  
2. Choose a compression method  
3. Click **Encode**  
4. Click **Decode** to verify lossless decompression  
"""
)

# ========== Layout ==========
col_left, col_right = st.columns([2, 1])

with col_left:
    st.subheader("📝 Original Text")
    original_text_input = st.text_area(
        "Input text",
        value=st.session_state.original_text or "",
        height=180,
        label_visibility="collapsed",
    )

    method = st.radio(
        "Compression Method",
        options=["Huffman", "LZW"],
        index=0 if (st.session_state.method in [None, "Huffman"]) else 1,
        horizontal=True,
    )

    # --- LZW configuration ---
    lzw_dict_mode = None
    lzw_start_index = None
    if method == "LZW":
        lzw_dict_mode = st.selectbox(
            "LZW dictionary initialization",
            [
                "Alphabet-based (unique chars from input)",
                "ASCII (0–255)",
            ],
            index=0,
        )

        if lzw_dict_mode.startswith("Alphabet"):
            lzw_start_index = st.radio(
                "Alphabet start index",
                options=[0, 1],
                index=1,  # default = 1
                horizontal=True,
            )

    col_b1, col_b2, col_b3 = st.columns(3)
    encode_clicked = col_b1.button("🔐 Encode")
    decode_clicked = col_b2.button("🔓 Decode")
    clear_clicked = col_b3.button("🧹 Clear")

    # ---------- Clear ----------
    if clear_clicked:
        # نمسح القيم من session_state، وStreamlit سيعيد تشغيل السكربت تلقائياً
        for k in DEFAULT_KEYS:
            st.session_state[k] = None
        st.session_state["original_text"] = ""
        st.session_state["encoded_str"] = ""
        st.session_state["decoded_str"] = ""

    # ---------- Encode ----------
    if encode_clicked:
        text = original_text_input.strip()
        if not text:
            st.warning("Input text is empty!")
        else:
            st.session_state.original_text = text
            st.session_state.method = method
            st.session_state.last_method = method

            original_bytes = len(text.encode("utf-8"))
            st.session_state.original_size = original_bytes

            if method == "Huffman":
                encoded_bits, tree, freq, codes = huffman_encode(text)

                st.session_state.huf_bits = encoded_bits
                st.session_state.huf_tree = tree
                st.session_state.huf_freq = freq
                st.session_state.huf_codes = codes

                st.session_state.lzw_codes = None
                st.session_state.lzw_variant = None
                st.session_state.lzw_alphabet = None
                st.session_state.lzw_start_index = None
                st.session_state.lzw_dict_entries = None

                st.session_state.encoded_str = encoded_bits

                compressed_bits = len(encoded_bits)
                compressed_bytes = math.ceil(compressed_bits / 8) if compressed_bits > 0 else 0

            else:  # LZW
                if lzw_dict_mode.startswith("Alphabet"):
                    codes, alphabet, dict_entries = lzw_encode_alphabet(
                        text, start_index=lzw_start_index
                    )
                    st.session_state.lzw_variant = "alphabet"
                    st.session_state.lzw_alphabet = alphabet
                    st.session_state.lzw_start_index = lzw_start_index
                else:
                    codes, dict_entries = lzw_encode_ascii(text)
                    st.session_state.lzw_variant = "ascii"
                    st.session_state.lzw_alphabet = None
                    st.session_state.lzw_start_index = None

                st.session_state.lzw_codes = codes
                st.session_state.lzw_dict_entries = dict_entries

                st.session_state.huf_bits = None
                st.session_state.huf_tree = None
                st.session_state.huf_freq = None
                st.session_state.huf_codes = None

                codes_str = " ".join(str(c) for c in codes)
                st.session_state.encoded_str = codes_str

                # assume 16 bits (2 bytes) per LZW code for size estimate
                compressed_bytes = len(codes) * 2

            st.session_state.compressed_size = compressed_bytes
            if original_bytes == 0:
                ratio = 0.0
            else:
                ratio = (1 - compressed_bytes / original_bytes) * 100
            st.session_state.ratio = ratio

    # ---------- Decode ----------
    if decode_clicked:
        if st.session_state.last_method is None:
            st.info("Please encode text first before decoding.")
        else:
            if st.session_state.last_method == "Huffman":
                if not st.session_state.huf_bits or st.session_state.huf_tree is None:
                    st.warning("No Huffman data to decode.")
                else:
                    decoded = huffman_decode(
                        st.session_state.huf_bits,
                        st.session_state.huf_tree
                    )
                    st.session_state.decoded_str = decoded
            else:  # LZW
                if not st.session_state.lzw_codes:
                    st.warning("No LZW data to decode.")
                else:
                    if st.session_state.lzw_variant == "alphabet":
                        decoded = lzw_decode_alphabet(
                            st.session_state.lzw_codes,
                            st.session_state.lzw_alphabet,
                            start_index=st.session_state.lzw_start_index,
                        )
                    else:
                        decoded = lzw_decode_ascii(st.session_state.lzw_codes)

                    st.session_state.decoded_str = decoded

            if st.session_state.decoded_str is not None:
                if (
                    st.session_state.original_text is not None
                    and st.session_state.decoded_str == st.session_state.original_text
                ):
                    st.success("Decoded text matches the original (lossless).")
                else:
                    st.warning("Decoded text does NOT match the original!")

with col_right:
    st.subheader("📊 Compression Statistics")

    original_size = st.session_state.original_size
    compressed_size = st.session_state.compressed_size
    ratio = st.session_state.ratio

    st.metric("Original Size", f"{original_size} bytes" if original_size is not None else "-")
    st.metric("Compressed Size", f"{compressed_size} bytes" if compressed_size is not None else "-")
    st.metric("Compression Ratio", f"{ratio:.2f} %" if ratio is not None else "-")

st.markdown("---")

col_e, col_d = st.columns(2)

with col_e:
    st.subheader("🔐 Encoded / Compressed Data")
    st.text_area(
        "Encoded output",
        value=st.session_state.encoded_str or "",
        height=200,
        label_visibility="collapsed",
    )
    st.caption(
        "Huffman → bitstring (0/1)\n"
        "LZW → list of integer codes, e.g. `3 1 2 2 1 4 6 1`"
    )

with col_d:
    st.subheader("🔓 Decoded Text")
    st.text_area(
        "Decoded output",
        value=st.session_state.decoded_str or "",
        height=200,
        label_visibility="collapsed",
    )

st.markdown("---")
st.subheader("🔎 Algorithm Details")

# ===== Huffman tree + table =====
if st.session_state.last_method == "Huffman" and st.session_state.huf_tree:
    st.markdown("#### Huffman tree (final)")

    dot = huffman_tree_to_dot(st.session_state.huf_tree)
    st.graphviz_chart(dot, use_container_width=True)

    freq = st.session_state.huf_freq or {}
    codes = st.session_state.huf_codes or {}

    rows = []
    for ch, fr in sorted(freq.items(), key=lambda x: (-x[1], x[0])):
        code = codes.get(ch, "")
        display_char = display_symbol(ch)
        rows.append(
            {
                "Symbol": display_char,
                "Frequency": fr,
                "Code": code,
                "Length (bits)": len(code),
            }
        )

    st.markdown("##### Huffman code table")
    st.dataframe(rows, use_container_width=True)

# ===== LZW dictionary table =====
if st.session_state.last_method == "LZW" and st.session_state.lzw_dict_entries:
    entries = st.session_state.lzw_dict_entries
    variant = st.session_state.lzw_variant

    rows = []
    for code, entry in entries:
        # for ASCII, hide initial 0–255 if you only care about new entries
        if variant == "ascii" and code < 256:
            continue
        rows.append({"Code": code, "Entry": entry})

    st.markdown("#### LZW dictionary (final)")
    if rows:
        st.dataframe(rows, use_container_width=True)
    else:
        st.write("Dictionary contains only the initial ASCII entries (0–255).")

st.markdown(
    """
---
**Notes**

- *Huffman*: internal nodes show total frequency; leaves show `symbol` and its frequency.  
  Left edges are labeled **0**, right edges **1** – a path from root to leaf gives the code.  
- **Important:** for the **same text**, different people (or programs) may build slightly different Huffman trees  
  and code tables, depending on how ties between equal frequencies are resolved.  
  As long as you always merge the two *lowest* frequencies at each step, your tree and codes are **still correct**.  
- *LZW*: dictionary table shows the final set of phrases after compression.
"""
)
