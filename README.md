Text Compression Studio – Huffman & LZW

A simple interactive web application that demonstrates text compression using Huffman Coding and LZW Compression.
The app allows users to:

Enter or paste text

Select a compression method (Huffman / LZW)

Encode the text

Decode back to verify lossless compression

View compression statistics

Visualize Huffman trees

View LZW dictionary tables

This application is built with Python and Streamlit.

🚀 Features
✔ Huffman Coding

Generates a full Huffman tree

Displays bitstring encoding

Shows symbol table (frequency, code, bit-length)

Ensures perfect lossless decoding

✔ LZW Compression

Supports alphabet-based dictionary

Supports ASCII dictionary

Customizable start index (0 or 1)

Displays final generated dictionary entries

✔ Compression Statistics

Original size (bytes)

Compressed size

Compression ratio (%)

✔ Error Handling

Handles empty input

Prevents decoding before encoding

Works with long texts without crashing

🛠 Tech Stack

Python

Streamlit

Graphviz (for tree rendering inside Streamlit)

📦 Run Locally
pip install -r requirements.txt
streamlit run compression_app_streamlit.py

🌐 Deploy

This project is ready for deployment on Streamlit Cloud.
Just upload the repository and select:

Main file: compression_app_streamlit.py

Requirements: requirements.txt

📜 License

Free to use for educational and research purposes.

👤 Developer

Created by Abdullah Alshabi.