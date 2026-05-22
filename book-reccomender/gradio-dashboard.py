import pandas as pd
import numpy as np
import re
from dotenv import load_dotenv

from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import CharacterTextSplitter
from langchain_chroma import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings

import gradio as gr

load_dotenv()

print("Loading dataset...")

# Load CSV
books = pd.read_csv("books_with_emotions.csv")


# Fix ISBN formatting
books["isbn13"] = books["isbn13"].astype(str).str.strip()

# Clean categories properly
if "categories" in books.columns:

    books["simple_category"] = (
        books["categories"]
        .fillna("Unknown")
        .str.split(";")
        .str[0]
        .str.strip()
        .str.title()
    )

else:

    books["simple_category"] = "Unknown"

# Fix thumbnails
books["large_thumbnail"] = books["thumbnail"] + "&fife=w800"

books["large_thumbnail"] = np.where(
    books["large_thumbnail"].isna(),
    "booknotfound.jpg",
    books["large_thumbnail"],
)

print("Loading tagged descriptions...")

# Load text file
raw_documents = TextLoader(
    "tagged_description.txt",
    encoding="utf-8"
).load()

print("Splitting documents...")

text_splitter = CharacterTextSplitter(
    chunk_size=10000,
    chunk_overlap=0,
    separator="\n"
)

documents = text_splitter.split_documents(raw_documents)

print("Loading embedding model...")

embedding = HuggingFaceEmbeddings(
    model_name="all-MiniLM-L6-v2"
)

print("Creating vector database...")

db_books = Chroma.from_documents(
    documents,
    embedding=embedding
)

print("Vector database ready!")

# --------------------------------------------------------

def retrieve_semantic_recommendations(
        query: str,
        category: str = None,
        tone: str = None,
        intial_top_k: int = 50,
        final_top_k: int = 16,
) -> pd.DataFrame:

    recs = db_books.similarity_search_with_score(
        query,
        k=intial_top_k
    )

    print("Total matches:", len(recs))

    books_list = []

    # Extract ISBN using regex
    for rec in recs:

        text = rec[0].page_content

        match = re.search(r"\b\d{13}\b", text)

        if match:
            books_list.append(match.group())

    print("Extracted ISBN sample:", books_list[:5])

    books_recs = books[
        books["isbn13"].isin(books_list)
    ]

    print("Books matched:", len(books_recs))

    # Category filter
    if category != "All":
        books_recs = books_recs[
            books_recs["simple_category"]
            .str.contains(category, case=False, na=False)
        ]

    # Tone sorting
    if tone == "Happy":
        books_recs = books_recs.sort_values(
            by="joy",
            ascending=False
        )

    elif tone == "Surprising":
        books_recs = books_recs.sort_values(
            by="surprise",
            ascending=False
        )

    elif tone == "Angry":
        books_recs = books_recs.sort_values(
            by="anger",
            ascending=False
        )

    elif tone == "Suspensefull":
        books_recs = books_recs.sort_values(
            by="fear",
            ascending=False
        )

    elif tone == "Sad":
        books_recs = books_recs.sort_values(
            by="sadness",
            ascending=False
        )

    return books_recs.head(final_top_k)

# --------------------------------------------------------

def recommend_book(query, category, tone):

    try:

        recommendations = retrieve_semantic_recommendations(
            query,
            category,
            tone
        )

        results = []

        for _, row in recommendations.iterrows():

            description = str(row["description"])

            truncated_description = " ".join(
                description.split()[:30]
            ) + "..."

            authors_split = str(row["authors"]).split(";")

            if len(authors_split) == 2:
                authors_str = (
                    f"{authors_split[0]} and {authors_split[1]}"
                )

            elif len(authors_split) > 2:
                authors_str = (
                    f"{','.join(authors_split[:-1])} "
                    f"and {authors_split[-1]}"
                )

            else:
                authors_str = row["authors"]

            caption = (
                f"{row['title']} by {authors_str}: "
                f"{truncated_description}"
            )

            results.append(
                (row["large_thumbnail"], caption)
            )

        return results

    except Exception as e:

        print("ERROR:", e)

        return []

# --------------------------------------------------------

print("Preparing UI...")

# Clean category dropdown (Top 15 useful categories)

categories = ["All"] + sorted(
    books["simple_category"]
    .value_counts()
    .head(15)
    .index
    .tolist()
)

tones = [
    "All",
    "Happy",
    "Surprising",
    "Angry",
    "Suspensefull",
    "Sad"
]

with gr.Blocks(theme=gr.themes.Glass()) as dashboard:

    gr.Markdown("# 📚 Semantic Book Recommender")

    with gr.Row():

        user_query = gr.Textbox(
            label="Enter a book description:",
            placeholder="e.g., A magical adventure about friendship"
        )

        category_dropdown = gr.Dropdown(
            choices=categories,
            label="Select a category:",
            value="All"
        )

        tone_dropdown = gr.Dropdown(
            choices=tones,
            label="Select emotional tone:",
            value="All"
        )

        submit_button = gr.Button(
            "Find Recommendations"
        )

    gr.Markdown("## 📖 Recommendations")

    output = gr.Gallery(
        label="Recommended Books",
        columns=8,
        rows=2
    )

    submit_button.click(
        fn=recommend_book,
        inputs=[
            user_query,
            category_dropdown,
            tone_dropdown
        ],
        outputs=output
    )

print("Launching Gradio...")

if __name__ == "__main__":
    dashboard.launch(share=True)