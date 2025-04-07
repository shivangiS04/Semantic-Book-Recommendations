# Semantic Book Recommender

A semantic book recommendation system built with Gradio, LangChain, and Cohere embeddings.

## Overview

This application allows users to discover books based on natural language descriptions. The system uses semantic search to find books that match the user's query, and supports filtering by category and emotional tone.

## Features

- **Semantic Search**: Enter a natural language description of the kind of book you're looking for
- **Category Filtering**: Filter recommendations by book categories
- **Emotional Tone Filtering**: Sort recommendations by emotional tone (Happy, Surprising, Angry, Suspenseful, Sad)
- **Visual Interface**: Browse recommendations with book covers and description snippets

## Installation

1. Clone this repository
2. Install the required packages:
   ```
   pip install -r requirements.txt
   ```
3. Add your Cohere API key wherever it it mentioned:
   ```
   COHERE_API_KEY=your_api_key_here
   ```

## Usage

Run the application:

```
python gradio_dashboard.py
```

Then:
1. Access the web interface at http://127.0.0.1:7860
2. Enter a description of the book you're looking for
3. (Optional) Select a category and emotional tone
4. Click "Find recommendations"
5. Browse the results in the gallery view

## How It Works

1. Book descriptions are embedded using Cohere's embedding model
2. These embeddings are stored in a vector database (Chroma)
3. When you enter a query, it's also embedded with the same model
4. The system finds books with the most similar embeddings to your query
5. Results are filtered by category and sorted by emotional tone as requested

## Data

The system uses:
- `books_with_emotions.csv`: Book metadata including emotional scores
- `tagged_description.txt`: Book descriptions for vector search
- Get original dataset from [here](https://www.kaggle.com/datasets/dylanjcastillo/7k-books-with-metadata)

## Requirements

- Python 3.7+
- pandas
- numpy
- langchain/langchain_community
- langchain_text_splitters
- langchain_chroma
- cohere
- gradio
- python-dotenv

## License

MIT License 