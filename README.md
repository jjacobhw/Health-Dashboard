# Diet Analyzer 🍎

A powerful AI-driven nutritional analysis tool that uses RAG (Retrieval-Augmented Generation) to analyze your diet and provide personalized, evidence-based recommendations.

## Features

- **Multi-format Support**: Upload diet logs in PDF, TXT, DOCX, or CSV formats
- **AI-Powered Analysis**: Uses Ollama LLM with RAG for intelligent nutritional insights
- **Category Breakdown**: Detailed analysis across 8 food categories
- **Modern UI**: Sleek dark theme with chat-like interface
- **Analysis History**: Track your dietary improvements over time
- **Evidence-Based**: Built-in nutritional knowledge base with scientific guidelines

## Prerequisites

- Python 3.8+
- Ollama installed and running locally
- Ollama model: `llama3.2:3b`

## Installation

1. **Clone or download the repository**

2. **Install required dependencies**:
```bash
pip install streamlit chromadb langchain-text-splitters langchain-ollama langchain-chroma langchain-core PyPDF2 python-docx pandas
```

3. **Install and configure Ollama**:
   - Download Ollama from [ollama.ai](https://ollama.ai)
   - Pull the required model:
   ```bash
   ollama pull llama3.2:3b
   ```
   - Ensure Ollama is running (default: http://localhost:11434)

## Usage

1. **Start the application**:
```bash
streamlit run nutritionist.py
```

2. **Initialize the system**:
   - Enter your Ollama API URL (default: http://localhost:11434)
   - Click "Initialize" to load the RAG system

3. **Upload your diet log**:
   - Prepare a document containing your food intake
   - Supported formats: PDF, TXT, DOCX, CSV
   - Upload through the file uploader

4. **Analyze your diet**:
   - Click "Analyze My Diet" to get comprehensive insights

## Food Categories

The analyzer categorizes foods into:
- **Vegetables**: Leafy greens, cruciferous vegetables, root vegetables
- **Fruits**: Fresh fruits, berries, tropical fruits
- **Proteins**: Meats, fish, eggs, legumes, plant-based proteins
- **Grains**: Whole grains, bread, pasta, cereals
- **Dairy**: Milk products, cheese, yogurt
- **Healthy Fats**: Nuts, seeds, avocados, oils
- **Processed Foods**: Fast food, packaged snacks, sugary items
- **Other**: Items not fitting other categories

## File Format Tips

### Text Files (.txt)
```
Breakfast: Oatmeal with blueberries and almonds
Lunch: Grilled chicken salad with olive oil
Dinner: Salmon with quinoa and broccoli
Snack: Apple with peanut butter
```

### CSV Files (.csv)
```csv
Meal,Food,Quantity
Breakfast,Oatmeal,1 bowl
Breakfast,Blueberries,1/2 cup
Lunch,Chicken breast,6 oz
Lunch,Mixed greens,2 cups
```

### PDF/DOCX
Any formatted document containing your food diary or meal plans.

## Technical Architecture

### System Overview

The Diet Analyzer uses a RAG (Retrieval-Augmented Generation) architecture to provide intelligent, context-aware nutritional analysis. Here's how the components work together:

```
┌─────────────────┐
│  User Interface │ (Streamlit)
└────────┬────────┘
         │
         ▼
┌─────────────────────────────────────────┐
│      Document Processing Layer           │
│  (PyPDF2, python-docx, pandas)          │
│  • Extracts text from multiple formats  │
│  • Parses food items and meals          │
│  • Filters and categorizes content      │
└────────┬────────────────────────────────┘
         │
         ▼
┌─────────────────────────────────────────┐
│      Embedding & Vector Storage          │
│         (ChromaDB + Ollama)              │
│  • Converts text to vector embeddings   │
│  • Stores in vector database            │
│  • Enables semantic search              │
└────────┬────────────────────────────────┘
         │
         ▼
┌─────────────────────────────────────────┐
│     RAG Retrieval System                 │
│        (LangChain)                       │
│  • Retrieves relevant nutrition info    │
│  • Combines with user's diet data       │
│  • Builds context for LLM               │
└────────┬────────────────────────────────┘
         │
         ▼
┌─────────────────────────────────────────┐
│     Language Model Generation            │
│      (Ollama - llama3.2:3b)             │
│  • Analyzes diet with context           │
│  • Generates personalized advice        │
│  • Returns structured insights          │
└─────────────────────────────────────────┘
```

### Core Components

#### 1. Frontend Layer (Streamlit)
- **Purpose**: User interface and interaction management
- **Features**:
  - Custom dark theme with gradient styling
  - Real-time file upload and processing
  - Interactive visualizations (metrics, charts)
  - Session state management for persistence
- **Key Files**: Custom CSS embedded in main script

#### 2. Document Processing Pipeline
- **Text Extraction**:
  - `PyPDF2`: Extracts text from PDF documents page-by-page
  - `python-docx`: Parses DOCX files and extracts paragraphs
  - `pandas`: Processes CSV files and converts to text
  - Native Python: Handles TXT files with UTF-8 encoding
  
- **Text Processing**:
  ```python
  Input → Tokenization → Stop Word Removal → 
  Food Extraction → Categorization → Analysis
  ```

- **Categorization Algorithm**:
  - Uses keyword matching against predefined categories
  - 8 main categories + "other" for uncategorized items
  - Case-insensitive matching with partial word detection

#### 3. Vector Database (ChromaDB)
- **Purpose**: Semantic storage and retrieval of nutritional knowledge
- **Configuration**:
  ```python
  - Collection: "nutrition_knowledge"
  - Anonymized telemetry: Disabled
  - Persistence: In-memory (session-based)
  ```
- **Contents**:
  - Pre-loaded nutritional guidelines and knowledge base
  - User's diet documents (added dynamically)
  - Chunked for optimal retrieval

#### 4. Embedding System (Ollama Embeddings)
- **Model**: llama3.2:3b
- **Process**:
  1. Text is split into chunks (500 chars, 50 overlap)
  2. Each chunk is converted to a dense vector embedding
  3. Vectors are stored in ChromaDB with metadata
  4. Enables semantic similarity search
- **Chunking Strategy**:
  ```python
  RecursiveCharacterTextSplitter(
      chunk_size=500,      # Optimal for context
      chunk_overlap=50,    # Maintains continuity
      length_function=len  # Character-based
  )
  ```

#### 5. RAG Framework (LangChain)
- **Components**:
  - **Retriever**: Fetches top-k relevant documents (k=3)
  - **Prompt Template**: Structures the query context
  - **Chain Type**: "stuff" - combines all retrieved docs
  
- **RAG Process**:
  ```
  User Query → Vector Search → Retrieve Context → 
  Build Prompt → LLM Generation → Return Answer
  ```

- **Prompt Engineering**:
  ```python
  Template includes:
  1. System role (expert nutritionist)
  2. Retrieved context from knowledge base
  3. User's specific query/diet data
  4. Structured output instructions
  ```

#### 6. Language Model (Ollama LLM)
- **Model**: llama3.2:3b (3 billion parameters)
- **Configuration**:
  - Temperature: 0.7 (balanced creativity/accuracy)
  - Base URL: Configurable (default: localhost:11434)
  - Streaming: Disabled for complete responses
  
- **Analysis Process**:
  1. Receives structured prompt with context
  2. Analyzes diet against nutritional guidelines
  3. Generates evidence-based recommendations
  4. Returns formatted response with sources

### Data Flow

#### Initialization Phase
```
1. Load nutritional knowledge base
2. Split into chunks (RecursiveCharacterTextSplitter)
3. Generate embeddings (Ollama)
4. Store in ChromaDB vector database
5. Initialize QA chain with retriever
```

#### Analysis Phase
```
1. User uploads diet document
2. Extract text based on file type
3. Parse and categorize food items
4. Create Document object from diet data
5. Add to vector store (embeddings generated)
6. Build analysis query with statistics
7. RAG retrieval: Search for relevant nutrition info
8. Combine user data + retrieved context
9. Send to LLM with structured prompt
10. Generate comprehensive analysis
11. Display results with visualizations
```

### Session State Management

The application maintains state across interactions:
```python
st.session_state = {
    'chroma_db': ChromaDB instance,
    'qa_chain': RetrievalQA chain,
    'analysis_history': List of past analyses,
    'nutrition_knowledge_loaded': Boolean flag,
    'ollama_url': API endpoint,
    'foods': Extracted food items,
    'file_text': Raw document text
}
```

### Performance Optimizations

1. **Lazy Loading**: RAG system initialized only when needed
2. **Caching**: Session state prevents re-initialization
3. **Chunking**: Optimal chunk sizes for embedding quality
4. **Top-K Retrieval**: Limits to 3 most relevant documents
5. **In-Memory Storage**: ChromaDB runs without disk persistence

### Security & Privacy

- **Local Processing**: All data stays on user's machine
- **No External APIs**: Uses local Ollama instance
- **Session-Based**: No persistent storage of user data
- **No Telemetry**: ChromaDB telemetry disabled

### Extensibility Points

The architecture supports easy extensions:
- **New File Formats**: Add extraction functions
- **Custom Categories**: Extend category dictionaries
- **Different Models**: Change Ollama model string
- **Enhanced Prompts**: Modify prompt templates
- **Additional Knowledge**: Update NUTRITION_KNOWLEDGE constant

## Configuration

### Ollama URL
Default: `http://localhost:11434`

Change in the UI or modify in code:
```python
st.session_state.ollama_url = "your_ollama_url"
```

### Model Selection
To use a different Ollama model, update:
```python
model="llama3.2:3b"  # Change to your preferred model
```

## License

This project is open-source and available for personal and educational use.

## Disclaimer

This tool provides general nutritional information and should not be used as a substitute for professional medical or nutritional advice. Always consult with qualified healthcare providers for personalized dietary recommendations.

## Support

For issues or questions:
- Check Ollama documentation: [docs.ollama.ai](https://docs.ollama.ai)
- Review Streamlit docs: [docs.streamlit.io](https://docs.streamlit.io)
- Verify LangChain setup: [python.langchain.com](https://python.langchain.com)

---

**Built with**: Streamlit • LangChain • Ollama • ChromaDB

**Version**: 1.0.0
