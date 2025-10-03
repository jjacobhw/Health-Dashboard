import streamlit as st
import chromadb
from chromadb.config import Settings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_ollama import OllamaEmbeddings
from langchain_chroma import Chroma
from langchain_ollama import OllamaLLM
from langchain.chains.retrieval_qa.base import RetrievalQA
from langchain_core.prompts import PromptTemplate
from langchain_core.documents import Document
import PyPDF2
import io
import json
from datetime import datetime

# Page configuration
st.set_page_config(
    page_title="AI Diet Health Dashboard with RAG",
    page_icon="🥗",
    layout="wide"
)

# Initialize session state
if 'chroma_db' not in st.session_state:
    st.session_state.chroma_db = None
if 'qa_chain' not in st.session_state:
    st.session_state.qa_chain = None
if 'analysis_history' not in st.session_state:
    st.session_state.analysis_history = []
if 'nutrition_knowledge_loaded' not in st.session_state:
    st.session_state.nutrition_knowledge_loaded = False

# Nutritional knowledge base
NUTRITION_KNOWLEDGE = """
# Nutritional Guidelines and Information

## Macronutrients
- Proteins: Essential for muscle repair, enzyme production, and immune function. Adults need 0.8g per kg of body weight daily.
- Carbohydrates: Primary energy source. Complex carbs (whole grains, vegetables) are preferred over simple sugars.
- Fats: Essential for hormone production and nutrient absorption. Focus on unsaturated fats from nuts, avocados, and fish.

## Micronutrients
- Vitamins: A, B-complex, C, D, E, K each play unique roles in health.
- Minerals: Calcium, iron, magnesium, zinc, potassium are crucial for various bodily functions.

## Food Groups
1. Vegetables: 5+ servings daily. Rich in fiber, vitamins, minerals, and antioxidants.
2. Fruits: 2-3 servings daily. Natural sugars with fiber, vitamins, and phytonutrients.
3. Proteins: Lean meats, fish, eggs, legumes, tofu. Essential for tissue repair.
4. Whole Grains: Brown rice, quinoa, oats. Provide sustained energy and fiber.
5. Dairy/Alternatives: Calcium and vitamin D for bone health.
6. Healthy Fats: Nuts, seeds, avocados, olive oil. Support brain and heart health.

## Dietary Patterns
- Mediterranean Diet: Rich in vegetables, fruits, whole grains, fish, olive oil.
- DASH Diet: Focuses on reducing sodium, increasing potassium-rich foods.
- Plant-Based: Emphasizes whole plant foods, may include or exclude animal products.

## Common Deficiencies
- Vitamin D: Common in northern climates, important for immune function.
- Iron: Particularly in menstruating women, vegetarians.
- B12: Risk for vegans, older adults.
- Omega-3: Important for heart and brain health.

## Processed Foods
- Ultra-processed foods often contain high sodium, sugar, unhealthy fats, and additives.
- Limit consumption of packaged snacks, sugary drinks, fast food.
- Focus on whole, minimally processed foods.

## Hydration
- Water: 8-10 glasses daily, more with exercise.
- Limit sugary drinks, excessive caffeine.

## Meal Timing
- Regular meal patterns support metabolism.
- Breakfast importance varies by individual.
- Evening eating should be balanced, not excessive.

## Special Considerations
- Athletes: Higher protein and calorie needs.
- Pregnancy: Increased folate, iron, calcium needs.
- Aging: May need more protein, calcium, vitamin D.
- Diabetes: Carbohydrate counting and timing important.
"""

def initialize_rag_system(ollama_url="http://localhost:11434"):
    """Initialize the RAG system with ChromaDB and LangChain"""
    try:
        # Initialize Ollama embeddings
        embeddings = OllamaEmbeddings(
            base_url=ollama_url,
            model="llama3.2:3b"
        )
        
        # Initialize ChromaDB client
        chroma_client = chromadb.Client(Settings(
            anonymized_telemetry=False,
            allow_reset=True
        ))
        
        # Create or get collection
        try:
            chroma_client.delete_collection("nutrition_knowledge")
        except:
            pass
        
        collection = chroma_client.create_collection(
            name="nutrition_knowledge",
            metadata={"description": "Nutritional information and guidelines"}
        )
        
        # Split nutrition knowledge into chunks
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=500,
            chunk_overlap=50,
            length_function=len
        )
        
        documents = [Document(page_content=NUTRITION_KNOWLEDGE)]
        splits = text_splitter.split_documents(documents)
        
        # Create vector store
        vectorstore = Chroma.from_documents(
            documents=splits,
            embedding=embeddings,
            collection_name="nutrition_knowledge",
            client=chroma_client
        )
        
        # Initialize Ollama LLM
        llm = OllamaLLM(
            base_url=ollama_url,
            model="llama3.2:3b",
            temperature=0.7
        )
        
        # Create custom prompt template
        prompt_template = """You are an expert nutritionist analyzing diet data. Use the following nutritional information to provide accurate, evidence-based advice.

Context: {context}

Question: {question}

Provide a detailed, professional analysis that includes:
1. Assessment of the current diet based on nutritional guidelines
2. Specific strengths and areas for improvement
3. Evidence-based recommendations
4. Potential health implications

Answer:"""

        PROMPT = PromptTemplate(
            template=prompt_template,
            input_variables=["context", "question"]
        )
        
        # Create retrieval QA chain
        qa_chain = RetrievalQA.from_chain_type(
            llm=llm,
            chain_type="stuff",
            retriever=vectorstore.as_retriever(search_kwargs={"k": 3}),
            chain_type_kwargs={"prompt": PROMPT},
            return_source_documents=True
        )
        
        return vectorstore, qa_chain, True
        
    except Exception as e:
        st.error(f"Error initializing RAG system: {str(e)}")
        return None, None, False

def extract_text_from_pdf(pdf_file):
    """Extract text from uploaded PDF"""
    try:
        pdf_reader = PyPDF2.PdfReader(io.BytesIO(pdf_file.read()))
        text = ""
        for page in pdf_reader.pages:
            text += page.extract_text()
        return text
    except Exception as e:
        st.error(f"Error reading PDF: {str(e)}")
        return None

def categorize_foods(food_list):
    """Categorize foods into groups"""
    categories = {
        'vegetables': ['lettuce', 'carrot', 'broccoli', 'spinach', 'kale', 'tomato', 
                      'cucumber', 'pepper', 'celery', 'cabbage', 'salad', 'greens', 'beans'],
        'fruits': ['apple', 'banana', 'orange', 'berry', 'berries', 'grape', 'melon', 
                  'peach', 'pear', 'mango', 'strawberry', 'blueberry'],
        'proteins': ['chicken', 'beef', 'pork', 'fish', 'salmon', 'tuna', 'egg', 
                    'tofu', 'turkey', 'shrimp', 'meat', 'lentil', 'bean'],
        'grains': ['rice', 'bread', 'pasta', 'cereal', 'oats', 'quinoa', 'wheat', 
                  'bagel', 'tortilla', 'noodle'],
        'dairy': ['milk', 'cheese', 'yogurt', 'butter', 'cream'],
        'processed': ['pizza', 'burger', 'fries', 'chips', 'candy', 'soda', 
                     'cookie', 'cake', 'donut'],
        'healthy_fats': ['avocado', 'nuts', 'seeds', 'olive', 'almond', 'walnut']
    }
    
    category_counts = {cat: 0 for cat in categories.keys()}
    categorized_items = {cat: [] for cat in categories.keys()}
    
    for food in food_list:
        food_lower = food.lower()
        for category, keywords in categories.items():
            if any(keyword in food_lower for keyword in keywords):
                category_counts[category] += 1
                categorized_items[category].append(food)
                break
    
    return category_counts, categorized_items

def calculate_health_score(category_counts):
    """Calculate health score based on food categories"""
    healthy = category_counts['vegetables'] + category_counts['fruits'] + category_counts['healthy_fats']
    unhealthy = category_counts['processed']
    protein = category_counts['proteins']
    
    score = min(100, max(0, (healthy * 10) - (unhealthy * 5) + (protein * 5) + 30))
    return round(score)

def analyze_diet_with_rag(qa_chain, food_data, category_counts, health_score):
    """Use RAG to analyze diet with retrieved nutritional knowledge"""
    
    # Prepare the query with diet information
    query = f"""
    Analyze this monthly diet data:
    
    Total food items: {len(food_data)}
    
    Category breakdown:
    - Vegetables: {category_counts['vegetables']} items
    - Fruits: {category_counts['fruits']} items
    - Proteins: {category_counts['proteins']} items
    - Grains: {category_counts['grains']} items
    - Dairy: {category_counts['dairy']} items
    - Processed foods: {category_counts['processed']} items
    - Healthy fats: {category_counts['healthy_fats']} items
    
    Current health score: {health_score}/100
    
    Sample foods: {', '.join(food_data[:15])}
    
    Provide a comprehensive nutritional analysis with specific recommendations.
    """
    
    try:
        result = qa_chain({"query": query})
        return result['result'], result.get('source_documents', [])
    except Exception as e:
        return f"Error during analysis: {str(e)}", []

def add_documents_to_vectorstore(vectorstore, documents, embeddings):
    """Add user's diet documents to the vector store for context"""
    try:
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=300,
            chunk_overlap=30
        )
        splits = text_splitter.split_documents(documents)
        vectorstore.add_documents(splits)
        return True
    except Exception as e:
        st.error(f"Error adding documents: {str(e)}")
        return False

# Main App
def main():
    st.title("🥗 AI Diet Health Dashboard with RAG")
    st.markdown("### Powered by LangChain, ChromaDB, and Ollama Llama 3.2 3B")
    
    # Sidebar for settings
    with st.sidebar:
        st.header("⚙️ Settings")
        ollama_url = st.text_input(
            "Ollama URL",
            value="http://localhost:11434",
            help="URL where Ollama is running"
        )
        
        if st.button("Initialize RAG System"):
            with st.spinner("Initializing RAG system..."):
                vectorstore, qa_chain, success = initialize_rag_system(ollama_url)
                if success:
                    st.session_state.chroma_db = vectorstore
                    st.session_state.qa_chain = qa_chain
                    st.session_state.nutrition_knowledge_loaded = True
                    st.success("✅ RAG system initialized!")
                else:
                    st.error("❌ Failed to initialize. Make sure Ollama is running.")
        
        st.markdown("---")
        st.markdown("### 📚 Knowledge Base Status")
        if st.session_state.nutrition_knowledge_loaded:
            st.success("✅ Nutrition knowledge loaded")
        else:
            st.warning("⚠️ RAG system not initialized")
        
        st.markdown("---")
        st.markdown("### 🔧 Setup Instructions")
        st.code("ollama pull llama3.2:3b", language="bash")
        st.code("ollama serve", language="bash")
    
    # Main content area
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.header("📄 Upload Your Diet Log")
        uploaded_file = st.file_uploader(
            "Upload a PDF containing your food log",
            type=['pdf'],
            help="Upload a PDF with your monthly food diary"
        )
        
        if uploaded_file:
            with st.spinner("Processing PDF..."):
                # Extract text from PDF
                pdf_text = extract_text_from_pdf(uploaded_file)
                
                if pdf_text:
                    st.success(f"✅ PDF processed: {len(pdf_text)} characters extracted")
                    
                    # Extract food items (simple parsing)
                    words = pdf_text.lower().split()
                    food_keywords = set(words) - {'the', 'and', 'with', 'for', 'week', 'day', 
                                                   'breakfast', 'lunch', 'dinner', 'snack'}
                    foods = list(food_keywords)[:50]  # Limit to 50 items
                    
                    st.session_state.foods = foods
                    st.session_state.pdf_text = pdf_text
                    
                    # Show extracted foods
                    with st.expander("📋 Extracted Foods", expanded=True):
                        st.write(f"**{len(foods)} food items detected**")
                        st.write(", ".join(foods[:30]))
                        if len(foods) > 30:
                            st.write(f"... and {len(foods) - 30} more")
    
    with col2:
        st.header("📊 Quick Stats")
        if 'foods' in st.session_state:
            foods = st.session_state.foods
            st.metric("Food Items", len(foods))
            
            category_counts, categorized = categorize_foods(foods)
            health_score = calculate_health_score(category_counts)
            
            st.metric("Health Score", f"{health_score}/100")
            
            # Show top categories
            st.markdown("**Top Categories:**")
            sorted_cats = sorted(category_counts.items(), key=lambda x: x[1], reverse=True)
            for cat, count in sorted_cats[:3]:
                if count > 0:
                    st.write(f"• {cat.title()}: {count}")
    
    # Analysis section
    if 'foods' in st.session_state and st.session_state.nutrition_knowledge_loaded:
        st.markdown("---")
        
        if st.button("🧠 Analyze Diet with RAG", type="primary", use_container_width=True):
            with st.spinner("Analyzing your diet with AI and nutritional knowledge base..."):
                foods = st.session_state.foods
                category_counts, categorized = categorize_foods(foods)
                health_score = calculate_health_score(category_counts)
                
                # Add user's diet to vector store for better context
                diet_doc = Document(
                    page_content=f"User's diet contains: {', '.join(foods)}. {st.session_state.pdf_text[:1000]}"
                )
                add_documents_to_vectorstore(
                    st.session_state.chroma_db,
                    [diet_doc],
                    OllamaEmbeddings(base_url=ollama_url, model="llama3.2:3b")
                )
                
                # Perform RAG analysis
                analysis, sources = analyze_diet_with_rag(
                    st.session_state.qa_chain,
                    foods,
                    category_counts,
                    health_score
                )
                
                # Store in history
                st.session_state.analysis_history.append({
                    'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                    'health_score': health_score,
                    'analysis': analysis,
                    'categories': category_counts
                })
                
                # Display results
                st.markdown("## 🎯 Analysis Results")
                
                # Health Score Card
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Overall Health Score", f"{health_score}/100")
                with col2:
                    color = "🟢" if health_score > 70 else "🟡" if health_score > 40 else "🔴"
                    st.metric("Status", f"{color} {'Excellent' if health_score > 70 else 'Good' if health_score > 40 else 'Needs Improvement'}")
                with col3:
                    total_healthy = category_counts['vegetables'] + category_counts['fruits']
                    st.metric("Healthy Foods", total_healthy)
                
                st.markdown("---")
                
                # AI Analysis
                st.markdown("### 🤖 AI-Powered Nutritional Analysis")
                st.info(analysis)
                
                # Category Breakdown
                st.markdown("### 📊 Category Breakdown")
                cols = st.columns(4)
                for idx, (category, count) in enumerate(category_counts.items()):
                    with cols[idx % 4]:
                        st.metric(category.replace('_', ' ').title(), count)
                
                # Retrieved Sources (optional)
                with st.expander("📚 Referenced Nutritional Guidelines"):
                    if sources:
                        for idx, doc in enumerate(sources[:3]):
                            st.markdown(f"**Source {idx + 1}:**")
                            st.text(doc.page_content[:200] + "...")
                    else:
                        st.write("No specific sources retrieved")
                
                # Detailed breakdown by category
                with st.expander("🔍 Detailed Food Categories"):
                    for category, items in categorized.items():
                        if items:
                            st.markdown(f"**{category.title()}** ({len(items)} items)")
                            st.write(", ".join(items[:10]))
                            if len(items) > 10:
                                st.write(f"... and {len(items) - 10} more")
    
    elif 'foods' in st.session_state and not st.session_state.nutrition_knowledge_loaded:
        st.warning("⚠️ Please initialize the RAG system from the sidebar first!")
    
    # Analysis History
    if st.session_state.analysis_history:
        st.markdown("---")
        st.header("📜 Analysis History")
        
        for idx, record in enumerate(reversed(st.session_state.analysis_history[-5:])):
            with st.expander(f"Analysis from {record['timestamp']} - Score: {record['health_score']}/100"):
                st.write(record['analysis'][:500] + "...")
                
                cols = st.columns(4)
                for jdx, (cat, count) in enumerate(list(record['categories'].items())[:4]):
                    with cols[jdx]:
                        st.metric(cat.title(), count)

if __name__ == "__main__":
    main()