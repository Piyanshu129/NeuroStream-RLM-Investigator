# RLM Data Investigator 🔍

A professional **Recursive Language Model (RLM)** dashboard designed for deep data investigation. This application uses a "Context-as-Variable" paradigm, where a reasoning LLM writes Python code to traverse and cross-reference multiple data sources dynamically.

![Dashboard Preview](https://via.placeholder.com/1200x600?text=RLM+Data+Investigator+UI)

## 🚀 Key Features

- **Recursive Reasoning**: Handles complex multi-step queries by spawning sub-agents for summarization and deep dives.
- **Context-as-Variable**: Instead of stuffing prompts with raw data, the LLM writes and executes code to fetch only what it needs.
- **Just-in-Time Graph Emission**: A real-time Knowledge Graph that grows as the agent "touches" data points in SQL, Vector DB, or Files.
- **Resilient Model Fallbacks**: Automatically retries and falls back to alternative LLMs (Llama-4, Nemotron) if the primary model (DeepSeek) is unavailable.
- **Case-Insensitive SQL**: Sophisticated SQLite handling to ensure natural language queries match data regardless of casing.

## 🛠️ Tech Stack

- **Backend**: FastAPI (Python) with SSE (Server-Sent Events) for real-time thought traces.
- **Frontend**: Next.js 15+, Tailwind CSS, and React Force Graph for visualization.
- **Brain**: OpenRouter API (DeepSeek R1/V3 + Llama series).
- **Data Layers**: 
  - **SQL**: SQLite for structured shipment and supplier data.
  - **Vector**: FAISS (Facebook AI Similarity Search) for semantic search over supplier agreements.
  - **File**: Raw text logs for shipping history.

## 🏁 Getting Started

### 1. Prerequisites
- Python 3.10+
- Node.js 18+
- [OpenRouter API Key](https://openrouter.ai/keys)

### 2. Installation
```powershell
# Clone the repository
git clone https://github.com/YOUR_USERNAME/RLM-Data-Investigator.git
cd RLM-Data-Investigator

# Install Backend Dependencies
pip install -r backend/requirements.txt

# Install Frontend Dependencies
cd frontend
npm install
```

### 3. Configuration
Create a `.env` file in the root directory:
```env
OPENROUTER_API_KEY=your_key_here
NEXT_PUBLIC_API_URL=http://localhost:8001
```

### 4. Initialize Data
```powershell
python -m backend.setup_dummy_data
```

### 5. Start the Engines
**Terminal 1 (Backend):**
```powershell
uvicorn backend.main:app --reload --port 8001
```

**Terminal 2 (Frontend):**
```powershell
cd frontend
npx next dev
```

## 🎥 Knowledge Graph Sources
- 🔵 **Blue**: SQL Database
- 🟣 **Purple**: Vector Database (FAISS)
- 🟠 **Orange**: Local Files
- 🟢 **Cyan**: Agent Logic Nodes

## 🛡️ License
MIT
