# 🎬 AnimAI Studio

AI-powered educational animation generator that creates professional Manim animations from natural language descriptions using a multi-agent pipeline.

## ✨ Features

- 🤖 **Multi-Agent Pipeline**: Teacher → Planner → Coder → Validator → Debugger orchestration
- 🧑‍🏫 **Teacher Agent**: Breaks down concepts with analogies, key terms, and step-by-step explanations before planning
- 🎯 **Intent Classification**: Automatically detects bare topics, simple explanations, and detailed requests
- 🎨 **Professional Design System**: Consistent dark theme with stunning visuals
- 🔄 **Self-Healing Code**: Automatic debugging and retry logic with structured failure logging
- 🎙️ **Native Manim Voiceover**: Narration rendered directly during Manim scene execution
- 📚 **RAG-Enhanced**: Retrieves relevant Manim patterns from documentation via FAISS
- 📈 **Learning System**: Improves over time from user feedback
- ✅ **Video Validator**: Checks generated code for segment structure, voiceover coverage, and cleanup patterns
- 📊 **Failure Analytics**: Persistent failure logging with tag-based analysis and viewer UI
- 🔒 **Safe Execution**: Docker-isolated code compilation

## 🏗️ Architecture

```
User Query
    │
    ▼
┌──────────────┐
│ Intent       │── bare_topic / simple_explanation / detailed
│ Classifier   │
└──────┬───────┘
       ▼
┌──────────────┐    ┌──────────────┐
│ Teacher      │───▶│ Planner      │── structured plan with voiceover script
│ Agent        │    │ Agent        │
└──────────────┘    └──────┬───────┘
                           ▼
                    ┌──────────────┐    ┌──────────────┐
                    │ Coder        │───▶│ Validator    │── checks segment structure
                    │ Agent        │    │ Agent        │
                    └──────┬───────┘    └──────────────┘
                           ▼
                    ┌──────────────┐    ┌──────────────┐
                    │ Docker       │◀──▶│ Debugger     │── self-healing retry loop
                    │ Sandbox      │    │ Agent        │
                    └──────┬───────┘    └──────────────┘
                           ▼
                     Final MP4 Video
```

## 📋 Prerequisites

- Python 3.9 or higher
- Docker Desktop installed and running
- FFmpeg installed
- Gemini API key
- Azure Speech resource (Azure for Students supported)

## 🚀 Setup

1. **Clone the repository**
   ```bash
   git clone https://github.com/Manish-Sharma26/AnimAi.git
   cd AnimAi
   ```

2. **Create virtual environment**
   ```bash
   # Windows
   python -m venv venv
   venv\Scripts\activate
   
   # Mac/Linux
   python3 -m venv venv
   source venv/bin/activate
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

4. **Set up environment variables**
   
   Create a `.env` file in the root directory:
   ```env
   GEMINI_API_KEY=your_gemini_api_key_here

   # TTS provider selection
   TTS_PROVIDER=azure
   TTS_FALLBACK_PROVIDER=gtts

   # Azure Speech (required when TTS_PROVIDER=azure)
   AZURE_SUBSCRIPTION_KEY=your_azure_speech_key
   AZURE_SERVICE_REGION=centralindia
   AZURE_TTS_VOICE=en-IN-NeerjaNeural
   AZURE_TTS_STYLE=general
   ```

5. **Build Docker image**
   ```bash
   docker build -t manim-voiceover .
   ```

6. **Initialize RAG system** (first time only)
   ```bash
   python rag/download_docs.py
   ```
   This downloads Manim documentation and builds the FAISS search index (~2-3 minutes).

7. **Run the application**
   ```bash
   streamlit run app.py
   ```

8. **Open your browser**
   
   Navigate to `http://localhost:8501`

## 🎯 Usage

1. Enter a description of the animation you want (e.g., "Animate bubble sort with array 5 2 8 1 9")
2. Click **Generate Plan** — the Teacher agent explains the concept, then the Planner builds a structured animation plan
3. Review and optionally edit the plan JSON
4. Click **Generate Video With Voice** to compile the narrated animation
5. Watch your animation and download the MP4
6. Give feedback to help the system learn!

## 📁 Project Structure

```
animai-studio/
├── app.py                      # Streamlit web interface
├── Dockerfile                  # Docker image configuration
├── PROJECT_REPORT.md           # Detailed project documentation
├── agent/                      # Multi-agent system
│   ├── orchestrator.py         # Main pipeline coordinator
│   ├── intent.py               # Query intent classifier (bare_topic / simple / detailed)
│   ├── teacher.py              # Concept explanation agent (analogies, key terms)
│   ├── planner.py              # Animation structure planner
│   ├── coder.py                # Manim code generator
│   ├── debugger.py             # Automatic error fixing with self-healing loop
│   ├── validator.py            # Post-generation video structure validator
│   ├── failure_logger.py       # Persistent failure logging and analytics
│   ├── topic_hints.py          # Topic-specific Manim patterns and hints
│   ├── llm.py                  # LLM API wrapper (Gemini)
│   └── feedback.py             # Learning from user feedback
├── sandbox/                    # Isolated execution
│   └── sandbox.py              # Docker-based Manim runner
├── rag/                        # Retrieval-Augmented Generation
│   ├── download_docs.py        # Doc scraper and indexer
│   ├── retriever.py            # FAISS vector similarity search
│   ├── manim_chunks.json       # Documentation chunks
│   └── manim_docs.index        # FAISS search index
├── scripts/                    # Utility and demo scripts
├── diagrams/                   # Architecture diagrams (Mermaid + PNG)
├── showcase/                   # Project showcase assets
├── failure_log_viewer.py       # Streamlit failure log viewer
├── export_diagrams.js          # Mermaid diagram → PNG exporter
└── outputs/                    # Generated videos and failure logs
```

## 🛠️ Technology Stack

- **Frontend**: Streamlit
- **LLM**: Google Gemini
- **Animation**: Manim Community Edition
- **Execution**: Docker
- **RAG**: FAISS + Sentence Transformers
- **Voiceover**: manim-voiceover + Azure Speech (primary) + gTTS (fallback)
- **Video Processing**: FFmpeg

## 🎨 Visual Styles

The system automatically chooses the best visual style:

- `array_boxes` - Data structures, search algorithms
- `bar_chart` - Sorting algorithms, comparisons
- `diagram` - Biology, chemistry, science concepts
- `graph_plot` - Mathematical functions, equations
- `physics_motion` - Moving objects, forces, trajectories
- `timeline` - History, sequences, processes
- `flowchart` - Decision trees, workflows

## 🤝 Contributing

Contributions welcome! The system learns from feedback, so the more you use it, the better it gets.

## 📝 License

MIT License - feel free to use for educational purposes!

## ⚠️ Security Note

- Never commit your `.env` file
- Keep your Gemini API key private
- Regenerate your API key if accidentally exposed

## 🐛 Troubleshooting

**Docker connection error**: Make sure Docker Desktop is running

**FFmpeg not found**: Install FFmpeg and add to system PATH

**Gemini API error**: Verify your API key in `.env` file

**Azure auth error**: Check `AZURE_SUBSCRIPTION_KEY` and `AZURE_SERVICE_REGION`

**gTTS fallback not used**: Ensure `TTS_FALLBACK_PROVIDER=gtts`

**No video generated**: Check Docker logs with `docker logs <container_id>`

**Failure logs**: Check `outputs/failure_logs/` or run `streamlit run failure_log_viewer.py` for analysis

## 📧 Support

For issues and questions, please open a GitHub issue.

---

Built with ❤️ for educators and students everywhere
