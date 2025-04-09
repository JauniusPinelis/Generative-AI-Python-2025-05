**8-Week Practical Course: Application Programming with GEN-AI and Python**

---

## **Week 1: Python Foundations & Environment Setup**

### **Day 1: Course Introduction & Environment Setup**
- Course overview, objectives, project goals.
- Python installation, PATH setup.
- Virtual Environments: `venv`/`uv` usage.
- IDE Setup: VSCode basics & extensions.
- Running "Hello, World!".

### **Day 2: Python Basics**
- Core Data Types: Numbers, Strings, Booleans.
- Data Structures: Lists, Tuples, Dictionaries, Sets.
- Control Flow: `if`/`elif`/`else`, `for`/`while` loops,
- Basic Debugging: `print`, tracebacks, IDE debugger intro.

### **Day 3: Python Essentials for GEN-AI**
- Functions: Definition (`def`), arguments, return values, scope.
- Error Handling: `try...except` blocks.
- Modules & Packages: `import`, standard library (`os`, `math`), package installation (`uv`).
- Code Style: PEP 8, modularity.
- **Lab:** Text manipulation utility functions.

### **Day 4: Working with Libraries & Data**
- HTTP Basics: Methods (GET, POST), status codes.
- `requests` Library: Making API calls, handling responses.
- `json` Library: Parsing (`json.loads()`) & serialization (`json.dumps()`).
- Intro to Data Handling: `numpy` (arrays), working with files
- **Lab:** Fetch & process data from a public API; optional load into files.

### **Day 5: Introduction to APIs & Basic Application Building**
- API Concepts: REST principles, authentication (API keys).
- Application Structure: Command-line app design.
- **Lab:** Build a CLI app that calls an API based on user input and displays results.

---

## **Week 2: Introduction to Generative AI & API Integration**

### **Day 1: What Is Generative AI?**
- Definition, history, key concepts (LLMs, prompts, tokens).
- Major models & players (OpenAI, Google, etc.).
- Ethical Considerations: Bias, misuse.
- Demo: Text & image generation tools (ChatGPT, Midjourney).

### **Day 2: REST APIs & Introduction to FastAPI**
- RESTful API review.
- Intro to FastAPI: Features, basic app setup (`uvicorn`).
- Path Operations: Routes (`@app.get`/`@app.post`), path/query parameters.
- Request Body: Pydantic models for validation.
- **Lab:** Simple FastAPI "Hello World" API with different endpoints; explore Swagger UI.

### **Day 3: Working with OpenAI's GPT APIs**
- OpenAI Platform: Models (GPT-4, GPT-3.5), pricing.
- API Key Management & Security.
- OpenAI Python Library: Installation (`uv add openai`), usage.
- Chat Completions API: `openai.chat.completions.create`, message roles.
- Key Parameters: `model`, `messages`, `temperature`, `max_tokens`.
- **Lab:** Python script to send prompts to GPT via API and print responses.

### **Day 4: OpenAI Models: Capabilities, Limitations & Prompting**
- Model Capabilities & Limitations (knowledge cutoff, hallucinations).
- Prompt Engineering: Basics (context, formatting, zero/few-shot).
- Intro to Tools/Function Calling.
- Intro to Assistants API (stateful conversations).
- **Lab:** Experiment with prompt engineering techniques; optional function calling example.

### **Day 5: Integrating OpenAI with FastAPI**
- App Structure: FastAPI interacting with OpenAI.
- Secure API Key Handling: Environment variables (`python-dotenv`).
- API Endpoint (`/generate`): Accepts prompt, calls OpenAI.
- Return Response: Send generated text back as JSON.
- **Lab:** FastAPI app endpoint taking a prompt, calling OpenAI, returning the response.

---

## **Week 3: Google Gemini & UI with Streamlit**

### **Day 0: Vibe coding tutorial (Cursor and Copilot)**
- Intro to AI code assistants: Cursor, Copilot features.
- Setup, basic usage (chat, code generation).
- Tips for AI pair programming.

### **Day 1: Exploring Google AI Studio & Vertex AI**
- Google AI Ecosystem: Vertex AI overview.
- Google AI Studio: Web-based prototyping with Gemini.
- Gemini Models: Pro, Flash capabilities.
- Hands-on: Prompt testing, parameters, safety settings in AI Studio.
- **Lab:** Use AI Studio for text tasks (generation, summary, Q&A) with Gemini.

### **Day 2: Programming with the Google Gemini API**
- Google AI Python SDK: Installation (`uv add google-generativeai`), setup.
- Authentication: API keys.
- API Calls: `GenerativeModel.generate_content`.
- Streaming Responses: Handling chunk-by-chunk generation.
- **Lab:** Python script using Gemini API for text generation, including streaming.

### **Day 3: Introduction to Streamlit for UI Development**
- Streamlit Basics: Rapid UI building, execution model, caching.
- Core Widgets: Text display (`st.write`), input (`st.text_input`, `st.button`), layout (`st.columns`, `st.sidebar`).
- Data Display: Pandas DataFrames (`st.dataframe`), basic charts.
- **Lab:** Build a simple Streamlit app with various input/output widgets and layout.

### **Day 4: Building Streamlit Chatbots with Gemini**
- Chat UI: `st.chat_input`, `st.chat_message`.
- State Management: `st.session_state` for history.
- Integration: Connect Streamlit input/output to Gemini API calls.
- Workflow: Get input -> Update history -> Call API -> Display response -> Update history.
- **Lab:** Simple Streamlit chatbot using `st.session_state` and Gemini API.

### **Day 5: Project Work & Review**
- Dedicated project work time.
- Instructor Q&A and support.
- Review Week 3: Gemini API, Streamlit, state management.
- Discussion: Challenges, ideas.

---

## **Week 4: Introduction to Retrieval-Augmented Generation (RAG)**

### **Day 1: What is RAG? Core Concepts & Use Cases**
- Problem: LLM limitations (knowledge cutoff, hallucinations).
- RAG Concept: Retrieve relevant info then generate.
- Architecture: Retriever + Generator (LLM).
- Benefits: Grounding, factual consistency, domain-specific knowledge.
- Use Cases: Chatbots, Q&A over private data, summarization.
- **Activity:** Brainstorm RAG applications & project ideas.

### **Day 2: RAG Implementation with Langchain**
- Intro to LangChain: Orchestration framework (Components, Chains).
- RAG Components: Document Loaders, Text Splitters, Embeddings, Vector Stores, Retrievers, LLMs.
- `RetrievalQA` Chain: Combining components for Q&A.
- **Lab:** Simple LangChain RAG pipeline: Load text -> Split -> Retrieve -> Generate answer with LLM (OpenAI/Gemini).

### **Day 3: Embeddings & Vector Databases**
- Embeddings: Representing semantic meaning (OpenAI, Sentence Transformers).
- Vector Stores: Storing embeddings for similarity search.
- LangChain Integration: Using Vector Stores (`Chroma`).
- **Lab:** Enhance RAG pipeline: Generate embeddings -> Store in ChromaDB -> Retrieve via similarity search.

### **Day 4: Improving RAG Performance**
- RAG Challenges: Retrieval & generation quality.
- Text Splitting Strategies: Impact of different chunking methods.
- Retrieval Techniques: MMR, metadata filtering, context compression.
- **Lab:** Experiment with different splitting/retrieval methods in the LangChain RAG pipeline; observe impact.

### **Day 5: Integrating RAG with Streamlit**
- Combining Week 3 & 4: Building a Streamlit UI for the RAG pipeline.
- Workflow: User query (Streamlit input) -> RAG Chain (LangChain) -> Display results (Streamlit output).
- Handling state (`st.session_state`) for conversation history with RAG.
- **Lab:** Create a Streamlit application that uses the RAG pipeline built earlier to answer questions based on the loaded documents, displaying both the answer and optionally the retrieved context.

---

## **Week 5: Open Source Models & Fine-Tuning**

### **Day 1: Introduction to Hugging Face**
- Hugging Face Hub: Models, datasets, Spaces.
- Transformers Library: Core concepts, pipeline for easy model usage.
- Loading pre-trained models and tokenizers.
- Basic tasks: Text generation, classification with `transformers`.
- **Lab:** Use the `transformers` library pipeline to perform text generation and sentiment analysis with pre-trained models.

### **Day 2: Running Models Locally with Ollama & LM Studio**
- Ollama Overview: Running LLMs locally via command line.
- Installation and setup.
- Pulling and running various open-source models (e.g., Llama 3, Mistral).
- LM Studio Overview: GUI for discovering, downloading, and running local LLMs.
- Comparing Ollama and LM Studio.
- **Lab:** Install Ollama, download a model, and interact with it via the command line. Optionally explore LM Studio.

### **Day 3: Interacting with Ollama via Python**
- Ollama Python Library: Installation (`uv add ollama`).
- Connecting to a running Ollama instance.
- Generating text, chat completions using the library.
- Streaming responses.
- Integrating local models into Python applications (e.g., basic FastAPI or Streamlit app).
- **Lab:** Write a Python script using the `ollama` library to interact with a locally running model. Experiment with different prompts and models.

### **Day 4: Fine-Tuning Theory**
- What is Fine-Tuning?: Adapting pre-trained models to specific tasks or domains.
- Why Fine-Tune?: Improved performance on niche tasks, domain adaptation, style transfer.
- Full Fine-Tuning vs. Parameter-Efficient Fine-Tuning (PEFT): Concepts, pros & cons.
- Data Preparation: Creating instruction datasets (prompt/response pairs).

### **Day 5: Fine-Tuning in Action with Unsloth & Kaggle/Colab**
- Unsloth Overview: Library for significantly faster LoRA/QLoRA fine-tuning.
- Setting up Environment: Using Kaggle or Google Colab GPUs.
- Data Formatting: Picking a dataset from HuggingFace
- Fine-tuning Script: Using Unsloth with `transformers` SFTTrainer.
- Running the fine-tuning job.
- Basic Inference/Testing the fine-tuned model.
- **Lab:** Fine-tune a small open-source model (e.g., Mistral-7B variant) on a sample instruction dataset using Unsloth on Kaggle/Colab.

---

## **Week 6: Multimodal Models**

### **Day 1: Vision Models (Image Understanding)**
- Intro to Multimodality: Combining text, images, audio.
- Vision Models: Image captioning, Visual Question Answering (VQA).
- APIs/Libraries: Using Gemini Vision, OpenAI Vision API, Hugging Face `transformers` for vision.
- **Lab:** Use an API (e.g., Gemini Pro Vision) to describe images or answer questions about them.

### **Day 2: Multimodal Reasoning**
- Reasoning Across Modalities: Combining information from text and images for complex tasks.
- Examples: Explaining visual jokes, following visual instructions, chart interpretation.
- Prompting Techniques for Multimodal Models.
- **Lab:** Experiment with prompts combining text and images to test reasoning capabilities (e.g., using Gemini or GPT-4 Vision).

### **Day 3: Image Generation Models**
- Text-to-Image Generation: Concepts behind diffusion models.
- Models & APIs: Stable Diffusion (local/API), DALL-E 3 (API), Google Imagen (API).
- Prompting for Image Generation: Style, content, negative prompts.
- **Lab:** Generate images using an API (e.g., OpenAI DALL-E) or a local setup (e.g., Diffusers library/Automatic1111).

### **Day 4: Audio Models**
- Speech-to-Text (STT): Transcribing audio to text (e.g., OpenAI Whisper).
- Text-to-Speech (TTS): Synthesizing speech from text (e.g., Google TTS, OpenAI TTS).
- APIs/Libraries for Audio Processing.
- **Lab:** Use Whisper API/library to transcribe an audio file; use a TTS API/library to generate speech from text.

### **Day 5: Project - Vision Analysis Application**
- Integrating Vision Models: Building a simple application using vision capabilities.
- Example Idea: Streamlit app that takes an image upload, analyzes it using Gemini/OpenAI Vision, and displays the description or answers user questions.
- Project Work Time & Q&A.
- **Lab:** Build a basic application (CLI or Streamlit) incorporating vision analysis from Day 1.

---

## **Week 7: AI Agents**

### **Day 1: Introduction to AI Agents**
- What are Agents?: LLMs that reason, plan, and interact with tools.
- Core Concepts: Reasoning loop (Observe, Think, Act), planning, memory.
- Agent Architectures: ReAct, Plan-and-Execute.
- Overview of Frameworks: LangChain Agents, OpenAI Assistants.

### **Day 2: Agents and Tools**
- Why Tools?: Extending agent capabilities (search, code execution, APIs).
- Defining Tools: Creating functions for agents to call.
- Tool Selection & Invocation: How agents use tools.
- Framework Integration: Creating and using tools.
- **Lab:** Create a simple custom tool (e.g., calculator) and integrate it with a basic agent.

### **Day 3: Model Context Protocol (MCP) Introduction**
- What is MCP?: A standard protocol for AI models/agents to interact with tools & resources.
- Core Concepts: Resources, Prompts, Tools, Sampling.
- Benefits: Interoperability between models, clients (editors, chat UIs), and tool servers.
- How it works: Client <-> Server communication (STDIO, SSE).
- Overview of SDKs (Python, TypeScript) & existing integrations (Cursor, LibreChat, etc.).
- **Lab:** Explore the MCP documentation (concepts, SDKs). Set up a simple MCP server using an SDK example (if feasible time-wise) or connect a client (like Cursor or Emacs) to a demo server.

### **Day 4: OpenAI Agents SDK**
- Overview: Lightweight framework for multi-agent workflows (distinct from Assistants API).
- Core Concepts: Agents (instructions, tools, guardrails), Handoffs (agent-to-agent transfer), Guardrails (validation), Tracing (debugging).
- Runner & Loop: `Runner.run()` executes the agent loop (LLM call -> process tools/handoffs -> repeat).
- Defining Tools: Using the `@function_tool` decorator.
- Compatibility: Works with models supporting OpenAI Chat Completions format.
- **Lab:** Install the SDK (`uv add openai-agents`). Run the provided examples (hello world, function tool, handoffs) and explore the basic agent loop.

### **Day 5: Project - Simple Research Agent**
- Goal: Build an agent that researches a topic using a web search tool.
- Components: LLM, Agent Framework (LangChain/Assistants), Search Tool (API wrapper).
- Workflow: Prompt -> Agent plans -> Agent calls search -> Agent synthesizes -> Response.
- **Lab:** Implement a basic research agent using a chosen framework and search tool.

---

## **Week 8: Enterprise System Development with Python and FastAPI**

### **Day 1: Advanced FastAPI & REST API Standards**
- FastAPI Features Recap: Dependency Injection, Background Tasks, Routers.
- RESTful API Design Principles: Resource naming, HTTP methods, status codes, HATEOAS.
- **Lab:** Refactor previous FastAPI examples using APIRouter.

### **Day 2: Database Integration with MongoDB & FastAPI**
- NoSQL Introduction: Document Databases (MongoDB) vs. SQL.
- MongoDB Basics: Documents, Collections, CRUD operations.
- Integrating MongoDB with FastAPI: Using libraries like `PyMongo`.
- Mapping Pydantic models to MongoDB documents.
- Asynchronous database operations in FastAPI endpoints.
- **Lab:** Connect a FastAPI application to MongoDB. Implement CRUD operations for a simple resource.

### **Day 3: Automated Testing with Pytest**
- Importance of Testing in Enterprise Applications.
- Testing Pyramid: Unit, Integration, End-to-End tests.
- Introduction to `pytest`: Writing test functions, assertions, fixtures.
- Unit Testing FastAPI Components: Testing Service layer logic, utility functions.
- Integration Testing: Testing interactions between layers (e.g., Service and Repository with a test database).
- Testing FastAPI Endpoints: Using `TestClient` for simulated HTTP requests.
- **Lab:** Write unit tests for the Service layer and integration/endpoint tests for the FastAPI application using `pytest` and `TestClient`.

### **Day 4: Containerization with Docker & Basic AWS Deployment**
- Introduction to Docker: Containers vs. VMs, Dockerfile basics.
- Containerizing a FastAPI Application: Writing a `Dockerfile`, building images.
- Docker Compose for multi-container setups (e.g., app + database).
- Cloud Deployment Concepts: Overview of AWS services (EC2, ECR, RDS/DocumentDB, potentially ECS/Fargate).
- Basic Deployment Strategy: Push Docker image to ECR, run container on an EC2 instance (or simple PaaS like App Runner).
- **Lab:** Create a `Dockerfile` for the FastAPI application, build the image, and optionally run it locally using Docker. Discuss steps for a basic AWS deployment.

### **Day 5: Recap and steps for further learning**
- **Recap:** Briefly review the key concepts covered throughout the week (Python, FastAPI, Databases, Docker, Cloud Basics).
- **Further Learning Suggestions for AI Engineers:**