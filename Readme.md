<div align="center">

<img src="https://capsule-render.vercel.app/api?type=waving&color=0:0d1117,40:0e2a45,100:0e75b6&height=260&section=header&text=Minahil%20Azaz&fontSize=76&fontColor=ffffff&fontAlignY=40&desc=AI%20Engineer%20%7C%20LLM%20Systems%20%7C%20RAG%20Pipelines%20%7C%20Agentic%20AI&descColor=58a6ff&descAlignY=60&animation=fadeIn" width="100%"/>

<img src="https://readme-typing-svg.herokuapp.com?font=JetBrains+Mono&weight=600&size=17&pause=1000&color=58A6FF&center=true&vCenter=true&width=750&lines=Building+production-grade+LLM+%26+ML+systems;RAG+Pipelines+%7C+ChromaDB+%7C+LangChain;Agentic+AI+%7C+LangGraph+%7C+Multi-Agent+Loops;Computer+Vision+%7C+Explainable+AI+%7C+XAI;Data+%E2%86%92+Model+%E2%86%92+Agent+%E2%86%92+API+%E2%86%92+Deploy" alt="Typing SVG" />

<br/>

![Open to Work](https://img.shields.io/badge/Open%20to%20Work-238636?style=flat-square&logo=checkmarx&logoColor=white)
&nbsp;
![MSc PhD](https://img.shields.io/badge/MSc%20%2F%20PhD%20Opportunities-0969DA?style=flat-square&logo=academia&logoColor=white)
&nbsp;
![Profile Views](https://komarev.com/ghpvc/?username=minahil-azaz&label=profile+views&color=58a6ff&style=flat-square)
&nbsp;
![Followers](https://img.shields.io/github/followers/minahil-azaz?label=followers&style=flat-square&color=8b949e)

</div>

---

## `$ whoami`

```python
class MinahilAzaz:
    location   = "Lahore, Pakistan"
    education  = "B.Sc. Computer Science — UET Lahore  |  GPA: 3.19/4.0"
    current    = "Full Stack AI Engineer @ RemoteDev Limited"
    prev       = ["AI Engineer @ Dot Republic Media", "GenAI Intern @ Xavor Corporation"]
    research   = "Lead Author — Springer Nature, ICCIS 2024"

    expertise  = [
        "LLM Systems & Prompt Engineering",
        "RAG Pipelines  (ChromaDB · FAISS · LangChain)",
        "Agentic AI     (LangGraph · Tool-use · Multi-step Reasoning)",
        "Computer Vision (PyTorch · OpenCV)",
        "Explainable AI  (SHAP · LIME · Grad-CAM · Attention)",
        "Full-Stack      (FastAPI · React · Django · Docker · AWS · GCP)",
    ]

    philosophy = "I don't just build models — I ship complete AI systems."
    goal       = "PhD in Explainable AI — healthcare & agriculture"
```

---

## 🏗️ System Architecture

```
┌──────────────────────────────────────────────────────────────────┐
│                   END-TO-END AI SYSTEM DESIGN                    │
├──────────────────────────────────────────────────────────────────┤
│                                                                  │
│   📦 Raw Data                                                    │
│       │                                                          │
│       ▼                                                          │
│   🔧 Preprocessing → 🧠 Model Training → 📊 Evaluation           │
│                                │                                 │
│               ┌────────────────┴───────────────────┐            │
│               ▼                                    ▼            │
│        📚 RAG Pipeline                    🔁 Agent Loop         │
│    Chunk → Embed → Store              Plan → Act → Reflect      │
│               │                                    │            │
│               └────────────────┬───────────────────┘            │
│                                ▼                                 │
│                      ⚡ FastAPI / REST Layer                      │
│                                │                                 │
│               ┌────────────────┴───────────────────┐            │
│               ▼                                    ▼            │
│        🌐 React Frontend                   ☁️ Cloud Deploy       │
│                                                                  │
└──────────────────────────────────────────────────────────────────┘
```

---

## 📚 RAG Pipeline

```
PDF / Text Corpus
      │
      ▼
PyMuPDFLoader
      │
      ▼
RecursiveCharacterTextSplitter  (chunk_size=500, overlap=50)
      │
      ▼
HuggingFaceEmbeddings  (all-MiniLM-L6-v2)
      │
      ▼
ChromaDB  →  Persistent Vector Store
      │
      ▼
Similarity Search  (k=4)
      │
      ▼
Mistral 7B Instruct  (HuggingFace Inference API)
      │
      ▼
Grounded Response  →  Citation-backed Answer
```

```python
# Requires: langchain-huggingface, langchain-chroma (modern imports)
from langchain_huggingface import HuggingFaceEmbeddings, HuggingFaceEndpoint
from langchain_chroma import Chroma
from langchain.chains import RetrievalQA
from langchain_community.document_loaders import PyMuPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter

class RAGPipeline:
    def __init__(self, pdf_path: str):
        docs     = PyMuPDFLoader(pdf_path).load()
        chunks   = RecursiveCharacterTextSplitter(
                       chunk_size=500, chunk_overlap=50
                   ).split_documents(docs)
        embedder = HuggingFaceEmbeddings(
                       model_name="sentence-transformers/all-MiniLM-L6-v2"
                   )
        self.db  = Chroma.from_documents(chunks, embedder)
        self.llm = HuggingFaceEndpoint(
                       repo_id="mistralai/Mistral-7B-Instruct-v0.3"
                   )

    def query(self, question: str) -> str:
        chain = RetrievalQA.from_chain_type(
            llm=self.llm,
            retriever=self.db.as_retriever(search_kwargs={"k": 4})
        )
        return chain.invoke({"query": question})["result"]
```

**Stack:** `LangChain` · `ChromaDB` · `PyMuPDF` · `HuggingFace Inference API` · `Mistral 7B Instruct`

---

## 🔁 Agentic AI — LangGraph

```
                      ┌──────────────┐
                      │   PLANNER    │  ← Decomposes topic into sub-goals
                      └──────┬───────┘
                             │
           ┌─────────────────┼─────────────────┐
           ▼                 ▼                 ▼
    ┌─────────────┐  ┌──────────────┐  ┌──────────────────┐
    │   SEARCHER  │  │  RETRIEVER   │  │   SYNTHESIZER    │
    │ Tavily API  │  │  ChromaDB    │  │  LLM Reasoning   │
    └──────┬──────┘  └──────┬───────┘  └────────┬─────────┘
           └─────────────────┼─────────────────┘
                             │
                      ┌──────▼───────┐
                      │   EXPORTER   │  ← WeasyPrint PDF + SSE Stream
                      └──────────────┘
```

```python
from langgraph.graph import StateGraph, END
from typing import TypedDict

class ResearchState(TypedDict):
    topic:             str
    plan:              list[str]
    search_results:    list[str]
    retrieved_context: list[str]
    final_report:      str

graph = StateGraph(ResearchState)
graph.add_node("planner",     planner_node)    # task decomposition
graph.add_node("searcher",    search_node)     # Tavily web search
graph.add_node("retriever",   retrieval_node)  # ChromaDB memory
graph.add_node("synthesizer", synthesize_node) # LLM writing
graph.add_node("exporter",    export_node)     # PDF + streaming

graph.set_entry_point("planner")
graph.add_edge("planner",     "searcher")
graph.add_edge("searcher",    "retriever")
graph.add_edge("retriever",   "synthesizer")
graph.add_edge("synthesizer", "exporter")
graph.add_edge("exporter",    END)

agent = graph.compile()
```

**Stack:** `LangGraph` · `FastAPI` · `React/Vite` · `ChromaDB` · `Tavily` · `WeasyPrint` · `SSE Streaming`

---

## ⚡ Tech Stack

<div align="center">

**AI / ML**

![Python](https://img.shields.io/badge/Python-3776AB?style=flat-square&logo=python&logoColor=white)
![PyTorch](https://img.shields.io/badge/PyTorch-EE4C2C?style=flat-square&logo=pytorch&logoColor=white)
![TensorFlow](https://img.shields.io/badge/TensorFlow-FF6F00?style=flat-square&logo=tensorflow&logoColor=white)
![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-F7931E?style=flat-square&logo=scikit-learn&logoColor=white)
![HuggingFace](https://img.shields.io/badge/HuggingFace-FFD21F?style=flat-square&logo=huggingface&logoColor=black)
![OpenCV](https://img.shields.io/badge/OpenCV-5C3EE8?style=flat-square&logo=opencv&logoColor=white)

**LLM / RAG / Agents**

![LangChain](https://img.shields.io/badge/LangChain-1C3C3C?style=flat-square&logo=langchain&logoColor=white)
![LangGraph](https://img.shields.io/badge/LangGraph-1C3C3C?style=flat-square&logo=langchain&logoColor=3fb950)
![ChromaDB](https://img.shields.io/badge/ChromaDB-FF6B35?style=flat-square&logo=databricks&logoColor=white)
![Tavily](https://img.shields.io/badge/Tavily-00B4D8?style=flat-square&logo=searchengin&logoColor=white)
![OpenAI](https://img.shields.io/badge/OpenAI-412991?style=flat-square&logo=openai&logoColor=white)

**Backend / Full Stack**

![FastAPI](https://img.shields.io/badge/FastAPI-009688?style=flat-square&logo=fastapi&logoColor=white)
![Django](https://img.shields.io/badge/Django-092E20?style=flat-square&logo=django&logoColor=white)
![React](https://img.shields.io/badge/React-61DAFB?style=flat-square&logo=react&logoColor=black)
![Node.js](https://img.shields.io/badge/Node.js-339933?style=flat-square&logo=node.js&logoColor=white)
![MongoDB](https://img.shields.io/badge/MongoDB-47A248?style=flat-square&logo=mongodb&logoColor=white)

**Cloud / DevOps**

![AWS](https://img.shields.io/badge/AWS-FF9900?style=flat-square&logo=amazonaws&logoColor=white)
![GCP](https://img.shields.io/badge/GCP-4285F4?style=flat-square&logo=googlecloud&logoColor=white)
![Azure](https://img.shields.io/badge/Azure-0089D6?style=flat-square&logo=microsoftazure&logoColor=white)
![Docker](https://img.shields.io/badge/Docker-2496ED?style=flat-square&logo=docker&logoColor=white)
![GitHub Actions](https://img.shields.io/badge/GitHub%20Actions-2088FF?style=flat-square&logo=githubactions&logoColor=white)

</div>

---

## 🧩 Core Competencies

| Domain | Details |
|--------|---------|
| 🔁 **Agentic AI** | LangGraph multi-step loops · tool-use · planner-executor · stateful graphs |
| 📚 **RAG Infrastructure** | PDF ingestion · semantic chunking · vector search · grounded generation |
| 🧠 **LLM Systems** | HuggingFace Transformers · fine-tuning · prompt optimization · evaluation |
| 👁️ **Computer Vision** | CNN classification · Siamese networks · object detection · segmentation |
| 🔍 **Explainable AI** | SHAP · LIME · Grad-CAM · attention visualization · feature attribution |
| ⚡ **AI Deployment** | FastAPI microservices · Docker · REST APIs · cloud (AWS · GCP · Azure) |

---

## 📝 Research & Publication

```
┌──────────────────────────────────────────────────────────────────────┐
│  📖  Springer Nature — Infosys Science Foundation Series            │
│                                                                      │
│  "Enhancing Information Security: Proactive Detection,              │
│   Vulnerability Remediation, and Systematic Risk Management"         │
│                                                                      │
│  Authors  :  Minahil Azaz  ·  Syeda Um e Farwa                     │
│  Venue    :  ICCIS 2024 — Computational Intelligent Systems         │
│  Pages    :  201–215  ·  First Online: January 29, 2026            │
│  DOI      :  10.1007/978-981-95-2212-5_14                          │
└──────────────────────────────────────────────────────────────────────┘
```

<div align="center">

[![Read on Springer](https://img.shields.io/badge/Read%20on%20Springer-00599C?style=flat-square&logo=springer&logoColor=white)](https://link.springer.com/chapter/10.1007/978-981-95-2212-5_14)

</div>

---

## 📊 GitHub Stats

<div align="center">

<img src="https://github-readme-stats.vercel.app/api?username=minahil-azaz&show_icons=true&theme=github_dark&hide_border=true&bg_color=0d1117&title_color=58a6ff&icon_color=3fb950&text_color=8b949e&count_private=true&include_all_commits=true" height="170"/>
&nbsp;
<img src="https://github-readme-stats.vercel.app/api/top-langs/?username=minahil-azaz&layout=compact&theme=github_dark&hide_border=true&bg_color=0d1117&title_color=58a6ff&text_color=8b949e&count_private=true" height="170"/>

<br/><br/>

<img src="https://streak-stats.demolab.com?user=minahil-azaz&theme=github-dark-blue&hide_border=true&background=0d1117&stroke=21262d&ring=58a6ff&fire=3fb950&currStreakLabel=58a6ff&sideNums=58a6ff&sideLabels=8b949e&dates=8b949e&starting_year=2024" width="55%"/>

</div>

---

## 🔥 Contribution Graph

<div align="center">

<img src="https://github-readme-activity-graph.vercel.app/graph?username=minahil-azaz&bg_color=0d1117&color=58a6ff&line=3fb950&point=ffffff&area=true&hide_border=true" width="100%"/>

</div>

---

## 🐍 Contribution Snake

<div align="center">

<picture>
  <source media="(prefers-color-scheme: dark)" srcset="https://raw.githubusercontent.com/minahil-azaz/minahil-azaz/output/github-contribution-grid-snake-dark.svg"/>
  <source media="(prefers-color-scheme: light)" srcset="https://raw.githubusercontent.com/minahil-azaz/minahil-azaz/output/github-contribution-grid-snake.svg"/>
  <img alt="contribution snake" src="https://raw.githubusercontent.com/minahil-azaz/minahil-azaz/output/github-contribution-grid-snake.svg"/>
</picture>

</div>

<details>
<summary>⚙️ Setup: Auto-generate the snake via GitHub Actions</summary>

Add `.github/workflows/snake.yml` to your profile repo to auto-generate the snake on a schedule.

</details>

---

## 🎯 Open To

<div align="center">

| Status | Role |
|:---:|:---|
| ✅ | **AI / ML Engineer** — LLM systems · RAG pipelines · production ML |
| ✅ | **MSc / PhD Research** — XAI · computer vision · applied ML |
| ✅ | **AI Product Collaborations** — agentic workflows · intelligent systems |

</div>

---

## 📬 Contact

<div align="center">

[![Gmail](https://img.shields.io/badge/minahilazaz45%40gmail.com-EA4335?style=flat-square&logo=gmail&logoColor=white)](mailto:minahilazaz45@gmail.com)
&nbsp;
[![LinkedIn](https://img.shields.io/badge/LinkedIn-0A66C2?style=flat-square&logo=linkedin&logoColor=white)](https://www.linkedin.com/in/minahil-azaz-397756241/)
&nbsp;
[![GitHub](https://img.shields.io/badge/GitHub-181717?style=flat-square&logo=github&logoColor=white)](https://github.com/minahil-azaz)

</div>

---

<div align="center">

<img src="https://capsule-render.vercel.app/api?type=waving&color=0:0e75b6,100:0d1117&height=120&section=footer&animation=fadeIn" width="100%"/>

```
Data → Model → RAG → Agent → API → Deploy → Scale
```

*AI Engineer focused on building scalable LLM systems, retrieval-augmented architectures, and agentic workflows.*

</div>
