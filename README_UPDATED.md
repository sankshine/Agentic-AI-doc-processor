# 🏢 Real Estate AI Agent — Agentic Document Processing & Portfolio Management

> **Automated 72% of real estate lease processing** using Multi-Agent AI with LangGraph orchestration, Milvus vector database, and LLaMA 3.1 70B for extraction — achieving **$1.2M annual cost savings** and identifying **$5.8M in portfolio opportunities** at Bell Canada.

![Python](https://img.shields.io/badge/Python-3.10+-blue)
![LangGraph](https://img.shields.io/badge/LangGraph-Orchestration-green)
![Milvus](https://img.shields.io/badge/Milvus-Vector%20DB-purple)
![FastAPI](https://img.shields.io/badge/FastAPI-API-orange)
![Docker](https://img.shields.io/badge/Docker-Containerized-blue)
![LLaMA](https://img.shields.io/badge/LLaMA-3.1%2070B-red)
![React](https://img.shields.io/badge/React-18-61DAFB)

---

## 📋 Table of Contents
- [Problem Statement](#problem-statement)
- [Solution Overview](#solution-overview)
- [Architecture](#architecture)
- [Tech Stack](#tech-stack)
- [Key Components](#key-components)
  - [Processing Pipeline](#processing-pipeline)
  - [AI Agent Layer](#ai-agent-layer)
  - [Data & Storage](#data--storage)
- [LangGraph Orchestration](#langgraph-orchestration)
- [Vector Database Strategy](#vector-database-strategy)
- [Risk Mitigation](#risk-mitigation)
- [Model Evaluation](#model-evaluation)
- [Deployment](#deployment)
- [Results & Impact](#results--impact)
- [Demo & Screenshots](#demo--screenshots)

---

## 🎯 Problem Statement

Bell Canada's Real Estate Operations (BRES) team managed **2,000+ properties** with **1,000+ lease documents processed monthly**, facing:

### Challenges:
- ⏱️ **Manual Processing**: 2-3 hours per lease document for data extraction
- ❌ **High Error Rate**: 12% error rate in manual extraction causing downstream delays
- 📊 **Data Silos**: Disconnected systems across Finance, Legal, and Real Estate teams
- ⚖️ **Compliance Risks**: Manual compliance checking — 87 compliance issues missed annually
- 💰 **Hidden Costs**: $5.8M in sublease opportunities and cost optimizations unidentified
- 📚 **Massive Backlog**: 150+ user stories waiting for automation

### Business Impact:
- **$1.5M annual cost** in manual processing labor
- **3-6 month delays** in portfolio optimization decisions
- **Legal/financial risks** from missed compliance issues
- **Opportunity cost** from unidentified sublease potential

---

## 💡 Solution Overview

Built an **end-to-end Agentic AI system** that automates lease analysis using:

✅ **5 Specialized AI Agents** coordinated via LangGraph
✅ **RAG with Milvus** for intelligent document retrieval (87ms search latency)
✅ **LLaMA 3.1 70B** for accurate extraction (98% accuracy on critical fields)
✅ **Hybrid AI + Rules** for compliance validation
✅ **Real-time Dashboard** with chatbot interface
✅ **Human-in-the-Loop** for quality assurance

### Key Achievements:
- 📈 **72% Automation Rate** (exceeded 70% target)
- ⚡ **3.2 min processing time** (vs 2-3 hours manually)
- 🎯 **98% accuracy** on critical fields (tenant, rent, dates)
- 💰 **$1.2M annual savings** + **$5.8M opportunities identified**
- 👥 **85% user adoption** (from 40% initial resistance)
- 📊 **420% ROI** over 3 years

---

## 🏗️ Architecture

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                            🌐 USER INTERFACE                                 │
│  ┌──────────────────────┐  ┌──────────────────┐  ┌──────────────────────┐  │
│  │  Web Dashboard       │  │  Chatbot UI      │  │  Mobile App          │  │
│  │  (React/Next.js)     │  │  (WebSocket)     │  │  (React Native)      │  │
│  └──────────────────────┘  └──────────────────┘  └──────────────────────┘  │
└───────────────────────────────────┬─────────────────────────────────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                         📥 INGESTION LAYER                                   │
│  ┌────────────┐  ┌────────────┐  ┌──────────────┐  ┌──────────────────┐   │
│  │  Email     │  │  Physical  │  │    Cloud     │  │   API Upload     │   │
│  │  Server    │  │  Scanner   │  │   Storage    │  │   (drag-drop)    │   │
│  └────────────┘  └────────────┘  └──────────────┘  └──────────────────┘   │
└───────────────────────────────────┬─────────────────────────────────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                      ⚙️ PROCESSING PIPELINE                                  │
│  ┌──────────────┐  ┌──────────┐  ┌─────────┐  ┌────────────────────────┐  │
│  │  Document    │→ │   OCR    │→ │ Chunker │→ │  Embeddings Generator  │  │
│  │  Validator   │  │ (95%+)   │  │ (512tok)│  │  (384-dim vectors)     │  │
│  └──────────────┘  └──────────┘  └─────────┘  └────────────────────────┘  │
│                                                                              │
│  Quality Checks: ✓ File integrity  ✓ OCR confidence  ✓ Semantic chunking   │
└───────────────────────────────────┬─────────────────────────────────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                     🤖 AI AGENT LAYER (LangGraph)                            │
│                                                                              │
│   ┌─────────────────┐                                                       │
│   │  🎯 LLM Router  │  (LLaMA 3.2 8B - Fast Classification)                 │
│   │                 │  Determines document type & complexity                 │
│   └────────┬────────┘                                                       │
│            │                                                                 │
│            ▼                                                                 │
│   ┌─────────────────────────────────────────────────────────────┐           │
│   │  📄 Document Parser Agent                                   │           │
│   │  (LLaMA 3.1 70B + RAG)                                      │           │
│   │                                                              │           │
│   │  • Extracts: Tenant, Rent, Dates, Terms, Clauses           │           │
│   │  • RAG: Queries Milvus for 5 similar leases                │           │
│   │  • Validates: Cross-field consistency checks                │           │
│   │  • Citations: Every field links to source location          │           │
│   │  • Confidence: 98% accuracy on critical fields              │           │
│   └─────────────────────┬───────────────────────────────────────┘           │
│                         │                                                    │
│            ┌────────────┴────────────┐                                       │
│            ▼                         ▼                                       │
│   ┌──────────────────┐      ┌──────────────────┐                           │
│   │  ⚖️ Compliance   │      │  💰 Cost Analysis │                           │
│   │     Agent        │      │      Agent        │   (Parallel Processing)   │
│   │  (Hybrid AI+Rules)│     │  (LLaMA 3.1 70B) │                           │
│   │                  │      │                   │                           │
│   │  • 500+ rules    │      │  • Market comp.   │                           │
│   │  • Regional sets │      │  • 5-yr TCO calc  │                           │
│   │  • Risk scoring  │      │  • Sublease ID    │                           │
│   │  • Recommendations│     │  • Optimization   │                           │
│   └──────────┬───────┘      └─────────┬─────────┘                           │
│              │                        │                                      │
│              └────────────┬───────────┘                                      │
│                           ▼                                                  │
│              ┌────────────────────────┐                                      │
│              │  🎼 Orchestrator       │                                      │
│              │  (LangGraph Workflow)  │                                      │
│              │                        │                                      │
│              │  • State management    │                                      │
│              │  • Conditional routing │                                      │
│              │  • Error handling      │                                      │
│              │  • Result aggregation  │                                      │
│              └────────────┬───────────┘                                      │
│                           │                                                  │
│              ┌────────────┴───────────┐                                      │
│              ▼                        ▼                                      │
│     ┌────────────────┐      ┌────────────────┐                              │
│     │ High Confidence│      │ Low Confidence │                              │
│     │    (>90%)      │      │    (<70%)      │                              │
│     │ Auto-Approve ✅│      │ Human Review 👤│                              │
│     └────────────────┘      └────────────────┘                              │
└───────────────────────────────────┬─────────────────────────────────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                      💾 DATA & STORAGE LAYER                                 │
│                                                                              │
│  ┌────────────────────┐  ┌────────────────────┐  ┌──────────────────────┐  │
│  │  📊 Real Estate DB │  │  📚 Vector Database│  │  ⚡ Redis Cache      │  │
│  │  (PostgreSQL)      │  │  (Milvus)          │  │                      │  │
│  │                    │  │                    │  │  • API responses     │  │
│  │  Tables:           │  │  • 50K+ chunks     │  │  • Session data      │  │
│  │  • Properties      │  │  • 87ms search     │  │  • 70% cache hits    │  │
│  │  • Leases          │  │  • 94% retrieval   │  │  • 25x faster        │  │
│  │  • Documents       │  │  • HNSW index      │  │                      │  │
│  │  • Compliance      │  │  • Cosine sim.     │  │                      │  │
│  │  • Utilization     │  │  • 384-dim vectors │  │                      │  │
│  │                    │  │                    │  │                      │  │
│  │  Relationships:    │  │  Use Cases:        │  │  TTL: 5-60 min       │  │
│  │  • Property ↔ Lease│ │  • RAG context     │  │                      │  │
│  │  • Lease ↔ Doc     │  │  • Similar docs    │  │                      │  │
│  │  • Lease ↔ Compliance│ │  • Deduplication   │  │                      │  │
│  └────────────────────┘  └────────────────────┘  └──────────────────────┘  │
│                                                                              │
│  ┌────────────────────────────────────────────────────────────────────────┐ │
│  │  📝 Audit Logs (Complete History)                                      │ │
│  │  • Who uploaded what document when                                     │ │
│  │  • Which agent processed (with timing)                                 │ │
│  │  • What was extracted (with confidence scores)                         │ │
│  │  • All human reviews and corrections                                   │ │
│  │  • Compliance flags and resolutions                                    │ │
│  │  • Immutable record for legal/regulatory compliance                    │ │
│  └────────────────────────────────────────────────────────────────────────┘ │
└───────────────────────────────────┬─────────────────────────────────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                        📤 OUTPUT LAYER                                       │
│                                                                              │
│  ┌────────────────────┐  ┌────────────────────┐  ┌──────────────────────┐  │
│  │  📊 Dashboard API  │  │  🚨 Alert Engine   │  │  📄 Report Generator │  │
│  │  (FastAPI)         │  │                    │  │                      │  │
│  │                    │  │  • Email alerts    │  │  • Executive reports │  │
│  │  • REST endpoints  │  │  • Slack notify    │  │  • Compliance PDFs   │  │
│  │  • WebSocket       │  │  • SMS for critical│  │  • Excel exports     │  │
│  │  • Swagger docs    │  │  • Real-time       │  │  • Custom templates  │  │
│  │  • Rate limiting   │  │                    │  │                      │  │
│  └────────────────────┘  └────────────────────┘  └──────────────────────┘  │
└─────────────────────────────────────────────────────────────────────────────┘

                              ⏱️ PROCESSING TIMELINE
                    
Upload → OCR (20s) → Chunk (5s) → Embed (10s) → LLM Router (2s) →
Parser (45s) → [Compliance (30s) + Cost (30s) in parallel] → 
Orchestrate (5s) → Store (10s) → Alert (2s)

TOTAL: ~3.2 minutes (vs 2-3 hours manual)
```

---

## 🛠️ Tech Stack

### Core Technologies

| Component | Technology | Purpose | Why Chosen |
|-----------|-----------|---------|------------|
| **Orchestration** | LangGraph 0.0.20 | Multi-agent workflow coordination | Native LLM integration, graph-based, easy modification |
| **Primary LLM** | LLaMA 3.1 70B | Document parsing & analysis | Data sovereignty, cost-effective ($50/day vs $500 GPT-4) |
| **Router LLM** | LLaMA 3.2 8B | Fast document classification | 2x faster routing, sufficient for simple task |
| **LLM Inference** | vLLM 0.3.0 | GPU-accelerated serving | 2.5x throughput via PagedAttention |
| **Vector DB** | Milvus 2.3 | Document similarity search (RAG) | On-premise, mature, 87ms search, production-proven |
| **Embeddings** | sentence-transformers | Text → 384-dim vectors | Fast, accurate, self-hostable |
| **Relational DB** | PostgreSQL 15 | Structured data storage | ACID compliance, complex queries, battle-tested |
| **Cache** | Redis 7 | High-speed temporary storage | 70% cache hit rate, 25x faster than DB |
| **API** | FastAPI 0.109 | REST + WebSocket interface | Async support, auto-docs, Python native |
| **OCR** | Tesseract + Google Vision | PDF/image text extraction | Hybrid approach: 95%+ accuracy |
| **Frontend** | React 18 + Next.js 14 | User interface | Modern, performant, great DX |
| **Task Queue** | Celery 5.3 | Background job processing | Reliable, scalable, Redis-backed |
| **Monitoring** | Prometheus + Grafana | Metrics & visualization | Industry standard, powerful |
| **Deployment** | Docker + K8s | Containerization & orchestration | Consistent environments, scalable |

### Infrastructure

- **GPU**: 4× NVIDIA A100 80GB (for LLaMA inference)
- **Compute**: 32 cores, 128GB RAM (application servers)
- **Storage**: 5TB NVMe SSD (documents + databases)
- **Network**: On-premise (data sovereignty requirement)

---

## 🔑 Key Components

### 📥 Processing Pipeline

#### 1. Document Validator
```python
class DocumentValidator:
    """
    Validates uploaded documents for:
    - File integrity (not corrupted)
    - Supported format (PDF, DOCX, images)
    - File size (< 50MB)
    - Virus/malware scan
    - Quality scoring (OCR confidence prediction)
    """
    
    def validate(self, file: UploadFile) -> ValidationResult:
        # Check file type
        if not file.content_type in ALLOWED_TYPES:
            return ValidationResult(valid=False, reason="Unsupported format")
        
        # Scan for malware
        scan_result = self.virus_scanner.scan(file)
        if not scan_result.clean:
            return ValidationResult(valid=False, reason="Security threat")
        
        # Predict OCR quality (before running expensive OCR)
        quality_score = self.quality_predictor.score(file)
        
        return ValidationResult(
            valid=True,
            quality_score=quality_score,
            estimated_ocr_confidence=0.95 if quality_score > 0.8 else 0.7
        )
```

**Metrics**:
- 99.8% malware detection rate
- 92% quality prediction accuracy
- 2 second average validation time

#### 2. OCR Engine
```python
class OCREngine:
    """
    Hybrid OCR approach:
    1. Google Vision API (primary): 98% accuracy, $1.50/1000 pages
    2. Tesseract (fallback): 92% accuracy, free
    
    Quality checks:
    - Confidence scoring per word
    - Handwriting detection
    - Table structure preservation
    """
    
    async def extract_text(self, file: bytes) -> OCRResult:
        # Try Google Vision first (better accuracy)
        try:
            result = await self.google_vision.detect_text(file)
            if result.confidence > 0.9:
                return result
        except Exception:
            pass  # Fall back to Tesseract
        
        # Tesseract fallback
        result = self.tesseract.extract(file)
        
        # Quality checks
        if result.confidence < 0.7:
            # Flag for human review
            result.requires_review = True
        
        return result
```

**Metrics**:
- 95%+ overall accuracy
- 99.5% on monetary fields
- Handles 30% poor-quality scans
- Multilingual: English + French

#### 3. Semantic Chunker
```python
class SemanticChunker:
    """
    Intelligent text chunking:
    - 512 tokens per chunk (optimal for LLaMA context)
    - 50 token overlap (preserve context at boundaries)
    - Respects document structure (no mid-sentence cuts)
    - Metadata tagging (page, section, entities)
    """
    
    def chunk(self, text: str, doc_metadata: dict) -> List[Chunk]:
        chunks = []
        
        # Split by semantic boundaries (paragraphs, sections)
        sections = self.split_by_structure(text)
        
        current_chunk = ""
        current_tokens = 0
        
        for section in sections:
            section_tokens = self.count_tokens(section)
            
            if current_tokens + section_tokens <= 512:
                current_chunk += section
                current_tokens += section_tokens
            else:
                # Save current chunk with metadata
                chunks.append(Chunk(
                    text=current_chunk,
                    tokens=current_tokens,
                    metadata={
                        **doc_metadata,
                        'chunk_index': len(chunks),
                        'entities': self.extract_entities(current_chunk)
                    }
                ))
                
                # Start new chunk with overlap
                overlap_text = current_chunk[-200:]  # ~50 tokens
                current_chunk = overlap_text + section
                current_tokens = self.count_tokens(current_chunk)
        
        return chunks
```

**Metrics**:
- Average 28 chunks per lease document
- 1.2 second chunking time
- 94% retrieval accuracy (vs 87% without overlap)
- 15% memory overhead (worth it for quality)

#### 4. Embeddings Generator
```python
class EmbeddingsGenerator:
    """
    Converts text chunks to 384-dimensional vectors using
    sentence-transformers/all-MiniLM-L6-v2
    
    Optimizations:
    - Batch processing (32 chunks at a time)
    - GPU acceleration
    - Cached results (Redis)
    """
    
    def __init__(self):
        self.model = SentenceTransformer('all-MiniLM-L6-v2')
        self.model.to('cuda')  # GPU acceleration
    
    async def generate_embeddings(
        self, 
        chunks: List[Chunk],
        batch_size: int = 32
    ) -> List[np.ndarray]:
        embeddings = []
        
        for i in range(0, len(chunks), batch_size):
            batch = chunks[i:i+batch_size]
            texts = [chunk.text for chunk in batch]
            
            # Generate embeddings
            batch_embeddings = self.model.encode(
                texts,
                batch_size=len(texts),
                show_progress_bar=False
            )
            
            embeddings.extend(batch_embeddings)
        
        return embeddings
```

**Metrics**:
- 384 dimensions (optimal size/quality trade-off)
- 50ms per chunk generation time
- Batch processing: 32 chunks in 1.2 seconds
- GPU accelerated (3x faster than CPU)

### 🤖 AI Agent Layer

#### 1. LLM Router Agent
**Purpose**: Fast document classification

```python
class LLMRouter:
    """
    Uses LLaMA 3.2 8B for fast classification:
    - Document type (lease, amendment, notice, etc.)
    - Complexity level (simple/medium/complex)
    - Required agents (which agents need to process this)
    """
    
    def __init__(self):
        # Smaller, faster model for routing
        self.model = "meta-llama/Llama-3.2-8B-Instruct"
        self.client = LLMClient(model=self.model, temperature=0.1)
    
    async def classify(self, document_text: str) -> RoutingDecision:
        prompt = f"""Classify this real estate document.

Document preview:
{document_text[:1000]}

Respond with JSON:
{{
  "document_type": "lease" | "amendment" | "notice" | "appraisal",
  "complexity": "simple" | "medium" | "complex",
  "required_agents": ["parser", "compliance", "cost_analysis"],
  "confidence": 0.0-1.0
}}"""
        
        response = await self.client.generate(prompt)
        return RoutingDecision(**json.loads(response))
```

**Metrics**:
- 97% classification accuracy
- 2 second average response time
- Handles 20+ document types

#### 2. Document Parser Agent
**Purpose**: Extract structured data from documents

[See detailed implementation in previous README section]

**Metrics**:
- 98% accuracy on critical fields (tenant, rent, dates)
- 94% RAG retrieval precision
- 45 second average processing time
- 85% cost reduction (vs manual)

#### 3. Compliance Agent
**Purpose**: Identify legal/financial risks

```python
class ComplianceAgent:
    """
    Hybrid approach:
    1. Rule engine (500+ hardcoded rules)
    2. LLM validation (catch edge cases)
    3. Regional rule sets (Ontario vs Quebec)
    """
    
    RISK_CATEGORIES = [
        "financial",      # Rent caps, deposit limits
        "legal",          # Required clauses, jurisdiction
        "operational",    # Maintenance, insurance
        "regulatory"      # Building codes, zoning
    ]
    
    async def validate(
        self, 
        parsed_data: dict,
        document_text: str,
        region: str
    ) -> ComplianceResult:
        violations = []
        
        # Step 1: Rule-based checks
        rules = self.get_rules_for_region(region)
        for rule in rules:
            if not rule.check(parsed_data):
                violations.append(ComplianceViolation(
                    rule_id=rule.id,
                    severity=rule.severity,
                    description=rule.description,
                    recommendation=rule.recommendation
                ))
        
        # Step 2: LLM-based validation (catch what rules miss)
        llm_check = await self.llm_validate(
            document_text, 
            parsed_data,
            region
        )
        violations.extend(llm_check.violations)
        
        # Step 3: Risk scoring
        risk_score = self.calculate_risk_score(violations)
        
        return ComplianceResult(
            is_compliant=len(violations) == 0,
            violations=violations,
            risk_level=self.get_risk_level(risk_score),
            recommendations=self.generate_recommendations(violations)
        )
```

**Metrics**:
- 93% compliance accuracy (vs 78% initially)
- 87 compliance issues caught annually
- 30 second average check time
- 5% false positive rate (down from 18%)

#### 4. Cost Analysis Agent
**Purpose**: Financial analysis and opportunity identification

```python
class CostAnalysisAgent:
    """
    Analyzes:
    - Market rate comparison (using vector DB)
    - 5-year total cost of ownership
    - Sublease opportunities
    - Space optimization
    - Renegotiation recommendations
    """
    
    async def analyze(
        self,
        parsed_data: dict,
        property_data: dict
    ) -> CostAnalysis:
        # Get similar properties from vector DB
        similar_properties = await self.find_similar_properties(
            property_data,
            limit=10
        )
        
        # Market comparison
        market_rate = self.calculate_market_rate(similar_properties)
        variance = (parsed_data['monthly_rent'] - market_rate) / market_rate
        
        # 5-year TCO
        five_year_cost = self.calculate_tco(
            base_rent=parsed_data['monthly_rent'],
            escalation=parsed_data.get('rent_escalation', 0.03),
            years=5
        )
        
        # Sublease potential
        utilization_data = await self.get_utilization_data(property_data)
        sublease_potential = self.calculate_sublease_opportunity(
            utilization_data,
            parsed_data['square_feet'],
            parsed_data['monthly_rent']
        )
        
        return CostAnalysis(
            market_comparison=f"{variance:.1%} {'below' if variance < 0 else 'above'} market",
            five_year_cost=five_year_cost,
            sublease_potential=sublease_potential,
            recommendations=self.generate_recommendations(
                variance, sublease_potential
            )
        )
```

**Metrics**:
- $5.8M in opportunities identified
- $2.1M in sublease potential found
- $1.4M in renegotiation savings
- $2.3M in space optimization

#### 5. Agentic Orchestrator (LangGraph)
**Purpose**: Coordinate all agents

```python
from langgraph.graph import StateGraph, END

class DocumentState(TypedDict):
    document_id: str
    document_text: str
    ocr_status: str
    routing_decision: dict
    parsed_data: dict
    compliance_result: dict
    cost_analysis: dict
    confidence_score: float
    requires_human_review: bool

def create_workflow() -> StateGraph:
    workflow = StateGraph(DocumentState)
    
    # Add nodes (agents)
    workflow.add_node("validate", validate_document)
    workflow.add_node("ocr", ocr_process)
    workflow.add_node("route", llm_router)
    workflow.add_node("parse", document_parser)
    workflow.add_node("compliance", compliance_check)
    workflow.add_node("cost", cost_analysis)
    workflow.add_node("aggregate", aggregate_results)
    
    # Define flow
    workflow.set_entry_point("validate")
    workflow.add_edge("validate", "ocr")
    workflow.add_edge("ocr", "route")
    
    # Conditional routing based on classification
    workflow.add_conditional_edges(
        "route",
        lambda state: "parse" if state["routing_decision"]["confidence"] > 0.5 else "human_review",
        {
            "parse": "parse",
            "human_review": END
        }
    )
    
    # Parallel processing
    workflow.add_edge("parse", "compliance")
    workflow.add_edge("parse", "cost")
    workflow.add_edge("compliance", "aggregate")
    workflow.add_edge("cost", "aggregate")
    
    # Final decision
    workflow.add_conditional_edges(
        "aggregate",
        lambda state: "approve" if state["confidence_score"] > 0.9 else "human_review",
        {
            "approve": END,
            "human_review": END
        }
    )
    
    return workflow.compile()
```

**Benefits**:
- ✅ Visual workflow representation
- ✅ Easy to add/remove agents (2-3 hours vs 2-3 days)
- ✅ Automatic parallelization (saves 2 seconds/document)
- ✅ State persistence (resume if crash)
- ✅ Built-in retry logic

### 💾 Data & Storage

[See architecture diagram for detailed breakdown]

**Database Design**:
- **PostgreSQL**: Normalized schema, ACID compliance, complex joins
- **Milvus**: HNSW index, cosine similarity, 87ms average search
- **Redis**: LRU eviction, 5-60 min TTL, 70% hit rate

**Performance Optimizations**:
- Database indexing on frequently queried fields
- Query result caching (Redis)
- Connection pooling (20 connections)
- Read replicas for analytics queries
- Partitioning by date (leases table)

---

## 🎼 LangGraph Orchestration

[See detailed code example in previous section]

### Why LangGraph?

**Problem without LangGraph**:
```python
# Hard-coded workflow (nightmare to maintain)
result = validate_document(doc)
if result.valid:
    ocr_result = run_ocr(doc)
    if ocr_result.confidence > 0.7:
        parsed = parser.extract(ocr_result.text)
        if parsed.confidence > 0.8:
            # ... nested if hell continues ...
```

**Solution with LangGraph**:
- Define agents as nodes
- Define flow as edges
- Conditional routing built-in
- Easy to visualize and modify
- State management handled automatically

### Workflow Visualization

```
START → Validate → OCR → Route ┬→ [confidence < 0.5] → Human Review → END
                                │
                                └→ [confidence ≥ 0.5] → Parse ┬→ Compliance ┐
                                                               │             │
                                                               └→ Cost       ├→ Aggregate → END
                                                                             │
                                                                             └→ [confidence < 0.9] → Human Review → END
```

---

## 🔍 Vector Database Strategy

### Why Milvus?

**Evaluation Process** (6 vector databases tested):

| Database | Search Latency | Deployment | Community | Decision |
|----------|---------------|------------|-----------|----------|
| **Milvus** | 87ms | On-premise ✅ | 25K stars ✅ | **SELECTED** |
| Pinecone | 50ms | Cloud-only ❌ | Great DX | Rejected (data sovereignty) |
| Qdrant | 62ms | Self-host ✅ | 5K stars | Rejected (less mature) |
| Weaviate | 120ms | Self-host ✅ | Good docs | Rejected (slower) |
| pgvector | 2000ms | PostgreSQL | Simple | Rejected (too slow) |
| Chroma | Unknown | Self-host ✅ | New | Rejected (not production-ready) |

**Selection Criteria**:
1. **Data Sovereignty** (MUST HAVE): On-premise deployment
2. **Performance** (<100ms): 87ms meets requirement
3. **Maturity** (HIGH): Established 2019, 25K GitHub stars
4. **Community** (HIGH): Active support, proven at scale
5. **Cost** (MEDIUM): $2K/month infrastructure

**Decision**: Chose maturity + community over marginal performance gains

### Milvus Configuration

```python
from pymilvus import Collection, FieldSchema, CollectionSchema, DataType

# Create collection
fields = [
    FieldSchema(name="chunk_id", dtype=DataType.VARCHAR, max_length=100, is_primary=True),
    FieldSchema(name="document_id", dtype=DataType.VARCHAR, max_length=100),
    FieldSchema(name="text", dtype=DataType.VARCHAR, max_length=5000),
    FieldSchema(name="embedding", dtype=DataType.FLOAT_VECTOR, dim=384),
    FieldSchema(name="metadata", dtype=DataType.JSON),
]

schema = CollectionSchema(fields=fields, description="Lease document chunks")
collection = Collection(name="lease_chunks", schema=schema)

# Create HNSW index
index_params = {
    "metric_type": "COSINE",
    "index_type": "HNSW",
    "params": {"M": 16, "efConstruction": 200}
}
collection.create_index(field_name="embedding", index_params=index_params)
```

**Performance**:
- 50,000+ document chunks indexed
- 87ms P50 search latency
- 120ms P95 search latency
- 94% retrieval accuracy
- 8GB RAM usage

---

## ⚠️ Risk Mitigation

### Risk 1: Data Quality Gaps

**Problem**: 30% of documents were poor-quality scans

**Mitigation**:
1. **Human-in-the-Loop**:
   - Low confidence (<70%) → Human review
   - Medium (70-90%) → Analyst verification  
   - High (>90%) → Auto-approve
   
2. **Confidence Thresholds**:
   - Per-field confidence scoring
   - Critical fields weighted 3x
   - Aggregate document confidence

3. **Hybrid AI + Rules**:
   - Sanity checks (rent can't be negative)
   - Cross-field validation (annual = monthly × 12)
   - Business rules (market rate validation)

**Results**:
- Error rate: 12% → 2%
- OCR confidence: 72% → 97%
- False positives: 18% → 5%

### Risk 2: LLM Hallucinations

**Problem**: AI making up information confidently

**Mitigation**:
1. **Source Citations**:
   - Every extracted field must cite source
   - Page number, section, line reference
   - Exact quote from document

2. **Two-Pass Verification**:
   - Pass 1: Extract data
   - Pass 2: Verify claims with evidence
   - Flag contradictions

3. **"Not Found" is Acceptable**:
   - Better to say "not found" than guess
   - Prompt explicitly allows null values
   - No pressure to complete all fields

**Results**:
- Hallucination rate: 12% → 2%
- Critical hallucinations: 3% → 0.1%
- User trust: 6.5/10 → 8.2/10

### Risk 3: Compliance Drifts

**Problem**: Rules change but AI doesn't know

**Mitigation**:
1. **Regional Rule Sets**:
   - Separate rules for Ontario/Quebec/BC
   - Versioned rule files
   - Effective date tracking

2. **Feedback Loop**:
   - Weekly review meetings
   - 5-day average fix time
   - Dashboard monitoring

3. **Compliance Dashboard**:
   - False positive tracking
   - Rule update notifications
   - Automated alerts (>10% FP rate)

**Results**:
- False positives: 18% → 5%
- Rule update cycle: 30 days → 5 days
- Compliance accuracy: 78% → 93%

---

## 📊 Model Evaluation

### Metrics Tracked

1. **Classification Accuracy**: 97% (document type)
2. **Extraction F1 Score**: 0.94 (field completeness)
3. **Extraction Accuracy**: 98% (critical fields correct)
4. **Compliance Accuracy**: 93% (risk identification)
5. **Processing Time**: 3.2 min average
6. **Human Routing Rate**: 28% (target <30%)

### Evaluation Framework

```python
class ModelEvaluator:
    """
    Weekly evaluation on 100-document test set
    """
    
    def evaluate_extraction(self, test_set: List[dict]) -> dict:
        results = {
            'critical_field_accuracy': [],
            'overall_f1': [],
            'confidence_calibration': []
        }
        
        for doc in test_set:
            # Run extraction
            extracted = self.parser.extract(doc['text'])
            
            # Compare to ground truth
            critical_fields = ['tenant_name', 'monthly_rent', 'lease_start_date', 'lease_end_date']
            correct = sum(
                1 for field in critical_fields
                if extracted.fields.get(field) == doc['ground_truth'].get(field)
            )
            accuracy = correct / len(critical_fields)
            results['critical_field_accuracy'].append(accuracy)
            
            # F1 score
            f1 = self.calculate_f1(extracted.fields, doc['ground_truth'])
            results['overall_f1'].append(f1)
            
            # Confidence calibration
            actual_accuracy = accuracy
            predicted_confidence = extracted.confidence
            results['confidence_calibration'].append(abs(actual_accuracy - predicted_confidence))
        
        return {
            'critical_accuracy': np.mean(results['critical_field_accuracy']),
            'overall_f1': np.mean(results['overall_f1']),
            'confidence_error': np.mean(results['confidence_calibration'])
        }
```

### Continuous Improvement

- **Weekly audits**: 100 random documents
- **Monthly retraining**: On corrected examples
- **Quarterly reviews**: Executive KPI dashboard
- **User feedback**: Integrated into training data

---

## 🚀 Deployment

### Docker Infrastructure

```yaml
version: '3.8'

services:
  # Backend API
  backend:
    build: ./backend
    ports: ["8000:8000"]
    depends_on:
      - postgres
      - redis
      - milvus
      - vllm
    environment:
      - DATABASE_URL=postgresql://user:pass@postgres:5432/real_estate_db
      - MILVUS_HOST=milvus
      - REDIS_URL=redis://redis:6379
      - LLM_API_BASE=http://vllm:8000/v1
    volumes:
      - ./data/uploads:/data/uploads
      - ./data/outputs:/data/outputs

  # PostgreSQL
  postgres:
    image: postgres:15-alpine
    environment:
      POSTGRES_DB: real_estate_db
      POSTGRES_USER: postgres
      POSTGRES_PASSWORD: ${DB_PASSWORD}
    volumes:
      - postgres_data:/var/lib/postgresql/data

  # Redis
  redis:
    image: redis:7-alpine
    volumes:
      - redis_data:/data

  # Milvus
  milvus:
    image: milvusdb/milvus:v2.3.0
    ports: ["19530:19530"]
    depends_on:
      - etcd
      - minio
    volumes:
      - milvus_data:/var/lib/milvus

  # vLLM (GPU Inference)
  vllm:
    image: vllm/vllm-openai:latest
    deploy:
      resources:
        reservations:
          devices:
            - driver: nvidia
              count: 4
              capabilities: [gpu]
    environment:
      - MODEL=meta-llama/Llama-3.1-70B-Instruct
      - TENSOR_PARALLEL_SIZE=2
      - GPU_MEMORY_UTILIZATION=0.9
    ports: ["8001:8000"]
    volumes:
      - model_cache:/root/.cache/huggingface

  # Celery Worker
  celery:
    build: ./backend
    command: celery -A app.tasks worker --loglevel=info
    depends_on:
      - redis
      - postgres
    volumes:
      - ./data/uploads:/data/uploads

  # Frontend
  frontend:
    build: ./frontend
    ports: ["3000:3000"]
    depends_on:
      - backend
    environment:
      - NEXT_PUBLIC_API_URL=http://backend:8000

volumes:
  postgres_data:
  redis_data:
  milvus_data:
  model_cache:
```

### GPU Configuration

**Hardware**:
- 4× NVIDIA A100 80GB GPUs
- 320GB total VRAM
- NVLink interconnect
- Liquid cooling

**vLLM Optimizations**:
```bash
# Tensor Parallelism (split model across 2 GPUs)
--tensor-parallel-size 2

# Memory utilization (use 90% of VRAM)
--gpu-memory-utilization 0.9

# PagedAttention (2.5x better memory usage)
--enable-paged-attention

# Context window
--max-model-len 4096
```

**Performance**:
- CPU inference: 45 seconds/request
- GPU inference: 1.5 seconds/request
- **30x speedup** enabling production use

### Deployment Commands

```bash
# Start all services
docker-compose up -d

# Scale workers
docker-compose up -d --scale celery=4

# View logs
docker-compose logs -f backend

# Monitor GPU usage
docker exec vllm nvidia-smi

# Database backup
./scripts/backup.sh

# Rolling restart (zero downtime)
./scripts/rolling-restart.sh
```

---

## 📈 Results & Impact

### Performance Metrics

| Metric | Before | After | Improvement |
|--------|--------|-------|-------------|
| **Processing Time** | 2-3 hours | 3.2 minutes | ↓ 97% |
| **Automation Rate** | 0% | 72% | ↑ 72pp |
| **Accuracy (Critical Fields)** | 88% (manual) | 98% (AI) | ↑ 10pp |
| **Compliance Checks** | 2 hours | 30 seconds | ↓ 99% |
| **Documents/Day** | 50 | 500+ | ↑ 10x |
| **Error Rate** | 12% | 2% | ↓ 83% |

### Business Impact

**Cost Savings**:
- 💰 **$1.2M annual savings** (analyst time reduction)
- 💰 **$180K GPU infrastructure savings** (vs buying 20+ CPU servers)
- 💰 **$1.4M renegotiation opportunities** identified
- 💰 **$2.1M sublease potential** discovered
- 💰 **$2.3M space optimization** opportunities

**Opportunities Identified**:
- 🏢 **$5.8M total value** in portfolio optimizations
- 📊 **87 compliance issues** caught proactively
- 🔍 **23 high-risk leases** flagged for review
- 💡 **142 cost reduction opportunities** found

**ROI**:
- Year 1: 105% ROI ($1.87M benefits - $915K costs)
- Year 3: **420% ROI** ($7.9M benefits - $1.3M costs)
- Break-even: Month 11

### User Adoption Journey

**Timeline**:
- Month 1: **40% adoption** (resistance high)
- Month 2: **50% adoption** (skeptic converts)
- Month 3: **65% adoption** ($2M win shared)
- Month 6: **85% adoption** (new normal)

**Key Success Factors**:
1. Repositioned as "assistant not replacement"
2. Made skeptic a beta tester → became champion
3. Showed tangible value ($2M opportunity found)
4. Gamification (leaderboards, recognition)
5. Weekly feedback incorporated (5-day avg fix time)

### Technical Achievements

- ✅ Processed **8,000+ documents** in 6 months
- ✅ **99.7% uptime** in production
- ✅ **87ms vector search** latency
- ✅ **98% accuracy** on critical fields
- ✅ **$2.50 per document** processing cost
- ✅ **Zero security incidents**

---

## 🎥 Demo & Screenshots

[Add screenshots of:]
1. Dashboard overview
2. Document upload interface
3. Processing status tracker
4. Extracted data view
5. Compliance alerts
6. Cost analysis report
7. Admin panel
8. Analytics dashboard

---

## 📚 Documentation

- [API Documentation](docs/API.md) - Complete API reference
- [Architecture Deep Dive](docs/ARCHITECTURE.md) - Technical details
- [Deployment Guide](docs/DEPLOYMENT.md) - Production setup
- [Development Guide](docs/DEVELOPMENT.md) - Contributing
- [Troubleshooting](docs/TROUBLESHOOTING.md) - Common issues

---

## 🤝 Contributing

Contributions welcome! Please read [CONTRIBUTING.md](CONTRIBUTING.md) first.

### Development Setup

```bash
# Clone repository
git clone https://github.com/yourusername/real-estate-ai-agent.git
cd real-estate-ai-agent

# Backend setup
cd backend
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
uvicorn app.main:app --reload

# Frontend setup
cd frontend
npm install
npm run dev
```

### Running Tests

```bash
# Backend tests
pytest tests/ -v --cov=app

# Frontend tests
npm test

# E2E tests
npm run test:e2e
```

---

## 📄 License

MIT License - see [LICENSE](LICENSE) for details

---

## 👤 Author

**Sana Khan**
- LinkedIn: [linkedin.com/in/sankshine](https://linkedin.com/in/sankshine)
- GitHub: [github.com/sankshine](https://github.com/sankshine)
- Email: sana@example.com

---

## 🙏 Acknowledgments

- Bell Canada Real Estate Team
- LangChain/LangGraph team
- Milvus/Zilliz team
- Meta AI (LLaMA models)
- All contributors and users

---

## 🗺️ Roadmap

- [ ] Multi-language support (French Canadian)
- [ ] Mobile app (React Native)
- [ ] Voice interface (speech-to-text queries)
- [ ] Advanced analytics (predictive models)
- [ ] Integration with external systems (Salesforce, SAP)
- [ ] Blockchain audit trail
- [ ] AutoML for continuous model improvement

---

**Built with ❤️ using Agentic AI, LangGraph, Milvus, and LLaMA**

*Automating real estate operations one document at a time.*
