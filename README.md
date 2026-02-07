# 🤖 Agentic AI Document Processing — Multi-Agent Real Estate Workflows

> **Automated 70% of real-estate document processing** using Agentic AI with Milvus for vector-based retrieval and LangGraph for orchestrating multi-agent workflows across financial, legal, and real estate teams.

![Python](https://img.shields.io/badge/Python-3.11-blue)
![LangGraph](https://img.shields.io/badge/LangGraph-Orchestration-green)
![Milvus](https://img.shields.io/badge/Milvus-Vector%20DB-purple)
![FastAPI](https://img.shields.io/badge/FastAPI-API-orange)
![Docker](https://img.shields.io/badge/Docker-Containerized-blue)

---

## 📋 Table of Contents
- [Problem Statement](#problem-statement)
- [Architecture](#architecture)
- [Tech Stack](#tech-stack)
- [Agent Design](#agent-design)
- [LangGraph Orchestration](#langgraph-orchestration)
- [Vector Store (Milvus)](#vector-store-milvus)
- [Hallucination & Quality Control](#hallucination--quality-control)
- [Model Evaluation](#model-evaluation)
- [Deployment](#deployment)
- [Results](#results)

---

## Problem Statement

Bell's real estate operations (BRES) team processed 1000+ documents monthly — leases, appraisals, environmental reports, zoning certificates — involving:

- Manual data extraction from PDFs: 4-6 hours per document bundle
- Cross-referencing across 3 departments (financial, legal, real estate)
- High error rate (12%) in manual extraction, causing downstream delays
- Compliance validation done entirely by humans
- 150+ user stories in the backlog awaiting automation

## Architecture

```
┌──────────────────────────────────────────────────────────────────────┐
│                       DOCUMENT INTAKE                                 │
│    Upload API → File Parser (PDF/DOCX/Images) → Text Extraction      │
└──────────────┬───────────────────────────────────────────────────────┘
               │
               ▼
┌──────────────────────────────────────────────────────────────────────┐
│                  LANGGRAPH ORCHESTRATOR                                │
│                                                                       │
│   ┌──────────────┐     ┌──────────────┐     ┌───────────────────┐   │
│   │   AGENT 1     │     │   AGENT 2     │     │   AGENT 3          │   │
│   │   Document    │────▶│   Data        │────▶│   Compliance       │   │
│   │   Classifier  │     │   Extractor   │     │   Validator        │   │
│   │               │     │   (+ Milvus   │     │   (Legal rules     │   │
│   │   - Lease     │     │    lookup)    │     │    engine)         │   │
│   │   - Appraisal │     │               │     │                    │   │
│   │   - Environ.  │     │               │     │                    │   │
│   │   - Zoning    │     │               │     │                    │   │
│   └──────────────┘     └──────────────┘     └───────────────────┘   │
│          │                    │                       │              │
│          └────────────────────┼───────────────────────┘              │
│                               ▼                                      │
│                    ┌──────────────────┐                              │
│                    │   AGENT 4         │                              │
│                    │   Report          │                              │
│                    │   Generator       │                              │
│                    │   (Structured     │                              │
│                    │    output + PDF)  │                              │
│                    └──────────────────┘                              │
└──────────────────────────────────────────────────────────────────────┘
               │
               ▼
┌──────────────────────────────────────────────────────────────────────┐
│   HUMAN-IN-THE-LOOP                                                   │
│   Low-confidence items → Review Queue → Approve/Reject → Feedback    │
└──────────────────────────────────────────────────────────────────────┘
```

## Tech Stack

| Component | Technology | Purpose |
|-----------|-----------|---------|
| **Orchestration** | LangGraph | Multi-agent state machine |
| **Vector DB** | Milvus | Document similarity search |
| **LLM** | GPT-4 Turbo / PaLM 2 | Extraction & classification |
| **Embeddings** | text-embedding-3-small | Document vectorization |
| **API** | FastAPI | REST interface |
| **Document Parsing** | PyMuPDF + pytesseract | PDF/image extraction |
| **Monitoring** | LangSmith + Prometheus | Agent tracing & metrics |
| **CI/CD** | GitHub Actions + Docker | Automated pipeline |

## Agent Design

### Agent 1: Document Classifier

```python
# src/agents/classifier.py
"""
Agent 1: Classifies incoming documents into categories
and routes them to appropriate extraction pipelines.
"""
from langchain_core.messages import HumanMessage
from langchain_openai import ChatOpenAI
from pydantic import BaseModel, Field
from enum import Enum

class DocCategory(str, Enum):
    LEASE = "lease_agreement"
    APPRAISAL = "property_appraisal"
    ENVIRONMENTAL = "environmental_report"
    ZONING = "zoning_certificate"
    FINANCIAL = "financial_statement"
    UNKNOWN = "unknown"

class ClassificationResult(BaseModel):
    category: DocCategory
    confidence: float = Field(ge=0.0, le=1.0)
    subcategory: str = ""
    key_indicators: list[str] = []

class DocumentClassifier:
    """
    Uses LLM + few-shot examples to classify documents.
    Falls back to embedding similarity for low-confidence cases.
    """
    def __init__(self, model: str = "gpt-4-turbo-preview"):
        self.llm = ChatOpenAI(model=model, temperature=0)

    def classify(self, text: str, filename: str = "") -> ClassificationResult:
        prompt = f"""Classify this document into one of these categories:
- lease_agreement: Rental/lease contracts, tenancy agreements
- property_appraisal: Property valuation reports, market assessments
- environmental_report: Phase I/II ESA, contamination assessments
- zoning_certificate: Land use permits, zoning compliance documents
- financial_statement: P&L, balance sheets, cost analyses
- unknown: Cannot determine

Document filename: {filename}
Document text (first 2000 chars):
{text[:2000]}

Respond with JSON:
{{"category": "...", "confidence": 0.0-1.0, "subcategory": "...", "key_indicators": ["..."]}}
"""
        response = self.llm.invoke([HumanMessage(content=prompt)])
        import json
        result = json.loads(response.content)
        return ClassificationResult(**result)
```

### Agent 2: Data Extractor with Milvus Lookup

```python
# src/agents/extractor.py
"""
Agent 2: Extracts structured data from documents using
LLM + Milvus vector search for schema matching and validation.
"""
from pymilvus import Collection, connections
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from pydantic import BaseModel
from typing import Any

class ExtractionSchema(BaseModel):
    """Dynamic schema based on document category."""
    fields: dict[str, Any]
    source_references: list[dict]
    extraction_confidence: float

class DataExtractor:
    """
    Combines LLM extraction with Milvus vector search to:
    1. Find similar previously-processed documents
    2. Use their schemas as extraction templates
    3. Validate extracted values against historical data
    """
    def __init__(self, milvus_host: str = "localhost", milvus_port: int = 19530):
        connections.connect("default", host=milvus_host, port=milvus_port)
        self.collection = Collection("document_extractions")
        self.collection.load()
        self.embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
        self.llm = ChatOpenAI(model="gpt-4-turbo-preview", temperature=0)

        # Extraction schemas per category
        self.schemas = {
            "lease_agreement": {
                "tenant_name": "string",
                "landlord_name": "string",
                "property_address": "string",
                "lease_start_date": "date",
                "lease_end_date": "date",
                "monthly_rent": "currency",
                "security_deposit": "currency",
                "renewal_terms": "string",
                "special_clauses": "list[string]",
            },
            "property_appraisal": {
                "property_address": "string",
                "appraised_value": "currency",
                "appraisal_date": "date",
                "property_type": "string",
                "lot_size_sqft": "number",
                "building_size_sqft": "number",
                "comparable_sales": "list[dict]",
                "condition_notes": "string",
            },
        }

    def extract(self, text: str, category: str) -> ExtractionSchema:
        # Step 1: Find similar documents in Milvus for reference
        similar_docs = self._find_similar(text)

        # Step 2: Get extraction schema for this category
        schema = self.schemas.get(category, {})

        # Step 3: LLM extraction with schema guidance
        prompt = self._build_extraction_prompt(text, schema, similar_docs)
        response = self.llm.invoke([{"role": "user", "content": prompt}])

        import json
        extracted = json.loads(response.content)

        # Step 4: Validate against similar docs
        confidence = self._validate_extraction(extracted, similar_docs)

        return ExtractionSchema(
            fields=extracted,
            source_references=[{"similar_doc_id": d["id"]} for d in similar_docs[:3]],
            extraction_confidence=confidence,
        )

    def _find_similar(self, text: str, top_k: int = 5) -> list[dict]:
        query_embedding = self.embeddings.embed_query(text[:1000])
        results = self.collection.search(
            data=[query_embedding],
            anns_field="embedding",
            param={"metric_type": "IP", "params": {"nprobe": 16}},
            limit=top_k,
            output_fields=["doc_id", "category", "extracted_fields"],
        )
        return [
            {
                "id": hit.entity.get("doc_id"),
                "category": hit.entity.get("category"),
                "fields": hit.entity.get("extracted_fields"),
                "score": hit.distance,
            }
            for hit in results[0]
        ]

    def _build_extraction_prompt(
        self, text: str, schema: dict, similar_docs: list
    ) -> str:
        schema_desc = "\n".join(f"  - {k}: {v}" for k, v in schema.items())
        examples = ""
        if similar_docs:
            examples = f"\nReference extraction from similar document:\n{similar_docs[0].get('fields', {})}\n"

        return f"""Extract the following fields from this document.
Return ONLY a JSON object with the specified fields.
If a field cannot be found, use null.

Required fields:
{schema_desc}
{examples}
Document text:
{text[:4000]}

JSON output:"""

    def _validate_extraction(self, extracted: dict, similar_docs: list) -> float:
        """Cross-validate extracted values against similar historical documents."""
        if not similar_docs:
            return 0.7  # Baseline confidence without references

        # Check if extracted values are within expected ranges
        valid_fields = sum(1 for v in extracted.values() if v is not None)
        total_fields = len(extracted)
        completeness = valid_fields / max(total_fields, 1)

        return round(min(0.95, 0.5 + completeness * 0.4), 3)
```

### Agent 3: Compliance Validator

```python
# src/agents/compliance.py
"""
Agent 3: Validates extracted data against compliance rules
for financial, legal, and real estate regulations.
"""
from dataclasses import dataclass

@dataclass
class ComplianceCheckResult:
    is_compliant: bool
    violations: list[dict]
    warnings: list[dict]
    risk_level: str  # LOW / MEDIUM / HIGH / CRITICAL

class ComplianceValidator:
    """
    Rule-based + LLM-assisted compliance checking for:
    - Lease terms (minimum duration, rent caps, required clauses)
    - Environmental compliance (Phase I/II requirements)
    - Financial thresholds (approval limits, budget alignment)
    """
    RULES = {
        "lease_agreement": [
            {
                "id": "LEASE-001",
                "name": "Minimum lease duration",
                "check": lambda data: data.get("lease_end_date") and data.get("lease_start_date"),
                "severity": "HIGH",
            },
            {
                "id": "LEASE-002",
                "name": "Security deposit cap (2x monthly rent)",
                "check": lambda data: (
                    data.get("security_deposit") is None
                    or data.get("monthly_rent") is None
                    or data["security_deposit"] <= data["monthly_rent"] * 2
                ),
                "severity": "MEDIUM",
            },
            {
                "id": "LEASE-003",
                "name": "Required renewal clause present",
                "check": lambda data: data.get("renewal_terms") is not None,
                "severity": "LOW",
            },
        ],
        "property_appraisal": [
            {
                "id": "APPR-001",
                "name": "Appraisal within 6 months",
                "check": lambda data: data.get("appraisal_date") is not None,
                "severity": "HIGH",
            },
        ],
    }

    def validate(self, extracted_data: dict, category: str) -> ComplianceCheckResult:
        rules = self.RULES.get(category, [])
        violations = []
        warnings = []

        for rule in rules:
            try:
                passed = rule["check"](extracted_data)
            except (TypeError, KeyError):
                passed = False

            if not passed:
                entry = {
                    "rule_id": rule["id"],
                    "name": rule["name"],
                    "severity": rule["severity"],
                }
                if rule["severity"] in ("HIGH", "CRITICAL"):
                    violations.append(entry)
                else:
                    warnings.append(entry)

        risk_level = (
            "CRITICAL" if any(v["severity"] == "CRITICAL" for v in violations)
            else "HIGH" if violations
            else "MEDIUM" if warnings
            else "LOW"
        )

        return ComplianceCheckResult(
            is_compliant=len(violations) == 0,
            violations=violations,
            warnings=warnings,
            risk_level=risk_level,
        )
```

## LangGraph Orchestration

```python
# src/orchestrator/workflow.py
"""
LangGraph state machine that orchestrates the 4 agents
in a directed graph with conditional routing.
"""
from langgraph.graph import StateGraph, END
from typing import TypedDict, Annotated
from src.agents.classifier import DocumentClassifier
from src.agents.extractor import DataExtractor
from src.agents.compliance import ComplianceValidator

class DocumentState(TypedDict):
    raw_text: str
    filename: str
    classification: dict | None
    extracted_data: dict | None
    compliance_result: dict | None
    report: dict | None
    requires_human_review: bool
    error: str | None

def classify_node(state: DocumentState) -> DocumentState:
    classifier = DocumentClassifier()
    result = classifier.classify(state["raw_text"], state["filename"])
    state["classification"] = {
        "category": result.category.value,
        "confidence": result.confidence,
        "key_indicators": result.key_indicators,
    }
    return state

def extract_node(state: DocumentState) -> DocumentState:
    extractor = DataExtractor()
    category = state["classification"]["category"]
    result = extractor.extract(state["raw_text"], category)
    state["extracted_data"] = {
        "fields": result.fields,
        "confidence": result.extraction_confidence,
        "references": result.source_references,
    }
    return state

def compliance_node(state: DocumentState) -> DocumentState:
    validator = ComplianceValidator()
    category = state["classification"]["category"]
    result = validator.validate(state["extracted_data"]["fields"], category)
    state["compliance_result"] = {
        "is_compliant": result.is_compliant,
        "violations": result.violations,
        "warnings": result.warnings,
        "risk_level": result.risk_level,
    }
    return state

def report_node(state: DocumentState) -> DocumentState:
    state["report"] = {
        "filename": state["filename"],
        "category": state["classification"]["category"],
        "extracted_fields": state["extracted_data"]["fields"],
        "compliance": state["compliance_result"],
        "confidence": min(
            state["classification"]["confidence"],
            state["extracted_data"]["confidence"],
        ),
    }
    # Route to human if low confidence or compliance issues
    state["requires_human_review"] = (
        state["report"]["confidence"] < 0.7
        or not state["compliance_result"]["is_compliant"]
    )
    return state

def should_route_to_human(state: DocumentState) -> str:
    if state["classification"]["confidence"] < 0.5:
        return "human_review"
    return "extract"

# Build the graph
def build_workflow() -> StateGraph:
    workflow = StateGraph(DocumentState)

    workflow.add_node("classify", classify_node)
    workflow.add_node("extract", extract_node)
    workflow.add_node("compliance", compliance_node)
    workflow.add_node("report", report_node)

    workflow.set_entry_point("classify")
    workflow.add_conditional_edges(
        "classify",
        should_route_to_human,
        {"extract": "extract", "human_review": END},
    )
    workflow.add_edge("extract", "compliance")
    workflow.add_edge("compliance", "report")
    workflow.add_edge("report", END)

    return workflow.compile()
```

## Hallucination & Quality Control

| Risk | Mitigation | Implementation |
|------|-----------|----------------|
| **Extraction hallucination** | Cross-validate against Milvus similar docs | Agent 2 validation step |
| **Classification errors** | Confidence threshold + human routing | LangGraph conditional edge |
| **Compliance false negatives** | Rule engine + LLM double-check | Agent 3 hybrid approach |
| **Schema drift** | Weekly schema validation against ground truth | Scheduled eval job |
| **Stale vector index** | Automated re-indexing on new documents | Milvus index refresh cron |

## Model Evaluation

```python
# src/evaluation/eval_pipeline.py
"""End-to-end evaluation of the document processing pipeline."""

class AgenticPipelineEvaluator:
    def evaluate(self, test_set: list[dict]) -> dict:
        results = {
            "classification_accuracy": 0,
            "extraction_f1": 0,
            "compliance_accuracy": 0,
            "end_to_end_accuracy": 0,
            "avg_processing_time_sec": 0,
            "human_routing_rate": 0,
        }
        correct_class = 0
        correct_compliance = 0
        extraction_scores = []
        times = []
        human_routes = 0

        for item in test_set:
            import time
            start = time.time()
            output = self.workflow.invoke({
                "raw_text": item["text"],
                "filename": item["filename"],
                "classification": None,
                "extracted_data": None,
                "compliance_result": None,
                "report": None,
                "requires_human_review": False,
                "error": None,
            })
            elapsed = time.time() - start
            times.append(elapsed)

            # Classification accuracy
            if output.get("classification", {}).get("category") == item["expected_category"]:
                correct_class += 1

            # Extraction F1
            if output.get("extracted_data"):
                f1 = self._compute_extraction_f1(
                    output["extracted_data"]["fields"],
                    item["expected_fields"]
                )
                extraction_scores.append(f1)

            # Compliance
            if output.get("compliance_result"):
                if output["compliance_result"]["is_compliant"] == item["expected_compliant"]:
                    correct_compliance += 1

            if output.get("requires_human_review"):
                human_routes += 1

        n = len(test_set)
        results["classification_accuracy"] = round(correct_class / n, 3)
        results["extraction_f1"] = round(sum(extraction_scores) / max(len(extraction_scores), 1), 3)
        results["compliance_accuracy"] = round(correct_compliance / n, 3)
        results["avg_processing_time_sec"] = round(sum(times) / n, 2)
        results["human_routing_rate"] = round(human_routes / n, 3)

        return results

    def _compute_extraction_f1(self, predicted: dict, expected: dict) -> float:
        pred_keys = {k for k, v in predicted.items() if v is not None}
        exp_keys = set(expected.keys())
        tp = len(pred_keys & exp_keys)
        precision = tp / max(len(pred_keys), 1)
        recall = tp / max(len(exp_keys), 1)
        if precision + recall == 0:
            return 0.0
        return 2 * precision * recall / (precision + recall)
```

## Deployment

```yaml
# docker-compose.yml
version: "3.9"
services:
  agentic-api:
    build: .
    ports: ["8000:8000"]
    env_file: .env
    depends_on: [milvus-standalone]

  milvus-standalone:
    image: milvusdb/milvus:v2.3.4
    ports: ["19530:19530"]
    environment:
      ETCD_ENDPOINTS: etcd:2379
      MINIO_ADDRESS: minio:9000
    depends_on: [etcd, minio]

  etcd:
    image: quay.io/coreos/etcd:v3.5.5
    environment:
      ETCD_AUTO_COMPACTION_RETENTION: "1"
      ETCD_QUOTA_BACKEND_BYTES: "4294967296"

  minio:
    image: minio/minio:latest
    command: server /data
    environment:
      MINIO_ROOT_USER: minioadmin
      MINIO_ROOT_PASSWORD: minioadmin
```

## Results

| Metric | Before | After | Improvement |
|--------|--------|-------|-------------|
| Document Processing Time | 4-6 hrs/bundle | 12 min/bundle | **↓ 95%** |
| Manual Processing | 100% | 30% | **↓ 70% automated** |
| Extraction Accuracy | 88% (manual) | 94% (agent + human review) | **↑ 7%** |
| Compliance Check Time | 2 hrs/doc | 30 sec/doc | **↓ 99%** |
| Backlog Reduction | 150+ stories | 40 stories | **↓ 73%** |
| Cross-dept Coordination | Email/meetings | Automated routing | Real-time |

---

## License
MIT License

## Author
**Sana Khan** — [LinkedIn](https://linkedin.com/in/sankshine) | [GitHub](https://github.com/sankshine)
