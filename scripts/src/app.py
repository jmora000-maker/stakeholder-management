import json
import os
from datetime import date
from pathlib import Path
import numpy as np
import pandas as pd
import streamlit as st
from openai import OpenAI
from pydantic import BaseModel, Field, ConfigDict
from typing import List, Dict, Set, Optional
import contextlib
import re

# --- PATH & ENVIRONMENT SETUP ---
today_obj = date.today()
today = today_obj.strftime("%B %d, %Y")

src_dir = Path(__file__).resolve().parent
data_folder = src_dir / "data"
vector_store_folder = src_dir / "vector_store"
output_folder = src_dir / "outputs"

for folder in [data_folder, vector_store_folder, output_folder]:
    folder.mkdir(parents=True, exist_ok=True)

stakeholder_gap_report_path = output_folder / "STAKEHOLDER_GAP_REPORT.txt"
database_file_destination = vector_store_folder / "global_vector_store.json"

meeting_notes_path = data_folder / "Meeting_Notes.md"
stakeholder_plan_path = data_folder / "Stakeholder_Engagement_Plan.md"
stakeholder_register_path = data_folder / "Stakeholder_Register.csv"

# Populate sample data if files do not exist
if not stakeholder_register_path.exists():
    pd.DataFrame([
        {"name": "Priya Sharma", "role": "Technical Lead", "influence": "High", "interest": "High",
         "desired_engagement": "Manage Closely"},
        {"name": "Helen Brooks", "role": "Compliance Director", "influence": "High", "interest": "Medium",
         "desired_engagement": "Keep Satisfied"},
        {"name": "David Vance", "role": "Operations Manager", "influence": "Medium", "interest": "High",
         "desired_engagement": "Keep Informed"}
    ]).to_csv(stakeholder_register_path, index=False)

if not meeting_notes_path.exists():
    meeting_notes_path.write_text("""# Project Sync Notes
Date: June 15, 2026
Attendees: Priya Sharma, David Vance, Liam Patel, Fatima Al-Sayed

## Discussion
- Priya Sharma raised repeated concerns regarding audit logging and access control architectures.
- Liam Patel coordinated infrastructure targets. Note: Liam is driving the vendor handoff framework.
- David Vance noted deep anxieties regarding training readiness and frontline deployment schedules.
- Fatima Al-Sayed flagged missing support scripts and overall enablement blockages.""", encoding="utf-8")

if not stakeholder_plan_path.exists():
    stakeholder_plan_path.write_text("""# Stakeholder Engagement Strategy
- Priya Sharma: Host tailored tech architectural readouts and align on access control mitigation structures. Owner: Tech PM. Cadence: Bi-weekly.
- Helen Brooks: Compliance engagement will be handled through existing governance channels.
- David Vance: Provide weekly hypercare visibility dashboards. Owner: Ops Lead.""", encoding="utf-8")

# --- INITIALIZE OPENAI CLIENT ---
api_key = os.environ.get("OPENAI_API_KEY", "mock-key-for-local-ui-safety")
is_vector_search_enabled = os.environ.get("OPENAI_API_KEY") is not None and os.environ.get(
    "OPENAI_API_KEY") != "mock-key-for-local-ui-safety"
client = OpenAI(api_key=api_key)


# --- UTILITY TO CAPTURE STDOUT ---
class StreamlitStdoutRedirector:
    def __init__(self, placeholder):
        self.placeholder = placeholder
        self.output_str = ""

    def write(self, text):
        self.output_str += text
        self.placeholder.code(self.output_str, language="text")

    def flush(self):
        pass


# --- INTERNAL FACT DOMAIN SCHEMAS ---
class Stakeholder(BaseModel):
    name: str
    role: str
    influence: str
    interest: str
    desired_engagement: str


class Concern(BaseModel):
    description: str
    stakeholder_name: str
    normalized_category: str
    severity: str
    source_artifact: str
    line_number: int
    snippet: str


class EngagementAction(BaseModel):
    action_strategy: str
    stakeholder_name: str
    has_owner: bool
    has_cadence: bool
    source_artifact: str
    line_number: int
    snippet: str


class MeetingMention(BaseModel):
    stakeholder_name: str
    context_snippet: str
    source_artifact: str
    line_number: int
    is_explicit_attendee: bool


class GapFinding(BaseModel):
    """Middle-tier model decoupling raw structural facts from LLM synthesis narrative"""
    gap_category: str
    stakeholder_name: str
    severity: str
    confidence: str
    observed_gap: str
    practical_impact: str
    recommended_action: str
    primary_deterministic_evidence: List[str]
    vector_evidence_queries: List[str]


# --- OUTWARD REVENUE-GRADE REPORT SCHEMAS ---
class EvidenceItem(BaseModel):
    source: str
    snippet: str


class StakeholderGapReport(BaseModel):
    gap_category: str = Field(
        description="The gap category in ALL CAPS: MISSING STAKEHOLDER, STRATEGIC EXECUTION GAP, or RECURRENT CONCERN MISMATCH.")
    stakeholder_name: str
    severity: str
    confidence: str
    observed_gap: str
    practical_impact: str
    recommended_action: str
    evidence: List[EvidenceItem]


class ExecutiveStakeholderGapReport(BaseModel):
    model_config = ConfigDict(extra="forbid")
    executive_summary: str = Field(
        description="2-3 professional, highly grounded program governance executive summary sentences.")
    categories: List[StakeholderGapReport]


# --- PIPELINE LAYER: CORE UTILITIES ---
def chunk_text(text: str, chunk_size: int = 300, overlap: int = 50) -> List[str]:
    if overlap >= chunk_size:
        overlap = chunk_size // 2
    words = text.split()
    chunks = []
    for i in range(0, len(words), chunk_size - overlap):
        chunk = " ".join(words[i:i + chunk_size])
        if chunk:
            chunks.append(chunk)
    return chunks


def get_embedding(text: str) -> List[float]:
    if not is_vector_search_enabled:
        return []
    try:
        response = client.embeddings.create(model="text-embedding-3-small", input=str(text))
        return response.data[0].embedding
    except Exception:
        return []


def cosine_similarity(v1: List[float], v2: List[float]) -> float:
    if not v1 or not v2:
        return 0.0
    a, b = np.array(v1), np.array(v2)
    norm = np.linalg.norm(a) * np.linalg.norm(b)
    return float(np.dot(a, b) / norm) if norm > 0 else 0.0


# --- PIPELINE LAYER: VECTOR ENGINE ---
class SimpleVectorStore:
    def __init__(self):
        self.entries = []

    def build_indices(self, chunks: List[Dict]):
        print(" -> Building vector indices...")
        print(f" -> Found {len(chunks)} chunks to index.")
        self.entries = []
        if not is_vector_search_enabled:
            return
        for chunk in chunks:
            embedding = get_embedding(chunk["text"])
            if embedding:
                self.entries.append({**chunk, "embedding": embedding})

    def search(self, query: str, top_k: int = 2) -> List[Dict]:
        if not is_vector_search_enabled or not self.entries:
            return []
        query_vector = get_embedding(query)
        if not query_vector:
            return []
        scored = []
        for entry in self.entries:
            sim = cosine_similarity(query_vector, entry["embedding"])
            scored.append((sim, entry))
        scored.sort(reverse=True, key=lambda x: x[0])
        return [{"source": item[1]["metadata"]["source"], "snippet": item[1]["text"]} for item in scored[:top_k]]


# --- PIPELINE LAYER: TAXONOMY & NORMALIZATION ---
class CorporateTaxonomyNormalizer:
    """Provides advanced multi-entity normalization for both identity variations and concern classifications."""

    def __init__(self):
        self.identity_map = {
            "head of finance": "Maria Chen",
            "finance director": "Maria Chen",
            "alicia": "Alicia Sponsor",
            "liam": "Liam Patel",
            "priya": "Priya Sharma",
            "david": "David Vance",
            "helen": "Helen Brooks",
            "fatima": "Fatima Al-Sayed"
        }

        self.concern_taxonomy = {
            "audit logging": ["audit logging", "audit log", "logging controls", "access control",
                              "security architecture"],
            "training readiness": ["training readiness", "deployment schedules", "anxieties", "frontline deployment",
                                   "enablement"],
            "support infrastructure": ["support scripts", "vendor handoff", "infrastructure targets",
                                       "enablement blockages", "scripts"]
        }

    def normalize_name(self, name: str) -> str:
        cleaned = str(name).strip().lower().replace("-", " ")
        cleaned = re.sub(r'[:\s*•\-\d)]', ' ', cleaned)
        cleaned = " ".join(cleaned.split())
        if not cleaned:
            return ""
        if cleaned in self.identity_map:
            return self.identity_map[cleaned]
        for alias, canonical in self.identity_map.items():
            if alias in cleaned or cleaned in alias:
                return canonical
        return name.title()

    def classify_concern(self, text: str) -> str:
        lowered = text.lower()
        for category, triggers in self.concern_taxonomy.items():
            if any(trigger in lowered for trigger in triggers):
                return category
        return "general operational friction"


# --- PIPELINE LAYER: ARTIFACT-SPECIFIC INGESTION & STRUCTURAL FACTS ENGINE ---
class StructuredProjectContext:
    """Dynamic Document Content Ingestion executing explicit parsers with dense metadata population."""

    def __init__(self, normalizer: CorporateTaxonomyNormalizer):
        self.normalizer = normalizer
        self.stakeholders: Dict[str, Stakeholder] = {}
        self.concerns: List[Concern] = []
        self.engagement_actions: List[EngagementAction] = []
        self.meeting_mentions: List[MeetingMention] = []
        self.discovered_names: Set[str] = set()
        self.raw_chunks: List[Dict] = []

    def ingest_data(self):
        # Phase 0: Dynamic Discovery Sweep Across Registers
        if stakeholder_register_path.exists():
            print(" -> Ingesting stakeholder register...")
            df = pd.read_csv(stakeholder_register_path)
            for _, row in df.iterrows():
                n = str(row.get("name", "")).strip()
                if n: self.discovered_names.add(self.normalizer.normalize_name(n))

        if stakeholder_plan_path.exists():
            print(" -> Ingesting stakeholder engagement plan...")
            for line in stakeholder_plan_path.read_text(encoding="utf-8").split("\n"):
                if line.strip().startswith("-") and ":" in line:
                    n = line.split(":", 1)[0].replace("-", "").strip()
                    if n: self.discovered_names.add(self.normalizer.normalize_name(n))

        if meeting_notes_path.exists():
            print(" -> Ingesting stakeholder meeting notes...")
            for line in meeting_notes_path.read_text(encoding="utf-8").split("\n"):
                if "attendees:" in line.lower():
                    for name_part in line.split(":", 1)[1].split(","):
                        n = name_part.strip()
                        if n: self.discovered_names.add(self.normalizer.normalize_name(n))

        # 1. PARSER SPECIFIC: Stakeholder Register (CSV Domain Model Engine)
        if stakeholder_register_path.exists():
            print(" -> Ingesting stakeholder register...")
            df = pd.read_csv(stakeholder_register_path)
            for idx, row in df.iterrows():
                raw_name = str(row.get("name", "")).strip()
                if not raw_name: continue
                norm_name = self.normalizer.normalize_name(raw_name)
                self.stakeholders[norm_name] = Stakeholder(
                    name=norm_name,
                    role=str(row.get("role", "Unknown")),
                    influence=str(row.get("influence", "Medium")),
                    interest=str(row.get("interest", "Medium")),
                    desired_engagement=str(row.get("desired_engagement", "Keep Informed"))
                )
                self.raw_chunks.append({
                    "text": f"Register Row entry target: {raw_name} with role {row.get('role')}",
                    "metadata": {"source": "Stakeholder_Register.csv", "type": "Register", "owner": norm_name,
                                 "line": idx}
                })

        # 2. PARSER SPECIFIC: Engagement Strategy (Strict Bullet Syntax)
        if stakeholder_plan_path.exists():
            print(" -> Ingesting stakeholder engagement plan...")
            plan_lines = stakeholder_plan_path.read_text(encoding="utf-8").split("\n")
            for idx, line in enumerate(plan_lines):
                if line.strip().startswith("-") and ":" in line:
                    parts = line.split(":", 1)
                    raw_target = parts[0].replace("-", "").strip()
                    norm_target = self.normalizer.normalize_name(raw_target)
                    strategy_text = parts[1].strip()

                    has_owner = any(x in strategy_text.lower() for x in ["owner:", "lead", "pm"])
                    has_cadence = any(
                        x in strategy_text.lower() for x in ["cadence:", "weekly", "monthly", "bi-weekly"])

                    self.engagement_actions.append(EngagementAction(
                        action_strategy=strategy_text,
                        stakeholder_name=norm_target,
                        has_owner=has_owner,
                        has_cadence=has_cadence,
                        source_artifact="Stakeholder_Engagement_Plan.md",
                        line_number=idx + 1,
                        snippet=line.strip()
                    ))
            for chunk in chunk_text("\n".join(plan_lines)):
                self.raw_chunks.append({
                    "text": chunk,
                    "metadata": {"source": "Stakeholder_Engagement_Plan.md", "type": "Strategy",
                                 "section": "Execution Plan"}
                })

        # 3. PARSER SPECIFIC: Meeting Notes (Context-Rich Sweep)
        if meeting_notes_path.exists():
            print(" -> Ingesting stakeholder meeting notes...")
            notes_lines = meeting_notes_path.read_text(encoding="utf-8").split("\n")
            is_attendee_line = False

            for idx, line in enumerate(notes_lines):
                lowered_line = line.lower()
                if "attendees:" in lowered_line:
                    is_attendee_line = True

                for target_name in self.discovered_names:
                    if target_name.lower() in lowered_line:
                        is_att = is_attendee_line and (target_name.lower() in lowered_line.split("attendees:")[-1])

                        self.meeting_mentions.append(MeetingMention(
                            stakeholder_name=target_name,
                            context_snippet=line.strip(),
                            source_artifact="Meeting_Notes.md",
                            line_number=idx + 1,
                            is_explicit_attendee=is_att
                        ))

                        if any(k in lowered_line for k in
                               ["concern", "anxiety", "risk", "issue", "flagged", "stalled", "vulnerability"]):
                            category = self.normalizer.classify_concern(line)
                            severity = "High" if "architecture" in lowered_line or "blockage" in lowered_line else "Medium"
                            self.concerns.append(Concern(
                                description=line.replace("-", "").strip(),
                                stakeholder_name=target_name,
                                normalized_category=category,
                                severity=severity,
                                source_artifact="Meeting_Notes.md",
                                line_number=idx + 1,
                                snippet=line.strip()
                            ))
                if line.strip() == "":
                    is_attendee_line = False

            for chunk in chunk_text("\n".join(notes_lines)):
                self.raw_chunks.append({
                    "text": chunk,
                    "metadata": {"source": "Meeting_Notes.md", "type": "Notes", "section": "Discussion Logs"}
                })


# --- PIPELINE LAYER: DETERMINISTIC AUDIT RULES ENGINE ---
class GapDetector:
    """Runs programmatic validations generating decoupled GapFinding models grounded in structural evidence."""

    def __init__(self, context: StructuredProjectContext, store: SimpleVectorStore):
        self.context = context
        self.store = store

    def execute_audit_checks(self) -> List[GapFinding]:
        print(f" -> Found {len(self.context.raw_chunks)} chunks to analyze.")
        print(f" -> Found {len(self.context.meeting_mentions)} mentions to analyze.")
        print(f" -> Found {len(self.context.concerns)} concerns to analyze.")
        print(f" -> Found {len(self.context.engagement_actions)} engagement actions to analyze.")
        print(f" -> Found {len(self.context.stakeholders)} stakeholders to analyze.")
        findings: List[GapFinding] = []

        # --- RULE 1: MISSING STAKEHOLDER ---
        mentioned_names = {m.stakeholder_name for m in self.context.meeting_mentions}
        registered_names = set(self.context.stakeholders.keys())

        for name in mentioned_names:
            if name not in registered_names and name:
                mentions = [m for m in self.context.meeting_mentions if m.stakeholder_name == name]
                primary_evidence = [f"[{m.source_artifact} Line {m.line_number}]: '{m.context_snippet}'" for m in
                                    mentions]

                is_attendee = any(m.is_explicit_attendee for m in mentions)
                has_substance = any(any(c.stakeholder_name == name for c in self.context.concerns) for m in mentions)

                if len(mentions) > 1 and is_attendee:
                    confidence = "High"
                elif is_attendee or has_substance:
                    confidence = "Medium"
                else:
                    confidence = "Low"

                findings.append(GapFinding(
                    gap_category="MISSING STAKEHOLDER",
                    stakeholder_name=name,
                    severity="High",
                    confidence=confidence,
                    observed_gap=f"Stakeholder '{name}' is driving program parameters or listed in meeting minutes but has zero representation in the official Stakeholder Register.",
                    practical_impact=f"Cross-functional governance lines risk severe communications disruption and unmonitored scope drift due to omission of technical or executive authority.",
                    recommended_action=f"Add '{name}' to the official Stakeholder Register, designate an internal team owner, and define communication boundaries.",
                    primary_deterministic_evidence=primary_evidence,
                    vector_evidence_queries=[f"{name} meeting involvement role architecture"]
                ))

        # --- RULE 2: STRATEGIC EXECUTION GAP ---
        for name, stakeholder in self.context.stakeholders.items():
            if stakeholder.influence.lower() == "high":
                actions = [a for a in self.context.engagement_actions if a.stakeholder_name == name]

                is_gap = False
                reasons = []
                primary_evidence = []

                if not actions:
                    is_gap = True
                    reasons.append("has no actionable engagement entries within the strategy files")
                    primary_evidence = [
                        f"Stakeholder Register record confirms High-Influence status for '{stakeholder.name}'."]
                else:
                    primary_evidence = [f"[{a.source_artifact} Line {a.line_number}]: '{a.snippet}'" for a in actions]
                    combined_text = " ".join([a.action_strategy.lower() for a in actions])

                    if any(x in combined_text for x in ["existing governance", "generic channels", "vague"]):
                        is_gap = True
                        reasons.append(
                            "relies completely on non-specific, passive management channels ('existing governance channels')")

                    missing_controls = []
                    if not any(a.has_owner for a in actions):
                        missing_controls.append("lacks an explicit strategy action owner")
                    if not any(a.has_cadence for a in actions):
                        missing_controls.append("omits an operational execution cadence")

                    if missing_controls:
                        is_gap = True
                        reasons.append(f"contains gaps in programmatic rigor ({', '.join(missing_controls)})")

                if is_gap:
                    anomaly_desc = f"High-influence project owner '{stakeholder.name}' " + " and ".join(reasons) + "."
                    findings.append(GapFinding(
                        gap_category="STRATEGIC EXECUTION GAP",
                        stakeholder_name=stakeholder.name,
                        severity="High",
                        confidence="High" if not actions else "Medium",
                        observed_gap=anomaly_desc,
                        practical_impact="Stewardship misalignment risks sudden approval blockages, delivery delays, and critical failure in strategic consensus tracking.",
                        recommended_action=f"Refine strategy mapping entries for '{stakeholder.name}' to insert concrete bi-weekly 1-on-1 reviews with clear owners assigned.",
                        primary_deterministic_evidence=primary_evidence,
                        vector_evidence_queries=[f"{stakeholder.name} strategy framework engagement coverage plan"]
                    ))

        # --- RULE 3: RECURRENT CONCERN MISMATCH ---
        for concern in self.context.concerns:
            name = concern.stakeholder_name
            actions = [a for a in self.context.engagement_actions if a.stakeholder_name == name]

            has_coverage = False
            for action in actions:
                action_target_category = self.context.normalizer.classify_concern(action.action_strategy)
                if action_target_category == concern.normalized_category and action_target_category != "general operational friction":
                    has_coverage = True
                    break

            if not has_coverage:
                primary_evidence = [f"[{concern.source_artifact} Line {concern.line_number}]: '{concern.snippet}'"]
                findings.append(GapFinding(
                    gap_category="RECURRENT CONCERN MISMATCH",
                    stakeholder_name=name,
                    severity=concern.severity,
                    confidence="High",
                    observed_gap=f"Active risk vector classification ('{concern.normalized_category.upper()}') flagged by '{name}' has zero tracking or mitigation steps inside current engagement plans.",
                    practical_impact="Repetitive technical or architecture friction points compound systemic administrative drag and degrade deployment velocity.",
                    recommended_action=f"Formally bridge this gap by mapping the '{concern.normalized_category}' concern to a verified milestone deliverable stream.",
                    primary_deterministic_evidence=primary_evidence,
                    vector_evidence_queries=[f"{name} {concern.normalized_category} risk tracking mitigation"]
                ))

        return findings


# --- ORCHESTRATION LAYER ---
def run_automated_pipeline() -> str:
    print("PIPELINE STARTED")

    print("STEP 1: Normalizing Stakeholders and Concerns")
    normalizer = CorporateTaxonomyNormalizer()

    print("STEP 2: Ingesting Raw Data")
    context = StructuredProjectContext(normalizer=normalizer)
    context.ingest_data()

    print("STEP 3: Building Vector Index")
    store = SimpleVectorStore()
    store.build_indices(context.raw_chunks)

    print("STEP 4: Executing Gap Detection")
    detector = GapDetector(context, store)
    internal_findings = detector.execute_audit_checks()

    print("STEP 5: Synthesizing Executive Stakeholder Gap Report")
    compiled_raw_payload = []
    for f in internal_findings:
        print(f" -> Found {len(f.primary_deterministic_evidence)} deterministic evidence items.")
        print(f" -> Found {len(f.vector_evidence_queries)} vector evidence queries.")
        vector_support = store.search(f.vector_evidence_queries[0], top_k=1)

        combined_evidence_items = []
        for pe in f.primary_deterministic_evidence:
            combined_evidence_items.append({"source": "Deterministic Fact Log", "snippet": pe})
        for vs in vector_support:
            combined_evidence_items.append({"source": f"RAG Context [{vs['source']}]", "snippet": vs["snippet"]})

        if not combined_evidence_items:
            combined_evidence_items.append({"source": "System Record",
                                            "snippet": "Calculated structural omission based on database log evaluation."})

        compiled_raw_payload.append({
            "gap_category": f.gap_category,
            "stakeholder_name": f.stakeholder_name,
            "severity": f.severity,
            "confidence": f.confidence,
            "observed_gap": f.observed_gap,
            "practical_impact": f.practical_impact,
            "recommended_action": f.recommended_action,
            "evidence": combined_evidence_items
        })

    print(" -> Generating Executive Stakeholder Gap Report")
    if is_vector_search_enabled:
        try:
            response = client.beta.chat.completions.parse(
                model="gpt-4o-mini",
                messages=[
                    {"role": "system",
                     "content": "You are an expert Senior Project Leader and Complexity Translator. Synthesize structural program audit rules and deterministic gaps into crisp corporate summaries. Do not extrapolate data outside the verified findings structure."},
                    {"role": "user",
                     "content": f"Review these calculated stakeholder program management gaps and produce the clean final structured synthesis report:\n{json.dumps(compiled_raw_payload, indent=2)}"}
                ],
                response_format=ExecutiveStakeholderGapReport,
                temperature=0.1
            )
            structured_report = response.choices[0].message.parsed
        except Exception:
            structured_report = fallback_synthesis(compiled_raw_payload)
    else:
        structured_report = fallback_synthesis(compiled_raw_payload)

    lines = [
        "==========================================================================",
        "                     STAKEHOLDER GAP AUDIT REPORT                         ",
        f"                     GENERATED: {today.upper()}                          ",
        "==========================================================================",
        "\n### EXECUTIVE SUMMARY ###",
        structured_report.executive_summary,
        "\n" + "-" * 74
    ]
    for cat in structured_report.categories:
        lines.extend([
            f"\nGAP CATEGORY       : {cat.gap_category}",
            f"TARGET OWNER       : {cat.stakeholder_name}",
            f"SEVERITY LEVEL     : {cat.severity} | CONFIDENCE: {cat.confidence}",
            f"OBSERVED ANOMALY   : {cat.observed_gap}",
            f"OPERATIONAL IMPACT : {cat.practical_impact}",
            f"RECOMMENDED ACTION : {cat.recommended_action}",
            "\nFOUNDATIONAL EVIDENCE PASSAGES:"
        ])
        for ev in cat.evidence:
            lines.append(f"  - [{ev.source}]: \"{ev.snippet[:140]}\"")
        lines.append("\n" + "-" * 74)

    final_report_text = "\n".join(lines)
    stakeholder_gap_report_path.write_text(final_report_text, encoding="utf-8")

    print("PIPELINE COMPLETED")
    return final_report_text


def fallback_synthesis(raw_findings: List[Dict]) -> ExecutiveStakeholderGapReport:
    categories = [StakeholderGapReport(**f) for f in raw_findings]
    return ExecutiveStakeholderGapReport(
        executive_summary="Automated structural diagnostics identified key stakeholder register exclusions, strategic alignment gaps, and untracked architectural concerns requiring program management mitigation.",
        categories=categories
    )


# --- STREAMLIT DASHBOARD INTERFACE ---
st.set_page_config(page_title="AI Stakeholder Gap Analysis", layout="wide")
st.title("Stakeholder Gap Analysis")
st.caption("Identify and mitigate stakeholder gaps in your organization's project delivery pipeline.")
st.markdown("---")

col1, col2 = st.columns(2)

with col1:
    st.subheader("System Configuration")

    uploaded_reg = st.file_uploader("Upload Stakeholder Register (.csv)", type=["csv"])
    uploaded_plan = st.file_uploader("Upload Engagement Strategy (.md)", type=["md"])
    uploaded_notes = st.file_uploader("Upload Meeting Notes (.md)", type=["md"])

    if uploaded_reg:
        pd.read_csv(uploaded_reg).to_csv(stakeholder_register_path, index=False)

    if uploaded_plan:
        stakeholder_plan_path.write_bytes(uploaded_plan.getvalue())

    if uploaded_notes:
        meeting_notes_path.write_bytes(uploaded_notes.getvalue())

    if data_folder.exists():
        files = [f.name for f in data_folder.iterdir() if f.is_file()]
        st.write("Current Stakeholder Files:", files)

    start_pipeline = st.button("Execute Stakeholder Gap Pipeline", use_container_width=True, type="primary")

    st.subheader("Pipeline Summary")
    console_logs = st.empty()
    console_logs.info("Click 'Execute Stakeholder Gap Pipeline' button to begin.")

with col2:
    st.subheader("Report Workspace")
    report_placeholder = st.empty()
    report_placeholder.info("The Stakeholder Gap Analysis will populate here upon synthesis.")

    if start_pipeline:
        redirector = StreamlitStdoutRedirector(console_logs)

        with st.spinner("Processing Stakeholder Gap Analysis..."):
            with contextlib.redirect_stdout(redirector):
                final_narrative = run_automated_pipeline()

        if final_narrative:
            with report_placeholder.container():
                st.markdown(
                    f"""
                    <div style="
                        background-color: #1e293b; 
                        color: #f8fafc; 
                        padding: 20px; 
                        border-radius: 8px; 
                        height: 550px; 
                        overflow-y: scroll; 
                        white-space: pre-wrap; 
                        font-family: inherit;
                        border: 1px solid #334155;
                        line-height: 1.5;
                    ">
                        <p style="font-size: 16px !important; margin: 0; padding: 0;">{final_narrative}</p>
                    </div>
                    """,
                    unsafe_allow_html=True
                )

                st.download_button(
                    label="Download Stakeholder Gap Report (.txt)",
                    data=final_narrative,
                    file_name="stakeholder_gap_report.txt",
                    mime="text/plain",
                    use_container_width=True
                )