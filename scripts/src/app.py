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
root_folder = src_dir.parent.parent
data_folder = root_folder / "data"
vector_store_folder = root_folder / "vector_store"
output_folder = root_folder / "outputs"

#Hardcoding files for demo
stakeholder_register_path = data_folder / "Stakeholder_Register.csv"
stakeholder_plan_path = data_folder / "Stakeholder_Engagement_Plan.md"
meeting_notes_path = data_folder / "Meeting_Notes.md"

folder_paths = [data_folder, vector_store_folder, output_folder]
for folder in folder_paths:
    folder.mkdir(parents=True, exist_ok=True)

stakeholder_gap_report_path = output_folder / "STAKEHOLDER_GAP_REPORT.txt"
database_file_destination = vector_store_folder / "global_vector_store.json"

KNOWN_STAKEHOLDERS = {"Liam Patel", "Fatima Al-Sayed", "John Doe"}

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
    def __init__(self, placeholder, max_chars: int = 8000):
        self.placeholder = placeholder
        self.output_str = ""
        self.max_chars = max_chars

    def reset(self):
        self.output_str = ""
        self.placeholder.empty()

    def write(self, text):
        if not text:
            return
        self.output_str += str(text)
        if len(self.output_str) > self.max_chars:
            self.output_str = self.output_str[-self.max_chars:]
        self.placeholder.code(self.output_str, language="text")

    # Add this exact method to satisfy sys.stdout
    def flush(self):
        pass


# --- INTERNAL FACT DOMAIN SCHEMAS ---
class Stakeholder(BaseModel):
    stakeholder_id: str
    name: str
    role: str
    influence: str
    interest: str
    desired_engagement: str
    source_artifact: str
    source_row: Optional[int] = None


class Concern(BaseModel):
    description: str
    stakeholder_name: str
    normalized_category: str
    severity: str
    source_artifact: str
    line_number: int
    snippet: str
    concern_keywords: List[str] = []


class EngagementAction(BaseModel):
    action_strategy: str
    stakeholder_name: str
    owner_text: Optional[str] = None
    cadence_text: Optional[str] = None
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
    mention_type: str  # attendee, discussion, concern


class GapFinding(BaseModel):
    finding_id: str
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
    line_number: Optional[int] = None
    artifact_type: Optional[str] = None


class StakeholderGapReport(BaseModel):
    gap_category: str = Field(
        description="The gap category in ALL CAPS: MISSING STAKEHOLDER, STRATEGIC EXECUTION GAP, or RECURRENT CONCERN MISMATCH.")
    stakeholder_name: str
    severity: str
    confidence: str
    observed_gap: str
    practical_impact: str
    recommended_action: str
    finding_id: str
    evidence: List[EvidenceItem]


class ExecutiveStakeholderGapReport(BaseModel):
    model_config = ConfigDict(extra="forbid")
    executive_summary: str
    findings: List[StakeholderGapReport]

# --- PIPELINE LAYER: CORE UTILITIES ---
def chunk_text(text: str, chunk_size: int = 300, overlap: int = 50) -> List[str]:
    if chunk_size <= 0:
        return []
    if overlap >= chunk_size:
        overlap = max(1, chunk_size // 2)

    words = text.split()
    if not words:
        return []

    step = max(1, chunk_size - overlap)
    chunks = []

    for i in range(0, len(words), step):
        chunk = " ".join(words[i:i + chunk_size]).strip()
        if chunk:
            chunks.append(chunk)

    return chunks



def get_embedding(text: str) -> List[float]:
    if not is_vector_search_enabled:
        return []

    cleaned = " ".join(str(text).split())
    if not cleaned:
        return []

    try:
        response = client.embeddings.create(
            model="text-embedding-3-small",
            input=cleaned
        )
        return response.data[0].embedding
    except Exception as e:
        print(f" -> Embedding generation failed: {e}")
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

    def save(self, path: Path):
        print(" -> Saving vector index.")
        with open(path, "w", encoding="utf-8") as f:
            json.dump(self.entries, f)

    def load(self, path: Path):
        print(" -> Loading vector index.")
        if path.exists():
            with open(path, "r", encoding="utf-8") as f:
                self.entries = json.load(f)

    def build_indices(self, chunks: List[Dict]):
        print(" -> Building vector indices.")
        print(f" -> Found {len(chunks)} chunks to index.")
        self.entries = []
        if not is_vector_search_enabled:
            return
        for chunk in chunks:
            embedding = get_embedding(chunk["text"])
            if embedding:
                self.entries.append({**chunk, "embedding": embedding})

    def search(self, query: str, top_k: int = 2, min_similarity: float = 0.20) -> List[Dict]:
        if not is_vector_search_enabled or not self.entries:
            return []

        query_vector = get_embedding(query)
        if not query_vector:
            return []

        scored = []
        for entry in self.entries:
            sim = cosine_similarity(query_vector, entry["embedding"])
            if sim >= min_similarity:
                scored.append((sim, entry))

        scored.sort(reverse=True, key=lambda x: x[0])
        return [
            {
                "source": item[1]["metadata"].get("source", "Unknown"),
                "snippet": item[1]["text"],
                "score": round(item[0], 3)
            }
            for item in scored[:top_k]
        ]


# --- PIPELINE LAYER: TAXONOMY & NORMALIZATION ---
class CorporateTaxonomyNormalizer:
    def __init__(self):
        self.blocklist = {"Actions", "Steering Committee", "Security Review", "Discussion"}
        self.identity_map = {
            "maria chen": "Maria Chen",
            "david": "David Okafor", # Map "David" to the name in your CSV
            "david okafor": "David Okafor",
            "priya": "Priya Nair", # Map "Priya" to the name in your CSV
            "priya nair": "Priya Nair",
            "helen brooks": "Helen Brooks",
            "jonas": "Jonas Weber",
            "jonas weber": "Jonas Weber",
            "fatima": "Fatima Hassan", # Map "Fatima" to the name in your CSV
            "fatima hassan": "Fatima Hassan",
            "fatima al-sayed": "Fatima Hassan"
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
        # If the name is in the blocklist, force it to return an empty string
        if name.title() in self.blocklist:
            return ""

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


        for known_name in KNOWN_STAKEHOLDERS:
            self.discovered_names.add(known_name)

        # 1. Register Discovery (Trusted Source)
        if stakeholder_register_path.exists():
            print(" -> Pass 1: Discovering names from register.")
            df = pd.read_csv(stakeholder_register_path)
            for _, row in df.iterrows():
                n = str(row.get("name", "")).strip()
                if n: self.discovered_names.add(self.normalizer.normalize_name(n))

        # 2. Plan Discovery (Update for Markdown Headers)
        if stakeholder_plan_path.exists():
            print(" -> Pass 1: Discovering names from plan.")
            content = stakeholder_plan_path.read_text(encoding="utf-8")
            # This regex looks for names between "###" and "—"
            matches = re.findall(r"###\s+(.*?)\s+—", content)
            for raw_name in matches:
                norm = self.normalizer.normalize_name(raw_name)
                if norm: self.discovered_names.add(norm)

        # 3. Meeting Discovery (STRICT: ONLY the Attendees line)
        if meeting_notes_path.exists():
            print(" -> Pass 1: Discovering names from meeting notes.")
            for line in meeting_notes_path.read_text(encoding="utf-8").split("\n"):
                if line.lower().startswith("attendees:"):
                    # Extract only the content after "attendees:"
                    raw_attendees = line.split(":", 1)[1]
                    for name_part in raw_attendees.split(","):
                        n = name_part.strip()
                        if n:
                            norm = self.normalizer.normalize_name(n)
                            # Only add if it's not a common project term
                            # You could add a blacklist here if needed
                            if norm not in ["Actions", "Steering Committee", "Security Review"]:
                                self.discovered_names.add(norm)

        # 1. PARSER SPECIFIC: Stakeholder Register
        if stakeholder_register_path.exists():
            print(" -> Pass 2: Ingesting Stakeholder Register.")
            df = pd.read_csv(stakeholder_register_path)
            for _, row in df.iterrows():
                raw_name = str(row.get("Stakeholder Name", "")).strip()
                if not raw_name: continue

                norm_name = self.normalizer.normalize_name(raw_name)

                # FIX: Add the missing required fields
                self.stakeholders[norm_name] = Stakeholder(
                    stakeholder_id=f"STK-{norm_name.replace(' ', '-').upper()}",  # Generated ID
                    name=norm_name,
                    role=str(row.get("Role", "Unknown")),
                    influence=str(row.get("Influence", "Medium")),
                    interest=str(row.get("Interest", "Medium")),
                    desired_engagement=str(row.get("Preferred Communication", "Keep Informed")),
                    source_artifact="Stakeholder_Register.csv"  # Required field
                )

                self.raw_chunks.append({
                    "text": f"Register entry: {raw_name}, Role: {row.get('Role')}, Dept: {row.get('Department')}",
                    "metadata": {"source": "Stakeholder_Register.csv", "type": "Register"}
                })

        # 2. PARSER SPECIFIC: Engagement Plan
        if stakeholder_plan_path.exists():
            print(" -> Pass 2: Ingesting Engagement Plan.")
            lines = stakeholder_plan_path.read_text(encoding="utf-8").split("\n")
            current_stakeholder = None

            for idx, line in enumerate(lines):
                # Look for the header format "### Name — Role"
                header_match = re.search(r"###\s+(.*?)\s+—", line)
                if header_match:
                    raw_name = header_match.group(1).strip()
                    current_stakeholder = self.normalizer.normalize_name(raw_name)
                    continue  # Move to next line

                # If we are under a stakeholder, look for actions
                if current_stakeholder and line.strip().startswith("-"):
                    strategy_text = line.replace("-", "").strip()

                    # Only add if it's an actual action, not a label
                    if strategy_text.lower() not in ["engagement approach:", "actions:", "desired outcome:"]:
                        has_owner = any(x in strategy_text.lower() for x in ["owner:", "lead", "pm"])
                        has_cadence = any(
                            x in strategy_text.lower() for x in ["weekly", "bi-weekly", "monthly", "quarterly"])

                        self.engagement_actions.append(EngagementAction(
                            action_strategy=strategy_text,
                            stakeholder_name=current_stakeholder,
                            has_owner=has_owner,
                            has_cadence=has_cadence,
                            source_artifact="Stakeholder_Engagement_Plan.md",
                            line_number=idx + 1,
                            snippet=line.strip()
                        ))

        # 3. PARSER SPECIFIC: Meeting Notes (Context-Rich Sweep)
        if meeting_notes_path.exists():
            print(" -> Pass 2: Ingesting Meeting Notes.")
            notes_lines = meeting_notes_path.read_text(encoding="utf-8").split("\n")
            is_attendee_line = False

            for idx, line in enumerate(notes_lines):
                # 1. Skip structural markdown headers
                if line.strip().startswith("#"):
                    continue

                # 2. Reset attendee status on empty lines
                if not line.strip():
                    is_attendee_line = False
                    continue

                lowered_line = line.lower()
                if "attendees:" in lowered_line:
                    is_attendee_line = True

                for target_name in self.discovered_names:
                    # Guard: Skip empty or invalid names
                    if not target_name or target_name in self.normalizer.blocklist:
                        continue

                    # Use Regex Word Boundaries (\b) for precise matching
                    pattern = rf"\b{re.escape(target_name)}\b"

                    if re.search(pattern, line, re.IGNORECASE):
                        # Determine attendee status (only if on the "attendees:" line)
                        is_att = is_attendee_line and (target_name.lower() in lowered_line.split("attendees:")[-1])

                        # --- LOGIC TO DEFINE MENTION TYPE ---
                        if is_att:
                            m_type = "attendee"
                        elif any(k in lowered_line for k in
                                 ["concern", "anxiety", "risk", "issue", "flagged", "stalled", "vulnerability"]):
                            m_type = "concern"
                        else:
                            m_type = "discussion"
                        # ------------------------------------

                        self.meeting_mentions.append(MeetingMention(
                            stakeholder_name=target_name,
                            context_snippet=line.strip(),
                            source_artifact="Meeting_Notes.md",
                            line_number=idx + 1,
                            is_explicit_attendee=is_att,
                            mention_type=m_type
                        ))

                        # Keep concern logic, but only if it's a valid stakeholder name
                        if m_type == "concern":
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

            # Create chunks for vector search
            for chunk in chunk_text("\n".join(notes_lines)):
                self.raw_chunks.append({
                    "text": chunk,
                    "metadata": {"source": "Meeting_Notes.md", "type": "Notes", "section": "Discussion Logs"}
                })


# --- PIPELINE LAYER: DETERMINISTIC AUDIT RULES ENGINE ---
class GapDetector:
    def __init__(self, context: StructuredProjectContext, store: SimpleVectorStore, normalizer: CorporateTaxonomyNormalizer):
        self.context = context
        self.store = store
        self.normalizer = normalizer # <--- THIS IS THE MISSING LINE

    def generate_strategic_heatmap(self) -> List[Dict]:
        print(" -> Generating Strategic Heatmap.")
        heatmap_data = []
        for name, stakeholder in self.context.stakeholders.items():
            concern_count = len([c for c in self.context.concerns if c.stakeholder_name == name])

            # Risk Level: High Influence + Multiple Concerns = CRITICAL
            if stakeholder.influence == "High" and concern_count > 0:
                risk_level = "CRITICAL - Immediate Action Required"
            elif concern_count > 0:
                risk_level = "ELEVATED - Requires Attention"
            else:
                risk_level = "STABLE"

            heatmap_data.append({
                "stakeholder": name,
                "influence": stakeholder.influence,
                "concern_count": concern_count,
                "risk_level": risk_level
            })
        return heatmap_data


    def execute_audit_checks(self) -> List[GapFinding]:
        findings: List[GapFinding] = []

        # --- RULE 1: MISSING STAKEHOLDER ---
        print(" -> Executing Rule 1: Missing Stakeholder Detection.")
        mentioned_names = {m.stakeholder_name for m in self.context.meeting_mentions}
        registered_names = set(self.context.stakeholders.keys())

        for name in mentioned_names:
            if name not in registered_names and name:
                mentions = [m for m in self.context.meeting_mentions if m.stakeholder_name == name]
                primary_evidence = [f"[{m.source_artifact} Line {m.line_number}]: '{m.context_snippet}'" for m in
                                    mentions]

                f_id = f"GAP-MISSING-{name.replace(' ', '-').upper()}"
                findings.append(GapFinding(
                    finding_id=f_id,
                    gap_category="MISSING STAKEHOLDER",
                    stakeholder_name=name,
                    severity="High",
                    confidence="High" if len(mentions) > 1 else "Medium",
                    observed_gap=f"Stakeholder '{name}' is active in meeting notes but is not present in the Stakeholder Register.",
                    practical_impact="Lack of formal oversight for active project participants creates a governance blind spot.",
                    recommended_action=f"Add '{name}' to the Stakeholder Register and assign a communication owner.",
                    primary_deterministic_evidence=primary_evidence,
                    vector_evidence_queries=[f"{name} role in project"]
                ))


        # --- RULE 2: STRATEGIC EXECUTION GAP ---
        print(" -> Executing Rule 2: Strategic Execution Gap Detection.")
        for name, stakeholder in self.context.stakeholders.items():
            # Normalize the lookup name to ensure it matches exactly how the Action was stored
            norm_lookup = self.normalizer.normalize_name(name)

            # Filter actions by checking if the action's stakeholder name matches our normalized name
            actions = [
                a for a in self.context.engagement_actions
                if self.normalizer.normalize_name(a.stakeholder_name) == norm_lookup
            ]

            is_gap = False
            reasons = []
            primary_evidence = [f"[{a.source_artifact} Line {a.line_number}]: '{a.snippet}'" for a in actions]

            # 1. Actionability Gap
            if not actions:
                is_gap = True
                reasons.append("has no actionable engagement entries in the strategy plan")
            else:
                combined_strategy = " ".join([a.action_strategy.lower() for a in actions])

                # 2. Ownership Gap (Check if anyone is actually assigned to handle this stakeholder)
                if not any(a.has_owner for a in actions):
                    is_gap = True
                    reasons.append("lacks an explicit engagement owner")

                # 3. Cadence Gap (Tiered Frequency Check)
                strategy_lower = combined_strategy.lower()
                if stakeholder.desired_engagement == "Manage Closely":
                    # Manage Closely must have Weekly or Bi-Weekly
                    if not any(x in strategy_lower for x in ["weekly", "bi-weekly"]):
                        is_gap = True
                        reasons.append(
                            "requires high-frequency 'Manage Closely' engagement, but only low-frequency cadence is defined")

                elif stakeholder.desired_engagement == "Keep Satisfied":
                    # Keep Satisfied must be at least Monthly
                    if not any(x in strategy_lower for x in ["weekly", "bi-weekly", "monthly"]):
                        is_gap = True
                        reasons.append("requires 'Keep Satisfied' engagement, but cadence is insufficient")

            if is_gap:
                f_id = f"GAP-EXEC-{name.replace(' ', '-').upper()}"
                findings.append(GapFinding(
                    finding_id=f_id,
                    gap_category="STRATEGIC EXECUTION GAP",
                    stakeholder_name=name,
                    severity="High",
                    confidence="High",
                    observed_gap=f"Stakeholder '{name}' " + " and ".join(
                        reasons) + ".",
                    practical_impact="Failure to maintain the appropriate engagement rigor for high-influence stakeholders directly threatens program continuity and strategic consensus.",
                    recommended_action=f"Update the Engagement Plan to assign a clear owner and increase frequency for '{name}' to match their '{stakeholder.desired_engagement}' status.",
                    primary_deterministic_evidence=primary_evidence,
                    vector_evidence_queries=[f"{name} engagement strategy ownership frequency"]
                ))

        # --- RULE 3: RECURRENT CONCERN MISMATCH ---
        print(" -> Executing Rule 3: Recurrent Concern Mismatch Detection.")
        for concern in self.context.concerns:
            name = concern.stakeholder_name
            # Find actions for this stakeholder and check if ANY action covers the concern category
            matching_actions = [
                a for a in self.context.engagement_actions
                if a.stakeholder_name == name and
                   self.context.normalizer.classify_concern(a.action_strategy) == concern.normalized_category
            ]

            if not matching_actions:
                f_id = f"GAP-CONCERN-{name.replace(' ', '-').upper()}-{concern.line_number}"
                findings.append(GapFinding(
                    finding_id=f_id,
                    gap_category="RECURRENT CONCERN MISMATCH",
                    stakeholder_name=name,
                    severity=concern.severity,
                    confidence="High",
                    observed_gap=f"The concern '{concern.normalized_category}' flagged by '{name}' has no mapped action in the engagement plan.",
                    practical_impact="Unaddressed recurrent concerns lead to project friction and reduced stakeholder trust.",
                    recommended_action=f"Add a specific mitigation action for '{concern.normalized_category}' to the Engagement Plan.",
                    primary_deterministic_evidence=[
                        f"[{concern.source_artifact} Line {concern.line_number}]: '{concern.snippet}'"],
                    vector_evidence_queries=[f"{name} {concern.normalized_category} mitigation steps"]
                ))

        return findings


def compile_raw_payload(internal_findings: List[GapFinding], store: SimpleVectorStore, heatmap: List[Dict]) -> Dict:
    compiled_raw_payload = []

    for f in internal_findings:
        # Use the passed store variable for vector lookups
        vector_support = store.search(f.vector_evidence_queries[0], top_k=1) if f.vector_evidence_queries else []

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
    return {
        "findings": compiled_raw_payload,
        "strategic_heatmap": heatmap
    }


def synthesize_report_with_llm(compiled_raw_payload: List[dict]) -> ExecutiveStakeholderGapReport:
    # Use the globally configured client and capability flag
    if is_vector_search_enabled:
        try:
            response = client.beta.chat.completions.parse(
                model="gpt-4o-mini",
                messages=[
                    {
                        "role": "system",
                        "content": (
                                "You are an expert Senior Program Leader. "
                                "You are provided with stakeholder gap findings and a 'Strategic Risk Heatmap'. "
                                "1. Use the Heatmap to identify the top 2-3 'CRITICAL' stakeholders. "
                                "2. In your Executive Summary, explicitly name these stakeholders and explain why they are at risk based on their Influence vs. Concern count. "
                                "3. Provide a clear prioritization plan."
                        )
                    },
                    {
                        "role": "user",
                        "content": f"Review these calculated stakeholder program management gaps and produce the clean final structured synthesis report:\n{json.dumps(compiled_raw_payload, indent=2)}"
                    }
                ],
                response_format=ExecutiveStakeholderGapReport,
                temperature=0.1
            )
            structured_report = response.choices[0].message.parsed
        except Exception as e:
            print(f" -> LLM parsing failed due to error: {e}. Slipping into fallback framework.")
            structured_report = fallback_synthesis(compiled_raw_payload)
    else:
        print(" -> Vector Search/API Key disabled. Utilizing native fallback framework.")
        structured_report = fallback_synthesis(compiled_raw_payload)

    return structured_report


def generate_executive_summary(structured_report: ExecutiveStakeholderGapReport) -> str:
    print(f" -> Re-applying structure and saving report to disk.")

    # Correctly access 'findings' instead of 'categories'
    total_findings = len(structured_report.findings)

    lines = [
        "================================================================================",
        "STAKEHOLDER ALIGNMENT REPORT",
        f"Report Date: {today}",
        f"Summary: Verified {total_findings} stakeholder alignment findings.",
        "================================================================================",
        "",
        "EXECUTIVE SUMMARY:",
        structured_report.executive_summary,
        "",
        "DETAILED FINDINGS BY CATEGORY",
        ""
    ]

    # Iterate over 'findings'
    for finding in structured_report.findings:
        lines.append(f"{finding.gap_category}:")
        lines.append(f"Stakeholder: {finding.stakeholder_name}")
        lines.append(f"Issue: {finding.observed_gap}")
        lines.append(f"Operational Impact: {finding.practical_impact}")
        lines.append(f"Recommendation: {finding.recommended_action}")
        lines.append("")

    final_report_text = "\n".join(lines)
    stakeholder_gap_report_path.write_text(final_report_text, encoding="utf-8")

    return final_report_text


# --- ORCHESTRATION LAYER ---
def run_automated_pipeline() -> str:
    print("PIPELINE STARTED")

    print("STEP 1: Normalizing Stakeholders and Concerns.")
    normalizer = CorporateTaxonomyNormalizer()

    print("STEP 2: Ingesting Raw Data.")
    context = StructuredProjectContext(normalizer=normalizer)
    context.ingest_data()

    print("STEP 3: Building Vector Index.")
    store = SimpleVectorStore()

    target_dir = database_file_destination.parent
    if not target_dir.exists():
        print(f" -> Creating missing directory: {target_dir}")
        target_dir.mkdir(parents=True, exist_ok=True)

    # 2. Check for the file
    if database_file_destination.exists():
        print(f" -> Found existing vector store: {database_file_destination.name}")
        store.load(database_file_destination)
    else:
        print(" -> No existing vector store found. Starting new ingestion...")
        store.build_indices(context.raw_chunks)
        store.save(database_file_destination)

    print("STEP 4: Executing Gap Detection.")
    detector = GapDetector(context, store, normalizer)

    internal_findings = detector.execute_audit_checks()

    heatmap = detector.generate_strategic_heatmap()

    print("STEP 5: Compiling Raw Payload.")
    print(f" -> Found {len(internal_findings)} internal findings.")

    # STEP 5: Compiling Raw Payload (Pass the heatmap)
    raw_payload = compile_raw_payload(internal_findings, store, heatmap)

    print("STEP 6: Synthesizing AI Report Narrative.")
    structured_report = synthesize_report_with_llm(raw_payload)

    print("STEP 7: Generating Executive Summary.")
    # Assign the returned text directly
    final_report_text = generate_executive_summary(structured_report)

    print("PIPELINE COMPLETED")
    return final_report_text


def fallback_synthesis(raw_findings: List[Dict]) -> ExecutiveStakeholderGapReport:
    # 'findings' matches the ExecutiveStakeholderGapReport model field
    findings = [StakeholderGapReport(**f) for f in raw_findings]
    return ExecutiveStakeholderGapReport(
        executive_summary="Automated structural diagnostics identified key stakeholder register exclusions, strategic alignment gaps, and untracked architectural concerns.",
        findings=findings
    )

# --- STREAMLIT DASHBOARD INTERFACE ---
st.set_page_config(page_title="AI Stakeholder Alignment", layout="wide")
st.title("Stakeholder Alignment Dashboard")
st.caption("Real-time diagnostics to bridge the gap between project strategy and operational delivery.")
st.markdown("---")

col1, col2 = st.columns(2)

with col1:
    st.subheader("System Configuration")

    # --- 2. Hardcoded File Loading Logic ---
    # Check if the data folder exists
    if data_folder.exists():

        st.text(f"Files found in '{data_folder.name}'")
        # iterdir() yields Path objects; we grab .name for just the filename
        files = [f.name for f in data_folder.iterdir()]
        st.write(files)

        # Verify the specific files exist before trying to read them
        if stakeholder_register_path.exists():
            # Read the CSV directly into a DataFrame
            df_register = pd.read_csv(stakeholder_register_path)
            # You can now use df_register throughout your app
        else:
            st.error(f"Missing file: {stakeholder_register_path.name}")

        if stakeholder_plan_path.exists():
            plan_content = stakeholder_plan_path.read_text(encoding="utf-8")

        if meeting_notes_path.exists():
            notes_content = meeting_notes_path.read_text(encoding="utf-8")
    else:
        st.error(f"Data directory '{data_folder}' does not exist. Please create it and add your files.")

    start_pipeline = st.button("Execute Stakeholder Alignment Pipeline", use_container_width=True, type="primary")

    st.subheader("Pipeline Summary")
    console_logs = st.empty()
    console_logs.info("Click 'Execute Stakeholder Alignment Pipeline' button to begin.")

with col2:
    st.subheader("Report Workspace")
    report_placeholder = st.empty()
    report_placeholder.info("The Stakeholder Alignment Report will populate here upon synthesis.")

    if start_pipeline:
        console_logs.empty()
        redirector = StreamlitStdoutRedirector(console_logs)
        # CLEAR the UI before the pipeline starts
        redirector.reset()

        with st.spinner("Processing Stakeholder Alignment Pipeline..."):
            with contextlib.redirect_stdout(redirector):
                final_narrative = run_automated_pipeline()

        if final_narrative:
            with report_placeholder.container():
                st.html(
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
                                """
                )

                st.download_button(
                    label="Download Stakeholder Alignment Report (.txt)",
                    data=final_narrative,
                    file_name="stakeholder_alignment_report.txt",
                    mime="text/plain",
                    use_container_width=True
                )