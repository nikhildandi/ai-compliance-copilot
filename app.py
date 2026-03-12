import os
import re
import json
import time
import streamlit as st
import streamlit.components.v1 as components
from pypdf import PdfReader
from dotenv import load_dotenv

from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_community.vectorstores import FAISS
from langchain_classic.chains.question_answering import load_qa_chain

load_dotenv()

st.set_page_config(page_title="AI Compliance Copilot", layout="wide")

st.markdown("""
<style>
.stApp {
    background: linear-gradient(135deg, #0f172a, #1e3a8a);
    color: white;
}
h1 { color: #60a5fa; }
h2, h3 { color: #93c5fd; }
.block-container { padding-top: 1.5rem; max-width: 1400px; }
[data-testid="stSidebar"] { background-color: #020617; }
.stTextInput > div > div > input {
    border: 1px solid rgba(147, 197, 253, 0.3) !important;
    border-radius: 8px !important;
    background: rgba(15, 23, 42, 0.8) !important;
    color: white !important;
    box-shadow: none !important;
    outline: none !important;
}
.stTextInput > div > div > input:focus {
    border: 1px solid rgba(96, 165, 250, 0.7) !important;
    box-shadow: 0 0 0 2px rgba(96, 165, 250, 0.15) !important;
}
.stTextInput > div[data-baseweb="input"] { border: none !important; box-shadow: none !important; }
.stTextInput > div > div > div[data-testid="InputInstructions"],
.stTextInput small, .stTextInput [class*="instructions"],
[data-testid="InputInstructions"], div[class*="instructions"], small[class*="caption"] {
    display: none !important; visibility: hidden !important; height: 0 !important; overflow: hidden !important;
}
.stButton > button {
    background: #1d4ed8 !important; color: white !important; border: none !important;
    border-radius: 8px !important; padding: 0.5rem 1.4rem !important;
    font-size: 0.9rem !important; font-weight: 600 !important;
    cursor: pointer !important; transition: background 0.2s ease !important;
    width: 100% !important; margin-top: 0.4rem !important;
}
.stButton > button:hover { background: #2563eb !important; }
.preview-card {
    background: rgba(15, 23, 42, 0.75); border: 1px solid rgba(147, 197, 253, 0.18);
    border-radius: 12px; padding: 14px 16px; height: 260px; overflow-y: auto;
    line-height: 1.65; font-size: 13px; white-space: pre-wrap; word-break: break-word;
    color: #cbd5e1; font-family: ui-monospace, monospace;
}
.answer-card {
    background: rgba(15, 23, 42, 0.75); border: 1px solid rgba(96, 165, 250, 0.22);
    border-radius: 12px; padding: 16px 18px; line-height: 1.7; font-size: 15px;
    margin-top: 0.5rem; margin-bottom: 0.75rem;
}
div[data-testid="stMetric"] {
    background: rgba(15, 23, 42, 0.45); border: 1px solid rgba(147, 197, 253, 0.12);
    padding: 10px 14px; border-radius: 12px;
}
[data-testid="stExpander"] {
    background: rgba(15, 23, 42, 0.5) !important; border: 1px solid rgba(147, 197, 253, 0.15) !important;
    border-radius: 10px !important; margin-top: 0.5rem !important;
}
[data-testid="stExpander"] summary {
    font-size: 0.82rem !important; color: #93c5fd !important;
    font-weight: 500 !important; padding: 0.5rem 0.75rem !important;
}
[data-testid="stExpander"] summary:hover { color: #60a5fa !important; }
[data-testid="stExpander"] > div > div { padding: 0.25rem 0.75rem 0.75rem !important; }
.evidence-card {
    background: rgba(15, 23, 42, 0.6); border: 1px solid rgba(147, 197, 253, 0.12);
    border-radius: 10px; padding: 12px 14px; margin-bottom: 10px;
}
.evidence-title { color: #93c5fd; font-weight: 600; margin-bottom: 6px; font-size: 12px; text-transform: uppercase; letter-spacing: 0.06em; }
.evidence-text { color: #cbd5e1; line-height: 1.6; font-size: 13px; white-space: pre-wrap; word-break: break-word; }
.gap-covered { background: rgba(16,185,129,0.1); border: 1px solid rgba(16,185,129,0.35); border-left: 4px solid #10b981; border-radius: 10px; padding: 12px 16px; margin-bottom: 10px; }
.gap-partial  { background: rgba(245,158,11,0.1);  border: 1px solid rgba(245,158,11,0.35);  border-left: 4px solid #f59e0b; border-radius: 10px; padding: 12px 16px; margin-bottom: 10px; }
.gap-missing  { background: rgba(239,68,68,0.1);   border: 1px solid rgba(239,68,68,0.35);   border-left: 4px solid #ef4444; border-radius: 10px; padding: 12px 16px; margin-bottom: 10px; }
.gap-control-id { font-size: 11px; font-weight: 700; letter-spacing: 0.08em; text-transform: uppercase; margin-bottom: 4px; }
.gap-covered .gap-control-id { color: #10b981; }
.gap-partial  .gap-control-id { color: #f59e0b; }
.gap-missing  .gap-control-id { color: #ef4444; }
.gap-control-name { font-size: 14px; font-weight: 600; color: #e2e8f0; margin-bottom: 4px; }
.gap-finding { font-size: 13px; color: #94a3b8; line-height: 1.55; }
.gap-status-badge { display: inline-block; font-size: 11px; font-weight: 600; padding: 2px 10px; border-radius: 20px; margin-bottom: 6px; }
.badge-covered { background: rgba(16,185,129,0.2); color: #10b981; }
.badge-partial  { background: rgba(245,158,11,0.2);  color: #f59e0b; }
.badge-missing  { background: rgba(239,68,68,0.2);   color: #ef4444; }
</style>
""", unsafe_allow_html=True)

# NIST CSF Controls
NIST_CSF_CONTROLS = [
    {"id": "ID.AM-1", "name": "Asset Inventory",            "function": "Identify", "description": "Physical devices and systems are inventoried"},
    {"id": "ID.AM-2", "name": "Software Inventory",         "function": "Identify", "description": "Software platforms and applications are inventoried"},
    {"id": "ID.GV-1", "name": "Security Policy",            "function": "Identify", "description": "Organizational security policy is established and communicated"},
    {"id": "ID.GV-3", "name": "Legal & Regulatory",         "function": "Identify", "description": "Legal and regulatory requirements are understood and managed"},
    {"id": "ID.RA-1", "name": "Risk Assessment",            "function": "Identify", "description": "Asset vulnerabilities are identified and documented"},
    {"id": "PR.AC-1", "name": "Identity Management",        "function": "Protect",  "description": "Identities and credentials are managed for authorized users"},
    {"id": "PR.AC-3", "name": "Remote Access",              "function": "Protect",  "description": "Remote access is managed and controlled"},
    {"id": "PR.AC-4", "name": "Access Permissions",         "function": "Protect",  "description": "Access permissions are managed incorporating least privilege"},
    {"id": "PR.AC-7", "name": "Authentication",             "function": "Protect",  "description": "Users and devices are authenticated commensurate with risk (e.g. MFA)"},
    {"id": "PR.AT-1", "name": "Security Awareness",         "function": "Protect",  "description": "All users are informed and trained on security responsibilities"},
    {"id": "PR.DS-1", "name": "Data-at-Rest Protection",    "function": "Protect",  "description": "Data-at-rest is protected through encryption or other controls"},
    {"id": "PR.DS-2", "name": "Data-in-Transit Protection", "function": "Protect",  "description": "Data-in-transit is protected through encryption (e.g. TLS)"},
    {"id": "PR.DS-5", "name": "Data Leakage Prevention",    "function": "Protect",  "description": "Protections against data leaks are implemented"},
    {"id": "PR.IP-1", "name": "Baseline Configuration",     "function": "Protect",  "description": "A baseline configuration is created and maintained for systems"},
    {"id": "PR.IP-3", "name": "Change Control",             "function": "Protect",  "description": "Configuration change control processes are in place"},
    {"id": "PR.IP-9", "name": "Incident Response Plan",     "function": "Protect",  "description": "Response and recovery plans are in place and managed"},
    {"id": "PR.MA-1", "name": "Asset Maintenance",          "function": "Protect",  "description": "Maintenance and repair of assets is performed and logged"},
    {"id": "PR.PT-1", "name": "Audit Logging",              "function": "Protect",  "description": "Audit/log records are determined, documented, and reviewed"},
    {"id": "DE.AE-1", "name": "Baseline Network Ops",       "function": "Detect",   "description": "A baseline of network operations is established and managed"},
    {"id": "DE.CM-1", "name": "Network Monitoring",         "function": "Detect",   "description": "The network is monitored to detect potential cybersecurity events"},
    {"id": "DE.CM-7", "name": "Unauthorized Activity",      "function": "Detect",   "description": "Monitoring for unauthorized personnel and connections is performed"},
    {"id": "RS.RP-1", "name": "Response Plan",              "function": "Respond",  "description": "A response plan is executed during or after an incident"},
    {"id": "RS.CO-2", "name": "Incident Reporting",         "function": "Respond",  "description": "Incidents are reported consistent with established criteria"},
    {"id": "RS.AN-1", "name": "Incident Investigation",     "function": "Respond",  "description": "Notifications from detection systems are investigated"},
    {"id": "RS.MI-1", "name": "Incident Containment",       "function": "Respond",  "description": "Incidents are contained to limit impact"},
    {"id": "RC.RP-1", "name": "Recovery Plan",              "function": "Recover",  "description": "A recovery plan is executed during or after an incident"},
    {"id": "RC.IM-1", "name": "Recovery Improvements",      "function": "Recover",  "description": "Recovery plans incorporate lessons learned"},
]

FUNCTION_COLORS = {
    "Identify": "#818cf8",
    "Protect":  "#34d399",
    "Detect":   "#fbbf24",
    "Respond":  "#f87171",
    "Recover":  "#60a5fa",
}

def run_gap_analysis(policy_text, llm):
    controls_json = json.dumps(
        [{"id": c["id"], "name": c["name"], "description": c["description"]} for c in NIST_CSF_CONTROLS],
        indent=2
    )
    prompt = f"""You are a GRC analyst. Assess the policy below against each NIST CSF control.
For each control assign a status of exactly one of: "covered", "partial", or "missing".
- covered: the policy explicitly and adequately addresses it
- partial: the topic is mentioned but lacks detail
- missing: not addressed at all

Return ONLY a valid JSON array. Each element: {{"id": "...", "status": "...", "finding": "one concise sentence max 20 words"}}.
No markdown, no extra text.

Controls:
{controls_json}

Policy (first 6000 chars):
{policy_text[:6000]}"""

    response = llm.invoke(prompt)
    raw = response.content.strip()
    raw = re.sub(r'^```json|^```|```$', '', raw, flags=re.MULTILINE).strip()
    return json.loads(raw)


# App
with st.sidebar:
    # New Chat button
    if st.button("+ New Analysis", key="new_chat"):
        for key in ["last_question", "prev_question", "last_response", "last_docs", "gap_results"]:
            if key in st.session_state:
                del st.session_state[key]
        st.rerun()

    st.markdown("<div style='margin-top: 1.5rem;'></div>", unsafe_allow_html=True)

    # Supported frameworks
    st.markdown("""
    <div style="font-size: 0.7rem; font-weight: 700; letter-spacing: 0.1em; text-transform: uppercase; color: #475569; margin-bottom: 0.75rem;">
        Supported Frameworks
    </div>
    """, unsafe_allow_html=True)

    frameworks = [
        ("🔵", "NIST CSF",   "Gap analysis ready"),
        ("🟢", "HIPAA",      "Q&A supported"),
        ("🟡", "SOC 2",      "Q&A supported"),
        ("🟠", "ISO 27001",  "Q&A supported"),
        ("🔴", "PCI-DSS",    "Q&A supported"),
        ("🟣", "GDPR",       "Q&A supported"),
    ]

    for icon, name, status in frameworks:
        st.markdown(f"""
        <div style="
            display: flex;
            align-items: center;
            justify-content: space-between;
            background: rgba(15, 23, 42, 0.5);
            border: 1px solid rgba(147, 197, 253, 0.1);
            border-radius: 8px;
            padding: 8px 12px;
            margin-bottom: 6px;
        ">
            <span style="font-size: 0.85rem; color: #e2e8f0; font-weight: 500;">{icon} {name}</span>
            <span style="font-size: 0.68rem; color: #475569;">{status}</span>
        </div>
        """, unsafe_allow_html=True)

    # Footer
    st.markdown("""
    <div style="position: fixed; bottom: 2rem; left: 0; width: 18rem; padding: 0 1.5rem;">
        <div style="border-top: 1px solid rgba(147, 197, 253, 0.12); padding-top: 0.75rem; font-size: 0.72rem; color: #475569; line-height: 1.6;">
            Built by <span style="color: #93c5fd; font-weight: 600;">Nikhil Dandi</span><br>
            AI Compliance Copilot v1.0
        </div>
    </div>
    """, unsafe_allow_html=True)

st.title("Compliance Intelligence Dashboard")
st.caption("AI-powered analysis for security and compliance documents")
st.divider()

uploaded_file = st.file_uploader("Upload a compliance or policy document", type=["pdf"])

if uploaded_file is not None:
    reader = PdfReader(uploaded_file)
    text = ""
    for page in reader.pages:
        extracted = page.extract_text()
        if extracted:
            text += extracted

    text = re.sub(r'(?<!\n)\n(?!\n)', ' ', text)
    text = re.sub(r' {2,}', ' ', text)
    text = re.sub(r'\n{3,}', '\n\n', text)

    if text.strip():
        st.success("Document processed successfully")

        splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
        chunks = splitter.split_text(text)
        embeddings = OpenAIEmbeddings()
        vectorstore = FAISS.from_texts(chunks, embedding=embeddings)

        st.divider()

        col1, col2 = st.columns([1, 2], gap="large")

        with col1:
            st.subheader("Document Information")
            mc1, mc2 = st.columns(2)
            with mc1:
                st.metric("Pages", len(reader.pages))
            with mc2:
                st.metric("Characters", len(text))
            st.markdown("#### Document Preview")
            preview_text = text[:2200].replace("<", "&lt;").replace(">", "&gt;")
            st.markdown(f'<div class="preview-card">{preview_text}</div>', unsafe_allow_html=True)

        with col2:
            tab_qa, tab_gap = st.tabs(["  AI Policy Analysis", "  NIST CSF Gap Analysis"])

            # Tab 1: Q&A
            with tab_qa:
                st.subheader("AI Policy Analysis")
                PLACEHOLDERS = [
                    "e.g. What are the MFA requirements?",
                    "e.g. What is the password expiration policy?",
                    "e.g. Who gets notified in a P1 incident?",
                    "e.g. What encryption standard is required for data at rest?",
                    "e.g. What is the HIPAA breach notification timeline?",
                    "e.g. What TLS versions are allowed?",
                    "e.g. What are the privileged access management rules?",
                    "e.g. How often must encryption keys be rotated?",
                    "e.g. What is the session timeout for PHI systems?",
                    "e.g. What activities are prohibited under acceptable use?",
                ]
                placeholder = PLACEHOLDERS[int(time.time() / 8) % len(PLACEHOLDERS)]

                question = st.text_input(
                    "Ask a question about the document",
                    placeholder=placeholder,
                    key="question_input"
                )
                submit = st.button("Analyze →", key="analyze_btn")

                if (submit and question) or (question and st.session_state.get("last_question") != question):
                    st.session_state["prev_question"] = st.session_state.get("last_question")
                    st.session_state["last_question"] = question
                    docs = vectorstore.similarity_search(question, k=3)
                    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
                    chain = load_qa_chain(llm, chain_type="stuff")
                    response = chain.run(input_documents=docs, question=question)
                    st.session_state["last_response"] = response
                    st.session_state["last_docs"] = docs

                if st.session_state.get("last_response"):
                    st.success("Analysis Complete")
                    st.markdown("### AI Answer")
                    st.markdown(f'<div class="answer-card">{st.session_state["last_response"]}</div>', unsafe_allow_html=True)
                    with st.expander("View Sources"):
                        for i, doc in enumerate(st.session_state["last_docs"], start=1):
                            raw = doc.page_content[:450].strip()
                            if len(doc.page_content) > 450:
                                raw = raw.rsplit(' ', 1)[0] + '...'
                            preview = raw.replace("<", "&lt;").replace(">", "&gt;")
                            st.markdown(f"""
                            <div class="evidence-card">
                                <div class="evidence-title">Evidence {i}</div>
                                <div class="evidence-text">{preview}</div>
                            </div>""", unsafe_allow_html=True)

            # Tab 2: Gap Analysis
            with tab_gap:
                st.subheader("NIST CSF Gap Analysis")
                st.caption("Checks your policy against 27 NIST Cybersecurity Framework controls and flags what's covered, partial, or missing.")

                if st.button("Run Gap Analysis →", key="gap_btn"):
                    with st.spinner("Analyzing policy against NIST CSF controls... this takes ~20 seconds"):
                        llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
                        results = run_gap_analysis(text, llm)
                        st.session_state["gap_results"] = results

                if st.session_state.get("gap_results"):
                    results = st.session_state["gap_results"]
                    result_map = {r["id"]: r for r in results}

                    covered = sum(1 for r in results if r["status"] == "covered")
                    partial = sum(1 for r in results if r["status"] == "partial")
                    missing = sum(1 for r in results if r["status"] == "missing")
                    total   = len(results)
                    score   = round((covered + partial * 0.5) / total * 100)

                    components.html(f"""
                    <div style="display: grid; grid-template-columns: repeat(4, 1fr); gap: 12px; margin-bottom: 1rem; font-family: sans-serif;">

                        <div style="background: rgba(15,23,42,0.45); border: 1px solid rgba(147,197,253,0.12); border-radius: 12px; padding: 16px 18px;">
                            <div style="font-size: 0.78rem; color: #94a3b8; margin-bottom: 6px;">Overall Score</div>
                            <div style="font-size: 2rem; font-weight: 700; color: #e2e8f0;">{score}%</div>
                        </div>

                        <div style="background: rgba(15,23,42,0.45); border: 1px solid rgba(147,197,253,0.12); border-radius: 12px; padding: 16px 18px;">
                            <div style="font-size: 0.78rem; color: #94a3b8; margin-bottom: 6px;">Covered</div>
                            <div style="display: flex; align-items: center; gap: 10px;">
                                <svg width="28" height="28" viewBox="0 0 28 28" fill="none" xmlns="http://www.w3.org/2000/svg">
                                    <circle cx="14" cy="14" r="13" stroke="#10b981" stroke-width="2"/>
                                    <polyline points="8,14 12,18 20,10" stroke="#10b981" stroke-width="2.2" stroke-linecap="round" stroke-linejoin="round" fill="none"/>
                                </svg>
                                <span style="font-size: 2rem; font-weight: 700; color: #10b981;">{covered}</span>
                            </div>
                        </div>

                        <div style="background: rgba(15,23,42,0.45); border: 1px solid rgba(147,197,253,0.12); border-radius: 12px; padding: 16px 18px;">
                            <div style="font-size: 0.78rem; color: #94a3b8; margin-bottom: 6px;">Partial</div>
                            <div style="display: flex; align-items: center; gap: 10px;">
                                <svg width="28" height="28" viewBox="0 0 28 28" fill="none" xmlns="http://www.w3.org/2000/svg">
                                    <polygon points="14,3 27,25 1,25" stroke="#f59e0b" stroke-width="2" stroke-linejoin="round" fill="none"/>
                                    <line x1="14" y1="11" x2="14" y2="18" stroke="#f59e0b" stroke-width="2.2" stroke-linecap="round"/>
                                    <circle cx="14" cy="22" r="1.2" fill="#f59e0b"/>
                                </svg>
                                <span style="font-size: 2rem; font-weight: 700; color: #f59e0b;">{partial}</span>
                            </div>
                        </div>

                        <div style="background: rgba(15,23,42,0.45); border: 1px solid rgba(147,197,253,0.12); border-radius: 12px; padding: 16px 18px;">
                            <div style="font-size: 0.78rem; color: #94a3b8; margin-bottom: 6px;">Missing</div>
                            <div style="display: flex; align-items: center; gap: 10px;">
                                <svg width="28" height="28" viewBox="0 0 28 28" fill="none" xmlns="http://www.w3.org/2000/svg">
                                    <circle cx="14" cy="14" r="13" stroke="#ef4444" stroke-width="2"/>
                                    <line x1="9" y1="9" x2="19" y2="19" stroke="#ef4444" stroke-width="2.2" stroke-linecap="round"/>
                                    <line x1="19" y1="9" x2="9" y2="19" stroke="#ef4444" stroke-width="2.2" stroke-linecap="round"/>
                                </svg>
                                <span style="font-size: 2rem; font-weight: 700; color: #ef4444;">{missing}</span>
                            </div>
                        </div>

                    </div>
                    """, height=100)

                    st.divider()

                    for func in ["Identify", "Protect", "Detect", "Respond", "Recover"]:
                        controls_in_func = [c for c in NIST_CSF_CONTROLS if c["function"] == func]
                        color = FUNCTION_COLORS[func]
                        func_covered = sum(1 for c in controls_in_func if result_map.get(c["id"], {}).get("status") == "covered")
                        func_total   = len(controls_in_func)

                        st.markdown(
                            f'<h3 style="color:{color}; margin-top:1.2rem;">{func} '
                            f'<span style="font-size:0.75rem; font-weight:400; color:#64748b;">({func_covered}/{func_total} covered)</span></h3>',
                            unsafe_allow_html=True
                        )

                        for control in controls_in_func:
                            result = result_map.get(control["id"])
                            if not result:
                                continue
                            status  = result["status"]
                            finding = result.get("finding", "")
                            st.markdown(f"""
                            <div class="gap-{status}">
                                <div class="gap-control-id">{control['id']}</div>
                                <div class="gap-control-name">{control['name']}</div>
                                <span class="gap-status-badge badge-{status}">{status.upper()}</span>
                                <div class="gap-finding">{finding}</div>
                            </div>""", unsafe_allow_html=True)
    else:
        st.error("No readable text was found in this PDF.")