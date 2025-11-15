import streamlit as st
import pandas as pd
import snowflake.connector
import google.generativeai as genai
from typing import TypedDict, List, Dict, Any, Optional
from langgraph.graph import StateGraph, END 
from langchain_google_genai import ChatGoogleGenerativeAI
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime
import json
import time
import os
import hashlib # Added for scalable clustering simulation
import uuid    # Added for unique ID generation
import numpy as np # Added for numerical operations
import re      # Added for text cleaning

# NOTE: For advanced clustering (HDBSCAN), uncomment the lines below and install libraries:
# import hdbscan
# from sentence_transformers import SentenceTransformer
# CLUSTER_MIN_SIZE = 5

# ==================== CONFIGURATION & SECRETS (HARDCODED) ====================

# WARNING: REPLACE THESE HARDCODED SECRETS WITH STREAMLIT SECRETS FOR DEPLOYMENT!
GEMINI_API_KEY = "AIzaSyCRbP8fryI0cCfmPoLh9aYQxekxINdg5iQ" 

SNOWFLAKE_CONFIG = {
    'account': 'AUYWMHB-UN24606',
    'user': 'MOHAMEDARSHATH3',
    'password': 'Arshath@302004', 
    'database': 'MOHAMED_ARSHATH_PROJECT_DB', 
    'schema': 'BASELINE_DATA',                  
    'warehouse': 'COMPUTE_WH'
}

TARGET_DB = SNOWFLAKE_CONFIG['database']
TARGET_SCHEMA = SNOWFLAKE_CONFIG['schema']
TARGET_TABLE_NAME = "OPTIMIZER_BASELINE_DATA" 
TARGET_TABLE_FULL = f"{TARGET_DB}.{TARGET_SCHEMA}.{TARGET_TABLE_NAME}"

USABLE_WAREHOUSES = [
    "COMPUTE_WH", "WH_SMALL_SYNTHETIC", "WH_XSMALL_SYNTHETIC", "WH_LARGE_SYNTHETIC", 
    "WH_MEDIUM_SYNTHETIC", "WH_XLARGE_SYNTHETIC", "SNOWFLAKE_LEARNING_WH"
]

# --- Deployment Checks ---
if not GEMINI_API_KEY or GEMINI_API_KEY == "AIzaSyCRbP8fryI0cCfmPoLh9aYQxekxINdg5iQ":
    st.warning("⚠️ Using placeholder/hardcoded API key. Replace for production use.")

if not SNOWFLAKE_CONFIG.get('password') or not SNOWFLAKE_CONFIG.get('account'):
    st.error("FATAL: Essential Snowflake configuration is missing. Cannot connect to Snowflake.")
    st.stop()


# ==============================================================================
# STREAMLIT SETUP AND AGENT STATE DEFINITION
# ==============================================================================

st.set_page_config(page_title="LLM Warehouse Optimizer", page_icon="🤖", layout="wide")

st.markdown("""
<style>
    .main-header {
        background: linear-gradient(90deg, #1e3a8a 0%, #3b82f6 100%);
        padding: 2rem;
        border-radius: 10px;
        color: white;
        text-align: center;
        margin-bottom: 2rem;
    }
    .stButton>button {
        width: 100%;
    }
</style>
""", unsafe_allow_html=True)

class AgentState(TypedDict):
    """The shared memory for the LangGraph workflow."""
    query_id: str
    query_text: str
    original_warehouse: str
    query_type: str
    original_execution_time_ms: float
    original_bytes_scanned: float
    original_efficiency_score: float
    original_estimated_cost: float 
    recommended_warehouse: str
    optimization_reason: str
    optimized_query: str
    optimization_suggestions: List[str]
    execution_result: str
    new_execution_time_ms: float
    actual_warehouse_used: str
    execution_status: str
    error_message: str

# ==================== HELPER FUNCTION: SAFE CONNECTION MANAGER ====================
def _create_snowflake_connection_safe(config: Dict):
    try:
        conn = snowflake.connector.connect(**config)
        warehouse_name = config.get('warehouse', 'COMPUTE_WH')
        cursor = conn.cursor()
        cursor.execute(f"ALTER WAREHOUSE {warehouse_name} RESUME IF SUSPENDED;")
        cursor.close()
        return conn
    except Exception as e:
        st.error(f"❌ Snowflake Connection Failed: {e}")
        raise e

# ==============================================================================
# WAREHOUSE OPTIMIZER AGENT CLASS (LANGGRAPH NODES)
# ==============================================================================

class WarehouseOptimizerAgent:
    def __init__(self, gemini_api_key: str, sf_config: dict):
        self.genai_model = ChatGoogleGenerativeAI(
            model="gemini-2.5-flash", 
            google_api_key=gemini_api_key,
            temperature=0.0
        )
        self.sf_config = sf_config
        self.workflow = self.build_workflow()

    def get_snowflake_connection(self, warehouse=None):
        return snowflake.connector.connect(
            account=self.sf_config['account'],
            user=self.sf_config['user'],
            password=self.sf_config['password'],
            database=self.sf_config['database'],
            schema=self.sf_config['schema'],
            warehouse=warehouse or self.sf_config['warehouse'] 
        )

    # --- LANGGRAPH NODE 1: RECOMMEND WAREHOUSE ---
    def recommend_warehouse(self, state: AgentState) -> AgentState:
        available_wh_list = ", ".join(USABLE_WAREHOUSES)
        
        prompt = f"""You are a Snowflake warehouse optimization expert. Analyze this query and recommend the best warehouse.

Available Warehouses (You MUST choose one of these exact names): {available_wh_list}

Current Query Details:
- Query ID: {state['query_id']}
- Current Warehouse: {state['original_warehouse']}
- Query Type: {state['query_type']}
- Execution Time: {state['original_execution_time_ms']} ms
- Bytes Scanned: {state['original_bytes_scanned']} bytes
- Efficiency Score: {state['original_efficiency_score']}
- Current Cost: ${state['original_estimated_cost']}

Query:
{state['query_text']}

Analyze complexity (SELECT, JOIN, GROUP BY, aggregations, subqueries, data volume) and recommend:
1. Best warehouse name (must be one of: {available_wh_list})
2. Clear reasoning in 2 to 3 lines

Guidelines:
- Match warehouse size to query complexity to balance speed and cost.

Response as JSON:
{{
    "recommended_warehouse": "WAREHOUSE_NAME_FROM_LIST",
    "reasoning": "detailed explanation"
}}"""
        try:
            response = self.genai_model.invoke(prompt)
            result_text = response.content.strip().split("```json")[-1].split("```")[0].strip()
            result = json.loads(result_text)
            recommended_wh = result['recommended_warehouse'].upper().strip()
            
            if recommended_wh not in [w.upper() for w in USABLE_WAREHOUSES]:
                state['recommended_warehouse'] = state['original_warehouse']
                state['optimization_reason'] = f"LLM suggested an invalid warehouse: {recommended_wh}. Using original warehouse."
            else:
                state['recommended_warehouse'] = recommended_wh
                state['optimization_reason'] = result['reasoning']
        except Exception as e:
            state['recommended_warehouse'] = state['original_warehouse']
            state['optimization_reason'] = f"Error in recommendation: {str(e)}. Using original warehouse."
            state['error_message'] = str(e)
        return state
    
    # --- LANGGRAPH NODE 2: OPTIMIZE QUERY ---
    def optimize_query(self, state: AgentState) -> AgentState:
        prompt = f"""
You are an expert Snowflake SQL Performance Engineer. Analyze and optimize the following query for maximum performance while maintaining identical results.
Query:
{state['query_text']}
Original Warehouse: {state['original_warehouse']}
Recommended Warehouse: {state['recommended_warehouse']}

Optimization Guidelines:
- Replace SELECT * with explicit columns.
- Suggest filtering strategies (e.g., pruning).
- Suggest reordering JOIN clauses if beneficial.
- Always maintain functional equivalence to the original query.

Response as JSON:
{{
    "optimized_query": "REWRITTEN SQL QUERY HERE",
    "optimizations": ["List of specific changes made (e.g., Removed SELECT * in line 5)."]
}}"""
        try:
            response = self.genai_model.invoke(prompt)
            result_text = response.content.strip().split("```json")[-1].split("```")[0].strip()
            result = json.loads(result_text)
            state['optimized_query'] = result['optimized_query']
            state['optimization_suggestions'] = result['optimizations']
        except Exception as e:
            state['optimized_query'] = state['query_text'] 
            state['optimization_suggestions'] = [f"Using original query due to optimization error: {str(e)}"]
            state['error_message'] = str(e)
        return state
    
    # --- LANGGRAPH NODE 3: EXECUTE IN SNOWFLAKE (SCALABLE CHECK ADDED) ---
    def execute_in_snowflake(self, state: AgentState) -> AgentState:
        """Step 3: Execute optimized query and retrieve official execution time."""
        
        # SCALABILITY BYPASS CHECK
        if state.get('execution_status') == 'SKIPPED_BY_CLUSTER':
            state['new_execution_time_ms'] = 0.0
            state['actual_warehouse_used'] = state['recommended_warehouse']
            return state
            
        conn = None
        try:
            conn = self.get_snowflake_connection(warehouse=state['recommended_warehouse'])
            cursor = conn.cursor()
            cursor.execute("SELECT CURRENT_WAREHOUSE()")
            state['actual_warehouse_used'] = cursor.fetchone()[0]
            start_time = time.time()
            cursor.execute(state['optimized_query'])
            execution_duration_ms = (time.time() - start_time) * 1000 
            state['new_execution_time_ms'] = execution_duration_ms
            state['execution_status'] = 'SUCCESS'
            state['execution_result'] = f"Query executed successfully on {state['actual_warehouse_used']}."
        except Exception as e:
            state['execution_status'] = 'FAILED'
            state['error_message'] = str(e)
            state['new_execution_time_ms'] = 0
        finally:
            if conn: conn.close()
        return state
    
    # --- NEW SCALABILITY METHOD: LLM ANALYSIS ONLY ---
    def analyze_only(self, state: AgentState) -> AgentState:
        """Runs only the recommendation and optimization nodes (Steps 1 & 2)."""
        state = self.recommend_warehouse(state)
        state = self.optimize_query(state)
        state['execution_status'] = 'ANALYZED'
        state['execution_result'] = 'Plan derived via cluster analysis.'
        state['new_execution_time_ms'] = 0.0 
        return state
    
    def build_workflow(self):
        workflow = StateGraph(AgentState)
        workflow.add_node("recommend_warehouse", self.recommend_warehouse)
        workflow.add_node("optimize_query", self.optimize_query)
        workflow.add_node("execute_query", self.execute_in_snowflake)
        workflow.set_entry_point("recommend_warehouse")
        workflow.add_edge("recommend_warehouse", "optimize_query")
        workflow.add_edge("optimize_query", "execute_query")
        workflow.add_edge("execute_query", END)
        return workflow.compile()
    
    def process_query_record(self, record: dict) -> AgentState:
        # Standardizes column names from pandas fetch/synthetic data to AgentState keys
        return AgentState(
            query_id=str(record.get('QUERY_ID', record.get('query_id', str(uuid.uuid4())))), 
            query_text=record.get('QUERY_TEXT', record.get('query_text', '')), 
            original_warehouse=record.get('ORIGINAL_WAREHOUSE', record.get('original_wh', self.sf_config['warehouse'])), 
            query_type=record.get('QUERY_TYPE', record.get('query_type', 'SELECT')),
            original_execution_time_ms=float(record.get('ORIGINAL_EXECUTION_TIME_MS', record.get('execution_time_ms', 0))), 
            original_bytes_scanned=float(record.get('ORIGINAL_BYTES_SCANNED', record.get('bytes_scanned', 0))),
            original_efficiency_score=float(record.get('ORIGINAL_EFFICIENCY_SCORE', record.get('efficiency_score', 0))), 
            original_estimated_cost=float(record.get('ORIGINAL_ESTIMATED_COST', record.get('estimated_cost', 0))),
            recommended_warehouse=record.get('RECOMMENDED_WAREHOUSE', record.get('recommended_wh', '')), 
            optimization_reason=record.get('OPTIMIZATION_REASON', record.get('llm_reason', '')), 
            optimized_query=record.get('OPTIMIZED_QUERY', record.get('optimized_query', '')),
            optimization_suggestions=record.get('OPTIMIZATION_SUGGESTIONS', []), 
            execution_result=record.get('EXECUTION_RESULT', ''), 
            new_execution_time_ms=float(record.get('NEW_EXECUTION_TIME_MS', 0)),
            actual_warehouse_used=record.get('ACTUAL_WAREHOUSE_USED', ''), 
            execution_status=record.get('EXECUTION_STATUS', 'PENDING'), 
            error_message=record.get('ERROR_MESSAGE', '')
        )

# ==============================================================================
# SCALABILITY CORE FUNCTIONS (CLUSTERING)
# ==============================================================================

def scalable_batch_analysis_with_clustering(df_queries: pd.DataFrame, agent: WarehouseOptimizerAgent):
    """
    Implements the core logic of the scalable optimization pipeline using HASHING 
    to simulate clustering/sampling and dramatically reduce LLM calls.
    """
    
    st.markdown("---")
    st.subheader("Simulated Clustering & Sampling")
    
    # 1. CLEAN AND HASH (SIMULATED CLUSTERING)
    df_queries['CLEAN_QUERY_TEXT'] = df_queries['QUERY_TEXT'].str.lower().str.replace(r'\s+', ' ', regex=True).str.strip()
    df_queries['QUERY_HASH'] = df_queries['CLEAN_QUERY_TEXT'].apply(lambda x: hashlib.sha1(x.encode()).hexdigest())

    # 2. SELECT REPRESENTATIVE SAMPLE
    representative_df = df_queries.drop_duplicates(subset=['QUERY_HASH'], keep='first').reset_index(drop=True)
    
    st.info(f"Input records: **{len(df_queries)}** | Unique patterns (clusters): **{len(representative_df)}**")
    
    # 3. BATCHED LLM OPTIMIZATION (on the small sample only)
    recommendations_for_reps = {}
    progress_bar = st.progress(0)
    rep_items = list(representative_df.iterrows())
    
    for i, (_, rep_data) in enumerate(rep_items):
        progress_bar.progress((i + 1) / len(rep_items), text=f"STEP 4/5: Analyzing representative pattern {i+1}/{len(rep_items)}...")
        
        # Prepare and run the LLM analysis (Node 1 & 2)
        initial_state = agent.process_query_record(rep_data.to_dict())
        final_state = agent.analyze_only(initial_state) # Use the efficient analyze_only method
        
        # Use the query_id as the key for mapping
        recommendations_for_reps[final_state['query_id']] = final_state

    # 4. MAPPING RESULTS BACK
    final_states_list = []
    
    for _, row in df_queries.iterrows():
        # Find the representative result using the hash
        rep_row = representative_df[representative_df['QUERY_HASH'] == row['QUERY_HASH']].iloc[0].to_dict()
        rep_id = rep_row['QUERY_ID']
            
        rep_result = recommendations_for_reps.get(rep_id)
        
        # Initialize the state with original data
        state = agent.process_query_record(row.to_dict())
        
        # Apply the optimization plan from the representative
        if rep_result:
            state['recommended_warehouse'] = rep_result['recommended_warehouse']
            state['optimization_reason'] = rep_result['optimization_reason']
            state['optimized_query'] = rep_result['optimized_query']
            state['optimization_suggestions'] = rep_result['optimization_suggestions']
            state['execution_status'] = 'SKIPPED_BY_CLUSTER' 
            state['execution_result'] = f"Plan applied from REP Query ID: {rep_id}"
        else:
            state['execution_status'] = 'MAPPING_FAILED'
            state['execution_result'] = f"Could not map plan from REP Query ID: {rep_id}"
        
        final_states_list.append(state)
        
    progress_bar.empty()
    st.success(f"🎉 Analysis complete! All {len(final_states_list)} records received an optimization plan.")
    return final_states_list


def set_page(page_name):
    """Updates session state to control which page function is rendered."""
    st.session_state.page = page_name

# ----------------- PAGE 1: DATA LOADING AND SELECTION -----------------
def page_1_load_data():
    
    st.header("📊 Load Data and Select Records")
    
    col1, col2 = st.columns([3, 1])
    with col1:
        st.info(f"Source Table: **{TARGET_TABLE_FULL}**")
    with col2:
        if st.button("🔄 Load All Records", type="primary"):
            try:
                with st.spinner("Connecting to Snowflake and loading records..."):
                    conn = _create_snowflake_connection_safe(SNOWFLAKE_CONFIG)
                    
                    query = f"""
                    SELECT 
                        QUERY_ID, QUERY_TEXT, ORIGINAL_WAREHOUSE, QUERY_TYPE,
                        ORIGINAL_EXECUTION_TIME_MS, ORIGINAL_BYTES_SCANNED,
                        ORIGINAL_EFFICIENCY_SCORE, 
                        ORIGINAL_COST AS ORIGINAL_ESTIMATED_COST 
                    FROM {TARGET_TABLE_FULL}
                    ORDER BY Random();
                    """
                    df = pd.read_sql(query, conn)
                    conn.close()
                    
                    df.columns = df.columns.str.upper()
                    st.session_state.all_records = df
                    st.success(f"✅ Loaded {len(df)} records successfully! Ready for analysis.")
            except Exception as e:
                st.error(f"❌ Error loading data. Ensure table {TARGET_TABLE_FULL} exists: {str(e)}")
                st.stop()

    if st.session_state.all_records is not None:
        df = st.session_state.all_records
        st.subheader(f"📋 All Records ({len(df)} total)")
        st.dataframe(df, use_container_width=True, height=300)
        st.markdown("---")

        st.subheader("🎯 Select Records for Analysis")
        num_records = st.slider(
            "How many records should the LLM analyze?",
            min_value=1,
            max_value=len(df),
            value=min(50, len(df)),
            key="num_records_slider",
            help="Select number of records. Clustering will ensure few LLM calls."
        )
        
        st.session_state.selected_records = df.head(num_records)
        st.info(f"📌 Selected {num_records} records.")
        st.dataframe(st.session_state.selected_records, use_container_width=True, height=200)

        st.markdown("---")
        if st.button("➡️ Proceed to Analysis and optimization", type="secondary", use_container_width=True):
            if len(st.session_state.selected_records) > 0:
                st.session_state.execution_results = [] 
                st.session_state.recommendations = []
                set_page("analysis")
                st.rerun()
            else:
                st.warning("Please load and select at least one record.")

# ----------------- PAGE 2: LLM ANALYSIS AND OPTIMIZATION -----------------
def page_2_analysis():
    st.header("🤖  LLM Analysis and Optimization (Scalable Clustering)")
    
    if st.button("⬅️ Back to Data Selection", key="back_to_page1"):
        set_page("data_load")
        st.rerun()

    num_records = len(st.session_state.selected_records)
    st.info(f"Analyzing {num_records} selected queries using representative sampling...")

    col1, col2 = st.columns(2)
    
    with col1:
        if st.button("🧠 Run Scalable LLM Analysis & Get Recommendations", type="primary", use_container_width=True):
            try:
                agent = WarehouseOptimizerAgent(GEMINI_API_KEY, SNOWFLAKE_CONFIG)
                
                # CALL THE SCALABLE FUNCTION HERE
                st.session_state.recommendations = scalable_batch_analysis_with_clustering(
                    st.session_state.selected_records.copy(), 
                    agent
                )
                
                st.rerun()
                
            except Exception as e:
                st.error(f"❌ Error during scalable analysis: {str(e)}")
    
    with col2:
        if st.button("🗑️ Clear Recommendations", use_container_width=True, key="clear_recs"):
            st.session_state.recommendations = []
            st.session_state.execution_results = []
            st.rerun()

    # Display recommendations
    if st.session_state.recommendations:
        st.subheader("💡 LLM Recommendations Details")
        
        df_recs = pd.DataFrame(st.session_state.recommendations)
        # Use query_text to determine unique patterns, reflecting the number of LLM calls
        total_analyzed = len(df_recs['optimized_query'].unique())
        st.markdown(f"**Total records selected:** {num_records} | **Unique patterns analyzed by LLM:** **{total_analyzed}**")
        
        for rec in st.session_state.recommendations:
            q_id = rec.get('query_id', 'N/A')
            with st.expander(f"🔍 Query: {q_id} | Recommended WH: {rec['recommended_warehouse']} | Status: {rec['execution_status']}", expanded=False):
                colA, colB = st.columns(2)
                
                with colA:
                    st.markdown("### 📊 Original Performance")
                    st.metric("Warehouse", rec['original_warehouse'])
                    st.metric("Execution Time", f"{rec['original_execution_time_ms']:.0f} ms")
                    st.metric("Cost", f"${rec['original_estimated_cost']:.4f}")
                    st.code(rec['query_text'], language="sql")
                
                with colB:
                    st.markdown("### 🚀 LLM Recommendation (Mapped)")
                    st.metric("Recommended WH", rec['recommended_warehouse'], 
                              delta=f"Change from {rec['original_warehouse']}" if rec['recommended_warehouse'] != rec['original_warehouse'] else "Same")
                    st.markdown("**Reasoning:**")
                    st.info(rec['optimization_reason'])
                    if rec['optimization_suggestions']:
                        st.markdown("**Query Optimizations:**")
                        for opt in rec['optimization_suggestions']:
                            st.markdown(f"✓ {opt}")
                    st.markdown("**Optimized Query:**")
                    st.code(rec['optimized_query'], language="sql")

        st.markdown("---")
        st.header("⚡  Execution Validation")
        if st.button("🚀 Proceed to Validation (Step 4)", type="primary", use_container_width=True):
            set_page("validation")
            st.rerun()

# ----------------- PAGE 3: VALIDATION AND RESULTS -----------------
def page_3_validation_and_results():
    st.header("📈 Execution Validation and Results")
    
    if st.button("⬅️ Back to Analysis", key="back_to_page2"):
        set_page("analysis")
        st.rerun()

    st.warning("⚠️ Executing validation runs the optimized queries on the recommended warehouses in Snowflake to get official time. NOTE: We only execute a **strategic sample** of 5 queries due to the clustering approach.")
    
    if not st.session_state.execution_results:
        # Filter down to a small, strategic sample (max 5 records)
        records_to_execute = [r for r in st.session_state.recommendations if r['execution_status'] == 'SKIPPED_BY_CLUSTER'][:5]
        
        if st.button(f"⚡ Run Execution Validation on Strategic Sample ({len(records_to_execute)} queries)", type="primary", use_container_width=True):
            try:
                agent = WarehouseOptimizerAgent(GEMINI_API_KEY, SNOWFLAKE_CONFIG)
                st.session_state.execution_results = []
                progress_bar = st.progress(0)
                
                executed_results = []
                
                for idx, rec in enumerate(records_to_execute):
                    progress_bar.progress((idx + 1) / len(records_to_execute), text=f"⚡ Executing Query {idx + 1}/{len(records_to_execute)}: {rec['query_id']}")
                    
                    cloned_state = dict(rec) 
                    cloned_state['execution_status'] = 'PENDING' 
                    
                    # Execute the full LangGraph flow for validation
                    result = agent.workflow.invoke(cloned_state) 
                    executed_results.append(result)
                    
                # Collect the non-executed results and append the executed ones
                non_executed = [r for r in st.session_state.recommendations if r not in records_to_execute]
                st.session_state.execution_results = executed_results + non_executed
                
                st.success("🎉 Sample validation complete!")
                st.rerun()
                
            except Exception as e:
                st.error(f"❌ Execution error: {str(e)}")
    
    if st.session_state.execution_results:
        valid_results = [r for r in st.session_state.execution_results if r['execution_status'] == 'SUCCESS']
        
        # --- RESULTS METRICS (Only for successfully validated samples) ---
        st.subheader("Summary Performance Metrics (Validated Sample)")
        successful = len(valid_results)
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric(" Successful Runs", f"{successful}/{len(valid_results) + len([r for r in st.session_state.execution_results if r['execution_status'] == 'FAILED'])}")
        with col2:
            avg_improvement = sum([
                ((r['original_execution_time_ms'] - r['new_execution_time_ms']) / r['original_execution_time_ms'] * 100)
                for r in valid_results
            ]) / max(len(valid_results), 1)
            st.metric("⚡ Avg Improvement", f"{avg_improvement:.1f}%")
        
        with col3:
            total_time_saved = sum([
                (r['original_execution_time_ms'] - r['new_execution_time_ms'])
                for r in valid_results
            ])
            st.metric("⏱️ Total Time Saved", f"{total_time_saved/1000:.1f}s")
        
        with col4:
            st.metric("📊 Total Optimization Plans", len(st.session_state.recommendations))
        
        # --- VISUALIZATION ---
        if len(valid_results) > 0:
            st.subheader("Execution Time Comparison (Validated Sample)")
            fig = go.Figure()
            query_ids = [r['query_id'][:8] for r in valid_results]
            original_times = [r['original_execution_time_ms'] for r in valid_results]
            new_times = [r['new_execution_time_ms'] for r in valid_results]
            
            fig.add_trace(go.Bar(name='Original Time', x=query_ids, y=original_times, marker_color='#ef4444'))
            fig.add_trace(go.Bar(name='Optimized Time', x=query_ids, y=new_times, marker_color='#22c55e'))
            
            fig.update_layout(
                title='Original vs. Optimized Execution Time (ms) - Validated Sample',
                xaxis_title='Query ID',
                yaxis_title='Time (ms)',
                barmode='group',
                height=400
            )
            st.plotly_chart(fig, use_container_width=True)

        # --- RESULTS DATAFRAME ---
        st.subheader("Detailed Results Table (Full List with Sample Validation)")
        results_df = pd.DataFrame([{
            'Query ID': r['query_id'],
            'Original WH': r['original_warehouse'],
            'Recommended WH': r['recommended_warehouse'],
            'Original Time (ms)': f"{r['original_execution_time_ms']:.0f}",
            'New Time (ms)': f"{r['new_execution_time_ms']:.0f}" if r['new_execution_time_ms'] > 0 else 'N/A',
            'Improvement': f"{((r['original_execution_time_ms'] - r['new_execution_time_ms']) / r['original_execution_time_ms'] * 100):.1f}%" if r['new_execution_time_ms'] > 0 and r['original_execution_time_ms'] > 0 else 'N/A',
            'Status': r['execution_status'],
            'Result': r['execution_result']
        } for r in st.session_state.execution_results])
        
        st.dataframe(results_df, use_container_width=True)
        
        csv = results_df.to_csv(index=False)
        st.download_button(
            label="📥 Download Results CSV",
            data=csv,
            file_name=f"warehouse_optimization_scalable_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
            mime="text/csv"
        )


# ==============================================================================
# MAIN EXECUTION ROUTER
# ==============================================================================

def main():
    st.markdown("""
    <div class="main-header">
        <h1>🤖 Snowflake Warehouse Optimizer with AI Agent</h1>
        <p>Analyzes query patterns, recommends scaled warehouses, optimizes SQL, and verifies automatically using Gemini & LangGraph.</p>
    </div>
    """, unsafe_allow_html=True)

    # --- INITIALIZE ALL SESSION STATE AT THE START ---
    if 'page' not in st.session_state: st.session_state.page = "data_load"
    if 'all_records' not in st.session_state: st.session_state.all_records = None
    if 'selected_records' not in st.session_state: st.session_state.selected_records = None
    if 'recommendations' not in st.session_state: st.session_state.recommendations = []
    if 'execution_results' not in st.session_state: st.session_state.execution_results = []
    
    if st.session_state.page == "data_load":
        page_1_load_data()
    elif st.session_state.page == "analysis":
        if st.session_state.get('selected_records') is None:
            set_page("data_load"); st.rerun()
        page_2_analysis()
    elif st.session_state.page == "validation":
        if not st.session_state.get('recommendations'):
            set_page("analysis"); st.rerun()
        page_3_validation_and_results()

if __name__ == "__main__":
    main()
