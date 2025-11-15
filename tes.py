import streamlit as st
import pandas as pd
import snowflake.connector
import google.generativeai as genai
from typing import TypedDict, List, Dict
from langgraph.graph import StateGraph, END # Required for Agentic Workflow
from langchain_google_genai import ChatGoogleGenerativeAI
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime
import json
import time
import os
# dotenv import REMOVED

# ==================== CONFIGURATION & SECRETS (HARDCODED) ====================

# 1. Gemini API Key (HARDCODED)
GEMINI_API_KEY = "AIzaSyCRbP8fryI0cCfmPoLh9aYQxekxINdg5iQ" 

# 2. Snowflake Configuration (HARDCODED to specific paths)
SNOWFLAKE_CONFIG = {
    'account': 'AUYWMHB-UN24606',
    'user': 'MOHAMEDARSHATH3',
    'password': 'Arshath@302004', 
    'database': 'MOHAMED_ARSHATH_PROJECT_DB', # FIXED: Target Database
    'schema': 'BASELINE_DATA',                # FIXED: Target Schema
    'warehouse': 'COMPUTE_WH'
}

# Target table path components
TARGET_DB = SNOWFLAKE_CONFIG['database']
TARGET_SCHEMA = SNOWFLAKE_CONFIG['schema']
TARGET_TABLE_NAME = "OPTIMIZER_BASELINE_DATA" 
TARGET_TABLE_FULL = f"{TARGET_DB}.{TARGET_SCHEMA}.{TARGET_TABLE_NAME}"


# Application constraints and resources
USABLE_WAREHOUSES = [
    "COMPUTE_WH", 
    "WH_SMALL_SYNTHETIC", 
    "WH_XSMALL_SYNTHETIC", 
    "WH_LARGE_SYNTHETIC", 
    "WH_MEDIUM_SYNTHETIC", 
    "WH_XLARGE_SYNTHETIC", 
    "SNOWFLAKE_LEARNING_WH"
]

# --- Deployment Checks ---
if not GEMINI_API_KEY:
    st.error("FATAL: GEMINI_API_KEY is missing. Check the code's hardcoded value.")
    st.stop()

if not SNOWFLAKE_CONFIG.get('password') or not SNOWFLAKE_CONFIG.get('account'):
    st.error("FATAL: Essential Snowflake configuration is missing. Cannot connect to Snowflake.")
    st.stop()


# ==============================================================================
# STREAMLIT SETUP AND AGENT STATE DEFINITION
# ==============================================================================

st.set_page_config(
    page_title="LLM Warehouse Optimizer",
    page_icon="🤖",
    layout="wide"
)

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
    """Establishes a Snowflake connection and automatically resumes the warehouse."""
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
        """Create Snowflake connection with optional dynamic warehouse override."""
        return snowflake.connector.connect(
            account=self.sf_config['account'],
            user=self.sf_config['user'],
            password=self.sf_config['password'],
            database=self.sf_config['database'],
            schema=self.sf_config['schema'],
            warehouse=warehouse or self.sf_config['warehouse'] 
        )
    
    # ----------------- LANGGRAPH NODE 1: RECOMMEND WAREHOUSE -----------------
    def recommend_warehouse(self, state: AgentState) -> AgentState:
        """Step 1: LLM recommends optimal warehouse based on query history."""
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
- Simple SELECT with filters → X-SMALL or SMALL warehouses (WH_XSMALL_SYNTHETIC, SNOWFLAKE_LEARNING_WH)
- Complex joins, multiple aggregations → LARGE warehouses (WH_LARGE_SYNTHETIC, WH_XLARGE_SYNTHETIC)

Response as JSON:
{{
    "recommended_warehouse": "WAREHOUSE_NAME_FROM_LIST",
    "reasoning": "detailed explanation"
}}"""
        
        try:
            response = self.genai_model.invoke(prompt)
            result_text = response.content.strip()
            
            if "```json" in result_text:
                result_text = result_text.split("```json")[1].split("```")[0].strip()
            elif "```" in result_text:
                result_text = result_text.split("```")[1].split("```")[0].strip()
            
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
    
    # ----------------- LANGGRAPH NODE 2: OPTIMIZE QUERY -----------------
    def optimize_query(self, state: AgentState) -> AgentState:
        """Step 2: LLM optimizes SQL query based on best practices."""
        
        prompt = f"""
You are an expert Snowflake SQL Performance Engineer. Analyze and optimize the following query for maximum performance while maintaining identical results.
... (Optimization prompt omitted for brevity) ...
Response as JSON:
{{
    "optimized_query": "REWRITTEN SQL QUERY HERE",
    "optimizations": ["List of specific changes made (e.g., Removed SELECT * in line 5)."]
}}"""
        
        try:
            response = self.genai_model.invoke(prompt)
            result_text = response.content.strip()
            
            if "```json" in result_text: result_text = result_text.split("```json")[1].split("```")[0].strip()
            elif "```" in result_text: result_text = result_text.split("```")[1].split("```")[0].strip()
            
            result = json.loads(result_text)
            state['optimized_query'] = result['optimized_query']
            state['optimization_suggestions'] = result['optimizations']
            
        except Exception as e:
            state['optimized_query'] = state['query_text'] 
            state['optimization_suggestions'] = [f"Using original query due to optimization error: {str(e)}"]
            state['error_message'] = str(e)
        
        return state
    
    # ----------------- LANGGRAPH NODE 3: EXECUTE IN SNOWFLAKE (ACCURATE TIMING) -----------------
    def execute_in_snowflake(self, state: AgentState) -> AgentState:
        """Step 3: Execute optimized query and retrieve official execution time."""
        
        conn = None
        try:
            conn = self.get_snowflake_connection(warehouse=state['recommended_warehouse'])
            cursor = conn.cursor()
    
            # Get actual warehouse name for logging
            cursor.execute("SELECT CURRENT_WAREHOUSE()")
            state['actual_warehouse_used'] = cursor.fetchone()[0]
            
            # Execute the optimized query
            start_time = time.time()
            cursor.execute(state['optimized_query'])
            
            execution_duration_ms = (time.time() - start_time) * 1000 
            
            state['new_execution_time_ms'] = execution_duration_ms
            state['execution_status'] = 'SUCCESS'
            state['execution_result'] = f"Query executed successfully on {state['actual_warehouse_used']}."
            
            cursor.close()
            
        except Exception as e:
            state['execution_status'] = 'FAILED'
            state['execution_result'] = f"Execution failed: {str(e)}"
            state['error_message'] = str(e)
            state['new_execution_time_ms'] = 0
            state['actual_warehouse_used'] = 'N/A'
        finally:
            if conn:
                conn.close()
        
        return state
    
    def build_workflow(self):
        """Defines the sequential LangGraph pipeline."""
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
        """Initializes the state and runs the compiled LangGraph workflow."""
        initial_state = AgentState(
            query_id=str(record['QUERY_ID']), query_text=record['QUERY_TEXT'], original_warehouse=record['ORIGINAL_WAREHOUSE'], query_type=record['QUERY_TYPE'],
            original_execution_time_ms=float(record['ORIGINAL_EXECUTION_TIME_MS']), original_bytes_scanned=float(record['ORIGINAL_BYTES_SCANNED']),
            original_efficiency_score=float(record['ORIGINAL_EFFICIENCY_SCORE']), original_estimated_cost=float(record['ORIGINAL_ESTIMATED_COST']),
            recommended_warehouse='', optimization_reason='', optimized_query='',
            optimization_suggestions=[], execution_result='', new_execution_time_ms=0,
            actual_warehouse_used='', execution_status='PENDING', error_message=''
        )
        return self.workflow.invoke(initial_state)

def set_page(page_name):
    """Updates session state to control which page function is rendered."""
    st.session_state.page = page_name

# ----------------- PAGE 1: DATA LOADING AND SELECTION -----------------
def page_1_load_data():
    
    # Define target table path
    TARGET_DB = SNOWFLAKE_CONFIG.get('database')
    TARGET_SCHEMA = SNOWFLAKE_CONFIG.get('schema')
    TARGET_TABLE_FULL = f"{TARGET_DB}.{TARGET_SCHEMA}.OPTIMIZER_BASELINE_DATA"
    
    st.header("📊 Load Data and Select Records")
    
    col1, col2 = st.columns([3, 1])
    with col1:
        st.info(f"Source Table: **{TARGET_TABLE_FULL}**")
    with col2:
        if st.button("🔄 Load All Records", type="primary"):
            try:
                with st.spinner("Connecting to Snowflake and loading records..."):
                    conn = _create_snowflake_connection_safe(SNOWFLAKE_CONFIG)
                    
                    # Target the verified baseline table and alias ORIGINAL_COST
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
                    
                    # Ensure column names are uppercase to match AgentState
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
            value=min(5, len(df)),
            key="num_records_slider",
            help="Select number of records for optimization."
        )
        
        st.session_state.selected_records = df.head(num_records)
        st.info(f"📌 Selected {num_records} records.")
        st.dataframe(st.session_state.selected_records, use_container_width=True, height=200)

        st.markdown("---")
        if st.button("➡️ Proceed to Analysis and optimization", type="secondary", use_container_width=True):
            if len(st.session_state.selected_records) > 0:
                st.session_state.execution_results = [] 
                set_page("analysis")
                st.rerun()
            else:
                st.warning("Please load and select at least one record.")

# ----------------- PAGE 2: LLM ANALYSIS AND OPTIMIZATION -----------------
def page_2_analysis():
    st.header("🤖  LLM Analysis and Optimization")
    
    if st.button("⬅️ Back to Data Selection", key="back_to_page1"):
        set_page("data_load")
        st.rerun()

    num_records = len(st.session_state.selected_records)
    st.info(f"Analyzing {num_records} selected queries...")

    col1, col2 = st.columns(2)
    
    with col1:
        if st.button("🧠 Run LLM Analysis & Get Recommendations", type="primary", use_container_width=True):
            try:
                agent = WarehouseOptimizerAgent(GEMINI_API_KEY, SNOWFLAKE_CONFIG)
                st.session_state.recommendations = []
                
                progress_bar = st.progress(0)
                status_text = st.empty()
                
                for idx, row in st.session_state.selected_records.iterrows():
                    status_text.text(f"🔍 Analyzing Query {idx + 1}/{num_records}: {row['QUERY_ID']}")
                    
                    # --- Convert Pandas row to AgentState compatible dict ---
                    record_dict = row.to_dict()
                    for key in ['ORIGINAL_EXECUTION_TIME_MS', 'ORIGINAL_BYTES_SCANNED', 'ORIGINAL_EFFICIENCY_SCORE', 'ORIGINAL_ESTIMATED_COST']:
                        record_dict[key] = float(record_dict.get(key, 0)) if pd.notna(record_dict.get(key)) else 0.0

                    initial_state = AgentState(
                        query_id=str(record_dict['QUERY_ID']), query_text=record_dict['QUERY_TEXT'], original_warehouse=record_dict['ORIGINAL_WAREHOUSE'], query_type=record_dict['QUERY_TYPE'],
                        original_execution_time_ms=record_dict['ORIGINAL_EXECUTION_TIME_MS'], original_bytes_scanned=record_dict['ORIGINAL_BYTES_SCANNED'],
                        original_efficiency_score=record_dict['ORIGINAL_EFFICIENCY_SCORE'], original_estimated_cost=record_dict['ORIGINAL_ESTIMATED_COST'],
                        recommended_warehouse='', optimization_reason='', optimized_query='',
                        optimization_suggestions=[], execution_result='', new_execution_time_ms=0,
                        actual_warehouse_used='', execution_status='PENDING', error_message=''
                    )
                    
                    # Execute LangGraph: Recommend -> Optimize
                    state = agent.recommend_warehouse(initial_state)
                    state = agent.optimize_query(state)
                    
                    st.session_state.recommendations.append(state)
                    progress_bar.progress((idx + 1) / num_records)
                
                status_text.text("✅ Analysis complete!")
                st.success(f"🎉 Generated recommendations for {len(st.session_state.recommendations)} queries!")
                st.rerun()
                
            except Exception as e:
                st.error(f"❌ Error during analysis: {str(e)}")
    
    with col2:
        if st.button("🗑️ Clear Recommendations", use_container_width=True, key="clear_recs"):
            st.session_state.recommendations = []
            st.session_state.execution_results = []
            st.rerun()

    # Display recommendations
    if st.session_state.recommendations:
        st.subheader("💡 LLM Recommendations Details")
        
        for rec in st.session_state.recommendations:
            with st.expander(f"🔍 Query: {rec['query_id']} | Recommended WH: {rec['recommended_warehouse']}", expanded=False):
                colA, colB = st.columns(2)
                
                with colA:
                    st.markdown("### 📊 Original Performance")
                    st.metric("Warehouse", rec['original_warehouse'])
                    st.metric("Execution Time", f"{rec['original_execution_time_ms']:.0f} ms")
                    st.metric("Cost", f"${rec['original_estimated_cost']:.4f}")
                    st.metric("Efficiency Score", f"{rec['original_efficiency_score']:.2f}")
                    st.code(rec['query_text'], language="sql")
                
                with colB:
                    st.markdown("### 🚀 LLM Recommendation")
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
        st.header("⚡  Execution Validation")
        if st.button("🚀 Proceed to Validation (Step 4)", type="primary", use_container_width=True):
            set_page("validation")
            st.rerun()

# ----------------- PAGE 3: VALIDATION AND RESULTS -----------------
def page_3_validation_and_results():
    st.header("📈 Execution Validation and Results")
    
    if st.button("⬅️ Back to Analysis", key="back_to_page2"):
        set_page("analysis")
        st.rerun()

    st.warning("⚠️ Executing validation runs the optimized queries on the recommended warehouses in Snowflake to get official time.")
    
    if not st.session_state.execution_results:
        if st.button("⚡ Run Execution Validation", type="primary", use_container_width=True):
            try:
                agent = WarehouseOptimizerAgent(GEMINI_API_KEY, SNOWFLAKE_CONFIG)
                st.session_state.execution_results = []
                progress_bar = st.progress(0)
                status_text = st.empty()
                total_recs = len(st.session_state.recommendations)
                
                for idx, rec in enumerate(st.session_state.recommendations):
                    status_text.text(f"⚡ Executing Query {idx + 1}/{total_recs}: {rec['query_id']}")
                    
                    # *** CLONING THE RECORD FOR ISOLATION ***
                    cloned_state = dict(rec) 
                    
                    # Execute the final node of the LangGraph flow
                    result = agent.execute_in_snowflake(cloned_state)
                    st.session_state.execution_results.append(result)
                    
                    progress_bar.progress((idx + 1) / total_recs)
                
                status_text.text("✅ All executions complete!")
                st.success("🎉 All queries attempted!")
                st.rerun()
                
            except Exception as e:
                st.error(f"❌ Execution error: {str(e)}")
    
    if st.session_state.execution_results:
        valid_results = [r for r in st.session_state.execution_results if r['new_execution_time_ms'] > 0 and r['original_execution_time_ms'] > 0]
        
        # --- RESULTS METRICS ---
        st.subheader("Summary Performance Metrics")
        successful = len([r for r in st.session_state.execution_results if r['execution_status'] == 'SUCCESS'])
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric(" Successful Runs", f"{successful}/{len(st.session_state.execution_results)}")
        with col2:
            avg_improvement = sum([
                ((r['original_execution_time_ms'] - r['new_execution_time_ms']) / r['original_execution_time_ms'] * 100)
                for r in valid_results
            ]) / max(len(valid_results), 1)
            st.metric("⚡ Avg Improvement", f"{avg_improvement:.1f}%")
        
        with col3:
            total_time_saved = sum([
                (r['original_execution_time_ms'] - r['new_execution_time_ms'])
                for r in st.session_state.execution_results
            ])
            st.metric("⏱️ Total Time Saved", f"{total_time_saved/1000:.1f}s")
        
        with col4:
            st.metric("📊 Total Queries Analyzed", len(st.session_state.execution_results))
        
        # --- VISUALIZATION ---
        if len(valid_results) > 0:
            st.subheader("Execution Time Comparison")
            fig = go.Figure()
            query_ids = [r['query_id'][:8] for r in valid_results]
            original_times = [r['original_execution_time_ms'] for r in valid_results]
            new_times = [r['new_execution_time_ms'] for r in valid_results]
            
            fig.add_trace(go.Bar(name='Original Time', x=query_ids, y=original_times, marker_color='#ef4444'))
            fig.add_trace(go.Bar(name='Optimized Time', x=query_ids, y=new_times, marker_color='#22c55e'))
            
            fig.update_layout(
                title='Original vs. Optimized Execution Time (ms)',
                xaxis_title='Query ID',
                yaxis_title='Time (ms)',
                barmode='group',
                height=400
            )
            st.plotly_chart(fig, use_container_width=True)

        # --- RESULTS DATAFRAME ---
        st.subheader("Detailed Results Table")
        results_df = pd.DataFrame([{
            'Query ID': r['query_id'],
            'Original WH': r['original_warehouse'],
            'Recommended WH': r['recommended_warehouse'],
            'Original Time (ms)': f"{r['original_execution_time_ms']:.0f}",
            'New Time (ms)': f"{r['new_execution_time_ms']:.0f}",
            'Improvement': f"{((r['original_execution_time_ms'] - r['new_execution_time_ms']) / r['original_execution_time_ms'] * 100):.1f}%" if r['new_execution_time_ms'] > 0 and r['original_execution_time_ms'] > 0 else 'N/A',
            'Status': r['execution_status'],
            'Result': r['execution_result']
        } for r in st.session_state.execution_results])
        
        st.dataframe(results_df, use_container_width=True)
        
        # Download results
        csv = results_df.to_csv(index=False)
        st.download_button(
            label="📥 Download Results CSV",
            data=csv,
            file_name=f"warehouse_optimization_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
            mime="text/csv"
        )


# ==============================================================================
# MAIN EXECUTION ROUTER
# ==============================================================================

def main():
    # Render the fixed, global header
    st.markdown("""
    <div class="main-header">
        <h1>🤖 Snowflake Warehouse Optimizer with AI Agent</h1>
        <p>Analyze queries, recommend warehouses, optimize SQL, and verify automatically using Gemini & LangGraph.</p>
    </div>
    """, unsafe_allow_html=True)

    # --- FIX: INITIALIZE ALL SESSION STATE AT THE START ---
    if 'page' not in st.session_state:
        st.session_state.page = "data_load"
    if 'all_records' not in st.session_state:
        st.session_state.all_records = None
    if 'selected_records' not in st.session_state:
        st.session_state.selected_records = None
    if 'recommendations' not in st.session_state:
        st.session_state.recommendations = []
    if 'execution_results' not in st.session_state:
        st.session_state.execution_results = []
    
    # --- ROUTING LOGIC ---
    if st.session_state.page == "data_load":
        page_1_load_data()
    elif st.session_state.page == "analysis":
        # Check for necessary data before rendering page 2
        if st.session_state.get('selected_records') is None:
            set_page("data_load")
            st.rerun()
        page_2_analysis()
    elif st.session_state.page == "validation":
        # Check for necessary data before rendering page 3
        if not st.session_state.get('recommendations'):
            set_page("analysis")
            st.rerun()
        page_3_validation_and_results()

if __name__ == "__main__":
    main()


