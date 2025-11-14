import streamlit as st
import pandas as pd
import snowflake.connector
import google.generativeai as genai
from typing import TypedDict, List, Dict
from langgraph.graph import StateGraph, END # Required for Agentic Workflow
from langchain_google_genai import ChatGoogleGenerativeAI
import plotly.graph_objects as go
from datetime import datetime
import json
import time
import os
from dotenv import load_dotenv # Required to load variables from .env file

# ==================== CONFIGURATION & SECRETS ====================

# --- LOAD ENVIRONMENT VARIABLES ---
load_dotenv()

# 1. Gemini API Key
GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY") 

# 2. Snowflake Configuration (Reads values directly from os.environ)
SNOWFLAKE_CONFIG = {
    'account': os.environ.get('SF_ACCOUNT'),
    'user': os.environ.get('SF_USER'),
    'password': os.environ.get('SF_PASSWORD'), 
    'database': os.environ.get('SF_DATABASE'),
    'schema': os.environ.get('SF_SCHEMA'),
    'warehouse': os.environ.get('SF_WAREHOUSE')
}

# Target table path components
TARGET_DB = SNOWFLAKE_CONFIG.get('database')
TARGET_SCHEMA = SNOWFLAKE_CONFIG.get('schema')
TARGET_TABLE_NAME = "OPTIMIZER_BASELINE_DATA" 

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
    st.error("FATAL: GEMINI_API_KEY environment variable is not set. Please set it in your hosting platform's environment settings.")
    st.stop()

if not SNOWFLAKE_CONFIG.get('password') or not SNOWFLAKE_CONFIG.get('account'):
    st.error("FATAL: Essential Snowflake connection variables are missing. Check your .env file.")
    st.stop()


# ==================== HELPER FUNCTION: SAFE CONNECTION MANAGER (FIXED) ====================
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

# ==================== AGENT STATE DEFINITION ====================
class AgentState(TypedDict):
    """The shared memory for the LangGraph workflow."""
    query_record: Dict
    available_warehouses: List[str]
    gemini_recommendation: Dict
    execution_metrics: Dict
    current_agent: str
    error: str

# ==================== AGENT 1: GEMINI LLM AGENT ====================
class GeminiAnalysisAgent:
    """Agent 1: AI Analysis and Recommendation (The brain)"""
    
    def __init__(self, api_key):
        self.api_key = api_key
        self.model = self._setup_gemini()
    
    def _setup_gemini(self):
        try:
            genai.configure(api_key=self.api_key)
            return genai.GenerativeModel('models/gemini-2.5-flash')
        except Exception as e:
            st.error(f"❌ Gemini setup failed: {e}")
            raise e
    
    def analyze(self, state: AgentState) -> Dict: # Returns recommendation dictionary
        """LangGraph Node: Analyze query and recommend warehouse"""
        st.info("🤖 **Agent 1 Started**: Analyzing query with Gemini AI...")
        
        try:
            prompt = self._create_expert_prompt(state["query_record"], state["available_warehouses"])
            response = self.model.generate_content(prompt)
            recommendation = self._parse_response(response.text, state["available_warehouses"])
            st.success(f"✅ **Agent 1 Completed**: Recommended {recommendation['recommended_warehouse']} (Confidence: {recommendation['confidence_score']:.0%})")
            return recommendation
            
        except Exception as e:
            st.error(f"❌ Analysis Failed for {state['query_record']['QUERY_ID']}: {e}")
            return self._fallback_recommendation(state)
    
    def _create_expert_prompt(self, query_metrics: Dict, warehouses: List) -> str:
        """Creates the detailed prompt for multi-factor analysis."""
        query_text = query_metrics.get('QUERY_TEXT', '')
        current_wh = query_metrics.get('ORIGINAL_WAREHOUSE', '')
        exec_time_ms = query_metrics.get('ORIGINAL_EXECUTION_TIME_MS', 0)
        bytes_scanned = query_metrics.get('ORIGINAL_BYTES_SCANNED', 0)
        
        # --- Contexts used in the prompt ---
        bytes_spilled_mb = query_metrics.get('SIM_BYTES_SPILLED_MB', 0) 
        sim_queue_time = query_metrics.get('SIM_QUEUED_TIME_MS', 0)
        bytes_scanned_mb = bytes_scanned / (1024 * 1024)

        return f"""
        You are an expert Snowflake query optimizer. Recommend the **OPTIMAL WAREHOUSE** considering all resources (CPU, Memory, I/O).
        ... (Full prompt details here) ...
        Response as JSON:
        {{
            "recommended_warehouse": "exact_warehouse_name",
            "confidence_score": 0.98,
            "justification": "summary of the bottleneck analysis (Memory/CPU/IO)", 
            "original_query_text": "{query_text}"
        }}
        """

    def _parse_response(self, response_text: str, warehouses: List) -> Dict:
        """Robustly parse and validate LLM response"""
        try:
            if '```json' in response_text:
                json_str = response_text.split('```json')[1].split('```')[0].strip()
            else:
                json_str = response_text.strip()
            
            recommendation = json.loads(json_str)
            if recommendation['recommended_warehouse'] not in warehouses:
                raise ValueError(f"Invalid warehouse: {recommendation['recommended_warehouse']}")
            return recommendation
        except Exception as e:
            raise ValueError(f"Failed to parse AI response: {e}")
    
    def _fallback_recommendation(self, state: AgentState) -> Dict:
        """Rule-based fallback for safety"""
        return {
            "recommended_warehouse": "WH_MEDIUM_SYNTHETIC",
            "confidence_score": 0.7,
            "justification": "Fallback due to AI parse error. Defaulting to safe, balanced Medium warehouse.",
            "original_query_text": state["query_record"].get('QUERY_TEXT', 'N/A')
        }

# ==================== AGENT 2: SNOWFLAKE EXECUTION AGENT ====================
class SnowflakeExecutionAgent:
    """Agent 2: Executes Query and Returns Status"""
    
    def __init__(self, config):
        self.conn = _create_snowflake_connection_safe(config)
        st.success("✅ Snowflake connection established")

    def execute_query(self, query_text: str, warehouse: str) -> bool:
        """Runs a single query with the recommended warehouse (used by click handler)."""
        try:
            cursor = self.conn.cursor()
            cursor.execute(f"USE WAREHOUSE {warehouse}")
            cursor.execute(query_text) 
            cursor.close()
            return True
        except Exception as e:
            return False

# ==================== LANGRAPH WORKFLOW BUILDER ====================
class AgenticAIWorkflow:
    """The LangGraph structure (Agents are called directly)"""
    
    def __init__(self, gemini_api_key, snowflake_config):
        self.agent1 = GeminiAnalysisAgent(gemini_api_key)
        self.agent2 = SnowflakeExecutionAgent(snowflake_config)
    
    def analyze_record(self, record: Dict, available_warehouses: List) -> Dict:
        """Direct call to Agent 1 for analysis."""
        state = AgentState(
            query_record=record, available_warehouses=available_warehouses, gemini_recommendation={},
            execution_metrics={}, current_agent="start", error=""
        )
        return self.agent1.analyze(state)
        
# ==================== DATA LOADING AND MAIN APP LOGIC ====================

def _load_and_analyze_queries(config: Dict, analysis_agent: GeminiAnalysisAgent, available_warehouses: List) -> pd.DataFrame:
    """Loads all data, simulates metrics, and runs Agent 1 analysis for ALL records."""
    
    st.info("🔄 Connecting to Snowflake and analyzing all 32 records...")
    
    try:
        conn = _create_snowflake_connection_safe(config)
        
        # --- Target the verified baseline table ---
        TARGET_DB_RUNTIME = config.get('database')
        TARGET_SCHEMA_RUNTIME = config.get('schema')
        TARGET_TABLE_FULL = f"{TARGET_DB_RUNTIME}.{TARGET_SCHEMA_RUNTIME}.OPTIMIZER_BASELINE_DATA"
        
        # FIX: Use alias to match AgentState field name
        query = f"""
        SELECT 
            QUERY_ID, QUERY_TEXT, ORIGINAL_WAREHOUSE, QUERY_TYPE,
            ORIGINAL_EXECUTION_TIME_MS, ORIGINAL_BYTES_SCANNED,
            ORIGINAL_EFFICIENCY_SCORE, 
            ORIGINAL_COST AS ORIGINAL_ESTIMATED_COST 
        FROM {TARGET_TABLE_FULL}
        ORDER BY RECORD_TIMESTAMP DESC;
        """
        
        df = pd.read_sql(query, conn)
        conn.close()
        
        # --- 1. SIMULATE MISSING METRICS FOR LLM INPUT (Needed for prompt complexity) ---
        scanned = df['ORIGINAL_BYTES_SCANNED']
        runtime = df['ORIGINAL_EXECUTION_TIME_MS']
        df['SIM_BYTES_SPILLED_MB'] = (runtime // 500) * 10 
        df['SIM_PARTITIONS_SCANNED'] = (scanned // 200000000) + 5 
        df['SIM_QUEUED_TIME_MS'] = runtime.apply(lambda x: 0 if x < 2000 else 500)
        df['SIM_COMPILATION_TIME_MS'] = scanned.apply(lambda x: 50 if x > 100000000 else 10)
        df.columns = df.columns.str.upper() # Ensure column names are uppercase
        
        # 2. Run AI Analysis Batch
        recommendations = []
        progress_text = st.empty()
        
        # **LIMIT ANALYSIS TO FIRST 5 RECORDS**
        df_display_limit = df.head(5).reset_index(drop=True)
        
        for i, record in enumerate(df_display_limit.to_dict('records')):
            progress_text.text(f"Analyzing record {i+1}/{len(df_display_limit)}...")
            
            # Agent 1 analysis
            reco_dict = analysis_agent.analyze(AgentState(
                query_record=record, available_warehouses=available_warehouses, 
                gemini_recommendation={}, execution_metrics={}, current_agent="start", error=""
            ))
            
            recommendations.append(reco_dict)
            
        # 3. Integrate Results
        df_reco = pd.DataFrame(recommendations)
        df_final = pd.concat([df_display_limit.reset_index(drop=True), df_reco.add_prefix('RECO_')], axis=1)
        
        progress_text.success("✅ Analysis Complete! Results loaded.")
        return df_final
        
    except Exception as e:
        st.error(f"❌ Failed to load and analyze data: {e}")
        return pd.DataFrame()

# ==================== INTERACTION HANDLERS ====================

def handle_execute_query(record_index: int, executor: SnowflakeExecutionAgent):
    """Handler to execute the query for a single row."""
    df = st.session_state.analyzed_df
    record = df.iloc[record_index]

    # Get the recommended warehouse
    recommended_wh = record['RECO_recommended_warehouse']
    
    # Execute the action handler
    success = executor.execute_query(
        query_text=record['QUERY_TEXT'],
        warehouse=recommended_wh
    )
    
    # Update status for display
    st.session_state[f'STATUS_{record_index}'] = '✅ EXECUTED' if success else '❌ FAILED'
    st.rerun() # Force Streamlit to refresh the table display

def main():
    st.set_page_config(page_title="AI Warehouse Recommender", layout="wide")
    st.title("🚀 AI Warehouse Recommender (Interactive Replacement)")
    st.markdown("---")
    
    # --- GLOBAL INIT ---
    if 'analyzed_df' not in st.session_state:
        st.session_state.analyzed_df = pd.DataFrame()
        
    available_warehouses = list(USABLE_WAREHOUSES)
    
    # Initialize agents (called directly in the logic)
    try:
        analysis_agent_instance = GeminiAnalysisAgent(GEMINI_API_KEY)
        executor_agent_instance = SnowflakeExecutionAgent(SNOWFLAKE_CONFIG)
    except Exception:
        return 
    
    # --- UI Logic ---
    st.subheader("🏭 Warehouse Replacement Tool")
    
    df_display = st.session_state.analyzed_df
    
    # --- LOAD/ANALYZE BUTTON ---
    if df_display.empty:
        if st.button("Step 1: Load All 32 Records & Get AI Recommendation (5 Records for Demo)", type="primary"):
            df_analyzed = _load_and_analyze_queries(SNOWFLAKE_CONFIG, analysis_agent_instance, available_warehouses)
            st.session_state.analyzed_df = df_analyzed
            
            # Initialize status flags for the execution step
            for i in range(len(df_analyzed)):
                st.session_state[f'STATUS_{i}'] = 'PENDING'
            st.rerun()
            return
        
    if not df_display.empty:
        st.markdown(f"### 🏭 Analysis Complete: {len(df_display)} Queries Ready for Replacement")
        st.warning("Click the 'Replace' checkbox next to any record to execute the query using the Recommended Warehouse.")
        
        # --- PREPARE INTERACTIVE TABLE FOR DISPLAY ---
        
        df_table = df_display.copy()
        
        # Display Columns
        df_table['Original WH'] = df_table['ORIGINAL_WAREHOUSE']
        df_table['Recommended WH'] = df_table['RECO_recommended_warehouse']
        
        # Add the status column
        status_list = [st.session_state.get(f'STATUS_{i}', 'PENDING') for i in df_table.index]
        df_table['Execution Status'] = status_list
        df_table['Replace'] = False # Checkbox column
        
        # Define the final columns (Confidence is intentionally excluded)
        display_cols = ['QUERY_ID', 'Original WH', 'Recommended WH', 'Execution Status', 'Replace']
        
        
        # --- STREAMLIT DATA EDITOR (MAIN INTERACTION) ---
        edited_df = st.data_editor(
            df_table[display_cols],
            column_config={
                "QUERY_ID": st.column_config.TextColumn("Query ID", disabled=True),
                "Execution Status": st.column_config.TextColumn("Status", disabled=True),
                "Replace": st.column_config.CheckboxColumn("Replace (Run Query)", help="Click to execute query on recommended WH", default=False),
            },
            hide_index=True,
            use_container_width=True,
            key='editor_table'
        )
        
        # --- HANDLE BUTTON CLICK/EDIT ---
        
        clicked_rows = edited_df[edited_df['Replace']]
        if not clicked_rows.empty:
            # Get the index of the first clicked row
            clicked_index_in_editor = clicked_rows.index[0] 
            
            # Reset the checkbox state
            if st.session_state.editor_table.get('Replace', pd.Series()).iloc[clicked_index_in_editor] == True:
                
                # Reset the checkbox state
                st.session_state.editor_table['Replace'][clicked_index_in_editor] = False
                
                # Execute the action handler for the original DataFrame index
                original_index = clicked_index_in_editor
                
                # Set status to RUNNING before handler executes
                st.session_state[f'STATUS_{original_index}'] = 'RUNNING...'
                st.rerun() 
                
                # Execute the query (this runs on the second rerun)
                handle_execute_query(original_index, executor_agent_instance)


if __name__ == "__main__":
    main()
