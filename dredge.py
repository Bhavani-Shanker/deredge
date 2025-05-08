import streamlit as st
import sqlite3
import pandas as pd
from PyPDF2 import PdfReader
from docx import Document
import json
import os
from dotenv import load_dotenv
from openai import AzureOpenAI
import chromadb
from sentence_transformers import SentenceTransformer
import nltk
from nltk.tokenize import word_tokenize
from rank_bm25 import BM25Okapi
import re
from ortools.linear_solver import pywraplp
from datetime import datetime, timedelta, date
from collections import defaultdict
from ast import literal_eval
import random
import numpy as np
import plotly.express as px
import calendar

# Ensure NLTK punkt_tab resource is available
try:
    nltk.data.find('tokenizers/punkt_tab')
except LookupError:
    nltk.download('punkt_tab')

load_dotenv()
ADMIN_CREDENTIALS = {"admin_id": "admin", "password": "admin123"}
conn = sqlite3.connect('files.db', check_same_thread=False)
c = conn.cursor()
c.execute('''CREATE TABLE IF NOT EXISTS files
             (id INTEGER PRIMARY KEY AUTOINCREMENT,
              filename TEXT UNIQUE,
              content TEXT,
              uploaded_at DATETIME DEFAULT CURRENT_TIMESTAMP)''')
conn.commit()
chroma_client = chromadb.PersistentClient(path="chroma_db")
collection = chroma_client.get_or_create_collection(name="documents")
embedding_model = SentenceTransformer("all-MiniLM-L6-v2")

class DredgingResourceAllocator:
    def __init__(self):
        self.employees = []
        self.dredgers = []
        self.projects = []
        self.skill_requirements = {}
        
    def load_data(self, employees_data, dredgers_data, projects_data):
        try:
            self.employees = employees_data.to_dict('records')
            for employee in self.employees:
                employee['skills'] = self._parse_skills(employee.get('skills', {}))
            self.dredgers = dredgers_data.to_dict('records')
            self.projects = projects_data.copy()
            self.projects['start_date'] = pd.to_datetime(self.projects['start_date'])
            self.projects['end_date'] = pd.to_datetime(self.projects['end_date'])
            self.projects = self.projects.to_dict('records')
            if not all(['employee_id' in e for e in self.employees]):
                raise ValueError("Employee data missing employee_id")
            if not all(['dredger_id' in d for d in self.dredgers]):
                raise ValueError("Dredger data missing dredger_id")
            if not all(['project_id' in p for p in self.projects]):
                raise ValueError("Project data missing project_id")
            self._process_skill_requirements()
            return True
        except Exception as e:
            st.error(f"Error loading data: {str(e)}")
            return False
    
    def _parse_skills(self, skills_data):
        if isinstance(skills_data, dict):
            return skills_data
        if isinstance(skills_data, str):
            try:
                return json.loads(skills_data.replace("'", '"'))
            except json.JSONDecodeError:
                try:
                    return literal_eval(skills_data)
                except:
                    return {}
        return {}
    
    def _process_skill_requirements(self):
        self.skill_requirements = {
            'Cutter Suction Dredger': {
                'min_crew': 4,
                'required_skills': {
                    'dredge_operation': 3,
                    'mechanical_maintenance': 2,
                    'environmental_compliance': 2
                }
            },
            'Trailing Suction Hopper Dredger': {
                'min_crew': 5,
                'required_skills': {
                    'dredge_operation': 3,
                    'hydrographic_surveying': 2,
                    'sediment_handling': 2
                }
            },
            'Backhoe Dredger': {
                'min_crew': 3,
                'required_skills': {
                    'dredge_operation': 2,
                    'mechanical_maintenance': 2,
                    'environmental_compliance': 1
                }
            }
        }
    
    def _get_employee_skill_level(self, employee, skill):
        return float(employee['skills'].get(skill, 0))
    
    def optimize_allocation(self, start_date, end_date):
        try:
            if isinstance(start_date, str):
                start_date = pd.to_datetime(start_date)
            elif isinstance(start_date, date):
                start_date = pd.to_datetime(start_date)
            if isinstance(end_date, str):
                end_date = pd.to_datetime(end_date)
            elif isinstance(end_date, date):
                end_date = pd.to_datetime(end_date)
            relevant_projects = [
                p for p in self.projects 
                if not (p['end_date'] < start_date or p['start_date'] > end_date)
            ]
            if not relevant_projects:
                return {"status": "No projects in specified period"}
            solver = pywraplp.Solver.CreateSolver('SCIP')
            if not solver:
                return {"status": "Failed to create solver"}
            assignments = {}
            for e in self.employees:
                for p in relevant_projects:
                    assignments[(e['employee_id'], p['project_id'])] = solver.IntVar(
                        0, 1, f"x_{e['employee_id']}_{p['project_id']}")
            objective = solver.Objective()
            for (e_id, p_id), var in assignments.items():
                employee = next((e for e in self.employees if e['employee_id'] == e_id), None)
                project = next((p for p in relevant_projects if p['project_id'] == p_id), None)
                if not employee or not project:
                    continue
                dredger = next((d for d in self.dredgers if d['dredger_id'] == project['dredger_id']), None)
                if not dredger:
                    continue
                skill_score = self._calculate_skill_match(employee, dredger['type'])
                cost = employee.get('daily_cost', 0) * (project['end_date'] - project['start_date']).days
                objective.SetCoefficient(var, skill_score - 0.1 * cost)
            objective.SetMaximization()
            self._add_availability_constraints(solver, assignments, relevant_projects, start_date, end_date)
            self._add_crew_size_constraints(solver, assignments, relevant_projects)
            self._add_skill_constraints(solver, assignments, relevant_projects)
            status = solver.Solve()
            if status == pywraplp.Solver.OPTIMAL:
                return self._prepare_results(assignments, relevant_projects)
            return {"status": f"No optimal solution found (status: {status})"}
        except Exception as e:
            return {"status": f"Optimization failed: {str(e)}"}
    
    def _add_availability_constraints(self, solver, assignments, projects, start_date, end_date):
        if isinstance(start_date, date):
            start_date = pd.to_datetime(start_date)
        if isinstance(end_date, date):
            end_date = pd.to_datetime(end_date)
        date_range = pd.date_range(start_date, end_date)
        for e in self.employees:
            for day in date_range:
                constraint = solver.Constraint(0, 1)
                for p in projects:
                    if p['start_date'] <= day <= p['end_date']:
                        var = assignments.get((e['employee_id'], p['project_id']))
                        if var:
                            constraint.SetCoefficient(var, 1)
    
    def _add_crew_size_constraints(self, solver, assignments, projects):
        for p in projects:
            dredger = next((d for d in self.dredgers if d['dredger_id'] == p['dredger_id']), None)
            if not dredger:
                continue
            dredger_type = dredger.get('type')
            if not dredger_type:
                continue
            min_crew = self.skill_requirements.get(dredger_type, {}).get('min_crew', 0)
            if min_crew <= 0:
                continue
            constraint = solver.Constraint(min_crew, solver.infinity())
            for e in self.employees:
                var = assignments.get((e['employee_id'], p['project_id']))
                if var:
                    constraint.SetCoefficient(var, 1)
    
    def _add_skill_constraints(self, solver, assignments, projects):
        for p in projects:
            dredger = next((d for d in self.dredgers if d['dredger_id'] == p['dredger_id']), None)
            if not dredger:
                continue
            dredger_type = dredger.get('type')
            if not dredger_type:
                continue
            req_skills = self.skill_requirements.get(dredger_type, {}).get('required_skills', {})
            for skill, min_level in req_skills.items():
                constraint = solver.Constraint(min_level, solver.infinity())
                for e in self.employees:
                    skill_level = self._get_employee_skill_level(e, skill)
                    if skill_level >= 1:
                        var = assignments.get((e['employee_id'], p['project_id']))
                        if var:
                            constraint.SetCoefficient(var, skill_level)
    
    def _calculate_skill_match(self, employee, dredger_type):
        if dredger_type not in self.skill_requirements:
            return 0
        total_score = 0
        for skill, min_level in self.skill_requirements[dredger_type]['required_skills'].items():
            skill_level = self._get_employee_skill_level(employee, skill)
            if skill_level >= 1:
                total_score += min(skill_level, min_level) * 1.0
                if skill_level > min_level:
                    total_score += (skill_level - min_level) * 0.5
        return total_score
    
    def _prepare_results(self, assignments, projects):
        allocation = defaultdict(list)
        for (e_id, p_id), var in assignments.items():
            if var.solution_value() > 0.5:
                allocation[p_id].append(e_id)
        return {
            "status": "OPTIMAL",
            "total_assignments": sum(var.solution_value() for var in assignments.values()),
            "allocations": dict(allocation)
        }
    
    def generate_report(self, allocation_result):
        if not isinstance(allocation_result, dict):
            return {"status": "Invalid allocation result format"}
        if allocation_result.get('status') != 'OPTIMAL':
            return {"status": f"No optimal allocation: {allocation_result.get('status', 'Unknown error')}"}
        if not allocation_result.get('allocations'):
            return {"status": "No allocations found in results"}
        report = []
        for p_id, e_ids in allocation_result['allocations'].items():
            project = next((p for p in self.projects if p['project_id'] == p_id), None)
            if not project:
                continue
            dredger = next((d for d in self.dredgers if d['dredger_id'] == project['dredger_id']), None)
            if not dredger:
                continue
            project_report = {
                'project_id': p_id,
                'dredger': dredger.get('name', 'Unknown'),
                'type': dredger.get('type', 'Unknown'),
                'location': project.get('location', 'Unknown'),
                'dates': f"{project.get('start_date', 'Unknown').strftime('%Y-%m-%d')} to {project.get('end_date', 'Unknown').strftime('%Y-%m-%d')}",
                'crew_count': len(e_ids),
                'crew': []
            }
            for e_id in e_ids:
                employee = next((e for e in self.employees if e['employee_id'] == e_id), None)
                if employee:
                    project_report['crew'].append({
                        'id': e_id,
                        'name': employee.get('name', 'Unknown'),
                        'position': employee.get('position', 'Unknown'),
                        'skills': employee.get('skills', {})
                    })
            report.append(project_report)
        return report if report else {"status": "No valid project allocations to report"}

def generate_random_maintenance_data(num_records=100):
    dredgers = ['Cutter Dredger 1', 'Hopper Dredger Alpha', 'Backhoe Dredger X', 'Dredger Poseidon', 'Dredger Triton']
    components = ['Cutter Head', 'Suction Pipe', 'Pump System', 'Hydraulic System', 'Dredge Pump', 
                  'Spud System', 'Electrical System', 'Monitoring System']
    maintenance_types = ['Preventive', 'Corrective', 'Predictive', 'Condition-based']
    statuses = ['Completed', 'Pending', 'Overdue', 'Cancelled']
    data = []
    for _ in range(num_records):
        dredger = random.choice(dredgers)
        component = random.choice(components)
        maintenance_type = random.choice(maintenance_types)
        status = random.choice(statuses)
        last_date = datetime.now() - timedelta(days=random.randint(1, 365))
        next_date = last_date + timedelta(days=random.randint(30, 365))
        cost = round(random.uniform(500, 15000), 2)
        data.append({
            'Dredger': dredger,
            'Component': component,
            'Maintenance Type': maintenance_type,
            'Last Maintenance Date': last_date,
            'Next Maintenance Date': next_date,
            'Status': status,
            'Cost ($)': cost,
            'Hours Spent': random.randint(2, 72)
        })
    return pd.DataFrame(data)

def generate_sensor_data(num_records=500):
    components = ['Cutter Head', 'Dredge Pump', 'Hydraulic System', 'Suction Pipe']
    parameters = {
        'Cutter Head': ['RPM', 'Torque', 'Temperature', 'Vibration'],
        'Dredge Pump': ['Pressure', 'Flow Rate', 'Temperature', 'Vibration'],
        'Hydraulic System': ['Pressure', 'Temperature', 'Fluid Level', 'Flow Rate'],
        'Suction Pipe': ['Sediment Density', 'Flow Rate', 'Pressure', 'Wear']
    }
    data = []
    for _ in range(num_records):
        component = random.choice(components)
        parameter = random.choice(parameters[component])
        if parameter == 'RPM':
            value = round(random.uniform(50, 300), 2)
            threshold = 250
        elif parameter == 'Torque':
            value = round(random.uniform(100, 1000), 2)
            threshold = 800
        elif parameter == 'Temperature':
            value = round(random.uniform(30, 90), 2)
            threshold = 80
        elif parameter == 'Vibration':
            value = round(random.uniform(0.1, 5), 2)
            threshold = 3.5
        elif parameter == 'Pressure':
            value = round(random.uniform(1, 15), 2)
            threshold = 12
        elif parameter == 'Flow Rate':
            value = round(random.uniform(5, 50), 2)
            threshold = 40
        elif parameter == 'Fluid Level':
            value = round(random.uniform(1, 10), 2)
            threshold = 2
        elif parameter == 'Sediment Density':
            value = round(random.uniform(1, 3), 2)
            threshold = 2.5
        elif parameter == 'Wear':
            value = round(random.uniform(0, 100), 2)
            threshold = 80
        timestamp = datetime.now() - timedelta(minutes=random.randint(1, 10080))
        alert = value > threshold
        data.append({
            'Timestamp': timestamp,
            'Component': component,
            'Parameter': parameter,
            'Value': value,
            'Threshold': threshold,
            'Alert': alert
        })
    return pd.DataFrame(data)

def app1():
    def extract_text_from_file(file):
        text = ""
        if file.type == "application/pdf":
            pdf_reader = PdfReader(file)
            for page in pdf_reader.pages:
                text += page.extract_text() + "\n"
        elif file.type == "text/plain":
            text = file.getvalue().decode("utf-8")
        elif file.type == "application/vnd.openxmlformats-officedocument.wordprocessingml.document":
            doc = Document(file)
            for para in doc.paragraphs:
                text += para.text + "\n"
        elif file.type == "application/json":
            json_data = json.load(file)
            text = json.dumps(json_data, indent=4)
        elif file.type == "text/markdown":
            text = file.getvalue().decode("utf-8")
        elif file.type in ["text/csv", "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"]:
            df = pd.read_csv(file) if file.type == "text/csv" else pd.read_excel(file)
            text = df.to_string()
        return text

    def chunk_text(text, chunk_size=1000, overlap=200):
        chunks = []
        start = 0
        text_length = len(text)
        while start < text_length:
            end = min(start + chunk_size, text_length)
            if end < text_length and end - start == chunk_size:
                last_period = max(text.rfind('.', start, end), text.rfind('\n', start, end))
                if last_period > start + chunk_size // 2: 
                    end = last_period + 1
            chunks.append(text[start:end])
            start = end - overlap if end < text_length else text_length
        return chunks

    def preprocess_text(text):
        text = re.sub(r'[^\w\s]', '', text.lower())
        return word_tokenize(text)

    def store_file_in_db(filename, content):
        c.execute("SELECT COUNT(*) FROM files WHERE filename = ?", (filename,))
        if c.fetchone()[0] == 0:
            c.execute("INSERT INTO files (filename, content) VALUES (?, ?)", (filename, content))
            conn.commit()
            chunks = chunk_text(content)
            for i, chunk in enumerate(chunks):
                chunk_id = f"{filename}_chunk_{i}"
                embedding = embedding_model.encode(chunk).tolist()
                collection.add(documents=[chunk], embeddings=[embedding], ids=[chunk_id], metadatas=[{"filename": filename, "chunk_index": i}])
            st.success(f"'{filename}' uploaded successfully!")
        else:
            st.warning(f"'{filename}' is already uploaded.")

    def retrieve_relevant_context(query, top_k=10):
        query_embedding = embedding_model.encode(query).tolist()
        results = collection.query(query_embeddings=[query_embedding], n_results=top_k*2)
        if not results["documents"][0]:
            return ""
        retrieved_docs = results["documents"][0]
        doc_ids = results["ids"][0]
        tokenized_query = preprocess_text(query)
        tokenized_docs = [preprocess_text(doc) for doc in retrieved_docs]
        bm25 = BM25Okapi(tokenized_docs)
        bm25_scores = bm25.get_scores(tokenized_query)
        vector_scores = [1.0 - results["distances"][0][i] for i in range(len(retrieved_docs))]
        max_bm25 = max(bm25_scores) if len(bm25_scores) > 0 else 1.0
        max_vector = max(vector_scores) if vector_scores else 1.0
        normalized_bm25 = [score/max_bm25 for score in bm25_scores]
        normalized_vector = [score/max_vector for score in vector_scores]
        alpha = 0.5
        combined_scores = [(alpha * v_score + (1-alpha) * bm_score)
                          for v_score, bm_score in zip(normalized_vector, normalized_bm25)]
        doc_score_pairs = list(zip(retrieved_docs, doc_ids, combined_scores))
        ranked_docs = sorted(doc_score_pairs, key=lambda x: x[2], reverse=True)[:top_k]
        context_chunks = []
        for doc, doc_id, score in ranked_docs:
            metadata = f"Source: {doc_id.split('_chunk_')[0]} (Score: {score:.3f})"
            context_chunks.append(f"{metadata}\n{doc}")
        return "\n\n" + "-"*50 + "\n\n".join(context_chunks)

    def azure_openai_query(question, context, api_key, endpoint):
        client = AzureOpenAI(
            api_key=api_key,
            api_version="2023-05-15",
            azure_endpoint=endpoint
        )
        system_prompt = """
            You are an AI assistant tasked with performing comprehensive extraction of information from ALL provided context chunks.
            When responding to a user's question, adhere strictly to the following guidelines:
            - Review EVERY context chunk provided thoroughly, ensuring you cover ALL occurrences of relevant information.
            - Extract and list EVERY relevant piece of information explicitly and separately, even if the same or similar information appears multiple times or across different chunks.
            - Do NOT stop after partial matches or the initial findings; CONTINUE reviewing ALL chunks until no additional relevant information remains.
            - Clearly format your responses in a structured manner (e.g., bullet points or numbered lists) for readability.
            Provide explicit source citations indicating:
            - The exact document filename.
            - The specific chunk or page number from which each piece of information was extracted.
            Your goal is complete accuracy and exhaustive retrieval—no relevant data should be omitted.
        """
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": f"Context:\n{context}\n\nQuestion: {question}\nAnswer:"}
        ]
        response = client.chat.completions.create(
            model="gpt-4o",
            messages=messages,
            temperature=0.7,
            max_tokens=1000,
            top_p=1,
            frequency_penalty=0,
            presence_penalty=0
        )
        return response.choices[0].message.content.strip()

    st.title("Dredging Contextual RAG Q&A Web Application")
    tab1, tab2, tab3 = st.tabs(["Admin", "Credentials", "Chat"])
    with tab1:
        st.subheader("Admin Panel")
        if 'admin_logged_in' not in st.session_state:
            st.session_state.admin_logged_in = False
        if not st.session_state.admin_logged_in:
            admin_id = st.text_input("Admin ID")
            password = st.text_input("Password", type="password")
            if st.button("Login"):
                if admin_id == ADMIN_CREDENTIALS["admin_id"] and password == ADMIN_CREDENTIALS["password"]:
                    st.session_state.admin_logged_in = True
                    st.success("Logged in as Admin")
                    st.rerun()
                else:
                    st.error("Invalid credentials")
        if st.session_state.admin_logged_in:
            uploaded_files = st.file_uploader("Upload files", accept_multiple_files=True, type=["pdf", "txt", "docx", "json", "md", "csv", "xlsx"])
            if uploaded_files:
                for file in uploaded_files:
                    text = extract_text_from_file(file)
                    store_file_in_db(file.name, text)
            st.subheader("Uploaded Files")
            c.execute("SELECT filename FROM files")
            files = c.fetchall()
            if files:
                df = pd.DataFrame(files, columns=["Filename"])
                for filename in df["Filename"]:
                    col1, col2 = st.columns([4, 1])
                    col1.text(filename)
                    if col2.button("Delete", key=filename):
                        c.execute("DELETE FROM files WHERE filename = ?", (filename,))
                        conn.commit()
                        try:
                            file_chunks = collection.get(where={"filename": filename})
                            if file_chunks and "ids" in file_chunks and file_chunks["ids"]:
                                collection.delete(ids=file_chunks["ids"])
                        except Exception as e:
                            st.error(f"Error deleting from ChromaDB: {e}")
                        st.rerun()
            else:
                st.info("No files uploaded yet.")
            if st.button("Logout"):
                st.session_state.admin_logged_in = False
                st.rerun()
    with tab2:
        st.subheader("Azure OpenAI Credentials")
        st.session_state.api_key = st.text_input("Enter Azure OpenAI API Key", type="password")
        st.session_state.endpoint = st.text_input("Enter Azure OpenAI Endpoint")
    with tab3:
        st.subheader("Chat Interface")
        if 'messages' not in st.session_state:
            st.session_state.messages = []
        for msg in st.session_state.messages:
            with st.chat_message("user" if msg["role"] == "user" else "assistant"):
                st.markdown(msg["content"])
        if 'show_context' not in st.session_state:
            st.session_state.show_context = False
        show_context = st.sidebar.checkbox("Show retrieved context", value=st.session_state.show_context)
        st.session_state.show_context = show_context
        question = st.chat_input("Ask me anything...")
        if question:
            st.chat_message("user").markdown(question)
            with st.status("Searching documents...", expanded=True) as status:
                context = retrieve_relevant_context(question)
                if not context:
                    status.update(label="No relevant documents found", state="error")
                    answer = "I couldn't find any relevant information in the knowledge base. Please try a different question or upload more documents."
                else:
                    if st.session_state.show_context:
                        st.sidebar.markdown("### Retrieved Context")
                        st.sidebar.markdown(context)
                    status.update(label="Generating answer...", state="running")
                    answer = azure_openai_query(question, context, st.session_state.api_key, st.session_state.endpoint)
                    status.update(label="Answer generated!", state="complete")
            st.chat_message("assistant").markdown(answer)
            st.session_state.messages.append({"role": "user", "content": question})
            st.session_state.messages.append({"role": "assistant", "content": answer})

def app2():
    def generate_random_data(num_employees=10, num_dredgers=3, num_projects=3):
        skill_pool = {
            'dredge_operation': [1, 2, 3, 4, 5],
            'hydrographic_surveying': [1, 2, 3, 4],
            'environmental_compliance': [2, 3, 4, 5],
            'mechanical_maintenance': [1, 2, 3, 4, 5],
            'sediment_handling': [1, 2, 3, 4],
            'safety_management': [2, 3, 4, 5],
            'project_management': [2, 3, 4]
        }
        positions = ['Dredge Master', 'Hydrographic Surveyor', 'Chief Engineer', 'Environmental Officer', 
                     'Maintenance Technician', 'Dredge Operator', 'Safety Manager', 'Project Coordinator']
        employees = []
        for i in range(1, num_employees + 1):
            num_skills = random.randint(2, 5)
            skills = {}
            for skill in random.sample(list(skill_pool.keys()), num_skills):
                skills[skill] = random.choice(skill_pool[skill])
            employees.append({
                'employee_id': 100 + i,
                'name': f"Employee {i}",
                'position': random.choice(positions),
                'skills': skills,
                'daily_cost': random.randint(250, 500)
            })
        dredger_types = ['Cutter Suction Dredger', 'Trailing Suction Hopper Dredger', 'Backhoe Dredger']
        dredgers = []
        for i in range(1, num_dredgers + 1):
            dredgers.append({
                'dredger_id': 200 + i,
                'name': f"Dredger {i}",
                'type': random.choice(dredger_types),
                'capacity': random.randint(1000, 20000)  # Cubic meters
            })
        locations = ['Port of Rotterdam Dredging', 'Miami Harbor Deepening', 
                     'Singapore Channel Maintenance', 'Dubai Coastal Expansion', 
                     'Sydney Harbor Dredging']
        projects = []
        base_date = datetime(2025, 3, 26)
        for i in range(1, num_projects + 1):
            start_date = base_date + timedelta(days=random.randint(0, 7))
            duration = random.randint(14, 60)  # Dredging projects may last longer
            end_date = start_date + timedelta(days=duration)
            projects.append({
                'project_id': 300 + i,
                'dredger_id': random.choice([d['dredger_id'] for d in dredgers]),
                'location': random.choice(locations),
                'start_date': start_date.strftime('%Y-%m-%d'),
                'end_date': end_date.strftime('%Y-%m-%d')
            })
        return (
            pd.DataFrame(employees),
            pd.DataFrame(dredgers),
            pd.DataFrame(projects))

    allocator = DredgingResourceAllocator()
    st.title("Dredging Resource Allocation System")
    st.sidebar.header("Configuration")
    use_random_data = st.sidebar.checkbox("Use Random Data", value=True)
    random.seed(st.sidebar.number_input("Random Seed", value=42))
    if use_random_data:
        num_employees = st.sidebar.slider("Number of Employees", 5, 50, 15)
        num_dredgers = st.sidebar.slider("Number of Dredgers", 1, 10, 3)
        num_projects = st.sidebar.slider("Number of Projects", 1, 10, 3)
        employees_df, dredgers_df, projects_df = generate_random_data(
            num_employees, num_dredgers, num_projects)
    else:
        employees_file = st.sidebar.file_uploader("Upload Employees CSV", type="csv")
        dredgers_file = st.sidebar.file_uploader("Upload Dredgers CSV", type="csv")
        projects_file = st.sidebar.file_uploader("Upload Projects CSV", type="csv")
        if not (employees_file and dredgers_file and projects_file):
            st.warning("Please upload all required files")
            return
        employees_df = pd.read_csv(employees_file)
        dredgers_df = pd.read_csv(dredgers_file)
        projects_df = pd.read_csv(projects_file)
    col1, col2 = st.columns(2)
    with col1:
        start_date = st.date_input("Start Date", date(2025, 3, 26))
    with col2:
        end_date = st.date_input("End Date", date(2025, 4, 10))
    st.subheader("Data Preview")
    tab1, tab2, tab3 = st.tabs(["Employees", "Dredgers", "Projects"])
    with tab1:
        st.dataframe(employees_df)
    with tab2:
        st.dataframe(dredgers_df)
    with tab3:
        st.dataframe(projects_df)
    if st.button("Run Optimization"):
        with st.spinner("Loading data and optimizing..."):
            if not allocator.load_data(employees_df, dredgers_df, projects_df):
                st.error("Failed to load data")
                return
            st.success("Data loaded successfully:")
            st.write(f"- Employees: {len(allocator.employees)}")
            st.write(f"- Dredgers: {len(allocator.dredgers)}")
            st.write(f"- Projects: {len(allocator.projects)}")
            result = allocator.optimize_allocation(start_date, end_date)
            st.subheader("Optimization Results")
            st.write(f"Status: {result.get('status')}")
            report = allocator.generate_report(result)
            if isinstance(report, list):
                st.success(f"Found {len(report)} project allocations:")
                for project in report:
                    with st.expander(f"Project {project['project_id']}: {project['dredger']} ({project['type']})"):
                        st.write(f"**Location:** {project['location']}")
                        st.write(f"**Dates:** {project['dates']}")
                        st.write(f"**Crew Members ({project['crew_count']}):**")
                        crew_df = pd.DataFrame(project['crew'])
                        st.dataframe(crew_df)
                        skills_data = []
                        for crew in project['crew']:
                            for skill, level in crew['skills'].items():
                                skills_data.append({
                                    'Name': crew['name'],
                                    'Skill': skill,
                                    'Level': level
                                })
                        if skills_data:
                            skills_df = pd.DataFrame(skills_data)
                            st.subheader("Crew Skills")
                            st.bar_chart(skills_df.pivot(index='Name', columns='Skill', values='Level'))
            else:
                st.warning(f"Report: {report.get('status')}")

def app3():
    if 'maintenance_data' not in st.session_state:
        st.session_state.maintenance_data = generate_random_maintenance_data(200)
    if 'sensor_data' not in st.session_state:
        st.session_state.sensor_data = generate_sensor_data(1000)
    st.sidebar.header("Filters")
    selected_dredger = st.sidebar.selectbox("Select Dredger", ['All'] + list(st.session_state.maintenance_data['Dredger'].unique()))
    selected_component = st.sidebar.selectbox("Select Component", ['All'] + list(st.session_state.maintenance_data['Component'].unique()))
    selected_status = st.sidebar.selectbox("Select Status", ['All'] + list(st.session_state.maintenance_data['Status'].unique()))
    selected_type = st.sidebar.selectbox("Select Maintenance Type", ['All'] + list(st.session_state.maintenance_data['Maintenance Type'].unique()))
    filtered_data = st.session_state.maintenance_data.copy()
    if selected_dredger != 'All':
        filtered_data = filtered_data[filtered_data['Dredger'] == selected_dredger]
    if selected_component != 'All':
        filtered_data = filtered_data[filtered_data['Component'] == selected_component]
    if selected_status != 'All':
        filtered_data = filtered_data[filtered_data['Status'] == selected_status]
    if selected_type != 'All':
        filtered_data = filtered_data[filtered_data['Maintenance Type'] == selected_type]
    st.title("🛠 Dredging Maintenance Management System")
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        total_maintenance = len(filtered_data)
        st.metric("Total Maintenance Records", total_maintenance)
    with col2:
        preventive_count = len(filtered_data[filtered_data['Maintenance Type'] == 'Preventive'])
        st.metric("Preventive Maintenance", preventive_count)
    with col3:
        overdue_count = len(filtered_data[filtered_data['Status'] == 'Overdue'])
        st.metric("Overdue Maintenance", overdue_count)
    with col4:
        total_cost = filtered_data['Cost ($)'].sum()
        st.metric("Total Cost ($)", f"{total_cost:,.2f}")
    tab1, tab2, tab3, tab4 = st.tabs(["Maintenance Records", "Preventive Schedule", "Predictive Analytics", "Add New Record"])
    with tab1:
        st.subheader("Maintenance Records")
        st.dataframe(filtered_data.sort_values('Next Maintenance Date', ascending=False), 
                    use_container_width=True, height=400)
        col1, col2 = st.columns(2)
        with col1:
            st.subheader("Maintenance by Type")
            type_counts = filtered_data['Maintenance Type'].value_counts()
            fig = px.pie(type_counts, values=type_counts.values, names=type_counts.index)
            st.plotly_chart(fig, use_container_width=True)
        with col2:
            st.subheader("Cost Distribution by Component")
            cost_by_component = filtered_data.groupby('Component')['Cost ($)'].sum().reset_index()
            fig = px.bar(cost_by_component, x='Component', y='Cost ($)', color='Component')
            st.plotly_chart(fig, use_container_width=True)
    with tab2:
        st.subheader("Preventive Maintenance Schedule")
        months = [calendar.month_name[i] for i in range(1, 13)]
        selected_month = st.selectbox("Select Month", months, index=datetime.now().month-1)
        month_num = months.index(selected_month) + 1
        current_year = datetime.now().year
        preventive_data = filtered_data[filtered_data['Maintenance Type'] == 'Preventive']
        monthly_schedule = preventive_data[
            (preventive_data['Next Maintenance Date'].dt.month == month_num) & 
            (preventive_data['Next Maintenance Date'].dt.year == current_year)
        ]
        if not monthly_schedule.empty:
            st.dataframe(monthly_schedule.sort_values('Next Maintenance Date'), 
                        use_container_width=True, height=400)
            fig = px.timeline(
                monthly_schedule, 
                x_start="Last Maintenance Date", 
                x_end="Next Maintenance Date", 
                y="Component",
                color="Dredger",
                title="Maintenance Schedule Timeline"
            )
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("No preventive maintenance scheduled for the selected month.")
    with tab3:
        st.subheader("Predictive Maintenance Analytics")
        alert_data = st.session_state.sensor_data[st.session_state.sensor_data['Alert'] == True]
        if not alert_data.empty:
            st.warning(f"⚠️ {len(alert_data)} active alerts detected!")
            st.dataframe(alert_data.sort_values('Timestamp', ascending=False), 
                        use_container_width=True, height=300)
            selected_component_alert = st.selectbox(
                "Select Component for Analysis", 
                alert_data['Component'].unique()
            )
            component_data = st.session_state.sensor_data[
                st.session_state.sensor_data['Component'] == selected_component_alert
            ]
            fig = px.line(
                component_data, 
                x='Timestamp', 
                y='Value', 
                color='Parameter',
                title=f"Sensor Data for {selected_component_alert}",
                markers=True
            )
            for parameter in component_data['Parameter'].unique():
                threshold = component_data[component_data['Parameter'] == parameter]['Threshold'].iloc[0]
                fig.add_hline(
                    y=threshold, 
                    line_dash="dot",
                    annotation_text=f"{parameter} Threshold",
                    annotation_position="bottom right",
                    line_color="red"
                )
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.success("No active alerts detected. All systems operating within normal parameters.")
    with tab4:
        st.subheader("Add New Maintenance Record")
        with st.form("maintenance_form"):
            col1, col2 = st.columns(2)
            with col1:
                dredger = st.selectbox("Dredger", st.session_state.maintenance_data['Dredger'].unique())
                component = st.selectbox("Component", st.session_state.maintenance_data['Component'].unique())
                maintenance_type = st.selectbox("Maintenance Type", st.session_state.maintenance_data['Maintenance Type'].unique())
                status = st.selectbox("Status", st.session_state.maintenance_data['Status'].unique())
            with col2:
                last_maintenance_date = st.date_input("Last Maintenance Date", datetime.now())
                next_maintenance_date = st.date_input("Next Maintenance Date", datetime.now() + timedelta(days=30))
                cost = st.number_input("Cost ($)", min_value=0.0, value=1000.0, step=100.0)
                hours_spent = st.number_input("Hours Spent", min_value=1, value=8, step=1)
            submitted = st.form_submit_button("Add Record")
            if submitted:
                new_record = {
                    'Dredger': dredger,
                    'Component': component,
                    'Maintenance Type': maintenance_type,
                    'Last Maintenance Date': last_maintenance_date,
                    'Next Maintenance Date': next_maintenance_date,
                    'Status': status,
                    'Cost ($)': cost,
                    'Hours Spent': hours_spent
                }
                new_df = pd.DataFrame([new_record])
                st.session_state.maintenance_data = pd.concat([st.session_state.maintenance_data, new_df], ignore_index=True)
                st.success("Maintenance record added successfully!")
                st.rerun()
    st.markdown("<br><br>", unsafe_allow_html=True)

def main():
    st.set_page_config(page_title="Dredging Management Suite", layout="wide")
    st.title("Dredging Management Suite")

    if 'selected_app' not in st.session_state:
        st.session_state.selected_app = "Dredging Contextual RAG Q&A"

    st.markdown("""
        <style>
        .app-button {
            padding: 10px 20px;
            margin: 5px;
            border: none;
            border-radius: 5px;
            cursor: pointer;
            font-size: 16px;
            width: 100%;
            text-align: center;
        }
        .app-button-unselected {
            background-color: #f0f0f0;
            color: #000000;
        }
        .app-button-selected {
            background-color: #4CAF50;
            color: white;
        }
        </style>
    """, unsafe_allow_html=True)

    col1, col2, col3 = st.columns(3)

    with col1:
        if st.button("Dredging Contextual RAG Q&A", key="app1_btn"):
            st.session_state.selected_app = "Dredging Contextual RAG Q&A"
        button_style = "app-button-selected" if st.session_state.selected_app == "Dredging Contextual RAG Q&A" else "app-button-unselected"
        st.markdown(f'<div class="{button_style} app-button">Dredging Contextual RAG Q&A</div>', unsafe_allow_html=True)

    with col2:
        if st.button("Dredging Resource Allocator", key="app2_btn"):
            st.session_state.selected_app = "Dredging Resource Allocator"
        button_style = "app-button-selected" if st.session_state.selected_app == "Dredging Resource Allocator" else "app-button-unselected"
        st.markdown(f'<div class="{button_style} app-button">Dredging Resource Allocator</div>', unsafe_allow_html=True)

    with col3:
        if st.button("Dredging Maintenance System", key="app3_btn"):
            st.session_state.selected_app = "Dredging Maintenance System"
        button_style = "app-button-selected" if st.session_state.selected_app == "Dredging Maintenance System" else "app-button-unselected"
        st.markdown(f'<div class="{button_style} app-button">Dredging Maintenance System</div>', unsafe_allow_html=True)

    st.markdown("---")
   
    if st.session_state.selected_app == "Dredging Resource Allocator":
        app2()
    elif st.session_state.selected_app == "Dredging Maintenance System":
        app3()
    elif st.session_state.selected_app == "Dredging Contextual RAG Q&A":
        app1()

if __name__ == "__main__":
    main()
