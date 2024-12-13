import streamlit as st
import pandas as pd
import csv
from collections import Counter
from langchain_groq import ChatGroq
from langchain.prompts import ChatPromptTemplate
from langchain.schema.output_parser import StrOutputParser

# Page configuration
st.set_page_config(
    page_title="CSV File Analyzer",
    page_icon="📊",
    layout="wide"
)

# Initialize session state
if 'df' not in st.session_state:
    st.session_state.df = None

def detect_delimiter(file_content, sample_size=4096):
    """Detect the delimiter in CSV content"""
    common_delimiters = [',', ';', '\t', '|', ':']
    
    # Take a sample of the content
    sample = file_content[:sample_size]
    
    # Method 1: Use csv.Sniffer
    try:
        dialect = csv.Sniffer().sniff(sample, delimiters=common_delimiters)
        if dialect.delimiter in common_delimiters:
            return dialect.delimiter
    except:
        pass
    
    # Method 2: Count potential delimiters
    counts = Counter(char for char in sample if char in common_delimiters)
    if counts:
        most_common = counts.most_common(1)[0]
        lines = sample.split('\n')[:5]
        if lines:
            first_line_count = lines[0].count(most_common[0])
            if first_line_count > 0 and all(line.count(most_common[0]) == first_line_count
                                          for line in lines[1:] if line.strip()):
                return most_common[0]
    
    return ','  # Default to comma if no delimiter is detected

def process_csv(uploaded_file):
    """Process the uploaded CSV file"""
    try:
        # Read file content
        content = uploaded_file.getvalue().decode('utf-8')
        delimiter = detect_delimiter(content)
        
        # Read CSV into DataFrame
        df = pd.read_csv(uploaded_file, sep=delimiter)
        return df, None
    except Exception as e:
        return None, f"Error processing file: {str(e)}"

def setup_langchain():
    """Setup LangChain with Groq"""
    try:
        llm = ChatGroq(
            model_name="llama3-8b-8192",
            api_key=st.secrets["groq_api_key"]
        )
        
        prompt = ChatPromptTemplate.from_messages([
            (
                "system",
                "You are an assistant designed to generate pandas queries. "
                "You will use the provided column names {column_names} and their data types {column_data_types} "
                "to generate a valid pandas query based on the user's input. "
                "If the user requests to display the entire dataset, simply return 'df'. "
                "Only return the generated pandas query; do not include any additional text or explanations."
            ),
            (
                "user",
                "{input}"
            )
        ])
        
        chain = prompt | llm | StrOutputParser()
        return chain
    except Exception as e:
        st.error(f"Error setting up LangChain: {str(e)}")
        return None

def execute_pandas_query(df, query_string):
    """Execute a pandas query and return the result"""
    try:
        query_string = query_string.strip()
        
        # Remove outer quotes
        while (query_string.startswith('"') and query_string.endswith('"')) or \
              (query_string.startswith("'") and query_string.endswith("'")):
            query_string = query_string[1:-1]
        
        # Remove backticks and clean up df references
        query_string = query_string.replace('`', '')
        if query_string.startswith('df.df.'):
            query_string = query_string[3:]
        
        # Execute query
        result = eval(query_string)
        return result, None
    except Exception as e:
        return None, f"Error executing query: {str(e)}"

# Main app
def main():
    st.title("📊 CSV File Analyzer")
    st.write("Upload your CSV file and analyze it using natural language queries!")
    
    # File uploader
    uploaded_file = st.file_uploader("Choose a CSV file", type="csv")
    
    if uploaded_file:
        with st.spinner("Processing CSV file..."):
            df, error = process_csv(uploaded_file)
            
            if error:
                st.error(error)
            else:
                st.session_state.df = df
                st.success("File uploaded successfully!")
                
                # Display basic information
                st.subheader("Dataset Overview")
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Rows", df.shape[0])
                with col2:
                    st.metric("Columns", df.shape[1])
                with col3:
                    st.metric("Memory Usage", f"{df.memory_usage(deep=True).sum() / 1024:.2f} KB")
                
                # Display sample data
                st.subheader("Sample Data")
                st.dataframe(df.head(), use_container_width=True)
                
                # Query interface
                st.subheader("Query Your Data")
                query = st.text_input("Enter your query in natural language:")
                
                if query:
                    chain = setup_langchain()
                    if chain:
                        with st.spinner("Generating query..."):
                            try:
                                # Generate pandas query
                                result = chain.invoke({
                                    "column_names": list(df.columns),
                                    "column_data_types": list(df.dtypes.values),
                                    "input": query
                                })
                                
                                # Execute query
                                st.code(result, language="python")
                                query_result, query_error = execute_pandas_query(df, result)
                                
                                if query_error:
                                    st.error(query_error)
                                else:
                                    st.subheader("Query Result")
                                    if isinstance(query_result, pd.DataFrame):
                                        st.dataframe(query_result, use_container_width=True)
                                    else:
                                        st.write(query_result)
                            
                            except Exception as e:
                                st.error(f"Error processing query: {str(e)}")

if __name__ == "__main__":
    main()
