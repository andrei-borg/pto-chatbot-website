from llama_index.vector_stores.chroma import ChromaVectorStore
from llama_index.core import VectorStoreIndex
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.llms.groq import Groq
from llama_index.core.chat_engine.context import ContextChatEngine
from llama_index.core.tools import QueryEngineTool
from llama_index.agent.openai import OpenAIAgent
from llama_index.core.memory import ChatMemoryBuffer
from llama_index.core.storage.chat_store import SimpleChatStore
from llama_index.postprocessor.colbert_rerank import ColbertRerank
import streamlit as st
import chromadb

st.write("## PTO Mack Chatbot")

if "llm_model" not in st.session_state:
    st.session_state["llm_model"] = "llama-3.1-8b-instant"

chat_store = SimpleChatStore()

chat_memory = ChatMemoryBuffer(
    token_limit=3000,
    chat_store_=chat_store,
    chat_store_key="user_1"
)

if "memory" not in st.session_state:
    st.session_state.memory = chat_memory

if "messages" not in st.session_state:
    st.session_state.messages = []

@st.cache_resource
def load_recursive_index():
    nomic_emb = HuggingFaceEmbedding(model_name="corto-ai/nomic-embed-text-v1", trust_remote_code=True) # Local emb
    chroma_disk = chromadb.PersistentClient(path="llama_parse_4osum_nomic")
    chroma_collection_disk = chroma_disk.get_collection("llama_parse")
    vector_store_disk = ChromaVectorStore(chroma_collection=chroma_collection_disk)
    recursive_index = VectorStoreIndex.from_vector_store(vector_store=vector_store_disk, embed_model=nomic_emb)
    return recursive_index

recursive_index = load_recursive_index()

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

with st.sidebar:
    groq_api_key = st.text_input("Paste Groq API-key here:", placeholder="API-key",
                                 type="password",
                                 help="You can create a free Groq API-key from https://console.groq.com/keys")

    # LLM Radio
    llm_option = st.radio("Select LLM (3.1 is the latest version from July 2024):", ["Llama-3.1-8B", "Llama-3.1-70B", "Llama-3-8B", "Llama-3-70B"],
                          captions=["New, smaller and faster", "New, bigger and more versatile"])

    # Embedding selection
    #embedding_option = st.selectbox("Choose embedding setup:", ("Nomic + GPT (Default)", "Nomic + Llama3"),
    #                                index=None, placeholder="Select embedding model...")
    
    if st.button("Apply settings", type="primary"):
        #if not groq_api_key and not embedding_option:
        #    st.error(":red[Please insert API-key and choose an embedding setup above!]")

        if not groq_api_key:
            st.error(":red[Please insert API-key above!]")

        #elif not embedding_option:
        #    st.error(":red[Please choose an embedding setup from the options above!]")
        
        # No errors
        else:
            with st.spinner("Saving settings..."):
                #if llm_option == "Llama-3.1-8B":
                #    st.session_state["llm_model"] = "llama-3.1-8b-instant"
                #elif llm_option == "Llama-3.1-70B":
                #    st.session_state["llm_model"] = "llama-3.1-70b-versatile"
                
                st.session_state["reranker"] = ColbertRerank(
                        top_n=4,
                        model="colbert-ir/colbertv2.0",
                        tokenizer="colbert-ir/colbertv2.0",
                        keep_retrieval_score=True,
                    )

                
                if llm_option == "Llama-3.1-8B":
                    st.session_state["llm_model"] = "llama-3.1-8b-instant"
                elif llm_option == "Llama-3.1-70B":
                    st.session_state["llm_model"] = "llama-3.1-70b-versatile"
                elif llm_option == "Llama-3-8B":
                    st.session_state["llm_model"] = "llama3-8b-8192"
                else:
                    st.session_state["llm_model"] = "llama3-70b-8192"

                st.success(":green[Settings saved!]")

if prompt := st.chat_input("Type question here"):
    if not groq_api_key:
        st.error(":red[Please add API-key!]")
    else:
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)
        
        with st.chat_message("assistant"):

            llm = Groq(model=st.session_state["llm_model"], api_key=groq_api_key)
            query_engine = recursive_index.as_query_engine(llm=llm, similarity_top_k=6,
                                                           node_postprocessor=[st.session_state["reranker"]])

            tool = QueryEngineTool.from_defaults(query_engine=query_engine,
                                                 name="Mack_Power_Take_Off_engine",
                                                 description="This query engine stores parameter code descriptions of Mack Power Take Off (PTO) instructions.")
            
            agent = OpenAIAgent.from_tools(tools=[tool],
                                           memory=st.session_state["memory"],
                                           #verbose=True,
                                           llm=llm,
                                           system_prompt="You are a usefull agent. When you are asked for a parameter \
                                            parameter name, it usually starts with 'P1'. An example parameter is 'P1V5K'.")
            answer = agent.chat(prompt)
            st.write(answer.response)
            #st.write(answer.sources)
        
        st.session_state.messages.append({"role": "assistant", "content": answer.response})