from unstructured.partition.pdf import partition_pdf  # required popplers and tesseract
import os
import base64
from langchain_core.messages import HumanMessage, AIMessage
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableLambda
from langchain.prompts import PromptTemplate
from langchain.schema.runnable import RunnableLambda, RunnablePassthrough
from langchain_community.vectorstores import Chroma
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from utils import *
import io
import streamlit as st
from dotenv import load_dotenv

# load Gemini_model api key
load_dotenv()

# to print image
def plt_img_base64(b64data):
    """
    Display base64-encoded image data
    """
    try:
        decoded_bytes = base64.b64decode(b64data)
        image = Image.open(io.BytesIO(decoded_bytes))

        # Display the image using st.image
        st.image(image, caption='Base64-encoded image')
    except Exception as e:
        st.write(f"Error displaying image: {e}")

def extract_pdf_path(uploaded_file):
    """
    Extracts the file path from the UploadedFile object.
    """
    # to Check if the attribute 'name' exists, and use it as the file path
    if hasattr(uploaded_file, 'name'):
        return uploaded_file.name
    else:
        st.warning("Unable to determine file path. Using a default filename.")
        return "default.pdf"  # Replace with your actual default filename




def img_prompt_func(data_dict):
    """
    Join the context into a single string
    """
    formatted_texts = "\n".join(data_dict["context"]["texts"])
    messages = []

    # Adding the text for analysis
    text_message = {
        "type": "text",
        "text": (
            "You are a reseracher tasking with providing factual answers from research papers.\n"
            "You will be given a mix of text, tables, and image(s) usually of charts or graphs.\n"
            "Use this information to provide answers related to the user question. \n"
            f"User-provided question: {data_dict['question']}\n\n"
            "Text and / or tables:\n"
            f"{formatted_texts}"
        ),
    }
    messages.append(text_message)
    # Adding image(s) to the messages if present
    if data_dict["context"]["images"]:
        for image in data_dict["context"]["images"]:
            image_message = {
                "type": "image_url",
                "image_url": {"url": f"data:image/jpeg;base64,{image}"},
            }
            messages.append(image_message)

    return [HumanMessage(content=messages)]



def multi_modal_rag_chain(retriever):
    """
    Multi-modal RAG chain
    """
    model_vision = ChatGoogleGenerativeAI(model="gemini-pro-vision",temperature=0, max_tokens=1024)

    # RAG pipeline
    chain = (
        {
            "context": retriever | RunnableLambda(split_image_text_types),
            "question": RunnablePassthrough(),
        }
        | RunnableLambda(img_prompt_func)
        | model_vision  # MM_LLM
        | StrOutputParser()
    )

    return chain

# data loading and text,table and image summarisation 
@st.cache(allow_output_mutation=True)
def data_loader(pdf_path):
    # extract images from documents using unstructured library
    image_path = "./figures"
    pdf_elements = partition_pdf(
        pdf_path, # here pdf_path
        chunking_strategy="by_title",
        extract_images_in_pdf=True,
        infer_table_structure=True,
        max_characters=3000,
        new_after_n_chars=2800,
        combine_text_under_n_chars=2000,
        image_output_dir_path=image_path
        )
    
    # extract tables and texts
    texts, tables = categorize_elements(pdf_elements)
    # Get text & table summaries
    text_summaries, table_summaries = generate_text_summaries(texts[0:19], tables, summarize_texts=True)
    # Image summaries
    img_base64_list, image_summaries = generate_img_summaries(image_path)

    return text_summaries, texts, table_summaries, tables, image_summaries, img_base64_list


# storing the summaries and raw info in the vector and doc stores respectively for retreval
@st.cache(allow_output_mutation=True)
def retriever_func(text_summaries,texts,table_summaries,tables,image_summaries,img_base64_list):

    # The vectorstore to use to index the summaries
    vectorstore = Chroma(
        collection_name="mm_rag_gemini",
        embedding_function=GoogleGenerativeAIEmbeddings(model="models/embedding-001"), # embedding model  
    )

    # Create retriever
    retriever_multi_vector_img = create_multi_vector_retriever(
        vectorstore,
        text_summaries,
        texts,
        table_summaries,
        tables,
        image_summaries,
        img_base64_list,
    )

    return retriever_multi_vector_img


# Streamlit UI
def main():
    st.title("Multi-Modal RAG ResearcherQA Bot ")

    # PDF upload file container
    pdf_file = st.file_uploader("Upload PDF file", type=["pdf"])
 
    # Check if PDF file is uploaded
    if pdf_file is not None:
        # Extract file path from UploadedFile object
        pdf_path = extract_pdf_path(pdf_file)
        # Initialize session state to store retriever data
        if 'retriever_multi_vector_img' not in st.session_state:
            # Load data from PDF
            pdf_data = data_loader(pdf_path)
            # Use retriever Function
            retriever_multi_vector_img = retriever_func(*pdf_data)
            st.session_state.retriever_multi_vector_img = retriever_multi_vector_img

    # User input with generate button
    user_input = st.text_input("Enter your question:")
    generate_button = st.button("Generate Answer")

    # Perform QA on button click
    if generate_button:
        if pdf_file is not None:
            # Get retriever data from session state
            retriever_multi_vector_img = st.session_state.retriever_multi_vector_img
            # Perform QA with the uploaded PDF file and user input
            query = f"{user_input}"
            # intermediate result
            docs = retriever_multi_vector_img.get_relevant_documents(query, limit=1) # intermediate reusults

            # Display intermediate result inside a hidable container
            with st.expander("Multi-Vector Retreiver result based on query"):
                for doc in docs:
                    if is_image_data(doc):
                        plt_img_base64(doc)
                    else:
                        st.write(doc)
                
            # Create RAG chain for final query
            chain_multimodal_rag = multi_modal_rag_chain(retriever_multi_vector_img)

            # Display final result in a text box
            st.text_area("Final result from LLM", chain_multimodal_rag.invoke(query))  


if __name__ == "__main__":
    main()