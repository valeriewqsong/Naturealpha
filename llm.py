import os

# # Import Azure OpenAI
# from langchain_openai import AzureChatOpenAI

from langchain_openai import ChatOpenAI

from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import AzureOpenAIEmbeddings
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import DataFrameLoader


from data_processing import read_data_files,extract_first_two_letters

import pandas as pd
from geopy.geocoders import Nominatim
from geopy.exc import GeopyError

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
os.environ["PATH"] += "OPENAI_API_KEY"


# Create an instance of Azure OpenAI
# Replace the deployment name with your own
# llm = AzureChatOpenAI(
#     deployment_name="gpt-4o",
# )

llm_chat = ChatOpenAI(model="gpt-4o")

def country_generator_chain(llm,docs,isin_data):

    prompt = ChatPromptTemplate.from_messages([("system","For every row in the document, list the corresponding country of the country code in this format:Country Code - Country. Each row in the document is a country code. Only use one new line to separate it. Document: {context}")])

    chain = create_stuff_documents_chain(llm,prompt)

    answer = chain.invoke({"context":docs})
        
    data_rows = [row.split(' - ') for row in answer.split('\n')]
    
    # Create DataFrame
    df = pd.DataFrame(data_rows, columns=["Code", "Country"])

    # Display the DataFrame
    df['Code'] = df['Code'].str.strip()  # Remove leading and trailing spaces
    df['Country'] = df['Country'].str.strip() 
    df = df.drop(columns=['Code'])
    
    
    isin_data_new = pd.DataFrame()
    isin_data_new["Company_Country"]= isin_data['company_name'] + ", " + df['Country']
    isin_data_new['Entity ISIN'] = isin_data['Entity ISIN']
    
    return isin_data_new


# Dataframe to documents

def dataframe_to_docs(dataset,header,textSplitter=RecursiveCharacterTextSplitter,chunk_size=1024,chunk_overlap=500):
        # header to be instantiated (maybe) or should it be a parameter?
        text_splitter = textSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        )

        loader = DataFrameLoader(dataset,page_content_column = header)
        docs = loader.load()

        text_splitter.split_documents(docs)

        return docs


# VECTOR DB + EMBEDDINGS

def vector_db(dataset):

    openai_embeddings = OpenAIEmbeddings(
        model="text-embedding-ada-002",
    )
    db = FAISS.from_documents(dataset,openai_embeddings)
    retriever = db.as_retriever(search_kwargs={"k": 1})  
    return retriever



def add_country_column(df, lat_col="latitude", lon_col="longitude", country_col="country"):
    """
    Adds a country column to a DataFrame based on latitude and longitude columns if the country column is empty.

    Parameters:
    df (pd.DataFrame): DataFrame with latitude and longitude columns.
    lat_col (str): Name of the latitude column.
    lon_col (str): Name of the longitude column.
    country_col (str): Name of the new country column to add.

    Returns:
    pd.DataFrame: DataFrame with the new country column added.
    """
    # Initialize the geolocator
    geolocator = Nominatim(user_agent="geoapi")

    def get_country(lat, lon):
        try:
            location = geolocator.reverse((lat, lon), exactly_one=True, language="en")
            if location and "country" in location.raw.get("address", {}):
                return location.raw["address"]["country"]
            else:
                return None
        except GeopyError as e:
            print(f"Error retrieving country for lat: {lat}, lon: {lon} - {e}")
            return None

    # Apply the get_country function to the DataFrame only where the country column is empty or missing
    if country_col not in df.columns:
        df[country_col] = None

    df[country_col] = df.apply(
        lambda row: get_country(row[lat_col], row[lon_col]) if pd.isna(row.get(country_col)) else row[country_col],
        axis=1
    )

    return df


def run(isin_data,assets_data,retriever):
    results_df = pd.DataFrame(columns=["Company Name", "Company Location","ISIN NAME","ISIN Country"])
    for x in range(0,len(assets_data)):
        if assets_data[x].metadata.get('country') != '':
            query= assets_data[x].page_content + ", " +assets_data[x].metadata.get('country') 

        else:
            query= assets_data[x].page_content
        
        docs = retriever.invoke(query)
        print("Asset name: {}, Company Name: {}".format(assets_data[x].page_content,docs))
        ISIN_num = docs[0].metadata.get('Entity ISIN')
        Country = docs[0].page_content.split(', ')[1]
        
    # Create a temporary DataFrame with the current query and docs
        temp_df = pd.DataFrame({"Company Name": [assets_data[x].page_content], "Company Location":[assets_data[x].metadata.get('country') ], "ISIN NAME": [ISIN_num],"ISIN Country": [Country]})
        
        # Concatenate the temporary DataFrame with the main results DataFrame
        results_df = pd.concat([results_df, temp_df], ignore_index=True)

        # results_df.to_csv('results.csv',index=False)
    
    return results_df
        
def remove_mismatched_rows(df, col1, col2):
    """
    Removes rows from the DataFrame where the values in two specified columns do not match.

    Parameters:
        df (pd.DataFrame): The input DataFrame.
        col1 (str): The name of the first column to compare.
        col2 (str): The name of the second column to compare.

    Returns:
        pd.DataFrame: A new DataFrame with mismatched rows removed.
    """
    if col1 not in df.columns or col2 not in df.columns:
        raise ValueError(f"Both {col1} and {col2} must be columns in the DataFrame.")

    # Filter rows where the values in the two columns match
    filtered_df = df[df[col1] == df[col2]]
    
    filtered_df.to_csv('identified_companies.csv',index=False)

    return filtered_df

def metric(original_data,new_data):
    original_data_size = len(original_data)
    new_data_size = len(new_data)
    
    percentage_diff = (new_data_size/ original_data_size) * 100 
    return percentage_diff        



def main():
    # Extract data from files
    assets_data = read_data_files('Hydro_oil_gas_asset_data_06122024.csv',index_col=0)
    isin_data = read_data_files('lookup_table_isins.csv',index_col=None)
    
    # Created a dataframe with the first two letters of the ISIN code
    country_code_df = extract_first_two_letters(isin_data)
    
    # Assets data frame updated with missing countries
    assets_data_new = add_country_column(assets_data)
    
    # Create documents
    company_docs = dataframe_to_docs(assets_data_new,header="name")
    country_code_doc = dataframe_to_docs(country_code_df,header="Country code")


    # Generate countries based on country code
    isin_data_new = country_generator_chain(llm_chat,country_code_doc,isin_data)
    # Create documents
    isin_data_docs = dataframe_to_docs(isin_data_new,header="Company_Country")
    
    # Vector DB Retriever creation
    retriever = vector_db(isin_data_docs)
    
    results = run(isin_data_docs,company_docs,retriever)
    
    final_data = remove_mismatched_rows(results,'Company Location','ISIN Country')
    
    print(final_data)
    
    print("Accuracy:", metric(assets_data,final_data))
    
if __name__ == "__main__":
    main()