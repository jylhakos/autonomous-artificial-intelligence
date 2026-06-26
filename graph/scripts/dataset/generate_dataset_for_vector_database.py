import pandas as pd
from sklearn.datasets import load_iris
from langchain_text_splitters import RecursiveCharacterTextSplitter

def generate_vector_db_ready_docs():
    # 1. Load Iris dataset using sklearn (reliable local approach)
    print("Loading Iris dataset from sklearn...")
    iris = load_iris()
    df = pd.DataFrame(iris.data, columns=iris.feature_names)
    df['class'] = iris.target_names[iris.target]

    # 2. Transform tabular rows into rich semantic text documents
    # Vector databases perform better when data reads like natural language
    semantic_documents = []
    
    for index, row in df.iterrows():
        doc_text = (
            f"Sample ID {index}: This biological specimen belongs to the genus Iris, "
            f"specifically classified under the species '{row['class']}'. "
            f"Morphological measurements indicate a sepal length of {row['sepal length (cm)']} cm "
            f"and a sepal width of {row['sepal width (cm)']} cm. "
            f"The floral structure further exhibits a petal length of {row['petal length (cm)']} cm "
            f"and a petal width of {row['petal width (cm)']} cm. "
            f"This specimen is part of a botanical research collection studying the morphological "
            f"variations within the Iris genus, which includes three main species: "
            f"Iris setosa, Iris versicolor, and Iris virginica."
        )
        semantic_documents.append(doc_text)

    # 3. Combine into a master text document
    combined_content = "\n\n".join(semantic_documents)
    
    output_filename = "document.txt"
    with open(output_filename, "w", encoding="utf-8") as f:
        f.write(combined_content)
        
    print(f"Successfully generated master text file: {output_filename}")

    # 4. Optional: Preview how a text splitter will chunk it for a Vector DB
    # Most vector pipelines use text splitters to enforce token limits
    splitter = RecursiveCharacterTextSplitter(chunk_size=300, chunk_overlap=30)
    chunks = splitter.create_documents([combined_content])
    
    print(f"\nVector Database Check:")
    print(f"- Total semantic documents generated: {len(semantic_documents)}")
    print(f"- Sample chunk for Vector Dababase embedding:\n{chunks[0].page_content}")

if __name__ == "__main__":
    generate_vector_db_ready_docs()
