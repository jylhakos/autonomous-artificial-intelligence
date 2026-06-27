from neo4j import GraphDatabase
from sklearn.datasets import load_iris
import pandas as pd

# Load Iris Dataset
iris = load_iris()
df = pd.DataFrame(iris.data, columns=iris.feature_names)
df['species'] = [iris.target_names[i] for i in iris.target]

# Neo4j Connection
URI = "bolt://localhost:7687"
AUTH = ("neo4j", "your_password")

query_string = """
UNWIND $records AS record
MERGE (s:Species {name: record.species})
CREATE (m:Measurement {
    sepalLength: record.`sepal length (cm)`,
    sepalWidth: record.`sepal width (cm)`,
    petalLength: record.`petal length (cm)`,
    petalWidth: record.`petal width (cm)`
})
CREATE (m)-[:IS_SPECIES]->(s)
"""

with GraphDatabase.driver(URI, auth=AUTH) as driver:
    with driver.session() as session:
        session.run(query_string, records=df.to_dict('records'))
        print("Iris relationships loaded successfully!")
