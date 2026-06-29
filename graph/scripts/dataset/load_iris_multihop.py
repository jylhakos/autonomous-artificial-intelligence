"""
Multi-Hop Graph Construction for Iris Dataset
Demonstrates complex relationship traversal similar to Employee -> Project -> Client -> Vendor

Graph Structure:
Specimen -> MeasurementGroup -> CharacteristicType -> Species -> Genus
"""

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


def categorize_measurement(value, measurement_type):
    """Categorize measurements into Small, Medium, Large"""
    if measurement_type in ['sepal length (cm)', 'petal length (cm)']:
        if value < 2.0:
            return 'Small'
        elif value < 5.0:
            return 'Medium'
        else:
            return 'Large'
    else:  # width measurements
        if value < 1.0:
            return 'Small'
        elif value < 2.5:
            return 'Medium'
        else:
            return 'Large'


def get_characteristic_type(measurement_name):
    """Map measurements to characteristic types"""
    if 'sepal' in measurement_name:
        return 'SepalCharacteristics'
    elif 'petal' in measurement_name:
        return 'PetalCharacteristics'
    return 'UnknownCharacteristic'


def create_multihop_graph():
    """Create a multi-layered graph with complex relationships"""
    
    with GraphDatabase.driver(URI, auth=AUTH) as driver:
        with driver.session() as session:
            
            # 1. Create Genus node (top level)
            print("Creating Genus node...")
            session.run("""
                MERGE (g:Genus {name: 'Iris', family: 'Iridaceae'})
            """)
            
            # 2. Create Species nodes and link to Genus
            print("Creating Species nodes...")
            for species_name in iris.target_names:
                session.run("""
                    MERGE (s:Species {name: $species_name})
                    WITH s
                    MATCH (g:Genus {name: 'Iris'})
                    MERGE (s)-[:BELONGS_TO_GENUS]->(g)
                """, species_name=species_name)
            
            # 3. Create CharacteristicType nodes and link to Species
            print("Creating CharacteristicType nodes...")
            characteristic_types = ['SepalCharacteristics', 'PetalCharacteristics']
            for char_type in characteristic_types:
                for species_name in iris.target_names:
                    session.run("""
                        MERGE (ct:CharacteristicType {
                            name: $char_type,
                            species: $species_name
                        })
                        WITH ct
                        MATCH (s:Species {name: $species_name})
                        MERGE (ct)-[:DEFINES_TRAIT_OF]->(s)
                    """, char_type=char_type, species_name=species_name)
            
            # 4. Create MeasurementGroup nodes for each combination
            print("Creating MeasurementGroup nodes...")
            measurement_groups = {
                'sepal length (cm)': 'SepalCharacteristics',
                'sepal width (cm)': 'SepalCharacteristics',
                'petal length (cm)': 'PetalCharacteristics',
                'petal width (cm)': 'PetalCharacteristics'
            }
            
            for idx, row in df.iterrows():
                for meas_name, char_type in measurement_groups.items():
                    size_category = categorize_measurement(row[meas_name], meas_name)
                    
                    session.run("""
                        MERGE (mg:MeasurementGroup {
                            type: $meas_name,
                            size: $size_category,
                            species: $species_name
                        })
                        WITH mg
                        MATCH (ct:CharacteristicType {
                            name: $char_type,
                            species: $species_name
                        })
                        MERGE (mg)-[:CATEGORIZED_AS]->(ct)
                    """, 
                    meas_name=meas_name, 
                    size_category=size_category,
                    char_type=char_type,
                    species_name=row['species'])
            
            # 5. Create Specimen nodes and measurements (base level)
            print("Creating Specimen nodes with measurements...")
            for idx, row in df.iterrows():
                specimen_id = f"Specimen_{idx}"
                
                # Create specimen
                session.run("""
                    CREATE (sp:Specimen {
                        id: $specimen_id,
                        sepal_length: $sepal_length,
                        sepal_width: $sepal_width,
                        petal_length: $petal_length,
                        petal_width: $petal_width,
                        species: $species_name
                    })
                """, 
                specimen_id=specimen_id,
                sepal_length=float(row['sepal length (cm)']),
                sepal_width=float(row['sepal width (cm)']),
                petal_length=float(row['petal length (cm)']),
                petal_width=float(row['petal width (cm)']),
                species_name=row['species'])
                
                # Link specimen to MeasurementGroups
                for meas_name, char_type in measurement_groups.items():
                    size_category = categorize_measurement(row[meas_name], meas_name)
                    
                    session.run("""
                        MATCH (sp:Specimen {id: $specimen_id})
                        MATCH (mg:MeasurementGroup {
                            type: $meas_name,
                            size: $size_category,
                            species: $species_name
                        })
                        MERGE (sp)-[:HAS_MEASUREMENT]->(mg)
                    """,
                    specimen_id=specimen_id,
                    meas_name=meas_name,
                    size_category=size_category,
                    species_name=row['species'])
            
            print("\n" + "="*70)
            print("Multi-hop Iris Graph created successfully!")
            print("="*70)
            
            # Verify the graph structure
            print("\nGraph Statistics:")
            result = session.run("""
                MATCH (n)
                RETURN labels(n)[0] as NodeType, count(n) as Count
                ORDER BY Count DESC
            """)
            for record in result:
                print(f"  {record['NodeType']}: {record['Count']} nodes")
            
            print("\nRelationship Statistics:")
            result = session.run("""
                MATCH ()-[r]->()
                RETURN type(r) as RelationType, count(r) as Count
                ORDER BY Count DESC
            """)
            for record in result:
                print(f"  {record['RelationType']}: {record['Count']} relationships")
            
            print("\n" + "="*70)
            print("Example Multi-Hop Query:")
            print("="*70)
            
            # Example multi-hop query
            result = session.run("""
                MATCH path = (sp:Specimen)-[:HAS_MEASUREMENT]->(mg:MeasurementGroup)
                             -[:CATEGORIZED_AS]->(ct:CharacteristicType)
                             -[:DEFINES_TRAIT_OF]->(s:Species)
                             -[:BELONGS_TO_GENUS]->(g:Genus)
                WHERE sp.id = 'Specimen_0'
                RETURN sp.id as Specimen,
                       mg.type as MeasurementType,
                       mg.size as Size,
                       ct.name as Characteristic,
                       s.name as Species,
                       g.name as Genus
                LIMIT 5
            """)
            
            print("\nTraversal: Specimen → MeasurementGroup → CharacteristicType → Species → Genus")
            print("-" * 70)
            for record in result:
                print(f"{record['Specimen']} → {record['MeasurementType']} ({record['Size']}) → "
                      f"{record['Characteristic']} → {record['Species']} → {record['Genus']}")


if __name__ == "__main__":
    create_multihop_graph()
