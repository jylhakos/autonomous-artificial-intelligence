"""
Enhanced GraphRAG Tools with Multi-Hop Reasoning Support
Demonstrates complex relationship traversal for Iris dataset
"""

from langchain.tools import tool
from neo4j import GraphDatabase
import re

# Neo4j Connection Configuration
URI = "bolt://localhost:7687"
AUTH = ("neo4j", "your_password")


@tool
def search_unstructured_text(query: str) -> str:
    """
    Search for semantic information about Iris species, morphological characteristics,
    and botanical descriptions from the unstructured text corpus.
    Use this for conceptual questions about flower biology, species descriptions,
    or general Iris dataset information.
    """
    # This would connect to your vector database
    # Placeholder implementation
    return f"Searching unstructured text for: {query}"


@tool
def query_graph_relationships(question: str) -> str:
    """
    Query the Iris knowledge graph for simple, direct relationships.
    Use this for basic questions about species, specimen counts, and direct connections.
    
    Examples:
    - "How many species are in the dataset?"
    - "What measurements does Specimen_5 have?"
    - "List all setosa specimens"
    """
    
    with GraphDatabase.driver(URI, auth=AUTH) as driver:
        with driver.session() as session:
            
            # Simple query patterns
            if "how many species" in question.lower():
                result = session.run("MATCH (s:Species) RETURN count(s) as count")
                count = result.single()["count"]
                return f"There are {count} species in the Iris dataset."
            
            elif "how many specimens" in question.lower():
                result = session.run("MATCH (sp:Specimen) RETURN count(sp) as count")
                count = result.single()["count"]
                return f"There are {count} specimens in the Iris dataset."
            
            elif "species names" in question.lower() or "list species" in question.lower():
                result = session.run("MATCH (s:Species) RETURN s.name as name")
                species = [record["name"] for record in result]
                return f"The Iris species are: {', '.join(species)}"
            
            else:
                # Default: return graph statistics
                result = session.run("""
                    MATCH (n)
                    RETURN labels(n)[0] as type, count(n) as count
                    ORDER BY count DESC
                """)
                stats = [f"{record['type']}: {record['count']}" for record in result]
                return "Graph statistics:\n" + "\n".join(stats)


@tool
def query_multihop_relationships(question: str) -> str:
    """
    Query the Iris knowledge graph using multi-hop relationship traversal.
    Use this for COMPLEX questions requiring traversal across multiple entity types
    and relationship levels.
    
    Supports traversal patterns like:
    Specimen → MeasurementGroup → CharacteristicType → Species → Genus
    
    Use this for:
    - Detailed specimen analysis with full taxonomic context
    - Cross-species characteristic comparisons
    - Pattern discovery across measurement categories
    - Hierarchical aggregations (specimen to genus level)
    - Finding similar specimens across different species
    - Taxonomic classification reasoning
    
    Examples:
    - "What are all the characteristics and taxonomy of Specimen_0?"
    - "Compare sepal characteristics across all species"
    - "Find specimens with similar petal patterns but different species"
    - "Which setosa specimens have large petal characteristics?"
    - "Show me the distribution of measurement sizes by species and characteristic type"
    """
    
    with GraphDatabase.driver(URI, auth=AUTH) as driver:
        with driver.session() as session:
            
            # Pattern 1: Full specimen characterization (5-hop traversal)
            if "specimen" in question.lower() and any(char.isdigit() for char in question):
                specimen_match = re.search(r'specimen[_\s]?(\d+)', question.lower())
                specimen_id = f"Specimen_{specimen_match.group(1)}" if specimen_match else "Specimen_0"
                
                result = session.run("""
                    MATCH path = (sp:Specimen {id: $specimen_id})
                                 -[:HAS_MEASUREMENT]->(mg:MeasurementGroup)
                                 -[:CATEGORIZED_AS]->(ct:CharacteristicType)
                                 -[:DEFINES_TRAIT_OF]->(s:Species)
                                 -[:BELONGS_TO_GENUS]->(g:Genus)
                    RETURN sp.id as specimen,
                           sp.sepal_length as sepal_length,
                           sp.sepal_width as sepal_width,
                           sp.petal_length as petal_length,
                           sp.petal_width as petal_width,
                           collect({
                               measurement: mg.type,
                               size: mg.size,
                               characteristic: ct.name
                           }) as measurements,
                           s.name as species,
                           g.name as genus
                """, specimen_id=specimen_id)
                
                records = [dict(record) for record in result]
                if records:
                    record = records[0]
                    response = f"\n=== Full Analysis of {record['specimen']} ===\n"
                    response += f"Species: {record['species']} (Genus: {record['genus']})\n"
                    response += f"\nRaw Measurements:\n"
                    response += f"  Sepal Length: {record['sepal_length']} cm\n"
                    response += f"  Sepal Width: {record['sepal_width']} cm\n"
                    response += f"  Petal Length: {record['petal_length']} cm\n"
                    response += f"  Petal Width: {record['petal_width']} cm\n"
                    response += f"\nCategorized Measurements:\n"
                    for meas in record['measurements']:
                        response += f"  {meas['measurement']}: {meas['size']} ({meas['characteristic']})\n"
                    return response
                else:
                    return f"No data found for {specimen_id}"
            
            # Pattern 2: Cross-species comparison (3-hop aggregation)
            elif "compare" in question.lower() or "comparison" in question.lower():
                characteristic_filter = ""
                if "sepal" in question.lower():
                    characteristic_filter = "WHERE ct.name = 'SepalCharacteristics'"
                elif "petal" in question.lower():
                    characteristic_filter = "WHERE ct.name = 'PetalCharacteristics'"
                
                result = session.run(f"""
                    MATCH (sp:Specimen)-[:HAS_MEASUREMENT]->(mg:MeasurementGroup)
                          -[:CATEGORIZED_AS]->(ct:CharacteristicType)
                          -[:DEFINES_TRAIT_OF]->(s:Species)
                    {characteristic_filter}
                    RETURN s.name as species,
                           ct.name as characteristic,
                           mg.size as size,
                           count(sp) as specimen_count,
                           avg(sp.sepal_length + sp.petal_length) as avg_total_length
                    ORDER BY s.name, ct.name, mg.size
                """)
                
                records = [dict(record) for record in result]
                if records:
                    response = "\n=== Species Comparison ===\n"
                    current_species = None
                    for record in records:
                        if current_species != record['species']:
                            current_species = record['species']
                            response += f"\n{current_species}:\n"
                        response += f"  {record['characteristic']} - {record['size']}: "
                        response += f"{record['specimen_count']} specimens "
                        response += f"(avg total length: {record['avg_total_length']:.2f} cm)\n"
                    return response
                else:
                    return "No comparison data found"
            
            # Pattern 3: Similar pattern detection (4-hop with bidirectional)
            elif "similar" in question.lower() or "pattern" in question.lower():
                result = session.run("""
                    MATCH (sp1:Specimen)-[:HAS_MEASUREMENT]->(mg1:MeasurementGroup)
                          -[:CATEGORIZED_AS]->(ct:CharacteristicType)
                          <-[:CATEGORIZED_AS]-(mg2:MeasurementGroup)
                          <-[:HAS_MEASUREMENT]-(sp2:Specimen)
                    WHERE sp1.id < sp2.id 
                      AND mg1.size = mg2.size 
                      AND mg1.type = mg2.type
                      AND sp1.species <> sp2.species
                    RETURN sp1.id as specimen1,
                           sp1.species as species1,
                           sp2.id as specimen2,
                           sp2.species as species2,
                           mg1.type as measurement_type,
                           mg1.size as shared_size,
                           ct.name as characteristic
                    LIMIT 15
                """)
                
                records = [dict(record) for record in result]
                if records:
                    response = "\n=== Specimens with Similar Patterns Across Species ===\n"
                    for record in records:
                        response += f"\n{record['specimen1']} ({record['species1']}) ↔ "
                        response += f"{record['specimen2']} ({record['species2']})\n"
                        response += f"  Shared: {record['measurement_type']} - {record['shared_size']} "
                        response += f"({record['characteristic']})\n"
                    return response
                else:
                    return "No similar patterns found across different species"
            
            # Pattern 4: Filtered multi-hop search
            elif "which" in question.lower() or "find" in question.lower():
                # Extract species if mentioned
                species_filter = ""
                for species in ["setosa", "versicolor", "virginica"]:
                    if species in question.lower():
                        species_filter = f"AND s.name = '{species}'"
                        break
                
                # Extract size if mentioned
                size_filter = ""
                for size in ["small", "medium", "large"]:
                    if size in question.lower():
                        size_filter = f"AND mg.size = '{size.capitalize()}'"
                        break
                
                # Extract characteristic type
                char_filter = ""
                if "sepal" in question.lower():
                    char_filter = "AND ct.name = 'SepalCharacteristics'"
                elif "petal" in question.lower():
                    char_filter = "AND ct.name = 'PetalCharacteristics'"
                
                result = session.run(f"""
                    MATCH path = (sp:Specimen)-[:HAS_MEASUREMENT]->(mg:MeasurementGroup)
                                 -[:CATEGORIZED_AS]->(ct:CharacteristicType)
                                 -[:DEFINES_TRAIT_OF]->(s:Species)
                                 -[:BELONGS_TO_GENUS]->(g:Genus)
                    WHERE 1=1 {species_filter} {size_filter} {char_filter}
                    RETURN sp.id as specimen,
                           s.name as species,
                           ct.name as characteristic,
                           mg.type as measurement,
                           mg.size as size,
                           length(path) as hops
                    LIMIT 20
                """)
                
                records = [dict(record) for record in result]
                if records:
                    response = f"\n=== Found {len(records)} matching specimens ===\n"
                    for record in records:
                        response += f"{record['specimen']} ({record['species']}): "
                        response += f"{record['measurement']} = {record['size']} "
                        response += f"[{record['characteristic']}, {record['hops']} hops]\n"
                    return response
                else:
                    return "No specimens match the specified criteria"
            
            # Pattern 5: Hierarchical aggregation (genus → species → characteristics)
            elif "distribution" in question.lower() or "aggregat" in question.lower():
                result = session.run("""
                    MATCH (sp:Specimen)-[:HAS_MEASUREMENT]->(mg:MeasurementGroup)
                          -[:CATEGORIZED_AS]->(ct:CharacteristicType)
                          -[:DEFINES_TRAIT_OF]->(s:Species)
                          -[:BELONGS_TO_GENUS]->(g:Genus)
                    RETURN g.name as genus,
                           s.name as species,
                           ct.name as characteristic,
                           mg.size as size,
                           count(sp) as count
                    ORDER BY g.name, s.name, ct.name, mg.size
                """)
                
                records = [dict(record) for record in result]
                if records:
                    response = "\n=== Hierarchical Distribution: Genus → Species → Characteristics ===\n"
                    current_genus = None
                    current_species = None
                    current_char = None
                    
                    for record in records:
                        if current_genus != record['genus']:
                            current_genus = record['genus']
                            response += f"\nGenus: {current_genus}\n"
                        
                        if current_species != record['species']:
                            current_species = record['species']
                            response += f"  Species: {current_species}\n"
                        
                        if current_char != record['characteristic']:
                            current_char = record['characteristic']
                            response += f"    {current_char}:\n"
                        
                        response += f"      {record['size']}: {record['count']} specimens\n"
                    
                    return response
                else:
                    return "No distribution data found"
            
            else:
                # Default: Show example multi-hop capabilities
                return """
Multi-hop querying is available! Try questions like:
- "What are all characteristics of Specimen_0?"
- "Compare sepal characteristics across all species"
- "Find specimens with similar petal patterns but different species"
- "Which setosa specimens have large petal characteristics?"
- "Show me the distribution of measurement sizes by species"

This allows traversal through:
Specimen → MeasurementGroup → CharacteristicType → Species → Genus
"""


if __name__ == "__main__":
    # Test the multi-hop tool
    print("Testing multi-hop queries...")
    
    test_questions = [
        "What are all characteristics of Specimen_0?",
        "Compare sepal characteristics across all species",
        "Find specimens with similar patterns",
        "Which setosa specimens have large petal characteristics?",
        "Show me the distribution"
    ]
    
    for question in test_questions:
        print(f"\n{'='*70}")
        print(f"Q: {question}")
        print(f"{'='*70}")
        result = query_multihop_relationships.invoke({"question": question})
        print(result)
