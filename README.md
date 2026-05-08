# VectorDB — Java Backend (Spring Boot)

Java port of the C++ VectorDB backend. Same REST API, same `index.html` frontend.

## Requirements
- Java 17+
- Maven 3.8+
- Ollama (for RAG): https://ollama.com

## Setup

```bash
# 1. Pull Ollama models
ollama pull nomic-embed-text
ollama pull llama3.2

# 2. Build the fat JAR
mvn clean package -DskipTests

# 3. Run from the folder that contains index.html
#    (copy index.html next to the JAR, or run from the C++ project folder)
java -jar target/vectordb-1.0.0.jar
```

Open http://localhost:8080

## Run with Maven (dev)
```bash
# Run from the directory containing index.html
mvn spring-boot:run -Dspring-boot.run.workingDirectory="..\Your-OWN-AI-main"
```

Or copy `index.html` into `VectorDB-Java/` and just run:
```bash
mvn spring-boot:run
```
