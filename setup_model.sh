#!/bin/bash
echo "Creating custom 'nomic-rag' model with increased context window..."
ollama create nomic-rag -f Modelfile
echo "Done! 'nomic-rag' is ready to use."
