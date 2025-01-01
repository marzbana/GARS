# Enhancing Code Generation with Grammar-Augmented Repeated Sampling

## Project Description
This project explores the enhancement of code generation capabilities in large language models (LLMs) by integrating grammar-augmented decoding and repeated sampling techniques. It leverages the SynCode framework, which enforces syntactical correctness during the generation process using deterministic finite automaton (DFA) constraints. To improve solution accuracy and coverage, repeated sampling strategies inspired by the "Large Language Monkeys" methodology are employed. The project evaluates its approach on a subset of the HumanEval benchmark, showcasing how smaller, cost-effective models can achieve comparable performance to larger counterparts.

## Tech Stack
- **SynCode Framework**: Grammar-augmented decoding for syntactical correctness.
- **Hugging Face Transformers**: For loading and running models like Stable-Code-3b, CodeLlama-7b-Instruct, and OpenCodeInterpreter-DS-6.7B.
- **Python**: Core programming language for model integration and evaluation.
- **HumanEval Dataset**: Benchmark dataset for evaluating code generation quality.

## Installation and Setup

Follow these steps to set up and run the project using Google Colab:

1. **Upload Notebook Files to Google Drive**:
   - Save the provided notebook files (`Project2_Stable_code_3b.ipynb`, `Project2_CodeLlama_7b_Instruct_hf.ipynb`, etc.) to your Google Drive.

2. **Open Google Colab**:
   - Go to [Google Colab](https://colab.research.google.com/) and sign in with your Google account.

3. **Import the Notebook**:
   - Click on "File" > "Open Notebook" > "Google Drive" and select the notebook file you want to run.

4. **Run Cells Sequentially**:
   - Execute the cells in the notebook one by one. Ensure that the required models are downloaded and any necessary configurations are completed as specified in the notebook.

## Challenges Faced

The most challenging aspect of this project was balancing computational costs with performance improvements. Incorporating grammar-enforced decoding required significant resource allocation, particularly when paired with repeated sampling. To overcome this, we optimized the SynCode framework for batch processing, enabling parallel inference across multiple samples, which reduced overhead without sacrificing accuracy.

## Future Work
This project lays the foundation for scaling LLM inference with enhanced correctness and reliability. Future research could explore:
- Broader language and domain applicability.
- Further optimization of computational costs.
- Integration with automatic correctness verification tools.
