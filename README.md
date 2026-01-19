# Alpha Discovery via Grammar-Guided Learning and Search

## Project Overview

This project is a Python-based integrated analysis toolkit designed for quantitative trading, data backtesting, and strategy research. It is built upon the theory of formal languages and consists of four hierarchical language processing modules, each corresponding to a different level of expression validity.

## Theoretical Framework

Our system is built on a nested language hierarchy:

• The set of all possible expressions

• The set of syntactically valid expressions

• The set of semantically valid expressions

• The set of semantically valid expressions with length not exceeding K 


## Core Modules

### 1. CFG-S

Corresponding language level:

A base Context-Free Grammar (CFG) parser responsible for filtering out syntactically correct expressions from all possible expressions, forming the syntactically valid language.

### 2. CFG-SS

Corresponding language level:

An enhanced version built on CFG-S, introducing semantic constraints and state management. It further filters syntactically correct expressions to obtain semantically meaningful expressions, forming the semantically valid language.

### 3. CFG-SSL

Corresponding language level:

A module that integrates complex syntax parsing with learning capabilities. It imposes a length constraint on semantically valid expressions, generating optimized expressions whose length does not exceed K.

### 4. RPN

A Reverse Polish Notation (RPN) calculator, used as a baseline or reference module.

## Environment Setup and Installation

### Prerequisites

• Python 3.8–3.11 (Python 3.9 or 3.10 recommended)

• pip (Python package manager)


### Installation Steps

1. Clone the repository:

   ```bash
   git clone <your-repo-url>
   cd alpha-discovery-via-grammar-guided-learning-and-search
   ```
2. Create a virtual environment:

   ```bash
   python -m venv venv
   source venv/bin/activate  # Linux/macOS
   # or
   .\venv\Scripts\activate   # Windows
   ```
3. Install dependencies:

   ```bash
   pip install -r requirements.txt
   ```
4. Verify the installation:

   ```bash
   python -c "import torch; print(f'PyTorch version: {torch.__version__}'); print(f'CUDA available: {torch.cuda.is_available()}')"
   ```

## Data Preparation

You can obtain the required stock data using one of the following approaches.

### Option 1: Using a Built-in Script (Recommended)

Run the built-in script to automatically fetch and preprocess data using **Baostock**:

```bash
python alphagen/data_collection/fetch_baostock_data.py
```

### Option 2: Use Qlib’s Data Pipeline (Optional)

Alternatively, you may leverage Qlib for more flexible and large-scale financial data handling.

GitHub repository:
https://github.com/microsoft/qlib


## Running the Project

To execute the integrated module pool, just enter the folder directory of different projects. For example:

```bash
cd ~/CFG-S
```

and run:

```bash
python run_pool.py
```

## Project Structure

```
.
├── CFG-S/             # Syntax-validity processing engine
├── CFG-SS/            # + Semantic-validity processing engine
├── CFG-SSL/           # + Constraint-optimized processing engine
├── RPN/               # Reverse Polish Notation calculator
├── requirements.txt   # Project dependencies
└── README.md          # Project documentation
```

## Notes

• Installation may take some time — ensure a stable network connection.

• The project depends on large libraries such as PyTorch and TensorFlow — ensure sufficient disk space.

• For CUDA-related issues, please verify GPU driver and CUDA version compatibility.


## Support

If you encounter any issues, please submit an issue on GitHub or contact the development team.
