# Real-Time AI Stock Advisor with Ollama (Llama 2) & Streamlit

This project provides real-time stock analysis and insights using large language models. It fetches stock data every minute, analyzes trends, and provides real-time, easy-to-understand explanations.

## Features

- Fetches real-time stock data for Apple (AAPL) and Dow Jones (DJI)
- Calculates technical indicators (EMA, RSI, Bollinger Bands)
- Generates natural language insights using Llama 2 via Ollama
- Interactive Streamlit web application with auto-refresh

## Prerequisites

- Python 3.7 or higher
- [Ollama](https://github.com/jmorganca/ollama) installed and configured
- Git (optional, for version control)

## Installation

### Clone the Repository

```bash
git clone https://github.com/satishgonella2024/ai-stock-advisor.git
cd ai-stock-advisor
```
## Set Up a Virtual Environment

```bash
python3 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```
## Install Dependencies

```bash
pip install -r requirements.txt
```

### Install and Configure Ollama

	•	Install Ollama: Follow the instructions on the Ollama GitHub page.
	•	Pull the Llama 2 Model:

```bash
ollama pull llama2
```

### Start the Ollama Server:

```bash
ollama serve
```


### Usage

Run the Streamlit application:

```bash
streamlit run app.py
```

Access the app in your web browser at http://localhost:8501.

#### Project Structure
```bash
ai-stock-advisor/
├── app.py
├── .gitignore
├── README.md
├── requirements.txt
└── venv/
```

	•	app.py: Main application script
	•	.gitignore: Specifies files for Git to ignore
	•	README.md: Project documentation
	•	requirements.txt: Python dependencies
	•	venv/: Virtual environment directory (ignored by Git)

Contributing

Contributions are welcome! Please open an issue or submit a pull request.

License

This project is licensed under the MIT License. See the LICENSE file for details.

Disclaimer

This application is for educational purposes and does not constitute financial advice.

Acknowledgments

	•	Ollama for the Llama 2 model
	•	Streamlit for the web framework
	•	Yahoo Finance for stock data via yfinance

---