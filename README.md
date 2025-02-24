# 📄 README - Gemini-Powered Chatbot with Wikipedia & Arxiv Tools

## 🧩 Overview
This project is a Streamlit-based chatbot that leverages Google's Gemini large language model (LLM) along with Wikipedia and Arxiv tools to provide detailed and accurate responses. It supports continuous chat, displaying both user and assistant messages in a conversational interface.

## 🚀 Features
- Conversational chatbot powered by **Gemini-1.5-pro**.
- Fetches information from **Wikipedia** and **Arxiv**.
- Continuous chat experience with message history.
- User-friendly interface with **Streamlit**.

---

## 📦 Prerequisites
Before you begin, ensure you have the following installed:

- Python **3.8+**  
- **pip** (Python package installer)  
- **virtualenv** (optional but recommended)

---

## 🛠️ Installation
1. **Clone the repository:**
   ```bash
   git clone <repository-url>
   cd <repository-directory>
   ```

2. **Create and activate a virtual environment (optional but recommended):**
   ```bash
   # Create virtual environment
   python -m venv .venv

   # Activate on Windows
   .venv\Scripts\activate

   # Activate on macOS/Linux
   source .venv/bin/activate
   ```

3. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

4. **Set up environment variables:**
   Create a `.env` file in the project root and add your Google API key:
   ```env
   GOOGLE_API_KEY=your_google_api_key_here
   ```
   > 📝 **Note:** You need access to the Gemini API through Google Cloud. Visit the [Google Cloud Console](https://console.cloud.google.com/) to obtain your API key.

---

## 🚦 Running the Application
Run the Streamlit app using the following command:

```bash
streamlit run chatbot.py
```

This will open a new browser window/tab with the chatbot interface. 

✅ **Enter your query in the chat input field** and press Enter to start the conversation.

---

## 💬 Usage Example
1. Type: `What is quantum computing?`  
2. The chatbot will respond with information from Wikipedia and, if available, relevant papers from Arxiv.
3. Continue chatting with follow-up questions.

---

## 🐛 Troubleshooting
### ⚠️ Common Issues
- **GOOGLE_API_KEY not found:**  
  Ensure your `.env` file is correctly placed in the root directory and contains a valid key.

- **429 Resource has been exhausted:**  
  This means you’ve exceeded your API quota. You can:
  - Check your quota on [Google Cloud Console](https://console.cloud.google.com/).
  - Reduce the number of requests.
  - Request a higher quota.

- **ModuleNotFoundError:**  
  Double-check that you've installed all dependencies via `pip install -r requirements.txt`.

---

## 🧪 Project Structure
```
├── chatbot.py          # Main Streamlit app file
├── requirements.txt    # Project dependencies
└── .env                # Environment variables (not tracked by git)
```

---

## 📚 Dependencies
- **Streamlit**: Interactive web app framework.
- **LangChain Community**: For Wikipedia and Arxiv tools integration.
- **LangGraph**: Manages the chatbot’s conversational state and flow.
- **langchain_google_genai**: Integrates Google's Gemini LLM.
- **python-dotenv**: Loads environment variables from `.env`.

---

## 📝 License
This project is open-source and available under the MIT License.

---

## 💡 Acknowledgments
- Powered by **Google Gemini API**.
- Data sourced from **Wikipedia** and **Arxiv**.
- Built with ❤️ using **Streamlit** and **LangChain**.

---

## 🙋‍♂️ Contact
For any questions or issues, please raise an issue in the repository.

---
✅ **Enjoy chatting! 🤖💬**
