# 🤖 LLM Comparator App (OpenRouter + OpenAI)

This Streamlit app allows you to **compare responses from multiple LLMs** (Large Language Models) including Mistral, LLaMA, Gemma, DeepSeek, and OpenAI’s GPT-4o, all from a single interface.

---

## 🚀 Features

- 🧠 Multi-model support: OpenRouter API + OpenAI API
- 💬 Prompt input with source-tracking instruction
- 📊 Compare responses across models
- 🛡️ Graceful error handling (e.g. 503 fallback)
- 📜 Prompt history stored as JSON
- 🔍 Filter history by model and prompt content
- 📤 Export prompt history as Excel file
- 🔐 Secure API key usage via Streamlit Secrets

---

## 🛠 Installation

```bash
git clone https://github.com/your-username/llm-comparator.git
cd llm-comparator
pip install -r requirements.txt
streamlit run app.py
```

### 📦 Requirements

- `pandas` for prompt history handling
- `openpyxl` to support Excel export
- Account with API on [openrouter.ai](https://openrouter.ai) to use free/paid LLMs:
    1. Sign up on https://openrouter.ai
    2. Go to profile > API Key
    3. Copy the key and add it to secrets (see below)
- Identify which LLMs to use for free with Openrouter : https://openrouter.ai/models

- Account with API access to [OpenAI](https://platform.openai.com)
- Check your spending on https://platform.openai.com/settings/organization/usage
- It is also possible to define a project with spending limits : https://help.openai.com/en/articles/9186755-managing-projects-in-the-api-platform#h_d2c8f84ece
---


## 🔐 Configuration

Create a `.streamlit/secrets.toml` file:

```toml
OPENROUTER_API_KEY = "pk-..."
OPENAI_API_KEY = "sk-..."
```

Or, use Streamlit Cloud's **Secrets Manager** if deployed online.

---

## 📜 Prompt History

Prompts are stored locally in `prompt_history.json` with:

- Timestamp
- Prompt content
- Model name and ID
- Raw model response

The app includes:
- A history viewer tab
- Filter by model
- Search prompt text
- Excel download (`prompt_history.xlsx`)

---

## 🚀 Deployment

### Option 1: Local

```bash
streamlit run app.py
```

### Option 2: Streamlit Community Cloud

- GitHub repo required
- Set secrets via the web UI
- Deploy in one click: https://streamlit.io/cloud
- App should be available through https://share.streamlit.io/
---

## 🧱 Tech Stack

- Python 3.11+
- Streamlit
- OpenRouter API
- OpenAI SDK
- pandas + openpyxl (for Excel export)

---

## 📄 License

MIT