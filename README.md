# 🌐 test_urls_list_availability

A Python project for analyzing the availability of URLs, detecting redirects, and generating interactive HTML reports. Designed with modularity, performance, and reliability in mind, the tool allows large-scale URL verification and reporting.

---

## 📦 Installation

### Requirements

- Python 3.8+
- Install dependencies:

```bash
pip install -r requirements.txt
```

---

## 🧠 Project Overview

This project analyzes the status of a set of URLs, identifies redirects and unreachable domains, and produces structured outputs (CSV, JSON, JSONL). It also provides domain-level statistics and is designed to handle large datasets efficiently with support for incremental saving and multithreading.
The project include as an experimentation an interactive HTML report

---

## 🛠️ Usage

### Run the URL availability check directly
```bash
python src/url_availability_checker/runner.py --website_txt_file_input inputs/your_urls.txt --path_to_output data/check_urls_availability/
```
website_txt_file_input : compulsory, input text file with list of URLs
path_to_output =compulsory, path to output data'

Optional arguments:
- `--jsonl_output`: Save results as JSONL (default: False)
- `--csv_output`: Save results as CSV (default: False)
- `--batch_size`: Number of URLs per batch (default: 100)
- `--workers`: Number of threads to use (default: 5)
- `--save_incrementally`: Save results after each URL fetch (default: False)



### Through custom implementation in examples/ 
Needs to set  src/ folder through PYTHONPATH
```bash
PYTHONPATH=src python3 examples/url_availability_checker_implementation_example.py
```

### Through custom MakeFile (example provided)
```bash
make run
```

---

## 🧩 Project Structure

```
test_urls_list_availability/
├── config/                                                    # Configuration files
│   ├── TLD_list.json                                          # JSON list of TLD (Top Level Directory)
├── data/
│   ├── output/                                                # Outputs of the availability checker in CSV, JSON, JSONL
│   ├── input/                                                 # Inputs :  URLs that need to through the availability checker
├── docs/                                                      # Documentation
├── examples/                                                  # Implementation examples
├── logs/                                                      # Logs
├── src/
│   ├── url_availability_checker/                              # Core logic for URL availability checker
│   │   ├── batch.py                                           # batch logic 
│   │   ├── config.py                                          # output config + constants 
│   │   ├── fetch.py                                           # fetch url 
│   │   ├── formatting.py                                      # format_data and data modeling 
│   │   ├── io_utils.py                                        # I/O functions 
│   │   ├── runner.py                                          # runner program to check URLs availability 
│   │   ├── statistics.py                                      # stats summary and logging 
│   │   ├── title_extraction.py                                # Code source to extract title from webpage 
│   │   ├── tld_utils.py                                       # Code source to extract URLs part and find TLD 
│   ├── generate_html_report/                                  # [POC] HTML report generator using Jinja library and a html template based on Bootstrap
│   ├── URL_availability_checker_implementation_example.py     # Implementation example to check URLs availability
├── CHANGELOG.md                                               # Project changelog
├── makefile                                                   # MakeFile
├── requirements.txt                                           # Python dependencies
├── README.md                                                  # Project documentation
├── ROADMAP.md                                                 # Project roadmap

```

---

## ✅ Highlights
### Architecture and Design
- Modular design for maintainability and extensions

### Network Resilience
- Robust against timeouts, broken links, malformed URLs
- HTTP status code analysis (200, 404, 500, etc.)
- Redirect Loop Handling: Detects more than 5 redirects to break infinite loops.
-
- Failed Domains Cache: Avoids repeated calls to known unreachable domains by created a list of failed_domains (during program execution only), using `urlparse(url).netloc` to extract domain
- Adaptive retry delays: Increases wait time between retries using adaptative delays (random and attempt retry based).

### Data Handling
- Incremental and batch-based saving : Avoids data loss by saving each result after fetch if the option is selected.
- Outputs available in different formats

### Content Analysis
- Title Extraction Strategy: Regex, or Falls back on `<h1>` or meta tag `<meta name="title">` if `<title>` is missing.

### Debugging & Monitoring
- Logging for error traceability, using a log file limited in size with rotating files (5 log files of 1MB)
- Detailed statistics

### Performance
- Increased performance through `ThreadPoolExecutor`

### Security Best Practices

- The tool avoids executing or interacting with unknown content.
- Network timeouts and redirects are safely handled with retry mechanisms.
- Sensitive operations (e.g. logging domains) are handled via filters to avoid leaks.
- User-Agent spoofing is applied for better coverage in some websites.

---

## 📘 How It Works

### `main.py`
- Parses arguments, sets up logging
- Loads input URLs and TLDs
- Processes URLs in batch with concurrency
- Saves results and generates reports

### `io_utils.py`
- Loads and saves URLs/results in various formats (CSV, JSON, JSONL)
- Sets up logging

### `tld_utils.py`
- Extracts domain, subdomain, TLD from URLs
- Cleans and normalizes URLs

### `title_extraction.py`
- Extracts page titles using `regex` and `BeautifulSoup`

### `get_urls_availaibility.py`
- Coordinates the entire availability checking process
- Splits URLs into batches
- Fetches URL content and handles errors
- Saves outputs and generates statistics

---

## 📍 Conclusion

This project is production-ready for large-scale URL checking with ideas for future growth. Proposed improvements include better configurability, enhanced monitoring, and automation.