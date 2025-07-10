# 📌 test_urls_list_availability – Project Roadmap

## 🚧 URGENT / NEXT
### TODO


### FIXME


---

## 🚀 Short-Term Goals (MVP)



---

## 🌟 Future Enhancements (With all the money and the time in the world)
### 🔄 Code Structure & Logic Separation
- [ ] Split the URL fetching logic from scraping logic
  - [ ] Handle data processing independently from HTTP access
  - [ ] Refactor exception handling to use clearer and more explicit `raise` statements
  - [ ] Improve logging and error reporting
    - [ ] Return distinct status messages for `timeout`, `connection error`, `invalid URL`

### ⚙️ Concurrency & Performance
- [ ] Improve concurrency logic: current `MAX_WORKERS` is static (default = 10)
- [ ] Dynamically adjust workers based on hardware resources:
  ```python
  import multiprocessing
  MAX_WORKERS = min(32, (multiprocessing.cpu_count() * 2) + 1)
  ```
- [ ] Test processor-based parallelization (especially on BNPP machines)
- [ ] Benchmark batch-saving results vs. per-URL saving
- [ ] Evaluate `pandas` for data manipulation

### 🔧 Configuration Management
- [ ] Move all parameters configurable via JSON/YAML
- [ ] Centralize settings like timeouts, retries, and worker count
- [ ] Add logging verbosity levels (`debug`, `info`, etc.)
- [ ] Classify errors in the final report (timeout, DNS error, etc.)

###  Advanced User Interface
- [ ] Add a GUI or Web UI

### 📈 Reporting & Monitoring
- [ ] Generate PDF reports with visuals
- [ ] Create a live dashboard for real-time URL tracking
- [ ] Evaluate `Airflow` as a task scheduler
- [ ] Allow scheduled checks (daily/weekly)
- [ ] Notify via email or system alerts for critical issues

### 🔄 Integration & Automation
- [ ] Support additional protocols (FTP, SFTP)
- [ ] Support JSON/XML API response parsing
- [ ] Integrate with CI/CD pipelines (e.g., GitLab CI)
- [ ] Enable historical comparison of URL status


### 🧪 Testing & Quality
- [ ] Refactor and ensure unit tests run correctly (fix module pathing issues)
- [ ] Revisit and improve test coverage
- [ ] Improve coverage for:
  - [ ] Redirects
  - [ ] Timeouts
  - [ ] Invalid URLs
  - [ ] Redirect loops












