# test_urls_list_availability Changelog
## Example 5.1.3 (2024-11-04) :
* Bugfixes
  * Reduce logging errors ([`921a825`](https://github.com/john-kurkowski/tldextract/commit/921a82523c0e4403d21d50b2c3410d9af43520ac))
  * Drop support for EOL Python 3.8 ([#340](https://github.com/john-kurkowski/tldextract/issues/340))
  * Support Python 3.13 ([#341](https://github.com/john-kurkowski/tldextract/issues/341))
  * Update bundled snapshot
* Documentation
  * Clarify how to use your own definitions
  * Clarify first-successful definitions vs. merged definitions
* Misc.
  * Switch from Black to Ruff ([#333](https://github.com/john-kurkowski/tldextract/issues/333))
  * Switch from pip to uv, during tox ([#324](https://github.com/john-kurkowski/tldextract/issues/324))


## 0.3.0 (2025-04-18)
* Documentation
  * Clarify README
  * Document change logs
  * Structure roadmap
  * Unify code comments in English
  * create the requirements.txt
* Architecture
  * Rename output and input directories, log filename in dedicated logs directory
  * Setup config directory will list of TLD
  * Isolate more functions from the old monolithic key program into separate python files 

## 0.2.0 (2025-01-05)
* Architecture
  * Isolation of similar functions in specific python files (ex: io_utils)
  * Enable output and input files in distinct directories
* Features
  * Handle URLs in batch of variable size
  * Generate statistics available in logs
  * Retrieve domain from URL
  * Format output data
  * Get title from webpage
  * Add arguments support
  * Check for arguments with valid (positive integers) values 
  * Add REGEX support to retrieve subdomain, domain, TLD, using a fixed set of TLD in JSON
  * Enable saving output results in JSON (always), CSV (optional), JSONL (optional)
  * Enable saving after each request or after each batch
* Bugfixes
  * Ensure the invalid argument name is displayed
  * Solve the issue with the logs displaying incorrect number of batch (rounding up was automatic)
  * Ensure the invalid argument name is displayed
  * Ensure the invalid argument name is displayed
* Misc
  * Separate file to handle TLD extraction
  * Separate file to handle title extraction



## 0.1.0 (2024-12)
* Proof of concept
* Experiment with HTML report
* Experiment with unit tests
