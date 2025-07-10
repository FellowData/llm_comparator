# Install dependencies
install:
	pip install -e .

# Run the app
run:
	PYTHONPATH=src python3 examples/url_availability_checker_implementation_example.py

# Clean logs and outputs
clean:
	rm -rf data/output/*
	rm -f logs/*.log