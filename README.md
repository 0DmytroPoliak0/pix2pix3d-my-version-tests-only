# pix2pix3d-my-version-tests-only
QA tests for https://github.com/0DmytroPoliak0/pix2pix3D-QA-version.git

@inproceedings{kangle2023pix2pix3d,
  title={3D-aware Conditional Image Synthesis},
  author={Deng, Kangle and Yang, Gengshan and Ramanan, Deva and Zhu, Jun-Yan},
  booktitle = {CVPR},
  year = {2023}
}


This repository contains the complete testing suite for our adapted version of the pix2pix3D framework. It includes all tests for validating the performance, accuracy, and reliability of our image generation models, covering both functional and non-functional aspects. The tests were designed to work with our MacOS-adapted version, which required several monkey patches and custom fixes to overcome the original CUDA-only constraints.

Overview
	•	White-Box Testing:
	•	Unit Tests: Validate core functions such as filename generation, image dimension checks, configuration validation, and seed reproducibility.
	•	Integration Tests: Ensure that different modules (like generate_samples.py) work together as expected.
	•	Use-Case Tests: Simulate real-world scenarios to check for proper output formatting and consistency.
	•	Non-Functional Testing:
	•	Performance Testing: Measure execution time, memory usage, and consistency across multiple runs.
	•	Security Testing: Test the system’s response to invalid or malicious inputs.
	•	Accuracy & Musa Testing:
	•	Evaluate image quality using metrics like IoU and SSIM, and combine them into a composite Musa Score that reflects overall output quality.
	•	Additional Testing Layers:
	•	Property-Based Testing: Using Hypothesis to generate diverse input scenarios and uncover subtle issues.
	•	Boundaries Testing: Ensure that computed metrics (IoU, SSIM, composite quality, performance) remain within their expected ranges.
	•	Stress Testing: Simulate heavy load conditions (using the STRESS_MODE flag) to assess system stability under high demand.
	•	Test Lifecycle:
	•	Automated hooks (via conftest.py) log test start/finish times and generate session reports to provide continuous feedback and support future CI integration.

⸻

Installation
	1.	Clone the Repository:git clone https://github.com/0DmytroPoliak0/pix2pix3d-my-version-tests-only.git
cd pix2pix3d-my-version-tests-only
	2.	Set Up the Virtual Environment:python -m venv venv
source venv/bin/activate
(install all requirments)
	3.	(Optional) Enable Stress Testing:export STRESS_MODE=true

 
Running Tests
	•	Run All Tests: pytest -s
	•	Run Specific Test Suites: example: pytest -s tests/security
	•	Boundaries Testing: pytest -s tests/boundaries
 	•	Coverage Report: coverage report
 


