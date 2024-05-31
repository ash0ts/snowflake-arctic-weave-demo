#!/bin/bash

echo "Sorting imports with isort..."
isort .

echo "Formatting code with black..."
black .

echo "Formatting code with autopep8..."
autopep8 --in-place --recursive .

echo "Linting with flake8..."
flake8 .

echo "Linting with ruff..."
ruff . --fix

echo "All operations completed."