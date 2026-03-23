import re
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor, TimeoutError

import requests
from flask import Flask, jsonify, request
from langchain_experimental.utilities import PythonREPL

app = Flask(__name__)
PYTHON_EXECUTOR_HOST = "0.0.0.0"
PYTHON_EXECUTOR_PORT = 7860


def is_standard_lib(package: str) -> bool:
    return package in sys.stdlib_module_names


def install(package: str) -> None:
    subprocess.check_call([sys.executable, "-m", "pip", "install", package])


@app.get("/")
def root():
    return {"service": "kgot-python-executor", "port": PYTHON_EXECUTOR_PORT}


@app.get("/health")
def health():
    return {"status": "ok"}


@app.route('/run', methods=['POST'])
def run_code() -> tuple[requests.Response, int]:
    timeout_seconds = 240

    required_modules = request.json.get('required_modules', [])
    if required_modules:
        for module in required_modules:
            if not is_standard_lib(module):
                install(module)

    code = request.json.get('code')
    if not code:
        return jsonify({"error": "No code provided"}), 400

    python_repl = PythonREPL()

    def execute_code() -> tuple[requests.Response, int]:
        def is_error_string(s: str) -> bool:
            error_pattern = re.compile(r'^[a-zA-Z_]+Error\((.*)\)$')
            return bool(error_pattern.match(s))

        result = python_repl.run(code)
        if is_error_string(result):
            return {"error": result}, 400
        return {"output": result}, 200

    executor = ThreadPoolExecutor(max_workers=1)
    future = executor.submit(execute_code)

    try:
        result, status_code = future.result(timeout=timeout_seconds)
    except TimeoutError:
        return jsonify({"error": "Code execution timed out"}), 408
    except Exception as e:
        return jsonify({"error": str(e)}), 400

    return jsonify(result), status_code


if __name__ == '__main__':
    from waitress import serve
    serve(app, host=PYTHON_EXECUTOR_HOST, port=PYTHON_EXECUTOR_PORT)
