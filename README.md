# viralytics_desafio3

Sistema de Interação Multimodal para Promoções em Loja.

## API

### Setup

```bash
cd api
pip install -r requirements.txt
```

#### With Homebrew Python

If you get an `externally-managed-environment` error, use a virtual environment:

```bash
cd api
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

> Activate the venv (`source .venv/bin/activate`) each time you open a new terminal before running the API.

### Run

```bash
python -m uvicorn main:app --host 0.0.0.0 --port 8000 --reload
python3 -m uvicorn main:app --host 0.0.0.0 --port 8000 --reload
```

- API docs: http://localhost:8000/docs
- Test connection: http://localhost:8000/test_robot_connection

> Use `0.0.0.0` to make the API accessible from other devices on the same network (e.g. the robot).
