"""One-click backend launcher for IDEs (PyCharm, VS Code, etc.).

Open this file and click the ▶ run button — no terminal needed. Equivalent
to running `uvicorn app.main:app --reload --port 8000` from `backend/`.
"""

import uvicorn

if __name__ == "__main__":
    uvicorn.run("app.main:app", host="127.0.0.1", port=8000, reload=True)
