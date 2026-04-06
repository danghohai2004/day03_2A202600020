import os
import requests
from datetime import datetime
from ddgs import DDGS

ddgs = DDGS()
def web_search(query: str) -> str:
    """Search the web using Brave Search API. Returns top 3 snippets."""
    res = ddgs.text(query, max_results=5)
    return "\n".join(t['body'] for t in res)

    api_key = os.getenv("BRAVE_API_KEY")
    if not api_key:
        return "Error: BRAVE_API_KEY not set in .env"

    url = "https://api.search.brave.com/res/v1/web/search"
    headers = {
        "Accept": "application/json",
        "Accept-Encoding": "gzip",
        "X-Subscription-Token": api_key,
    }
    params = {"q": query, "count": 3}

    try:
        resp = requests.get(url, headers=headers, params=params, timeout=10)
        resp.raise_for_status()
        data = resp.json()

        results = []
        for item in data.get("web", {}).get("results", [])[:3]:
            title = item.get("title", "")
            snippet = item.get("description", "")
            results.append(f"- {title}: {snippet}")

        if not results:
            return "No results found."
        return "\n".join(results)

    except requests.RequestException as e:
        return f"Search error: {str(e)}"


def calculator(expression: str) -> str:
    """Evaluate a math expression safely. E.g., calculator[150000 * 2 + 50000]"""
    try:
        # Only allow safe math operations
        allowed = set("0123456789+-*/.() ")
        if not all(c in allowed for c in expression):
            return f"Error: Invalid characters in expression: {expression}"
        result = eval(expression)
        return str(result)
    except Exception as e:
        return f"Calculation error: {str(e)}"


def get_system_time(ignore: object) -> str:
    """Returns the current date and day of week."""
    now = datetime.now()
    return now.strftime("%A, %B %d, %Y")
