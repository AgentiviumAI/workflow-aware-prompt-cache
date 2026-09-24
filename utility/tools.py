import httpx
import asyncio
from bs4 import BeautifulSoup
from ddgs import DDGS
from config import MAX_SEARCH_CONTENT_CHARS

async def fetch_html(url: str, client: httpx.AsyncClient) -> str:
    # Filter out direct PDF links
    if url.lower().endswith('.pdf'):
        return ""
        
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
    }
    try:
        response = await client.get(url, headers=headers, timeout=5.0, follow_redirects=True)
        response.raise_for_status()
        
        # Additional check to ensure it's HTML, not binary
        content_type = response.headers.get('Content-Type', '')
        if 'text/html' not in content_type:
            return ""
            
        return response.text
    except Exception:
        return ""

async def search_web(query: str, max_results: int = 3) -> str:
    # Run synchronous DDGS in a separate thread to prevent event loop blocking
    try:
        search_results = await asyncio.to_thread(lambda: list(DDGS().text(query, max_results=max_results)))
    except Exception as e:
        return f"Search failed: {str(e)}"
    
    urls = [res.get('href') for res in search_results if res.get('href')]
    if not urls:
        return "No results found."

    results = []
    async with httpx.AsyncClient(verify=False) as client:
        # Fetch websites concurrently
        html_contents = await asyncio.gather(*[fetch_html(url, client) for url in urls])
        
        for url, html in zip(urls, html_contents):
            if html:
                soup = BeautifulSoup(html, 'html.parser')
                # Extract text mainly from paragraphs to avoid boilerplate/menu junk
                paragraphs = soup.find_all('p')
                text = " ".join([p.get_text(strip=True) for p in paragraphs if p.get_text(strip=True)])
                
                if text:
                    truncated_text = text[:MAX_SEARCH_CONTENT_CHARS]
                    results.append(f"Source: {url}\nContent: {truncated_text}")
    
    if not results:
        return "No results found or access blocked by websites."
    
    return "\n\n".join(results)