from __future__ import annotations

import io

import httpx
from pydantic import BaseModel, Field

from agent.tools.base import BaseTool

MAX_CONTENT_LENGTH = 4000


class DocumentFetchInput(BaseModel):
    url: str = Field(description="URL to fetch (HTML page or PDF)")
    extract_text: bool = Field(default=True, description="Extract text content")


class DocumentFetchTool(BaseTool):
    name = "document_fetch"
    description = (
        "Fetch and parse content from a URL. Supports HTML pages and PDF documents. "
        "Returns extracted text content."
    )

    def get_schema(self) -> type[BaseModel]:
        return DocumentFetchInput

    async def _execute(self, url: str, extract_text: bool = True) -> str:
        async with httpx.AsyncClient(
            timeout=30.0, follow_redirects=True
        ) as client:
            response = await client.get(url)
            response.raise_for_status()

        content_type = response.headers.get("content-type", "")

        if "pdf" in content_type or url.lower().endswith(".pdf"):
            text = self._parse_pdf(response.content)
        else:
            text = self._parse_html(response.text)

        if not extract_text:
            return f"Fetched {len(response.content)} bytes from {url}"

        if len(text) > MAX_CONTENT_LENGTH:
            text = text[:MAX_CONTENT_LENGTH] + f"\n\n[Truncated — {len(text)} chars total]"

        return text if text.strip() else "No text content extracted."

    def _parse_pdf(self, content: bytes) -> str:
        from pypdf import PdfReader

        reader = PdfReader(io.BytesIO(content))
        pages = []
        for page in reader.pages:
            text = page.extract_text()
            if text:
                pages.append(text)
        return "\n\n".join(pages)

    def _parse_html(self, html: str) -> str:
        from bs4 import BeautifulSoup

        soup = BeautifulSoup(html, "html.parser")
        for tag in soup(["script", "style", "nav", "footer", "header", "aside"]):
            tag.decompose()
        return soup.get_text(separator="\n", strip=True)
