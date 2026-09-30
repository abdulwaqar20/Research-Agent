import trafilatura

url = "https://en.wikipedia.org/wiki/2010_flash_crash"
downloaded = trafilatura.fetch_url(url)
text = trafilatura.extract(downloaded)

print(text[:1000])
print("---")
print(f"total length: {len(text)} chars")