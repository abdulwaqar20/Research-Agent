from ddgs import DDGS

with DDGS() as ddgs:
    results = list(ddgs.text("what caused the 2010 flash crash", max_results=5))

for r in results:
    print(r["title"])
    print(r["href"])
    print(r["body"][:150])
    print("---")