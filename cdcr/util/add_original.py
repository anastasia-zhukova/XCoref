import json
import sys
import os

OUTPUT_ROOT = os.path.realpath(os.path.join(os.path.dirname(__file__),  "..", "..", "data", "original"))

if __name__ == "__main__":
    topic_name = sys.argv[1]
    json_file = sys.argv[2]
    print(f"Parsing articles from {json_file} and saving them in topic-folder {topic_name}.")
    articles = json.load(open(json_file))
    path = os.path.join(OUTPUT_ROOT, topic_name)
    print(f"Root path will be {path}.")
    i = 1
    os.makedirs(path, exist_ok=True)
    for article in articles:
        domain = article["source_domain"].split(".")[0]
        filename = f"{domain}_{i}.json"
        json.dump(article, open(os.path.join(path, filename), "w"), indent=2)
        i += 1
    print(f"Created {i-1} articles in {path}.")
