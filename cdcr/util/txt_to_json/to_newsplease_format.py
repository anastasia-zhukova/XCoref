import json, os, re


def read_sample():
    NEWSPLEASE_SAMPLE = os.path.join(os.path.dirname(__file__), "newsplease_sample.json")
    with open(NEWSPLEASE_SAMPLE, 'r') as infile:
        sample = json.load(infile)
    for key, val in sample.items():
        sample[key] = None
    return sample


def convert_file(file_url):
    with open(file_url, 'r') as input_file:
        file = input_file.readlines()
        input_file.close()
    new_file_delimiter = r'^(\d+\W?\s*){0}[a-zA-Z]+'
    publisher_pattern = new_file_delimiter.format("{1}")
    article_pattern = new_file_delimiter.format("{2}")
    dir = re.search(r'[a-z]*_\d*', os.path.basename(file_url)).group(0)
    new_full_dir = os.path.join(os.path.dirname(file_url), dir)
    if not os.path.exists(new_full_dir):
        os.makedirs(new_full_dir)
    sample = read_sample()
    text = ""
    file_id = 0
    for line in file:
        # "1." -> new publisher
        if re.match(new_file_delimiter.format("{1,}"), line):
            if text != "":
                article_data["text"] = text
                article_data["title"] = title
                article_data["description"] = publisher
                with open(os.path.join(new_full_dir, file_name), 'w') as outfile:
                    json.dump(article_data, outfile)
            if re.match(publisher_pattern,line):
                match = re.split(r'\W', line)
                file_id = 0
                publisher = "_".join(word for i, word in enumerate(match) if i > 0 and word != "")
                continue
            # "1.1." -> new file with an article
            if re.match(article_pattern, line):
                article_data = sample.copy()
                file_id += 1
                file_name = dir + "_" + publisher + "_" + str(file_id) + ".json"
                title = line[:-1].split(". ")[1] + ". "
                text = ""
                continue
        # fields: publisher (exact name?), text, title
        text += line
    article_data["text"] = text
    article_data["title"] = title
    article_data["description"] = publisher
    with open(os.path.join(new_full_dir, file_name), 'w') as outfile:
        json.dump(article_data, outfile)


if __name__ == "__main__":
    convert_file("C:\\Users\\annaz\\PycharmProjects\\WCL\\data\\original\\daca_25_selected_articles.txt")