import json

with open("C:\\Users\\annaz\\PycharmProjects\\newsalyze-backend_2\\newsalyze\\entities\\eecdcr\\data\\interim\\cybulska_setup\\ECB_Test_Entity_gold_mentions.json", "r") as file:
    entities = json.load(file)

with open(
        "C:\\Users\\annaz\\PycharmProjects\\newsalyze-backend_2\\newsalyze\\entities\\eecdcr\\data\\interim\\cybulska_setup\\ECB_Test_Event_gold_mentions.json",
        "r") as file:
    events = json.load(file)
doc_set = set()

for mention in entities + events:
    doc_set.add(mention["doc_id"])

with open("C:\\Users\\annaz\\PycharmProjects\\newsalyze-backend_2\\data\\ECBplus-prep\\test_events.json", "w") as file:
    json.dump(sorted(list(doc_set)), file)
print("Done!")
