import ujson as json
import xml.etree.ElementTree as etree

def flatten_list(my_list):
    flat_list = []
    for sublist in my_list:
        for item in sublist:
            flat_list.append(item)
    return flat_list
   
def parse_data_xml(tree, gold_rankings_file):
    root = tree.getroot()
    # lexelts = root.getchildren()
    problems = dict() # [instance]->{context:, target:, rank_cands:[]}
    
    print("Loading gold rankings...")
    gold_rankings = dict() # [ids] -> ranks
    for line in gold_rankings_file:  
        # Sentence 301 rankings: {team} {side}
        piece = line.strip().split(" ")
        rank_id = int(piece[1])
        gold_rankings[rank_id] = []
        for rank_cand in piece[3:]:
            cand = rank_cand.replace("{","").replace("}","")
            if "," in cand: # {a, b}
                cands = cand.split(",")
                gold_rankings[rank_id].append(cands)
            else:
                gold_rankings[rank_id].append([cand])


    print("Loading instance tree...")
    for lexelt in root: #lexelts:
        if lexelt.tag != "lexelt":
            continue

        word = lexelt.attrib["item"].split(".")[0] # item="show.v"
        # print(word)

        for instance in lexelt: #.getchildren():
            if instance.tag == "instance":
                inst_id = int(instance.attrib["id"])
                text1, target, text2 = "","",""
                # print(inst_id)
                for element in instance.getchildren():
                    # print(element.text)
                    if element.tag == "context1":
                        # print(element.text)
                        if element.text is not None:
                            text1 = element.text.strip() 
                    if element.tag == "head":
                        # print(element.text)
                        target = element.text #.strip()
                    if element.tag == "context2":
                        if element.text is not None:
                            text2 = element.text.strip()                
                    
                clean_context = text1 + " " + target + " " + text2

                problems[inst_id] = {}
                problems[inst_id]["target"] = word
                problems[inst_id]["context"] = clean_context
                problems[inst_id]["rank_cands"] = gold_rankings[inst_id]
                problems[inst_id]["candidates"] = flatten_list(gold_rankings[inst_id])

    return problems
        
tree = etree.parse("./datasets/semeval/my_contexts.xml")
with open("./datasets/semeval/substitutions.gold-rankings", "r") as f:
    gold_rankings = f.readlines()
data = parse_data_xml(tree, gold_rankings)
print(data[301])


def cache_cleaned_data(datalist, name):
    # cleaned_ppdb.json
    # cleaned semeval_rankings.json
    with open(name, "w") as f:
        json.dump(datalist, f)


print("Caching the Examples")
cache_cleaned_data(data, "./dataset/semeval/semeval_rankings.json")
print("Finished")

