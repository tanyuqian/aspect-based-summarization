import csv
import fire
import requests
from tqdm import tqdm
import json


extracted_concept_net = json.load(open('extracted_concept_net.json'))

BAN_RELATIONS = [
    'Antonym', 'DistinctFrom', 'NotCapableOf', 'NotDesires', 'NotHasProperty']


def search_concept_net(keyword):
    keyword = keyword.lower().replace(' ', '_')

    neighbors = []
    for relation in extracted_concept_net:
        if relation in BAN_RELATIONS or \
                keyword not in extracted_concept_net[relation]:
            continue

        for (h0, h1, weight) in extracted_concept_net[relation][keyword]:
            if h0 == h1 == keyword:
                continue

            neighbors.append({
                'entity': h0 if h1 == keyword else h1,
                'reasoning': f'ConceptNet: [[{h0}]] {relation} [[{h1}]]',
                'relation_weight': weight
            })

    for i in range(len(neighbors)):
        neighbors[i]['entity'] = neighbors[i]['entity'].replace('_', ' ')

    return neighbors


if __name__ == '__main__':
    results = search_concept_net('harry potter')

    for x in results:
        print(x)

# if __name__ == '__main__':
#     extracted_concept_net = {}
#
#     with open('/media/tan/T7 Touch/conceptnet-assertions-5.7.0.csv') as csv_file:
#         csv_reader = csv.reader(csv_file, delimiter='\t')
#
#         edges = []
#         rels = []
#         for i, (a, r, h0, h1, info) in enumerate(tqdm(csv_reader)):
#             if not h0.startswith('/c/en/') or not h1.startswith('/c/en/'):
#                 continue
#
#             assert r.startswith('/r/')
#
#             info = eval(info)
#
#             r = r.split('/')[2]
#             h0 = h0.split('/')[3]
#             h1 = h1.split('/')[3]
#             weight = info['weight']
#
#             if r not in extracted_concept_net:
#                 extracted_concept_net[r] = {}
#             if h0 not in extracted_concept_net[r]:
#                 extracted_concept_net[r][h0] = []
#             if h1 not in extracted_concept_net[r]:
#                 extracted_concept_net[r][h1] = []
#
#             if [h0, h1, weight] not in extracted_concept_net[r][h0]:
#                 extracted_concept_net[r][h0].append([h0, h1, weight])
#             if [h0, h1, weight] not in extracted_concept_net[r][h1]:
#                 extracted_concept_net[r][h1].append([h0, h1, weight])
#
#         json.dump(extracted_concept_net, open(
#             'extracted_concept_net.json', 'w'), indent=4)