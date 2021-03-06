import spacy
import nltk
import numpy as np
import wikipedia
from sklearn.feature_extraction.text import TfidfVectorizer

from supervisions.concept_net import search_concept_net


class Supervisor:
    def __init__(self):
        self._wiki_cache = {}
        self._concept_net_cache = {}

        self._ner_model = spacy.load('xx_ent_wiki_sm')

        self._tfidf_vectorizer = None
        self._document_vocab = None
        self._analyzer = None

    def get_entities(self, text):
        doc = self._ner_model(text)

        return list(set([ent.text.lower() for ent in doc.ents]))

    def guess_aspects(self, text):
        in_text_entities = self.get_entities(text)

        related_entities, relation_weights = [], []
        for in_text_entity in in_text_entities:
            neighbors = self.get_concept_net_neighbors(in_text_entity)

            ents, weights = [], []
            for neighbor in neighbors:
                if neighbor['entity'] not in \
                        in_text_entities + related_entities and \
                        np.isfinite(neighbor['relation_weight']):
                    ents.append(neighbor['entity'])
                    weights.append(neighbor['relation_weight'])

            if len(ents) > 0:
                related_entities.extend(np.random.choice(
                    ents, size=1, p=np.array(weights) / np.sum(weights)))

        return in_text_entities, related_entities

    def guess_summary(self, aspect, global_summary):
        neighbors = [{
            'entity': aspect,
            'reasoning': f'aspect [[{aspect}]] is in the text',
            'relation_weight': float('inf')}]
        neighbors.extend(self.get_concept_net_neighbors(aspect))

        picked_sents, reasonings = [], []
        for sent in nltk.sent_tokenize(global_summary):
            for neighbor in neighbors:
                if neighbor['entity'].lower() in sent.lower():
                    picked_sents.append(sent)
                    if neighbor['reasoning'] not in reasonings:
                        reasonings.append(neighbor['reasoning'])
                    break

        if len(picked_sents) > 0:
            return {
                'aspect_summary': ' '.join(picked_sents),
                'reasonings': reasonings
            }
        else:
            return None

    def get_concept_net_neighbors(self, keyword):
        if keyword not in self._concept_net_cache:
            self._concept_net_cache[keyword] = search_concept_net(keyword)

        return self._concept_net_cache[keyword]

    def get_wiki_words(self, aspect, document, n_limit):
        document_words = self.get_doc_words(document=document)

        wiki_page = self.get_wiki_page(aspect)
        if wiki_page is None:
            return []

        selected_words = []
        wiki_words = self._analyzer(wiki_page.content)
        for word in document_words:
            if word in wiki_words:
                selected_words.append(word)

            if len(selected_words) == n_limit:
                break

        return selected_words

    def get_wiki_page(self, title):
        if not title in self._wiki_cache:
            try:
                self._wiki_cache[title] = wikipedia.page(title)
            except:
                self._wiki_cache[title] = None
        return self._wiki_cache[title]

    def build_tfidf_vectorizer(self, documents):
        print('fitting tfidf ...', end=' ')
        self._tfidf_vectorizer = TfidfVectorizer(stop_words='english')
        self._tfidf_vectorizer.fit(documents)
        self._document_vocab = self._tfidf_vectorizer.get_feature_names()
        self._analyzer = self._tfidf_vectorizer.build_analyzer()
        print('done')

    def get_doc_words(self, document):
        vec = self._tfidf_vectorizer.transform([document])
        word_id_score_list = [[self._document_vocab[word_id], word_id, score]
                              for word_id, score in zip(vec.indices, vec.data)]

        word_id_score_list.sort(key=lambda t: t[-1], reverse=True)

        return [word for word, _, _ in word_id_score_list]
