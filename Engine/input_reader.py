import json
from abc import abstractmethod, ABC
from collections import OrderedDict
from logging import Logger
from typing import Iterable, List

from tqdm import tqdm
from transformers import BertTokenizer

from Engine import util
from Engine.terms import Dataset, TermType, RelationType, Term, Relation, Document


class BaseInputReader(ABC):
    def __init__(self, types_path: str, tokenizer: BertTokenizer, neg_term_count: int = None,
                 neg_rel_count: int = None, max_span_size: int = None, logger: Logger = None):
        types = json.load(open(types_path), object_pairs_hook=OrderedDict)  # term + relation types

        self._term_types = OrderedDict()
        self._idx2term_type = OrderedDict()
        self._relation_types = OrderedDict()
        self._idx2relation_type = OrderedDict()

        # terms
        # add 'None' term type
        none_term_type = TermType('None', 0, 'None', 'No Term')
        self._term_types['None'] = none_term_type
        self._idx2term_type[0] = none_term_type

        # specified term types
        for i, (key, v) in enumerate(types['terms'].items()):
            term_type = TermType(key, i + 1, v['short'], v['verbose'])
            self._term_types[key] = term_type
            self._idx2term_type[i + 1] = term_type

        # relations
        # add 'None' relation type
        none_relation_type = RelationType('None', 0, 'None', 'No Relation')
        self._relation_types['None'] = none_relation_type
        self._idx2relation_type[0] = none_relation_type

        # specified relation types
        for i, (key, v) in enumerate(types['relations'].items()):
            relation_type = RelationType(key, i + 1, v['short'], v['verbose'], v['symmetric'])
            self._relation_types[key] = relation_type
            self._idx2relation_type[i + 1] = relation_type

        self._neg_term_count = neg_term_count
        self._neg_rel_count = neg_rel_count
        self._max_span_size = max_span_size

        self._datasets = dict()

        self._tokenizer = tokenizer
        self._logger = logger

        self._vocabulary_size = tokenizer.vocab_size
        self._context_size = -1

    @abstractmethod
    def read(self, datasets):
        pass

    def get_dataset(self, label) -> Dataset:
        return self._datasets[label]

    def get_term_type(self, idx) -> TermType:
        term = self._idx2term_type[idx]
        return term

    def get_relation_type(self, idx) -> RelationType:
        relation = self._idx2relation_type[idx]
        return relation

    def _calc_context_size(self, datasets: Iterable[Dataset]):
        sizes = []

        for dataset in datasets:
            for doc in dataset.documents:
                sizes.append(len(doc.encoding))

        context_size = max(sizes)
        return context_size

    def _log(self, text):
        if self._logger is not None:
            self._logger.info(text)

    @property
    def datasets(self):
        return self._datasets

    @property
    def term_types(self):
        return self._term_types

    @property
    def relation_types(self):
        return self._relation_types

    @property
    def relation_type_count(self):
        return len(self._relation_types)

    @property
    def term_type_count(self):
        return len(self._term_types)

    @property
    def vocabulary_size(self):
        return self._vocabulary_size

    @property
    def context_size(self):
        return self._context_size

    def __str__(self):
        string = ""
        for dataset in self._datasets.values():
            string += "Dataset: %s\n" % dataset
            string += str(dataset)

        return string

    def __repr__(self):
        return self.__str__()


class JsonInputReader(BaseInputReader):
    def __init__(self, types_path: str, tokenizer: BertTokenizer, neg_term_count: int = None,
                 neg_rel_count: int = None, max_span_size: int = None, logger: Logger = None):
        super().__init__(types_path, tokenizer, neg_term_count, neg_rel_count, max_span_size, logger)

    def read(self, dataset_paths):
        for dataset_label, dataset_path in dataset_paths.items():
            dataset = Dataset(dataset_label, self._relation_types, self._term_types, self._neg_term_count,
                              self._neg_rel_count, self._max_span_size)
            self._parse_dataset(dataset_path, dataset)
            self._datasets[dataset_label] = dataset

        self._context_size = self._calc_context_size(self._datasets.values())

    def _parse_dataset(self, dataset_path, dataset):
        documents = json.load(open(dataset_path))
        for document in tqdm(documents, desc="Parse dataset '%s'" % dataset.label):
            self._parse_document(document, dataset)

    def _parse_document(self, doc, dataset) -> Document:
        jtokens = doc['tokens']
        jrelations = doc['relations']
        jterms = doc['entities']
        jpols = doc['polarities']
        jdep_label = doc['dep_label']
        jdep_label_indices = doc['dep_label_indices']
        jdep = doc['dep']
        jpos = doc['pos']
        jpos_indices = doc['pos_indices']

        # parse tokens
        doc_tokens, doc_encoding = self._parse_tokens(jtokens, dataset)

        # parse term mentions
        terms = self._parse_terms(jterms, doc_tokens, dataset)

        # parse relations
        relations = self._parse_relations(jrelations, terms, dataset)

        # create document
        document = dataset.create_document(doc_tokens, terms, relations, jpols, doc_encoding, jdep_label,
                                           jdep_label_indices, jdep, jpos, jpos_indices)

        return document

    def _parse_tokens(self, jtokens, dataset):
        doc_tokens = []

        # full document encoding including special tokens ([CLS] and [SEP]) and byte-pair encodings of original tokens
        doc_encoding = [self._tokenizer.convert_tokens_to_ids('[CLS]')]

        # parse tokens
        for i, token_phrase in enumerate(jtokens):
            token_encoding = self._tokenizer.encode(token_phrase, add_special_tokens=False)
            span_start, span_end = (len(doc_encoding), len(doc_encoding) + len(token_encoding))

            token = dataset.create_token(i, span_start, span_end, token_phrase)

            doc_tokens.append(token)
            doc_encoding += token_encoding

        doc_encoding += [self._tokenizer.convert_tokens_to_ids('[SEP]')]

        return doc_tokens, doc_encoding

    def _parse_terms(self, jterms, doc_tokens, dataset) -> List[Term]:
        terms = []

        for term_idx, jterm in enumerate(jterms):
            term_type = self._term_types[jterm['type']]
            start, end = jterm['start'], jterm['end']

            # create term mention
            tokens = doc_tokens[start:end+1]
            phrase = " ".join([t.phrase for t in tokens])
            term = dataset.create_term(term_type, tokens, phrase)
            terms.append(term)

        return terms

    def _parse_relations(self, jrelations, terms, dataset) -> List[Relation]:
        relations = []

        for jrelation in jrelations:
            relation_type = self._relation_types[jrelation['type']]

            head_idx = jrelation['head']
            tail_idx = jrelation['tail']

            # create relation
            head = terms[head_idx]
            tail = terms[tail_idx]

            reverse = int(tail.tokens[0].index) < int(head.tokens[0].index)

            # for symmetric relations: head occurs before tail in sentence
            if relation_type.symmetric and reverse:
                head, tail = util.swap(head, tail)

            relation = dataset.create_relation(relation_type, head_term=head, tail_term=tail, reverse=reverse)
            relations.append(relation)

        return relations
