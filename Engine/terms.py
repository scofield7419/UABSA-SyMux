from collections import OrderedDict
from typing import List
from torch.utils.data import Dataset as TorchDataset

from Engine import sampling


class RelationType:
    def __init__(self, identifier, index, short_name, verbose_name, symmetric=False):
        self._identifier = identifier
        self._index = index
        self._short_name = short_name
        self._verbose_name = verbose_name
        self._symmetric = symmetric

    @property
    def identifier(self):
        return self._identifier

    @property
    def index(self):
        return self._index

    @property
    def short_name(self):
        return self._short_name

    @property
    def verbose_name(self):
        return self._verbose_name

    @property
    def symmetric(self):
        return self._symmetric

    def __int__(self):
        return self._index

    def __eq__(self, other):
        if isinstance(other, RelationType):
            return self._identifier == other._identifier
        return False

    def __hash__(self):
        return hash(self._identifier)

class Polarity:
    def __init__(self, identifier, index, short_name, verbose_name, symmetric=False):
        self._identifier = identifier
        self._index = index
        self._short_name = short_name
        self._verbose_name = verbose_name
        self._symmetric = symmetric

    @property
    def identifier(self):
        return self._identifier

    @property
    def index(self):
        return self._index

    @property
    def short_name(self):
        return self._short_name

    @property
    def verbose_name(self):
        return self._verbose_name

    @property
    def symmetric(self):
        return self._symmetric

    def __int__(self):
        return self._index

    def __eq__(self, other):
        if isinstance(other, RelationType):
            return self._identifier == other._identifier
        return False

    def __hash__(self):
        return hash(self._identifier)


class TermType:
    def __init__(self, identifier, index, short_name, verbose_name):
        self._identifier = identifier
        self._index = index
        self._short_name = short_name
        self._verbose_name = verbose_name

    @property
    def identifier(self):
        return self._identifier

    @property
    def index(self):
        return self._index

    @property
    def short_name(self):
        return self._short_name

    @property
    def verbose_name(self):
        return self._verbose_name

    def __int__(self):
        return self._index

    def __eq__(self, other):
        if isinstance(other, TermType):
            return self._identifier == other._identifier
        return False

    def __hash__(self):
        return hash(self._identifier)


class Token:
    def __init__(self, tid: int, index: int, span_start: int, span_end: int, phrase: str):
        self._tid = tid  # ID within the corresponding dataset
        self._index = index  # original token index in document

        self._span_start = span_start  # start of token span in document (inclusive)
        self._span_end = span_end  # end of token span in document (exclusive)
        self._phrase = phrase

    @property
    def index(self):
        return self._index

    @property
    def span_start(self):
        return self._span_start

    @property
    def span_end(self):
        return self._span_end

    @property
    def span(self):
        return self._span_start, self._span_end

    @property
    def phrase(self):
        return self._phrase

    def __eq__(self, other):
        if isinstance(other, Token):
            return self._tid == other._tid
        return False

    def __hash__(self):
        return hash(self._tid)

    def __str__(self):
        return self._phrase

    def __repr__(self):
        return self._phrase


class TokenSpan:
    def __init__(self, tokens):
        self._tokens = tokens

    @property
    def span_start(self):
        return self._tokens[0].span_start

    @property
    def span_end(self):
        return self._tokens[-1].span_end

    @property
    def span(self):
        return self.span_start, self.span_end

    def __getitem__(self, s):
        if isinstance(s, slice):
            return TokenSpan(self._tokens[s.start:s.stop:s.step])
        else:
            try:
                return self._tokens[s]
            except:
                print(self._tokens)
                print(len(self._tokens))
                print(s)

    def __iter__(self):
        return iter(self._tokens)

    def __len__(self):
        return len(self._tokens)


class Term:
    def __init__(self, eid: int, term_type: TermType, tokens: List[Token], phrase: str):
        self._eid = eid  # ID within the corresponding dataset

        self._term_type = term_type

        self._tokens = tokens
        self._phrase = phrase

    def as_tuple(self):
        return self.span_start, self.span_end, self._term_type

    @property
    def term_type(self):
        return self._term_type

    @property
    def tokens(self):
        return TokenSpan(self._tokens)

    @property
    def span_start(self):
        return self._tokens[0].span_start

    @property
    def span_end(self):
        return self._tokens[-1].span_end

    @property
    def span(self):
        return self.span_start, self.span_end

    @property
    def phrase(self):
        return self._phrase

    def __eq__(self, other):
        if isinstance(other, Term):
            return self._eid == other._eid
        return False

    def __hash__(self):
        return hash(self._eid)

    def __str__(self):
        return self._phrase


class Relation:
    def __init__(self, rid: int, relation_type: RelationType, head_term: Term,
                 tail_term: Term, reverse: bool = False):
        self._rid = rid  # ID within the corresponding dataset
        self._relation_type = relation_type

        self._head_term = head_term
        self._tail_term = tail_term

        self._reverse = reverse

        self._first_term = head_term if not reverse else tail_term
        self._second_term = tail_term if not reverse else head_term

    def as_tuple(self):
        head = self._head_term
        tail = self._tail_term
        head_start, head_end = (head.span_start, head.span_end)
        tail_start, tail_end = (tail.span_start, tail.span_end)

        t = ((head_start, head_end, head.term_type),
             (tail_start, tail_end, tail.term_type), self._relation_type)
        return t

    @property
    def relation_type(self):
        return self._relation_type

    @property
    def head_term(self):
        return self._head_term

    @property
    def tail_term(self):
        return self._tail_term

    @property
    def first_term(self):
        return self._first_term

    @property
    def second_term(self):
        return self._second_term

    @property
    def reverse(self):
        return self._reverse

    def __eq__(self, other):
        if isinstance(other, Relation):
            return self._rid == other._rid
        return False

    def __hash__(self):
        return hash(self._rid)


class Document:
    def __init__(self, doc_id: int, tokens: List[Token], terms: List[Term], relations: List[Relation], polarities: List[Polarity],
                 encoding: List[int], dep_label: List[int], dep_label_indices: List[int], dep: List[int],
                 pos: List[str], pos_indices: List[int]):
        self._doc_id = doc_id  # ID within the corresponding dataset

        self._tokens = tokens
        self._terms = terms
        self._relations = relations

        # byte-pair document encoding including special tokens ([CLS] and [SEP])
        self._encoding = encoding

        self._dep_label = dep_label
        self._dep_label_indices = dep_label_indices
        self._dep = dep

        self._pos = pos
        self._pos_indices = pos_indices

    @property
    def doc_id(self):
        return self._doc_id

    @property
    def terms(self):
        return self._terms

    @property
    def relations(self):
        return self._relations

    @property
    def tokens(self):
        return TokenSpan(self._tokens)

    @property
    def encoding(self):
        return self._encoding

    @property
    def dep_label(self):
        return self._dep_label

    @property
    def dep_label_indices(self):
        return self._dep_label_indices

    @property
    def dep(self):
        return self._dep

    @property
    def pos_indices(self):
        return self._pos_indices

    @property
    def pos(self):
        return self._pos

    @encoding.setter
    def encoding(self, value):
        self._encoding = value

    def __eq__(self, other):
        if isinstance(other, Document):
            return self._doc_id == other._doc_id
        return False

    def __hash__(self):
        return hash(self._doc_id)


class BatchIterator:
    def __init__(self, terms, batch_size, order=None, truncate=False):
        self._terms = terms
        self._batch_size = batch_size
        self._truncate = truncate
        self._length = len(self._terms)
        self._order = order

        if order is None:
            self._order = list(range(len(self._terms)))

        self._i = 0

    def __iter__(self):
        return self

    def __next__(self):
        if self._truncate and self._i + self._batch_size > self._length:
            raise StopIteration
        elif not self._truncate and self._i >= self._length:
            raise StopIteration
        else:
            terms = [self._terms[n] for n in self._order[self._i:self._i + self._batch_size]]
            self._i += self._batch_size
            return terms


class Dataset(TorchDataset):
    TRAIN_MODE = 'train'
    EVAL_MODE = 'eval'

    def __init__(self, label, rel_types, term_types, neg_term_count,
                 neg_rel_count, max_span_size):
        self._label = label
        self._rel_types = rel_types
        self._term_types = term_types
        self._neg_term_count = neg_term_count
        self._neg_rel_count = neg_rel_count
        self._max_span_size = max_span_size
        self._mode = Dataset.TRAIN_MODE

        self._documents = OrderedDict()
        self._terms = OrderedDict()
        self._relations = OrderedDict()

        # current ids
        self._doc_id = 0
        self._rid = 0
        self._eid = 0
        self._tid = 0

    def iterate_documents(self, batch_size, order=None, truncate=False):
        return BatchIterator(self.documents, batch_size, order=order, truncate=truncate)

    def iterate_relations(self, batch_size, order=None, truncate=False):
        return BatchIterator(self.relations, batch_size, order=order, truncate=truncate)

    def create_token(self, idx, span_start, span_end, phrase) -> Token:
        token = Token(self._tid, idx, span_start, span_end, phrase)
        self._tid += 1
        return token

    def create_document(self, tokens, term_mentions, relations, polarities, doc_encoding, dep_label, dep_label_indices, dep,
                        pos, pos_indices) -> Document:
        document = Document(self._doc_id, tokens, term_mentions, relations, polarities, doc_encoding, dep_label,
                            dep_label_indices, dep, pos, pos_indices)
        self._documents[self._doc_id] = document
        self._doc_id += 1

        return document

    def create_term(self, term_type, tokens, phrase) -> Term:
        mention = Term(self._eid, term_type, tokens, phrase)
        self._terms[self._eid] = mention
        self._eid += 1
        return mention

    def create_relation(self, relation_type, head_term, tail_term, reverse=False) -> Relation:
        relation = Relation(self._rid, relation_type, head_term, tail_term, reverse)
        self._relations[self._rid] = relation
        self._rid += 1
        return relation

    def __len__(self):
        return len(self._documents)

    def __getitem__(self, index: int):
        doc = self._documents[index]

        if self._mode == Dataset.TRAIN_MODE:
            return sampling.create_train_sample(doc, self._neg_term_count, self._neg_rel_count,
                                                self._max_span_size, len(self._rel_types))
        else:
            return sampling.create_eval_sample(doc, self._max_span_size)

    def switch_mode(self, mode):
        self._mode = mode

    @property
    def label(self):
        return self._label

    @property
    def input_reader(self):
        return self._input_reader

    @property
    def documents(self):
        return list(self._documents.values())

    @property
    def terms(self):
        return list(self._terms.values())

    @property
    def relations(self):
        return list(self._relations.values())

    @property
    def document_count(self):
        return len(self._documents)

    @property
    def term_count(self):
        return len(self._terms)

    @property
    def relation_count(self):
        return len(self._relations)
