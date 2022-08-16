import random

import torch

from Engine import util


def create_train_sample(doc, neg_term_count: int, neg_rel_count: int, max_span_size: int, rel_type_count: int):
    encodings = doc.encoding
    token_count = len(doc.tokens)
    context_size = len(encodings)

    # positive terms
    pos_term_spans, pos_term_types, pos_term_masks, pos_term_sizes = [], [], [], []
    for e in doc.terms:
        pos_term_spans.append(e.span)
        pos_term_types.append(e.term_type.index)
        pos_term_masks.append(create_term_mask(*e.span, context_size))
        pos_term_sizes.append(len(e.tokens))

    # positive relations
    pos_rels, pos_rel_spans, pos_rel_types, pos_rel_masks = [], [], [], []
    pos_rels3, pos_rel_masks3, pos_rel_spans3 = [], [], []
    pos_pair_mask = []  # which triplet rel is true
    for rel in doc.relations:
        s1, s2 = rel.head_term.span, rel.tail_term.span
        pos_rels.append((pos_term_spans.index(s1), pos_term_spans.index(s2)))
        pos_rel_spans.append((s1, s2))
        pos_rel_types.append(rel.relation_type)
        pos_rel_masks.append(create_rel_mask(s1, s2, context_size))

    def is_in_relation(head, tail, relations):
        for rel in relations:
            s1, s2 = rel.head_term, rel.tail_term
            if s1 == head and s2 == tail:
                return 1
        return 0

    for x in range(len(doc.relations)):
        s1, s2 = doc.relations[x].head_term, doc.relations[x].tail_term
        x1, x2 = pos_term_spans.index(s1.span), pos_term_spans.index(s2.span)
        t_p_rels3 = []
        t_p_rels3_mask = []

        t_p_rel_span3 = []
        for idx, e in enumerate(doc.terms):
            if idx != x1 and idx != x2:
                if is_in_relation(s1, e, doc.relations) or is_in_relation(s2, e, doc.relations) or is_in_relation(e, s1, doc.relations) or is_in_relation(e, s2, doc.relations):
                    t_p_rels3.append((x1, x2, idx))
                    t_p_rels3_mask.append(create_rel_mask3(s1.span, s2.span, e.span, context_size))
                    t_p_rel_span3.append((s1.span, s2.span, e.span))
                    # t_p_rel_types3.append(1)
        if len(t_p_rels3) > 0:
            pos_rels3.append(t_p_rels3)
            pos_rel_masks3.append(t_p_rels3_mask)
            pos_pair_mask.append(1)
            pos_rel_spans3.append(t_p_rel_span3)
            # pos_rel_types3.append(t_p_rel_types3)
        else:
            pos_rels3.append([(x1, x2, 0)])
            pos_rel_masks3.append([(create_rel_mask3(s1.span, s2.span, (0, 0), context_size))])
            pos_pair_mask.append(0)

    assert len(pos_rels) == len(pos_rels3) == len(pos_pair_mask)

    # negative terms
    neg_term_spans, neg_term_sizes = [], []
    for size in range(1, max_span_size + 1):
        for i in range(0, (token_count - size) + 1):
            span = doc.tokens[i:i + size].span
            if span not in pos_term_spans:
                neg_term_spans.append(span)
                neg_term_sizes.append(size)

    # sample negative terms
    neg_term_samples = random.sample(list(zip(neg_term_spans, neg_term_sizes)),
                                       min(len(neg_term_spans), neg_term_count))
    neg_term_spans, neg_term_sizes = zip(*neg_term_samples) if neg_term_samples else ([], [])

    neg_term_masks = [create_term_mask(*span, context_size) for span in neg_term_spans]
    neg_term_types = [0] * len(neg_term_spans)

    # negative relations
    # use only strong negative relations, i.e. pairs of actual (labeled) terms that are not related
    # neg_rels3 = []
    neg_rel_spans = []
    neg_rel_spans3 = []
    neg_pair_mask = []

    for i1, s1 in enumerate(pos_term_spans):
        for i2, s2 in enumerate(pos_term_spans):
            rev = (s2, s1)
            rev_symmetric = rev in pos_rel_spans and pos_rel_types[pos_rel_spans.index(rev)].symmetric

            # do not add as negative relation sample:
            # neg. relations from an term to itself
            # term pairs that are related according to gt
            # term pairs whose reverse exists as a symmetric relation in gt
            if s1 != s2 and (s1, s2) not in pos_rel_spans and not rev_symmetric:
                neg_rel_spans.append((s1, s2))

                p_rel_span3 = []
                for i3, s3 in enumerate(pos_term_spans):
                    # three spans are different from each other and not exist in pos_rel_span3
                    if s1 != s2 and s1 != s3 and s2 != s3 and (s1, s2, s3) not in pos_rel_spans3:
                        p_rel_span3.append((s1, s2, s3))
                if len(p_rel_span3) > 0:
                    neg_rel_spans3.append(p_rel_span3)
                    neg_pair_mask.append(1)
                else:
                    neg_rel_spans3.append([(s1, s2, (0, 0))])
                    neg_pair_mask.append(0)

    # sample negative relations

    assert len(neg_rel_spans) == len(neg_rel_spans3) == len(neg_pair_mask)

    neg_rel_spans_samples = random.sample(list(zip(neg_rel_spans, neg_rel_spans3, neg_pair_mask)), min(len(neg_rel_spans), neg_rel_count))
    neg_rel_spans, neg_rel_spans3, neg_pair_mask = zip(*neg_rel_spans_samples) if neg_rel_spans_samples else ([], [], [])

    neg_rels = [(pos_term_spans.index(s1), pos_term_spans.index(s2)) for s1, s2 in neg_rel_spans]
    neg_rels3 = [[(pos_term_spans.index(s1), pos_term_spans.index(s2), pos_term_spans.index(s3)) for s1, s2, s3 in x] for x in neg_rel_spans3]

    assert len(neg_rels3) == len(neg_rel_spans3) == len(neg_pair_mask)

    neg_rel_masks = [create_rel_mask(*spans, context_size) for spans in neg_rel_spans]
    neg_rel_masks3 = [[create_rel_mask3(*sps, context_size) for sps in spans] for spans in neg_rel_spans3]
    neg_rel_types = [0] * len(neg_rel_spans)
    # neg_rel_types3 = [0] * len(neg_rel_spans3)

    # merge
    term_types = pos_term_types + neg_term_types
    term_masks = pos_term_masks + neg_term_masks
    term_sizes = pos_term_sizes + list(neg_term_sizes)
    term_spans = pos_term_spans + list(neg_term_spans)

    rels = pos_rels + neg_rels
    rel_types = [r.index for r in pos_rel_types] + neg_rel_types
    rel_masks = pos_rel_masks + neg_rel_masks

    rels3 = pos_rels3 + neg_rels3
    # rel_types3 = pos_rel_types3 + neg_rel_types3
    rel_masks3 = pos_rel_masks3 + neg_rel_masks3
    pair_mask = pos_pair_mask + list(neg_pair_mask)

    assert len(term_masks) == len(term_sizes) == len(term_types)
    try:
        assert len(rels) == len(rel_masks) == len(rel_types) == len(rels3) == len(pair_mask)
    except:
        print(len(rels))
        print(len(rels3))
        print(len(pair_mask))

    encodings = torch.tensor(encodings, dtype=torch.long)

    # masking of tokens
    context_masks = torch.ones(context_size, dtype=torch.bool)

    # also create samples_masks:
    # tensors to mask term/relation samples of batch
    # since samples are stacked into batches, "padding" terms/relations possibly must be created
    # these are later masked during loss computation
    if term_masks:
        term_types = torch.tensor(term_types, dtype=torch.long)
        term_masks = torch.stack(term_masks)
        term_sizes = torch.tensor(term_sizes, dtype=torch.long)
        term_sample_masks = torch.ones([term_masks.shape[0]], dtype=torch.bool)
        term_spans = torch.tensor(term_spans, dtype=torch.long)
    else:
        # corner case handling (no pos/neg terms)
        term_types = torch.zeros([1], dtype=torch.long)
        term_masks = torch.zeros([1, context_size], dtype=torch.bool)
        term_sizes = torch.zeros([1], dtype=torch.long)
        term_sample_masks = torch.zeros([1], dtype=torch.bool)
        term_spans = torch.tensor([1, 2], dtype=torch.long)

    if rels:
        rels = torch.tensor(rels, dtype=torch.long)
        rel_masks = torch.stack(rel_masks)
        rel_types = torch.tensor(rel_types, dtype=torch.long)
        rel_sample_masks = torch.ones([rels.shape[0]], dtype=torch.bool)
    else:
        # corner case handling (no pos/neg relations)
        rels = torch.zeros([1, 2], dtype=torch.long)
        rel_types = torch.zeros([1], dtype=torch.long)
        rel_masks = torch.zeros([1, context_size], dtype=torch.bool)
        rel_sample_masks = torch.zeros([1], dtype=torch.bool)

    if rels3:
        max_tri = max([len(x) for x in rels3])
        for idx, r in enumerate(rels3):
            r_len = len(r)
            if r_len < max_tri:
                rels3[idx].extend([rels3[idx][0]] * (max_tri - r_len))
                rel_masks3[idx].extend([rel_masks3[idx][0]] * (max_tri - r_len))
        rels3 = torch.tensor(rels3, dtype=torch.long)
        try:
            rel_masks3 = torch.stack([torch.stack(x) for x in rel_masks3])
        except:
            print(rel_masks3)
        rel_sample_masks3 = torch.ones([rels3.shape[0]], dtype=torch.bool)
        pair_mask = torch.tensor(pair_mask, dtype=torch.bool)
    else:
        rels3 = torch.zeros([1, 3], dtype=torch.long)
        rel_masks3 = torch.zeros([1, context_size], dtype=torch.bool)
        rel_sample_masks3 = torch.zeros([1], dtype=torch.bool)
        pair_mask = torch.tensor(pair_mask, dtype=torch.bool)

    # relation types to one-hot encoding
    rel_types_onehot = torch.zeros([rel_types.shape[0], rel_type_count], dtype=torch.float32)
    rel_types_onehot.scatter_(1, rel_types.unsqueeze(1), 1)
    rel_types_onehot = rel_types_onehot[:, 1:]  # all zeros for 'none' relation

    simple_graph = None
    graph = None
    try:
        simple_graph = torch.tensor(get_simple_graph(context_size, doc.dep), dtype=torch.long)  # only the relation
    except:
        print(context_size)
        print(token_count)
        print(encodings)
        print(doc.dep)
        print(doc.dep_label_indices)
    try:
        graph = torch.tensor(get_graph(context_size, doc.dep, doc.dep_label_indices),
                             dtype=torch.long)  # relation and the type of relation
    except:
        print(context_size)
        print(token_count)
        print(encodings)
        print(doc.dep)
        print(doc.dep_label_indices)

    pos = torch.tensor(get_pos(context_size, doc.pos_indices), dtype=torch.long)

    return dict(encodings=encodings, context_masks=context_masks, term_masks=term_masks,
                term_sizes=term_sizes, term_types=term_types, term_spans=term_spans,
                rels=rels, rel_masks=rel_masks, rel_types=rel_types_onehot,
                rels3=rels3, rel_sample_masks3=rel_sample_masks3, rel_masks3=rel_masks3,
                pair_mask=pair_mask,
                term_sample_masks=term_sample_masks, rel_sample_masks=rel_sample_masks,
                simple_graph=simple_graph, graph=graph, pos=pos)


def create_eval_sample(doc, max_span_size: int):
    encodings = doc.encoding
    token_count = len(doc.tokens)
    context_size = len(encodings)

    # create term candidates
    term_spans = []
    term_masks = []
    term_sizes = []

    for size in range(1, max_span_size + 1):
        for i in range(0, (token_count - size) + 1):
            span = doc.tokens[i:i + size].span
            term_spans.append(span)
            term_masks.append(create_term_mask(*span, context_size))
            term_sizes.append(size)

    # create tensors
    # token indices
    _encoding = encodings
    encodings = torch.zeros(context_size, dtype=torch.long)
    encodings[:len(_encoding)] = torch.tensor(_encoding, dtype=torch.long)

    # masking of tokens
    context_masks = torch.zeros(context_size, dtype=torch.bool)
    context_masks[:len(_encoding)] = 1

    # terms
    if term_masks:
        term_masks = torch.stack(term_masks)
        term_sizes = torch.tensor(term_sizes, dtype=torch.long)
        term_spans = torch.tensor(term_spans, dtype=torch.long)

        # tensors to mask term samples of batch
        # since samples are stacked into batches, "padding" terms possibly must be created
        # these are later masked during evaluation
        term_sample_masks = torch.tensor([1] * term_masks.shape[0], dtype=torch.bool)
    else:
        # corner case handling (no terms)
        term_masks = torch.zeros([1, context_size], dtype=torch.bool)
        term_sizes = torch.zeros([1], dtype=torch.long)
        term_spans = torch.zeros([1, 2], dtype=torch.long)
        term_sample_masks = torch.zeros([1], dtype=torch.bool)

    simple_graph = torch.tensor(get_simple_graph(context_size, doc.dep), dtype=torch.long)  # only the relation
    graph = torch.tensor(get_graph(context_size, doc.dep, doc.dep_label_indices),
                         dtype=torch.long)  # relation and the type of relation
    pos = torch.tensor(get_pos(context_size, doc.pos_indices), dtype=torch.long)

    return dict(encodings=encodings, context_masks=context_masks, term_masks=term_masks,
                term_sizes=term_sizes, term_spans=term_spans, term_sample_masks=term_sample_masks,
                simple_graph=simple_graph, graph=graph, pos=pos)


def create_term_mask(start, end, context_size):
    mask = torch.zeros(context_size, dtype=torch.bool)
    mask[start:end] = 1
    return mask


def create_rel_mask(s1, s2, context_size):
    start = s1[1] if s1[1] < s2[0] else s2[1]
    end = s2[0] if s1[1] < s2[0] else s1[0]
    mask = create_term_mask(start, end, context_size)
    return mask


def create_rel_mask3(s1, s2, s3, context_size):
    mask = torch.zeros(context_size, dtype=torch.bool)
    start = min(s1[0], s1[1], s2[0], s2[1], s3[0], s3[1])
    end = max(s1[0], s1[1], s2[0], s2[1], s3[0], s3[1])
    mask[start:end] = 1
    return mask


def collate_fn_padding(batch):
    padded_batch = dict()
    keys = batch[0].keys()

    for key in keys:
        samples = [s[key] for s in batch]
        if not batch[0][key].shape:
            padded_batch[key] = torch.stack(samples)
        else:
            padded_batch[key] = util.padded_stack([s[key] for s in batch])

    return padded_batch


def get_graph(seq_len, feature_data, feature2id):
    ret = [[0] * seq_len for _ in range(seq_len)]
    for i, item in enumerate(feature_data):
        if int(item) > seq_len-1 or int(item) == 0:
            continue
        ret[i + 1][int(item) - 1] = feature2id[i] + 2
        ret[int(item) - 1][i + 1] = feature2id[i] + 2
        ret[i + 1][i + 1] = 1
    return ret


def get_simple_graph(seq_len, feature_data):
    ret = [[0] * seq_len for _ in range(seq_len)]
    for i, item in enumerate(feature_data):
        if int(item) > seq_len-1:
            continue
        ret[i + 1][int(item) - 1] = 1
        ret[int(item) - 1][i + 1] = 1
        ret[i + 1][i + 1] = 1
    return ret


def get_pos(seq_len, pos_indices):
    ret = [0] * seq_len
    for i, item in enumerate(pos_indices):
        ret[i + 1] = pos_indices[i] + 1
    return ret