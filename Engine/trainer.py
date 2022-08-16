import argparse
import math
import os

import torch
from torch.nn import DataParallel
from torch.optim import Optimizer
import transformers
from torch.utils.data import DataLoader
from transformers import AdamW, BertConfig
from transformers import BertTokenizer

from Engine import models
from Engine import sampling
from Engine import util
from Engine.terms import Dataset
from Engine.evaluator import Evaluator
from Engine.input_reader import JsonInputReader, BaseInputReader
from Engine.loss import SyMuxLoss, Loss
from tqdm import tqdm
from Engine.base_trainer import BaseTrainer

SCRIPT_PATH = os.path.dirname(os.path.realpath(__file__))


class SyMuxTrainer(BaseTrainer):
    """ Joint term and relation extraction training and evaluation """

    def __init__(self, args: argparse.Namespace):
        super().__init__(args)

        # byte-pair encoding
        self._tokenizer = BertTokenizer.from_pretrained(args.tokenizer_path,
                                                        do_lower_case=args.lowercase,
                                                        cache_dir=args.cache_path)

        # path to export predictions to
        self._predictions_path = os.path.join(self._log_path, 'predictions_%s_epoch_%s.json')

        # path to export relation extraction examples to
        self._examples_path = os.path.join(self._log_path, 'examples_%s_%s_epoch_%s.html')

    def train(self, train_path: str, valid_path: str, types_path: str, input_reader_cls: BaseInputReader):
        args = self.args
        train_label, valid_label = 'train', 'valid'

        self._logger.info("Datasets: %s, %s" % (train_path, valid_path))
        self._logger.info("Model type: %s" % args.model_type)

        # create log csv files
        self._init_train_logging(train_label)
        self._init_eval_logging(valid_label)

        # read datasets
        input_reader = input_reader_cls(types_path, self._tokenizer, args.neg_term_count,
                                        args.neg_relation_count, args.max_span_size, self._logger)
        input_reader.read({train_label: train_path, valid_label: valid_path})
        self._log_datasets(input_reader)

        train_dataset = input_reader.get_dataset(train_label)
        train_sample_count = train_dataset.document_count
        updates_epoch = train_sample_count // args.train_batch_size
        updates_total = updates_epoch * args.epochs

        validation_dataset = input_reader.get_dataset(valid_label)

        self._logger.info("Updates per epoch: %s" % updates_epoch)
        self._logger.info("Updates total: %s" % updates_total)

        # create model
        model_class = models.get_model(self.args.model_type)

        # load model
        config = BertConfig.from_pretrained(self.args.model_path, cache_dir=self.args.cache_path)
        util.check_version(config, model_class, self.args.model_path)

        config.model_version = model_class.VERSION
        model = model_class.from_pretrained(self.args.model_path,
                                            config=config,
                                            cls_token=self._tokenizer.convert_tokens_to_ids('[CLS]'),
                                            relation_types=input_reader.relation_type_count - 1,
                                            term_types=input_reader.term_type_count,
                                            max_pairs=self.args.max_pairs,
                                            prop_drop=self.args.prop_drop,
                                            size_embedding=self.args.size_embedding,
                                            freeze_transformer=self.args.freeze_transformer,
                                            args=self.args,
                                            beta=self.args.beta,
                                            alpha=self.args.alpha,
                                            sigma=self.args.sigma)

        model.to(self._device)

        # create optimizer
        optimizer_params = self._get_optimizer_params(model)
        optimizer = AdamW(optimizer_params, lr=args.lr, weight_decay=args.weight_decay, correct_bias=False)
        # create scheduler
        scheduler = transformers.get_linear_schedule_with_warmup(optimizer,
                                                                 num_warmup_steps=args.lr_warmup * updates_total,
                                                                 num_training_steps=updates_total)

        # "AE",
        # "ALSC",
        # "AESC",
        # "OE",
        # "AOE",
        # "AOPE",
        # "TE"

        # create loss function
        pol_criterion = torch.nn.BCEWithLogitsLoss(reduction='none')
        term_criterion = torch.nn.CrossEntropyLoss(reduction='none')
        compute_loss_AE = SyMuxLoss(pol_criterion, term_criterion, model, optimizer, scheduler, args.max_grad_norm)
        compute_loss_OE = SyMuxLoss(pol_criterion, term_criterion, model, optimizer, scheduler, args.max_grad_norm)
        compute_loss_AOE = SyMuxLoss(pol_criterion, term_criterion, model, optimizer, scheduler, args.max_grad_norm)
        compute_loss_AOPE = SyMuxLoss(pol_criterion, term_criterion, model, optimizer, scheduler, args.max_grad_norm)
        compute_loss_ALSC = SyMuxLoss(pol_criterion, term_criterion, model, optimizer, scheduler, args.max_grad_norm)
        compute_loss_AESC = SyMuxLoss(pol_criterion, term_criterion, model, optimizer, scheduler, args.max_grad_norm)
        compute_loss_TE = SyMuxLoss(pol_criterion, term_criterion, model, optimizer, scheduler, args.max_grad_norm)

        # eval validation set
        if args.init_eval:
            self._eval(model, validation_dataset, input_reader, 0, updates_epoch)

        # train
        best_f1 = 0.0
        for epoch in range(args.epochs):
            # train epoch
            self._train_epoch(model, compute_loss_AE, compute_loss_OE, compute_loss_AOE,
                              compute_loss_AOPE, compute_loss_ALSC, compute_loss_AESC, compute_loss_TE,
                              optimizer, train_dataset, updates_epoch, epoch)

            # eval validation sets
            if not args.final_eval or (epoch == args.epochs - 1):
                rel_nec_eval = self._eval(model, validation_dataset, input_reader, epoch + 1, updates_epoch)
                if best_f1 < rel_nec_eval[-1]:
                    # save final model
                    best_f1 = rel_nec_eval[-1]
                    extra = dict(epoch=args.epochs, updates_epoch=updates_epoch, epoch_iteration=0)
                    global_iteration = args.epochs * updates_epoch
                    self._save_model(self._save_path, model, self._tokenizer, global_iteration,
                                     optimizer=optimizer if self.args.save_optimizer else None, save_as_best=True,
                                     extra=extra, include_iteration=False)

        self._logger.info("Logged in: %s" % self._log_path)
        self._logger.info("Saved in: %s" % self._save_path)
        self._close_summary_writer()

    def eval(self, dataset_path: str, types_path: str, input_reader_cls: BaseInputReader):
        args = self.args
        dataset_label = 'test'

        self._logger.info("Dataset: %s" % dataset_path)
        self._logger.info("Model: %s" % args.model_type)

        # create log csv files
        self._init_eval_logging(dataset_label)

        # read datasets
        input_reader = input_reader_cls(types_path, self._tokenizer,
                                        max_span_size=args.max_span_size, logger=self._logger)
        input_reader.read({dataset_label: dataset_path})
        self._log_datasets(input_reader)

        # create model
        model_class = models.get_model(self.args.model_type)

        config = BertConfig.from_pretrained(self.args.model_path, cache_dir=self.args.cache_path)
        util.check_version(config, model_class, self.args.model_path)

        model = model_class.from_pretrained(self.args.model_path,
                                            config=config,
                                            cls_token=self._tokenizer.convert_tokens_to_ids('[CLS]'),
                                            relation_types=input_reader.relation_type_count - 1,
                                            term_types=input_reader.term_type_count,
                                            max_pairs=self.args.max_pairs,
                                            prop_drop=self.args.prop_drop,
                                            size_embedding=self.args.size_embedding,
                                            freeze_transformer=self.args.freeze_transformer,
                                            args=self.args,
                                            beta=self.args.beta,
                                            alpha=self.args.alpha,
                                            sigma=self.args.sigma)

        model.to(self._device)

        # evaluate
        self._eval(model, input_reader.get_dataset(dataset_label), input_reader)

        self._logger.info("Logged in: %s" % self._log_path)
        self._close_summary_writer()

    def _train_epoch(self, model: torch.nn.Module, compute_loss_AE: Loss, compute_loss_OE: Loss, compute_loss_AOE: Loss,
                     compute_loss_AOPE: Loss, compute_loss_ALSC: Loss, compute_loss_AESC: Loss, compute_loss_TE: Loss,
                     optimizer: Optimizer, dataset: Dataset,
                     updates_epoch: int, epoch: int):
        self._logger.info("Train epoch: %s" % epoch)

        # create data loader
        dataset.switch_mode(Dataset.TRAIN_MODE)
        data_loader = DataLoader(dataset, batch_size=self.args.train_batch_size, shuffle=True, drop_last=True,
                                 num_workers=self.args.sampling_processes, collate_fn=sampling.collate_fn_padding)

        model.zero_grad()

        iteration = 0
        total = dataset.document_count // self.args.train_batch_size
        for batch in tqdm(data_loader, total=total, desc='Train epoch %s' % epoch):
            model.train()
            batch = util.to_device(batch, self._device)

            # forward step
            AE_clf, OE_clf, AOE_clf, AOPE_clf, ALSC_clf, AESC_clf, TE_clf = model(encodings=batch['encodings'], context_masks=batch['context_masks'],
                                            term_masks=batch['term_masks'], term_sizes=batch['term_sizes'],
                                            term_spans=batch['term_spans'], term_types=batch['term_types'],
                                            relations=batch['rels'], rel_masks=batch['rel_masks'],
                                            simple_graph=batch['simple_graph'], graph=batch['graph'],
                                            relations3=batch['rels3'], rel_masks3=batch['rel_masks3'],
                                            pair_mask=batch['pair_mask'], pos=batch['pos'])

            # compute loss for each subtasks
            batch_loss_AE = compute_loss_AE.compute(term_logits=AE_clf, pol_logits=None,
                                                    rel_types=batch['rel_types'], term_types=batch['term_types'],
                                                    term_sample_masks=batch['term_sample_masks'],
                                                    rel_sample_masks=batch['rel_sample_masks'])
            batch_loss_OE = compute_loss_OE.compute(term_logits=OE_clf, pol_logits=None,
                                                      rel_types=batch['rel_types'], term_types=batch['term_types'],
                                                      term_sample_masks=batch['term_sample_masks'],
                                                      rel_sample_masks=batch['rel_sample_masks'])
            batch_loss_AOE = compute_loss_AOE.compute(term_logits=AOE_clf, pol_logits=None,
                                                       rel_types=batch['rel_types'], term_types=batch['term_types'],
                                                       term_sample_masks=batch['term_sample_masks'],
                                                       rel_sample_masks=batch['rel_sample_masks'])
            batch_loss_AOPE = compute_loss_AOPE.compute(term_logits=AOPE_clf, pol_logits=None,
                                                        rel_types=batch['rel_types'], term_types=batch['term_types'],
                                                        term_sample_masks=batch['term_sample_masks'],
                                                        rel_sample_masks=batch['rel_sample_masks'])
            batch_loss_ALSC = compute_loss_ALSC.compute(term_logits=ALSC_clf, pol_logits=None,
                                                        rel_types=batch['rel_types'], term_types=batch['term_types'],
                                                        term_sample_masks=batch['term_sample_masks'],
                                                        rel_sample_masks=batch['rel_sample_masks'])
            batch_loss_AESC = compute_loss_AESC.compute(term_logits=AESC_clf, pol_logits=None,
                                                        rel_types=batch['rel_types'], term_types=batch['term_types'],
                                                        term_sample_masks=batch['term_sample_masks'],
                                                        rel_sample_masks=batch['rel_sample_masks'])
            batch_loss_TE = compute_loss_TE.compute(term_logits=TE_clf, pol_logits=None,
                                                      rel_types=batch['rel_types'], term_types=batch['term_types'],
                                                      term_sample_masks=batch['term_sample_masks'],
                                                      rel_sample_masks=batch['rel_sample_masks'])

            batch_loss = (batch_loss_AE + batch_loss_OE + batch_loss_AOE + batch_loss_AOPE +
                          batch_loss_ALSC + batch_loss_AESC + batch_loss_TE)

            # logging
            iteration += 1
            global_iteration = epoch * updates_epoch + iteration

            if global_iteration % self.args.train_log_iter == 0:
                self._log_train(optimizer, batch_loss, epoch, iteration, global_iteration, dataset.label)

        return iteration

    def _eval(self, model: torch.nn.Module, dataset: Dataset, input_reader: JsonInputReader,
              epoch: int = 0, updates_epoch: int = 0, iteration: int = 0):
        self._logger.info("Evaluate: %s" % dataset.label)

        if isinstance(model, DataParallel):
            # currently no multi GPU support during evaluation
            model = model.module

        # create evaluator
        evaluator = Evaluator(dataset, input_reader, self._tokenizer,
                              self.args.rel_filter_threshold, self.args.no_overlapping, self._predictions_path,
                              self._examples_path, self.args.example_count, epoch, dataset.label)

        # create data loader
        dataset.switch_mode(Dataset.EVAL_MODE)
        data_loader = DataLoader(dataset, batch_size=self.args.eval_batch_size, shuffle=False, drop_last=False,
                                 num_workers=self.args.sampling_processes, collate_fn=sampling.collate_fn_padding)

        with torch.no_grad():
            model.eval()

            # iterate batches
            total = math.ceil(dataset.document_count / self.args.eval_batch_size)
            for batch in tqdm(data_loader, total=total, desc='Evaluate epoch %s' % epoch):
                # move batch to selected device
                batch = util.to_device(batch, self._device)

                # run model (forward pass)
                AE_out, OE_out, AOE_out, AOPE_out, ALSC_out, AESC_out, TE_out  = model(encodings=batch['encodings'], context_masks=batch['context_masks'],
                               term_masks=batch['term_masks'], term_sizes=batch['term_sizes'],
                               term_spans=batch['term_spans'], term_sample_masks=batch['term_sample_masks'],
                               evaluate=True, simple_graph=batch['simple_graph'], graph=batch['graph'],
                               pos=batch['pos'])  # pos=batch['pos']
                # term_clf, rel_clf, rels = result

                # evaluate batch
                evaluator.eval_batch(AE_out, OE_out, AOE_out, AOPE_out, ALSC_out, AESC_out, TE_out, batch)

        global_iteration = epoch * updates_epoch + iteration
        ner_eval, rel_eval, rel_nec_eval = evaluator.compute_scores()
        self._log_eval(*ner_eval, *rel_eval, *rel_nec_eval,
                       epoch, iteration, global_iteration, dataset.label)

        if self.args.store_predictions and not self.args.no_overlapping:
            evaluator.store_predictions()

        if self.args.store_examples:
            evaluator.store_examples()
        return rel_nec_eval

    def _get_optimizer_params(self, model):
        param_optimizer = list(model.named_parameters())
        no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
        optimizer_params = [
            {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)],
             'weight_decay': self.args.weight_decay},
            {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}]

        return optimizer_params

    def _log_train(self, optimizer: Optimizer, loss: float, epoch: int,
                   iteration: int, global_iteration: int, label: str):
        # average loss
        avg_loss = loss / self.args.train_batch_size
        # get current learning rate
        lr = self._get_lr(optimizer)[0]

        # log to tensorboard
        self._log_tensorboard(label, 'loss', loss, global_iteration)
        self._log_tensorboard(label, 'loss_avg', avg_loss, global_iteration)
        self._log_tensorboard(label, 'lr', lr, global_iteration)

        # log to csv
        self._log_csv(label, 'loss', loss, epoch, iteration, global_iteration)
        self._log_csv(label, 'loss_avg', avg_loss, epoch, iteration, global_iteration)
        self._log_csv(label, 'lr', lr, epoch, iteration, global_iteration)

    def _log_eval(self, ner_prec_micro: float, ner_rec_micro: float, ner_f1_micro: float,
                  ner_prec_macro: float, ner_rec_macro: float, ner_f1_macro: float,

                  rel_prec_micro: float, rel_rec_micro: float, rel_f1_micro: float,
                  rel_prec_macro: float, rel_rec_macro: float, rel_f1_macro: float,

                  rel_nec_prec_micro: float, rel_nec_rec_micro: float, rel_nec_f1_micro: float,
                  rel_nec_prec_macro: float, rel_nec_rec_macro: float, rel_nec_f1_macro: float,
                  epoch: int, iteration: int, global_iteration: int, label: str):

        # log to tensorboard
        self._log_tensorboard(label, 'eval/ner_prec_micro', ner_prec_micro, global_iteration)
        self._log_tensorboard(label, 'eval/ner_recall_micro', ner_rec_micro, global_iteration)
        self._log_tensorboard(label, 'eval/ner_f1_micro', ner_f1_micro, global_iteration)
        self._log_tensorboard(label, 'eval/ner_prec_macro', ner_prec_macro, global_iteration)
        self._log_tensorboard(label, 'eval/ner_recall_macro', ner_rec_macro, global_iteration)
        self._log_tensorboard(label, 'eval/ner_f1_macro', ner_f1_macro, global_iteration)

        self._log_tensorboard(label, 'eval/pol_prec_micro', rel_prec_micro, global_iteration)
        self._log_tensorboard(label, 'eval/pol_recall_micro', rel_rec_micro, global_iteration)
        self._log_tensorboard(label, 'eval/pol_f1_micro', rel_f1_micro, global_iteration)
        self._log_tensorboard(label, 'eval/pol_prec_macro', rel_prec_macro, global_iteration)
        self._log_tensorboard(label, 'eval/pol_recall_macro', rel_rec_macro, global_iteration)
        self._log_tensorboard(label, 'eval/pol_f1_macro', rel_f1_macro, global_iteration)

        self._log_tensorboard(label, 'eval/pol_nec_prec_micro', rel_nec_prec_micro, global_iteration)
        self._log_tensorboard(label, 'eval/pol_nec_recall_micro', rel_nec_rec_micro, global_iteration)
        self._log_tensorboard(label, 'eval/pol_nec_f1_micro', rel_nec_f1_micro, global_iteration)
        self._log_tensorboard(label, 'eval/pol_nec_prec_macro', rel_nec_prec_macro, global_iteration)
        self._log_tensorboard(label, 'eval/pol_nec_recall_macro', rel_nec_rec_macro, global_iteration)
        self._log_tensorboard(label, 'eval/pol_nec_f1_macro', rel_nec_f1_macro, global_iteration)

        # log to csv
        self._log_csv(label, 'eval', ner_prec_micro, ner_rec_micro, ner_f1_micro,
                      ner_prec_macro, ner_rec_macro, ner_f1_macro,

                      rel_prec_micro, rel_rec_micro, rel_f1_micro,
                      rel_prec_macro, rel_rec_macro, rel_f1_macro,

                      rel_nec_prec_micro, rel_nec_rec_micro, rel_nec_f1_micro,
                      rel_nec_prec_macro, rel_nec_rec_macro, rel_nec_f1_macro,
                      epoch, iteration, global_iteration)

    def _log_datasets(self, input_reader):
        self._logger.info("Relation type count: %s" % input_reader.relation_type_count)
        self._logger.info("Term type count: %s" % input_reader.term_type_count)

        self._logger.info("Terms:")
        for e in input_reader.term_types.values():
            self._logger.info(e.verbose_name + '=' + str(e.index))

        self._logger.info("Relations:")
        for r in input_reader.relation_types.values():
            self._logger.info(r.verbose_name + '=' + str(r.index))

        for k, d in input_reader.datasets.items():
            self._logger.info('Dataset: %s' % k)
            self._logger.info("Document count: %s" % d.document_count)
            self._logger.info("Relation count: %s" % d.relation_count)
            self._logger.info("Term count: %s" % d.term_count)

        self._logger.info("Context size: %s" % input_reader.context_size)

    def _init_train_logging(self, label):
        self._add_dataset_logging(label,
                                  data={'lr': ['lr', 'epoch', 'iteration', 'global_iteration'],
                                        'loss': ['loss', 'epoch', 'iteration', 'global_iteration'],
                                        'loss_avg': ['loss_avg', 'epoch', 'iteration', 'global_iteration']})

    def _init_eval_logging(self, label):
        self._add_dataset_logging(label,
                                  data={'eval': ['ner_prec_micro', 'ner_rec_micro', 'ner_f1_micro',
                                                 'ner_prec_macro', 'ner_rec_macro', 'ner_f1_macro',
                                                 'rel_prec_micro', 'rel_rec_micro', 'rel_f1_micro',
                                                 'rel_prec_macro', 'rel_rec_macro', 'rel_f1_macro',
                                                 'rel_nec_prec_micro', 'rel_nec_rec_micro', 'rel_nec_f1_micro',
                                                 'rel_nec_prec_macro', 'rel_nec_rec_macro', 'rel_nec_f1_macro',
                                                 'epoch', 'iteration', 'global_iteration']})
