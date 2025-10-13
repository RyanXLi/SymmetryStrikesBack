import numpy as np
import os, time, datetime
import torch
import torch.utils.tensorboard
import wandb
import torch.profiler
import importlib
import shutil
import utils.util as util
import utils.util_vis as util_vis
import utils.eval_3D as eval_3D

from torch.nn.parallel import DistributedDataParallel as DDP
from utils.util import print_eval, setup, cleanup
from utils.util import EasyDict as edict
from copy import deepcopy
from model.compute_graph import graph_shape

from tqdm import tqdm
import datetime
from collections import deque

import json

# ============================ main engine for training and evaluation ============================

class Runner():

    def __init__(self, opt):
        super().__init__()
        if os.path.isdir(opt.output_path) and opt.resume == False and opt.device == 0:
            for filename in os.listdir(opt.output_path):
                if "tfevents" in filename: os.remove(os.path.join(opt.output_path, filename))
                if "html" in filename: os.remove(os.path.join(opt.output_path, filename))
                if "vis" in filename: shutil.rmtree(os.path.join(opt.output_path, filename))
                if "embedding" in filename: shutil.rmtree(os.path.join(opt.output_path, filename))
        if opt.device == 0: 
            os.makedirs(opt.output_path,exist_ok=True)
        setup(opt.device, opt.world_size, opt.port)
        opt.batch_size = opt.batch_size // opt.world_size
        self.iteration_durations = deque(maxlen=10)
        self.last_iteration_time = None
    
    def get_viz_data(self, opt):
        # get data for visualization
        viz_data_list = []
        sample_range = len(self.viz_loader)
        viz_interval = sample_range // opt.eval.n_vis
        for i in range(sample_range):
            current_batch = next(self.viz_loader_iter)
            if i % viz_interval != 0: continue
            viz_data_list.append(current_batch)
        return viz_data_list
    
    def load_dataset(self, opt, eval_split="test"):
        data_train = importlib.import_module('data.{}'.format(opt.data.dataset_train))
        data_test = importlib.import_module('data.{}'.format(opt.data.dataset_test))
        if opt.device == 0: print("loading training data...")
        self.train_data = data_train.Dataset(opt, split="train")
        self.train_loader = self.train_data.setup_loader(opt, shuffle=True, use_ddp=True, drop_last=True)
        self.num_batches = len(self.train_loader)
        if opt.device == 0: print("loading test data...")
        self.test_data = data_test.Dataset(opt, split=eval_split)
        self.test_loader = self.test_data.setup_loader(opt, shuffle=False, use_ddp=True, drop_last=True, batch_size=opt.eval.batch_size)
        self.num_batches_test = len(self.test_loader)
        if len(self.test_loader.sampler) * opt.world_size < len(self.test_data):
            self.aux_test_dataset = torch.utils.data.Subset(self.test_data,
                                    range(len(self.test_loader.sampler) * opt.world_size, len(self.test_data)))
            self.aux_test_loader = torch.utils.data.DataLoader(
                self.aux_test_dataset, batch_size=opt.eval.batch_size, shuffle=False, drop_last=False,
                num_workers=opt.data.num_workers)
        if opt.device == 0:
            print("creating data for visualization...")
            self.viz_loader = self.test_data.setup_loader(opt, shuffle=False, use_ddp=False, drop_last=False, batch_size=1)
            self.viz_loader_iter = iter(self.viz_loader)
            self.viz_data = self.get_viz_data(opt)

    def build_networks(self, opt):
        if opt.device == 0: print("building networks...")
        self.graph = DDP(graph_shape.Graph(opt).to(opt.device), device_ids=[opt.device], find_unused_parameters=(not opt.optim.fix_dpt or not opt.optim.fix_clip))

# =================================================== set up training =========================================================

    def setup_optimizer(self, opt):
        if opt.device == 0: print("setting up optimizers...")
        if opt.optim.fix_dpt:
            # when we do not need to train the dpt depth, every param will start from scratch
            scratch_param_decay = []
            scratch_param_nodecay = []
            # loop over all params
            for name, param in self.graph.named_parameters():
                # skip and fixed params
                if not param.requires_grad or 'dpt_depth' in name or 'intr_' in name:
                    continue
                # do not add wd on bias or low-dim params
                if param.ndim <= 1 or name.endswith(".bias"):
                    scratch_param_nodecay.append(param)
                else:
                    scratch_param_decay.append(param)
            # create the optim dictionary
            optim_dict = [
                {'params': scratch_param_nodecay, 'lr': opt.optim.lr, 'weight_decay': 0.},
                {'params': scratch_param_decay, 'lr': opt.optim.lr, 'weight_decay': opt.optim.weight_decay}
            ]
        else:
            # when we need to train dpt as well, related params should go to finetune list
            finetune_param_nodecay = []
            scratch_param_nodecay = []
            finetune_param_decay = []
            scratch_param_decay = []
            for name, param in self.graph.named_parameters():
                # skip and fixed params
                if not param.requires_grad:
                    continue
                # put dpt params into finetune list
                if 'dpt_depth' in name or 'intr_' in name or 'dino' in name:
                    if param.ndim <= 1 or name.endswith(".bias"):
                        finetune_param_nodecay.append(param)
                    else:
                        finetune_param_decay.append(param)
                # all other params go to scratch list
                else:
                    if param.ndim <= 1 or name.endswith(".bias"):
                        scratch_param_nodecay.append(param)
                    else:
                        scratch_param_decay.append(param)
            # create the optim dictionary
            optim_dict = [
                {'params': finetune_param_nodecay, 'lr': opt.optim.lr_ft, 'weight_decay': 0.},
                {'params': finetune_param_decay, 'lr': opt.optim.lr_ft, 'weight_decay': opt.optim.weight_decay},
                {'params': scratch_param_nodecay, 'lr': opt.optim.lr, 'weight_decay': 0.},
                {'params': scratch_param_decay, 'lr': opt.optim.lr, 'weight_decay': opt.optim.weight_decay}
            ]
            
        self.optim = torch.optim.AdamW(optim_dict, betas=(0.9, 0.95))
        if opt.optim.sched: 
            self.sched = torch.optim.lr_scheduler.CosineAnnealingLR(self.optim, opt.max_epoch)
        if opt.optim.amp:
            self.scaler = torch.cuda.amp.GradScaler()

    def restore_checkpoint(self, opt, best=False, evaluate=False):
        epoch_start, iter_start = None, None
        if opt.resume:
            if opt.device == 0: print("resuming from previous checkpoint...")
            epoch_start, iter_start, best_val, best_ep = util.restore_checkpoint(opt, self, resume=opt.resume, best=best, evaluate=evaluate)
            self.best_val = best_val
            self.best_ep = best_ep
        elif opt.load is not None:
            if opt.device == 0: print("loading weights from checkpoint {}...".format(opt.load))
            epoch_start, iter_start, best_val, best_ep = util.restore_checkpoint(opt, self, load_name=opt.load)
        else:
            if opt.device == 0: print("initializing weights from scratch...")
        self.epoch_start = epoch_start or 0
        self.iter_start = iter_start or 0

    def setup_visualizer(self, opt, test=False):
        if opt.device == 0: 
            print("setting up visualizers...")
            date_time = datetime.datetime.now()
            wandb.init(
                project="symm",
                config=opt,
                dir=opt.output_path,
                resume=opt.resume,
                name=date_time.strftime("%d%m-") + opt.yaml
            )

            if opt.tb:
                if test == False:
                    self.tb = torch.utils.tensorboard.SummaryWriter(log_dir=opt.output_path, flush_secs=10)
                else:
                    embedding_folder = os.path.join(opt.output_path, 'embedding')
                    os.makedirs(embedding_folder, exist_ok=True)
                    self.tb = torch.utils.tensorboard.SummaryWriter(log_dir=embedding_folder, flush_secs=10)
    
    def train(self, opt):
        # before training
        torch.cuda.set_device(opt.device)
        torch.cuda.empty_cache()
        if opt.device == 0: print("TRAINING START")
        self.train_metric_logger = util.MetricLogger(delimiter="  ")
        self.train_metric_logger.add_meter('lr', util.SmoothedValue(window_size=1, fmt='{value:.6f}'))
        self.iter_skip = self.iter_start % len(self.train_loader)
        self.it = self.iter_start
        self.skip_dis = False
        if not opt.resume: 
            self.best_val = np.inf
            self.best_ep = 1
        # training
        # evaluate first before training
        if self.iter_start == 0 and not opt.debug: self.evaluate(opt, ep=0, training=True)
        for self.ep in range(self.epoch_start, opt.max_epoch):
            self.train_epoch(opt)
        # after training
        if opt.device == 0: self.save_checkpoint(opt, ep=self.ep, it=self.it, best_val=self.best_val, best_ep=self.best_ep)
        if opt.tb and opt.device == 0:
            self.tb.flush()
            self.tb.close()
            wandb.finish()
        if opt.device == 0: 
            print("TRAINING DONE")
            print("Best CD: %.4f @ epoch %d" % (self.best_val, self.best_ep))
        cleanup()
    
    def train_epoch(self, opt):
        # before train epoch
        self.train_loader.sampler.set_epoch(self.ep)
        if opt.device == 0:
            print("training epoch {}".format(self.ep+1))
        batch_progress = range(self.num_batches)
        self.graph.train()
        # train epoch
        loader = iter(self.train_loader)

        
        if opt.debug and opt.profile:
            with torch.profiler.profile(
                    schedule=torch.profiler.schedule(wait=3, warmup=3, active=5, repeat=2),
                    on_trace_ready=torch.profiler.tensorboard_trace_handler('debug/profiler_log'),
                    record_shapes=True,
                    profile_memory=True,
                    with_stack=False
            ) as prof:
                for batch_id in batch_progress:
                    if batch_id >= (1 + 1 + 3) * 2:
                        # exit the program after 2 iterations of the warmup, active, and repeat steps
                        exit()
                    
                    # if resuming from previous checkpoint, skip until the last iteration number is reached
                    if self.iter_skip>0:
                        self.iter_skip -= 1
                        continue
                    batch = next(loader)
                    # train iteration
                    var = edict(batch)
                    opt.H, opt.W = opt.image_size
                    var = util.move_to_device(var, opt.device)
                    loss = self.train_iteration(opt, var, batch_progress)
                    prof.step()
        else:
            for batch_id in batch_progress:
                # if resuming from previous checkpoint, skip until the last iteration number is reached
                if self.iter_skip>0:
                    self.iter_skip -= 1
                    continue
                batch = next(loader)
                # train iteration
                var = edict(batch)
                opt.H, opt.W = opt.image_size
                var = util.move_to_device(var, opt.device)
                loss = self.train_iteration(opt, var, batch_progress)
            
        # after train epoch
        if opt.optim.sched: self.sched.step()
        if (self.ep + 1) % opt.freq.eval == 0:
            if opt.device == 0: print("validating epoch {}".format(self.ep+1))
            current_val = self.evaluate(opt, ep=self.ep+1, training=True)
            if current_val < self.best_val and opt.device == 0:
                self.best_val = current_val
                self.best_ep = self.ep + 1
                self.save_checkpoint(opt, ep=self.ep, it=self.it, best_val=self.best_val, best_ep=self.best_ep, best=True, latest=True)

    def train_iteration(self, opt, var, batch_progress):
        # before train iteration
        torch.distributed.barrier()
        # train iteration
        with torch.autocast(device_type='cuda', dtype=torch.float16, enabled=opt.optim.amp):
            var, loss = self.graph.forward(opt, var, training=True, get_loss=True)
            loss = self.summarize_loss(opt, var, loss)
            loss_scaled = loss.all / opt.optim.accum
            
        # backward
        if opt.optim.amp:
            self.scaler.scale(loss_scaled).backward()
            # skip update if accumulating gradient
            if (self.it + 1) % opt.optim.accum == 0:
                self.scaler.unscale_(self.optim)
                # gradient clipping
                if opt.optim.clip_norm:
                    norm = torch.nn.utils.clip_grad_norm_(self.graph.parameters(), opt.optim.clip_norm)
                    if opt.debug: print("Grad norm: {}".format(norm))
                self.scaler.step(self.optim)
                self.scaler.update()
                self.optim.zero_grad()
        else:
            loss_scaled.backward()
            if (self.it + 1) % opt.optim.accum == 0:
                if opt.optim.clip_norm:
                    norm = torch.nn.utils.clip_grad_norm_(self.graph.parameters(), opt.optim.clip_norm)
                    if opt.debug: print("Grad norm: {}".format(norm))
                self.optim.step()
                self.optim.zero_grad()
            
        # after train iteration
        lr = self.sched.get_last_lr()[0] if opt.optim.sched else opt.optim.lr
        self.train_metric_logger.update(lr=lr)
        self.train_metric_logger.update(loss=loss.all)
        self.train_metric_logger.update(**{key: loss[key].mean() for key in loss if key != "all"})

        if opt.device == 0: 
            self.graph.eval()
            if (self.it) % opt.freq.ckpt_latest == 0 and not opt.debug: 
                self.save_checkpoint(opt, ep=self.ep, it=self.it, best_val=self.best_val, best_ep=self.best_ep, latest=True)
            if (self.it) % opt.freq.scalar == 0 and not opt.debug: 
                self.log_scalars(opt, var, loss, step=self.it * opt.batch_size, split="train")
            if (self.it) % (opt.freq.save_vis * (self.it//10000*10+1)) == 0 and not opt.debug: 
                pass
            if (self.it) % opt.freq.print == 0:
                total_iterations = self.num_batches * opt.max_epoch
                current_time = datetime.datetime.now().time()
                if self.last_iteration_time is None:
                    self.last_iteration_time = time.time()
                    print(f'[{current_time}] Train Iter {self.it}/{total_iterations}: {self.train_metric_logger}')
                else:
                    n_iter_time = time.time() - self.last_iteration_time
                    self.iteration_durations.append(n_iter_time)
                    self.last_iteration_time = time.time()
                    average_duration = sum(self.iteration_durations) / len(self.iteration_durations)
                    estimated_total_time = average_duration * total_iterations / opt.freq.print
                    time_left = estimated_total_time - (average_duration * self.it / opt.freq.print)
                    time_left = datetime.timedelta(seconds=int(time_left))
                    print(f'[{current_time}] Time left: {time_left} - Train Iter {self.it}/{total_iterations}: {self.train_metric_logger}')
            self.graph.train()
        self.it += 1
        return loss
        
    @torch.no_grad()
    def vis_train_iter(self, opt):
        for i in range(len(self.viz_data)):
            var_viz = edict(deepcopy(self.viz_data[i]))
            var_viz = util.move_to_device(var_viz, opt.device)
            var_viz = self.graph.module(opt, var_viz, training=False, get_loss=False)
            eval_3D.eval_metrics(opt, var_viz, vis_only=True)
            vis_folder = "vis_log/iter_{}".format(self.it)
            os.makedirs("{}/{}".format(opt.output_path, vis_folder), exist_ok=True)
            util_vis.dump_images(opt, var_viz.idx, "image_input", var_viz.rgb_input_map, masks=None, from_range=(0, 1), folder=vis_folder)
            util_vis.dump_images(opt, var_viz.idx, "mask_input", var_viz.mask_input_map, folder=vis_folder)
            util_vis.dump_meshes_viz(opt, var_viz.idx, "mesh_viz", var_viz.mesh_pred, folder=vis_folder)
            if 'depth_pred' in var_viz:
                util_vis.dump_depths(opt, var_viz.idx, "depth_est", var_viz.depth_pred, var_viz.mask_input_map, rescale=True, folder=vis_folder)
            if 'depth_input_map' in var_viz:
                util_vis.dump_depths(opt, var_viz.idx, "depth_input", var_viz.depth_input_map, var_viz.mask_input_map, rescale=True, folder=vis_folder)
            if 'attn_vis' in var_viz:
                util_vis.dump_attentions(opt, var_viz.idx, "attn", var_viz.attn_vis, folder=vis_folder)
            if 'gt_surf_points' in var_viz and 'seen_points' in var_viz:
                util_vis.dump_pointclouds_compare(opt, var_viz.idx, "seen_surface", var_viz.seen_points, var_viz.gt_surf_points, folder=vis_folder)

    def summarize_loss(self, opt, var, loss, non_act_loss_key=[]):
        loss_all = 0.
        assert("all" not in loss)
        # weigh losses
        for key in loss:
            assert(key in opt.loss_weight)
            if opt.loss_weight[key] is not None:
                # assert not torch.isinf(loss[key].mean()), "loss {} is Inf".format(key)
                # assert not torch.isnan(loss[key].mean()), "loss {} is NaN".format(key)
                if torch.isinf(loss[key].mean()):
                    print("loss {} is Inf".format(key))
                    continue
                elif torch.isnan(loss[key].mean()):
                    print("loss {} is NaN".format(key))
                    continue
                loss_all += float(opt.loss_weight[key])*loss[key].mean() if key not in non_act_loss_key else 0.0*loss[key].mean()
        loss.update(all=loss_all)
        return loss

# =================================================== set up evaluation =========================================================

    def aggregate_symm_eval_metric_dicts(self, symm_eval_metric_dicts):
        result = {}
        for key, value in symm_eval_metric_dicts[0].items():
            result[key] = value
        for dict in symm_eval_metric_dicts[1:]:
            for key, value in dict.items():
                result[key] += value
        return result
    
    def gather_symm_eval_metrics(self, symm_eval, opt):
        for key, value in symm_eval.items():
            # print(key, torch.distributed.get_rank())
            try:
                torch.distributed.all_reduce(symm_eval[key], op=torch.distributed.ReduceOp.SUM)
            except:
                print("Error in gather_symm_eval_metrics")
                print(key)
                print(value)
                print(torch.distributed.get_rank())
                raise
        return symm_eval
    
    def get_batch_symm_dict(self, opt, var):
        with torch.no_grad():
            symm_dict = {}
            batch_size = var.idx.shape[0]
            offsets = var.symm_outputs['offset']
            normals = var.symm_outputs['normal']
            probs = var.symm_outputs['cls_logits'].sigmoid()
            gt_center = var.symm_targets['center_coords']
            gt_normal = var.symm_targets['gt_normal_normalized']
            num_gt = var.symm_targets['num_actual_gt']
            for b in range(batch_size):
                key = var.idx[b].item()
                symm_dict[key] = {}
                symm_dict[key]['offset'] = offsets[b].detach().cpu().numpy().tolist()
                symm_dict[key]['normal'] = normals[b].detach().cpu().numpy().tolist()
                symm_dict[key]['probs'] = probs[b].detach().cpu().numpy().tolist()
                symm_dict[key]['gt_center'] = gt_center[b].detach().cpu().numpy().tolist()
                symm_dict[key]['gt_normal'] = gt_normal[b].detach().cpu().numpy().tolist()
                symm_dict[key]['num_gt'] = num_gt[b].detach().cpu().item()

            return symm_dict
        
    def get_batch_refined_dict(self, opt, var, shape_results, batch_symm_dict):
        with torch.no_grad():
            refined_dict = {}
            key = var.idx[0].item()
            refined_dict[key] = {}
            for k,v in shape_results.items():
                refined_dict[key][k] = v
            for k,v in batch_symm_dict[key].items():
                refined_dict[key][k] = v


            refined_results = eval_3D.get_refined_dicts(opt, var)
            parsed_refined_results = eval_3D.parse_refined_results(opt, refined_results)
            for k,v in parsed_refined_results.items():
                refined_dict[key][k] = v

            return refined_dict

    @torch.no_grad()
    def evaluate(self, opt, ep, training=False):
        self.graph.eval()
        
        # lists for metrics
        symm_eval_metric_dicts = []
        loss_eval = edict()
        metric_eval = dict(dist_acc=0., dist_cov=0.)
        eval_metric_logger = util.MetricLogger(delimiter="  ")

        save_symm_dict = {}
        save_refined_dict = {}
        
        # result file on the fly
        if not training and not opt.eval.save_raw_symm_only and not opt.eval.save_refined_only: 
            assert opt.device == 0
            full_results_file = open(os.path.join(opt.output_path, '{}_full_results.txt'.format(opt.data.dataset_test)), 'w')
            full_results_file.write("IND,CD,ACC,COMP,")
            full_results_file.write(",".join(["F-score@{:.2f}".format(opt.eval.f_thresholds[i]*100) for i in range(len(opt.eval.f_thresholds))])) 
        
        # dataloader on the test set
        with torch.cuda.device(opt.device):
            for it, batch in enumerate(self.test_loader):
                
                # inference the model
                var = edict(batch)
                var = self.evaluate_batch(opt, var, ep, it, single_gpu=False)

                if opt.eval.also_save_raw_symm:
                    batch_symm_dict = self.get_batch_symm_dict(opt, var)
                    save_symm_dict.update(batch_symm_dict)

                if opt.eval.save_raw_symm_only:
                    if opt.device == 0 and it % opt.freq.print_eval == 0: 
                        print('[{}] '.format(datetime.datetime.now().time()), end='')
                        print(f'Eval Iter {it}/{len(self.test_loader)} @ EP {ep}: {eval_metric_logger}')

                    batch_symm_dict = self.get_batch_symm_dict(opt, var)
                    save_symm_dict.update(batch_symm_dict)
                    continue

                if opt.eval.save_refined_only: # need to be false cause no shape to refine with
                    # only works with bs = 1
                    assert opt.eval.icp == False
                    assert opt.eval.flip_mesh == False
                    if opt.device == 0 and it % opt.freq.print_eval == 0: 
                        print('[{}] '.format(datetime.datetime.now().time()), end='')
                        print(f'Eval Iter {it}/{len(self.test_loader)} @ EP {ep}: {eval_metric_logger}')


                    dist_acc, dist_cov = eval_3D.eval_metrics(opt, var)
                    batch_symm_dict = self.get_batch_symm_dict(opt, var)
                    continue


                # record CD for evaluation
                dist_acc, dist_cov = eval_3D.eval_metrics(opt, var)
                f1, geo, center_dist, symm_f = eval_3D.eval_symm(opt, var)
                
                # accumulate the scores
                symm_eval_metric_dicts.append(var.symm_eval_metric)
                eval_metric_logger.update(ACC=dist_acc)
                eval_metric_logger.update(COMP=dist_cov)
                eval_metric_logger.update(CD=(dist_acc+dist_cov) / 2)
                eval_metric_logger.update(F1=f1)
                eval_metric_logger.update(GEO=geo)
                eval_metric_logger.update(CENTER=center_dist)
                eval_metric_logger.update(SYMM_FAT5=symm_f[1]) # TODO: fix
                
                if opt.device == 0 and it % opt.freq.print_eval == 0: 
                    print('[{}] '.format(datetime.datetime.now().time()), end='')
                    print(f'Eval Iter {it}/{len(self.test_loader)} @ EP {ep}: {eval_metric_logger}')

            if opt.eval.save_raw_symm_only or opt.eval.also_save_raw_symm:
                # save save_symm_dict to a json file
                print("saving save_symm_dict to json file")
                save_symm_dict_file = os.path.join(opt.output_path, '{}_symm_results.json'.format(opt.data.dataset_test))
                with open(save_symm_dict_file, 'w') as f:
                    json.dump(save_symm_dict, f)
                return float('inf')
            if opt.eval.save_refined_only:
                # save save_refined_dict to a json file
                print("saving save_refined_dict to json file")
                save_refined_dict_file = os.path.join(opt.output_path, '{}_refined_results.json'.format(opt.data.dataset_test))
                with open(save_refined_dict_file, 'w') as f:
                    json.dump(save_refined_dict, f)
                return float('inf')

            # collect the eval results into tensors
            symm_eval = self.aggregate_symm_eval_metric_dicts(symm_eval_metric_dicts)

        if opt.world_size > 1:
            # gather the metrics
            torch.distributed.barrier()
            symm_eval_all = self.gather_symm_eval_metrics(symm_eval, opt)
            torch.distributed.barrier()
        else:
            symm_eval_all = symm_eval
        # handle last batch, if any
        if len(self.test_loader.sampler) * opt.world_size < len(self.test_data):
            symm_eval_all = [symm_eval_all]
            for batch in tqdm(self.aux_test_loader):
                # inference the model
                var = edict(batch)
                var = self.evaluate_batch(opt, var, ep, it, single_gpu=False)

                # record CD for evaluation
                dist_acc, dist_cov = eval_3D.eval_metrics(opt, var)
                f1, geo, center_dist, symm_f = eval_3D.eval_symm(opt, var)
                # accumulate the scores
                symm_eval_all.append(var.symm_eval_metric)
                    
            symm_eval_all = self.aggregate_symm_eval_metric_dicts(symm_eval_all)

        if not training: 
            full_results_file.close()
        # printout and save the metrics     
        if opt.device == 0:
            symm_eval_final = eval_3D.generate_symm_eval(symm_eval_all, opt)
            eval_3D.print_symm_eval(symm_eval_final, opt)
        
            if training:
                # log/visualize results to tb/vis
                self.log_scalars(opt, var, loss_eval, metric=metric_eval, step=self.it * opt.batch_size, split="eval")
            
            return float('inf')
        return float('inf')

    def evaluate_batch(self, opt, var, ep=None, it=None, single_gpu=False):
        var = util.move_to_device(var, opt.device)
        if single_gpu:
            var  = self.graph.module(opt, var, training=False, get_loss=False)
        else:
            var = self.graph(opt, var, training=False, get_loss=False)
        return var

    @torch.no_grad()
    def log_scalars(self, opt, var, loss, metric=None, step=0, split="train"):
        if split=="train":
            dist_acc, dist_cov = eval_3D.eval_metrics(opt, var)
            metric = dict(dist_acc=dist_acc, dist_cov=dist_cov)
        for key, value in loss.items():
            if key=="all": continue
            self.tb.add_scalar("{0}/loss_{1}".format(split, key), value.mean(), step)
            wandb.log({"{0}/loss_{1}".format(split, key): value.mean()}, step=step)
        if metric is not None:
            for key, value in metric.items():
                self.tb.add_scalar("{0}/{1}".format(split, key), value, step)
                wandb.log({"{0}/{1}".format(split, key): value}, step=step)
        # log the attention average values
        if 'attn_geo_avg' in var:
            self.tb.add_scalar("{0}/attn_geo_avg".format(split), var.attn_geo_avg, step)
            wandb.log({"{0}/attn_geo_avg".format(split): var.attn_geo_avg}, step=step)
        if 'attn_geo_seen' in var:
            self.tb.add_scalar("{0}/attn_geo_seen".format(split), var.attn_geo_seen, step)
        if 'attn_geo_occl' in var:
            self.tb.add_scalar("{0}/attn_geo_occl".format(split), var.attn_geo_occl, step)
        if 'attn_geo_bg' in var:
            self.tb.add_scalar("{0}/attn_geo_bg".format(split), var.attn_geo_bg, step)

    @torch.no_grad()
    def visualize(self, opt, var, step=0, split="train"):
        if 'pose_input' in var:
            pose_input = var.pose_input
        elif 'pose_gt' in var:
            pose_input = var.pose_gt
        else:
            pose_input = None
        util_vis.tb_image(opt, self.tb, step, split, "image_input_map", var.rgb_input_map, masks=None, from_range=(0, 1), poses=pose_input)
        util_vis.tb_image(opt, self.tb, step, split, "image_input_map_est", var.rgb_input_map, masks=None, from_range=(0, 1), 
                          poses=var.pose_pred if 'pose_pred' in var else var.pose)
        util_vis.tb_image(opt, self.tb, step, split, "mask_input_map", var.mask_input_map)
        if 'depth_pred' in var:
            util_vis.tb_image(opt, self.tb, step, split, "depth_est_map", var.depth_pred)
        if 'depth_input_map' in var:
            util_vis.tb_image(opt, self.tb, step, split, "depth_input_map", var.depth_input_map)

    @torch.no_grad()
    def dump_results(self, opt, var, ep, write_new=False, train=False):
        # create the dir
        current_folder = "dump_{}".format(opt.data.dataset_test) if train == False else "vis_{}".format(ep)
        os.makedirs("{}/{}/".format(opt.output_path, current_folder), exist_ok=True)
        
        # save the results
        if 'pose_input' in var:
            pose_input = var.pose_input
        elif 'pose_gt' in var:
            pose_input = var.pose_gt
        else:
            pose_input = None
        util_vis.dump_images(opt, var.idx, "image_input", var.rgb_input_map, masks=None, from_range=(0, 1), poses=pose_input, folder=current_folder)
        util_vis.dump_images(opt, var.idx, "mask_input", var.mask_input_map, folder=current_folder)
        util_vis.dump_meshes(opt, var.idx, "mesh", var.mesh_pred, folder=current_folder)
        util_vis.dump_meshes_viz(opt, var.idx, "mesh_viz", var.mesh_pred, folder=current_folder) # image frames + gifs
        if 'depth_pred' in var:
            util_vis.dump_depths(opt, var.idx, "depth_est", var.depth_pred, var.mask_input_map, rescale=True, folder=current_folder)
        if 'depth_input_map' in var:
            util_vis.dump_depths(opt, var.idx, "depth_input", var.depth_input_map, var.mask_input_map, rescale=True, folder=current_folder)
        if 'gt_surf_points' in var and 'seen_points' in var:
            util_vis.dump_pointclouds_compare(opt, var.idx, "seen_surface", var.seen_points, var.gt_surf_points, folder=current_folder)
        if 'attn_vis' in var:
            util_vis.dump_attentions(opt, var.idx, "attn", var.attn_vis, folder=current_folder)
        if 'attn_pc' in var:
            util_vis.dump_pointclouds(opt, var.idx, "attn_pc", var.attn_pc["points"], var.attn_pc["colors"], folder=current_folder)
        if 'dpc' in var:
            util_vis.dump_pointclouds_compare(opt, var.idx, "pointclouds_comp", var.dpc_pred, var.dpc.points, folder=current_folder)

    def save_checkpoint(self, opt, ep=0, it=0, best_val=np.inf, best_ep=1, latest=False, best=False):
        util.save_checkpoint(opt, self, ep=ep, it=it, best_val=best_val, best_ep=best_ep, latest=latest, best=best)
        if not latest:
            print("checkpoint saved: ({0}) {1}, epoch {2} (iteration {3})".format(opt.group, opt.name, ep, it))
        if best:
            print("Saving the current model as the best...")
