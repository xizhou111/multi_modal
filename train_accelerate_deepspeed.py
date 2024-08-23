import json
import os
import logging
import math

import torch
import torch.optim as optim
from torch.utils.data import DataLoader

from argparse import ArgumentParser

# from accelerate import Accelerator
from accelerator import MyAccelerator as Accelerator
from accelerate import DistributedDataParallelKwargs
from accelerate.utils import DummyOptim, DummyScheduler

from tensorboardX import SummaryWriter

from transformers import get_scheduler, set_seed
from transformers.trainer_utils import SchedulerType

from dataset import MultiModelDataset
from blip2_models.blip2_qformer import Blip2Qformer

from tqdm.auto import tqdm



def evaluate(model, val_loader, accelerator):
    model.eval()
    accelerator.print("******************************")
    accelerator.print("***** Running evaluation *****")
    with torch.no_grad():
        val_image_ids_list = []
        val_text_features_list = []
        val_image_features_list = []
        for val_batch in tqdm(val_loader, desc="extracting val data features", disable=not accelerator.is_main_process):
            text_features = model.module.extract_features(val_batch, mode='text')['text_embeds_proj'][:,0,:].cpu()
            image_features = model.module.extract_features(val_batch, mode='image')['image_embeds_proj'].mean(dim=1).cpu()

            val_image_ids_list.extend(val_batch['image_id'])
            val_text_features_list.append(text_features)
            val_image_features_list.append(image_features)

        val_text_features = torch.cat(val_text_features_list, dim=0) # [num_samples, 512]
        val_image_features = torch.cat(val_image_features_list, dim=0) # [num_samples, 512]

        # 计算i2t的相似度
        sim_i2t = torch.matmul(val_image_features, val_text_features.t()) # [num_samples, num_samples]

        # 计算i2t的召回率
        recalls_i2t = {
            "r1": 0,
            "r3": 0,
            "r5": 0,
            "r10": 0,
            "r50": 0,
            "r70": 0,
            "r100": 0,
            "r300": 0,
        }
        for i, sim in tqdm(enumerate(sim_i2t), desc="Computing i2t recall", total=len(val_image_ids_list), disable=not accelerator.is_main_process):
            topk_sim, topk_indices = sim.topk(k=300)
            for k in [1, 3, 5, 10, 50, 70, 100, 300]:
                if val_image_ids_list[i] in [val_image_ids_list[idx] for idx in topk_indices[:k]]:
                    recalls_i2t[f"r{k}"] += 1

        for k in [1, 3, 5, 10, 50, 70, 100, 300]:
            recalls_i2t[f"r{k}"] /= len(val_image_ids_list)
            
    return recalls_i2t
            

def main(args):
    # 加载数据
    with open(args.train_data_path, 'r') as f:
        train_data = json.load(f)
    with open(args.val_data_path, 'r') as f:
        val_data = json.load(f)

    train_dataset = MultiModelDataset(train_data)
    val_dataset = MultiModelDataset(val_data)

    per_device_train_batch_size = args.per_device_train_batch_size
    train_loader = DataLoader(train_dataset, batch_size=per_device_train_batch_size, shuffle=True, num_workers=4, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=2*per_device_train_batch_size, shuffle=False, num_workers=4, pin_memory=True)

    # 设置随机种子
    set_seed(args.seed)

    # 加载模型
    model = Blip2Qformer(
        vit_model=args.vit_model_name,
        vit_precision=args.vit_precision,
        freeze_vit=args.freeze_vit,
    )

    # 初始化accelerator
    ddp_kwargs = DistributedDataParallelKwargs(find_unused_parameters=True)
    accelerator = Accelerator(kwargs_handlers=[ddp_kwargs], 
                              gradient_accumulation_steps=args.gradient_accumulation_steps,
                              mixed_precision=args.mixed_precision,
                              )

    # 定义优化器
    optimizer_cls = (
        optim.AdamW
        if accelerator.state.deepspeed_plugin is None
        or "optimizer" not in accelerator.state.deepspeed_plugin.deepspeed_config
        else DummyOptim
    )
    optimizer = optimizer_cls(model.parameters(), lr=args.lr)

    num_update_steps_per_epoch = math.ceil(len(train_loader) / accelerator.gradient_accumulation_steps)
    overrode_max_train_steps = False
    if args.max_train_steps is None:
        args.max_train_steps = args.num_train_epochs * num_update_steps_per_epoch
        overrode_max_train_steps = True
    else:
        args.num_train_epochs = math.ceil(args.max_train_steps / num_update_steps_per_epoch)

    # 定义学习率调度器
    if (
        accelerator.state.deepspeed_plugin is None
        or "scheduler" not in accelerator.state.deepspeed_plugin.deepspeed_config
    ):
        lr_scheduler = get_scheduler(
            name=args.lr_scheduler_type,
            optimizer=optimizer,
            num_warmup_steps=args.num_warmup_rate * args.max_train_steps,
            num_training_steps=args.max_train_steps,
        )
    else:
        lr_scheduler = DummyScheduler(
            optimizer=optimizer,
            num_warmup_steps=args.num_warmup_rate * args.max_train_steps,
            num_training_steps=args.max_train_steps,
        )
        
    # 准备accelerator
    model, optimizer, train_loader, lr_scheduler = accelerator.prepare(
        model, optimizer, train_loader, lr_scheduler
    )

    # 重新计算max_train_steps
    num_update_steps_per_epoch = math.ceil(len(train_loader) / accelerator.gradient_accumulation_steps)
    if overrode_max_train_steps:
        args.max_train_steps = args.num_train_epochs * num_update_steps_per_epoch
    # Afterwards we recalculate our number of training epochs
    args.num_train_epochs = math.ceil(args.max_train_steps / num_update_steps_per_epoch)

    # Training!
    total_batch_size = args.per_device_train_batch_size * accelerator.num_processes * accelerator.gradient_accumulation_steps

    accelerator.print("***** Running training *****")
    accelerator.print(f"Num examples = {len(train_dataset)}")
    accelerator.print(f"Num Epochs = {args.num_train_epochs}")
    accelerator.print(f"Instantaneous batch size per device = {args.per_device_train_batch_size}")
    accelerator.print(f"Total train batch size (w. parallel, distributed & accumulation) = {total_batch_size}")
    accelerator.print(f"Gradient Accumulation steps = {accelerator.gradient_accumulation_steps}")
    accelerator.print(f"Total optimization steps = {args.max_train_steps}")
    accelerator.print(f"LR Scheduler = {accelerator._schedulers}")
    accelerator.print(f"Optimizer = {optimizer}")
    accelerator.print(f"Accelerator state from the current environment:\n{accelerator.state}")
    accelerator.print(f"DeepSpeed Engine = {accelerator.state.deepspeed_plugin}")


    date = os.popen('date +"%Y-%m-%d-%H-%M-%S"').read().strip()
    log_dir = os.path.join(args.log_dir, date)
    if accelerator.is_main_process:
        os.makedirs(log_dir, exist_ok=True)
        tf_writer = SummaryWriter(log_dir=log_dir)

    train_bar = tqdm(range(args.max_train_steps), desc="Training", leave=False, disable=not accelerator.is_main_process)
    overall_step = 0
    start_epoch = 0

    if args.resume_from_checkpoint:
        accelerator.load_state(args.resume_from_checkpoint, strict=False)
        accelerator.print(f"Resume from checkpoint: {args.resume_from_checkpoint}")

        path = os.path.basename(args.resume_from_checkpoint)
        resume_step = int(path.split('-')[-1])
        start_epoch = resume_step // num_update_steps_per_epoch
        resume_step -= start_epoch * num_update_steps_per_epoch

        train_bar.update(resume_step)

    # 开始训练
    for epoch in range(start_epoch, args.num_train_epochs):
        model.train()
        if args.resume_from_checkpoint and epoch == start_epoch and resume_step is not None:
            activate_dataloader = accelerator.skip_first_batches(train_loader, resume_step)
            overall_step += resume_step
        else:
            activate_dataloader = train_loader

        for step, batch in enumerate(activate_dataloader):

            with accelerator.accumulate(model):

                output = model(batch)
                loss = output['loss']
                loss_itc = output['loss_itc']
                loss_itm = output['loss_itm']

                accelerator.backward(loss)

                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad()

                loss_reduced = accelerator.gather(loss).mean().float()
                loss_itc_reduced = accelerator.gather(loss_itc).mean().float()
                loss_itm_reduced = accelerator.gather(loss_itm).mean().float()
                
                if accelerator.sync_gradients:
                    train_bar.update(1)
                    train_bar.set_postfix({"loss": loss_reduced.item(), "loss_itc": loss_itc_reduced.item(), "loss_itm": loss_itm_reduced.item()})
                    overall_step += 1

                    if accelerator.is_main_process:
                        if overall_step % args.log_steps == 0:
                            train_bar.write(f"Epoch {epoch}, Step {overall_step}, lr: {optimizer.param_groups[0]['lr']}, loss: {loss_reduced.item()}, loss_itc: {loss_itc_reduced.item()}, loss_itm: {loss_itm_reduced.item()}")
                    
                        tf_writer.add_scalar("train/loss", loss_reduced, overall_step)
                        tf_writer.add_scalar("train/loss_itc", loss_itc_reduced, step)
                        tf_writer.add_scalar("train/loss_itm", loss_itm_reduced, overall_step)
                        tf_writer.add_scalar("train/learning_rate", optimizer.param_groups[0]['lr'], overall_step)

                if overall_step % args.checkpoint_steps == 0:
                    # 保存模型以及状态
                    accelerator.wait_for_everyone()
                    output_dir = os.path.join(args.output_dir, f"checkpoint-{overall_step}")
                    accelerator.save_model(model, output_dir)
                    accelerator.save_state(output_dir)
                    accelerator.print(f"Save model to {output_dir}")

                    # 验证
                    recalls_i2t = evaluate(model, val_loader, accelerator)

                    if accelerator.is_main_process:
                        for k, v in recalls_i2t.items():
                            tf_writer.add_scalar(f"val/i2t_recall@{k}", v, overall_step)
                        accelerator.print(recalls_i2t)

                    accelerator.wait_for_everyone()


    # 训练结束，保存模型
    accelerator.wait_for_everyone()
    output_dir = os.path.join(args.output_dir, "final_checkpoint")
    accelerator.save_model(model, output_dir)
    accelerator.save_state(output_dir)

    # 验证
    recalls_i2t = evaluate(model, val_loader, accelerator)
    if accelerator.is_main_process:
        for k, v in recalls_i2t.items():
            tf_writer.add_scalar(f"val/i2t_recall@{k}", v, overall_step)
        accelerator.print(recalls_i2t)
            


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument("--log_dir", type=str, required=True, help="log_dir")
    parser.add_argument("--output_dir", type=str, required=True, help="output_dir")

    parser.add_argument("--num_train_epochs", type=int, default=2, help="num_train_epochs")
    parser.add_argument("--per_device_train_batch_size", type=int, default=8, help="per_device_train_batch_size")
    parser.add_argument( "--max_train_steps", type=int, default=None, help="Total number of training steps to perform. If provided, overrides num_train_epochs.")
    parser.add_argument("--log_steps", type=int, default=1, help="how many steps to log once")
    parser.add_argument("--lr", type=float, default=2e-6, help="learning_rate") # 2e-5/image_ids: 5e-6
    parser.add_argument("--seed", type=int, default=42, help="seed")
    parser.add_argument("--lr_scheduler_type", type=SchedulerType,
        default="cosine",
        help="The scheduler type to use.",
        choices=["linear", "cosine", "cosine_with_restarts", "polynomial", "constant", "constant_with_warmup"],
    )
    parser.add_argument("--num_warmup_rate", type=float, default=0.1, help="num_warmup_steps")

    parser.add_argument("--train_data_path", type=str, default='/mnt/cfs/ssw/zcl/multi_modal/data/new/blip2_data_train.json', help="train_data_path")
    parser.add_argument("--val_data_path", type=str, default='/mnt/cfs/ssw/zcl/multi_modal/data/new/blip2_data_val.json', help="val_data_path")

    parser.add_argument("--vit_model_name", type=str, default='eva_clip_g', help="vit_model_name")
    parser.add_argument("--vit_precision", type=str, default='fp32', help="vit_precision")
    parser.add_argument("--freeze_vit", type=bool, default=False, help="freeze_vit")

    parser.add_argument(
        "--mixed_precision",
        type=str,
        default=None,
        choices=["no", "fp16", "bf16", "fp8"],
        help="Whether to use mixed precision. Choose between fp16 and bf16 (bfloat16). Bf16 requires PyTorch >= 1.10. and an Nvidia Ampere GPU.",
    )

    parser.add_argument("--checkpoint_steps", type=int, default=1000, help="checkpoint_steps")
    parser.add_argument("--gradient_accumulation_steps", type=int, default=1, help="gradient_accumulation_steps")


    parser.add_argument("--resume_from_checkpoint", type=str, default=None, help="resume_from_checkpoint")

    args = parser.parse_args()

    main(args)