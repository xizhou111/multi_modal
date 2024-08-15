import json
import os
import torch
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader

from argparse import ArgumentParser

# from accelerate import Accelerator
from accelerator import MyAccelerator as Accelerator
from accelerate import DistributedDataParallelKwargs
from torch.cuda.amp import GradScaler

from tensorboardX import SummaryWriter

from transformers import get_scheduler, set_seed
from transformers.trainer_utils import SchedulerType

from dataset import MultiModelDataset
from blip2_models.blip2_qformer import Blip2Qformer

from tqdm.auto import tqdm

def main(args):
    # 加载数据
    with open(args.train_data_path, 'r') as f:
        train_data = json.load(f)
    with open(args.val_data_path, 'r') as f:
        val_data = json.load(f)
    with open(args.tiku_data_path, 'r') as f:
        tiku_data = json.load(f)

    train_dataset = MultiModelDataset(train_data)
    val_dataset = MultiModelDataset(val_data)
    tiku_dataset = MultiModelDataset(tiku_data)

    batch_size = args.batch_size
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=2*batch_size, shuffle=False, num_workers=4, pin_memory=True)
    tiku_loader = DataLoader(tiku_dataset, batch_size=2*batch_size, shuffle=False, num_workers=4, pin_memory=True)

    # 设置随机种子
    set_seed(args.seed)

    # 加载模型
    model = Blip2Qformer(
        vit_model=args.vit_model_name,
    )

    # 定义优化器
    optimizer = optim.AdamW(model.parameters(), lr=args.lr) # image_ids

    # 初始化加速器
    # accelerator = Accelerator(log_with="tensorboard")
    ddp_kwargs = DistributedDataParallelKwargs(find_unused_parameters=True)
    accelerator = Accelerator(kwargs_handlers=[ddp_kwargs])

    num_epoch = args.num_epoch
    num_training_steps = num_epoch * len(train_loader)
    lr_scheduler = get_scheduler(
        SchedulerType.COSINE,
        optimizer=optimizer,
        num_warmup_steps=args.num_warmup_rate * num_training_steps,
        num_training_steps=num_training_steps
    )

    model, optimizer, train_loader, lr_scheduler = accelerator.prepare(
        model, optimizer, train_loader, lr_scheduler
    )

    date = os.popen('date +"%Y-%m-%d-%H-%M-%S"').read().strip()
    log_dir = os.path.join(args.log_dir, date)

    if accelerator.is_main_process:
        if not os.path.exists(log_dir):
            os.makedirs(log_dir)
    accelerator.wait_for_everyone()

    tf_writer = SummaryWriter(log_dir=log_dir)

    num_training_steps = num_epoch * len(train_loader)
    train_bar = tqdm(range(num_training_steps), desc="Training", leave=False, disable=not accelerator.is_main_process)

    overall_step = 0
    start_epoch = 0

    if args.resume_from_checkpoint:
        accelerator.print(f"Resume from checkpoint: {args.resume_from_checkpoint}")
        accelerator.load_state(args.resume_from_checkpoint, strict=False)

        path = os.path.basename(args.resume_from_checkpoint)
        resume_step = int(path.split('-')[-1])
        start_epoch = resume_step // len(train_loader)
        resume_step -= start_epoch * len(train_loader)

        train_bar = tqdm(range(resume_step, num_training_steps), desc="Training", leave=False, disable=not accelerator.is_main_process)


    # 开始训练
    for epoch in range(start_epoch, num_epoch):
        model.train()
        if args.resume_from_checkpoint and epoch == start_epoch and resume_step is not None:
            activate_dataloader = accelerator.skip_first_batches(train_loader, resume_step)
            overall_step += resume_step
        else:
            activate_dataloader = train_loader

        for step, batch in enumerate(activate_dataloader):
            # step = i + epoch * len(train_loader) + 1

            output = model(batch)
            loss = output['loss']
            loss_itc = output['loss_itc']
            loss_itm = output['loss_itm']

            accelerator.backward(loss)

            optimizer.step()
            lr_scheduler.step()
            optimizer.zero_grad()

            loss_reduced = accelerator.gather(loss).mean()
            loss_itc_reduced = accelerator.gather(loss_itc).mean()
            loss_itm_reduced = accelerator.gather(loss_itm).mean()

            overall_step += 1

            if accelerator.is_main_process:
                train_bar.update(1)
                train_bar.set_postfix({"loss": loss_reduced.item(), "loss_itc": loss_itc_reduced.item(), "loss_itm": loss_itm_reduced.item()})
                if overall_step % 5 == 0:
                    train_bar.write(f"Epoch {epoch}, Step {overall_step}, lr: {optimizer.param_groups[0]['lr']}, loss: {loss_reduced.item()}, loss_itc: {loss_itc_reduced.item()}, loss_itm: {loss_itm_reduced.item()}")

                tf_writer.add_scalar("train/loss", loss_reduced, overall_step)
                # tf_writer.add_scalar("train/loss_itc", loss_itc_reduced, step)
                # tf_writer.add_scalar("train/loss_itm", loss_itm_reduced, step)
                tf_writer.add_scalar("train/learning_rate", optimizer.param_groups[0]['lr'], overall_step)

                # 更新loss
                # accelerator.log({"loss": loss_reduced}, step=step)

            if overall_step % args.checkpoint_step == 0:

                # 保存模型以及状态
                accelerator.wait_for_everyone()
                if accelerator.is_main_process:
                    output_dir = os.path.join(args.output_dir, f"checkpoint-{overall_step}")
                    accelerator.save_model(model, output_dir)
                    accelerator.save_state(output_dir)
                    accelerator.print(f"Save model to {output_dir}")

                accelerator.wait_for_everyone()
                # 验证
                model.eval()
                accelerator.print("#################")
                accelerator.print("Start validation")
                accelerator.print("#################")
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

                if accelerator.is_main_process:
                    for k, v in recalls_i2t.items():
                        tf_writer.add_scalar(f"val/i2t_recall@{k}", v, overall_step)
                    accelerator.print(recalls_i2t)

                accelerator.wait_for_everyone()


    # 训练结束，保存模型
    accelerator.wait_for_everyone()
    if accelerator.is_main_process:
        output_dir = os.path.join(args.output_dir, "final_checkpoint")
        accelerator.save_model(model, output_dir)
        accelerator.save_state(output_dir)
            
            

if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument("--log_dir", type=str, required=True, help="log_dir")
    parser.add_argument("--output_dir", type=str, required=True, help="output_dir")
    parser.add_argument("--num_epoch", type=int, default=2, help="num_epoch")
    parser.add_argument("--batch_size", type=int, default=25, help="batch_size")
    parser.add_argument("--lr", type=float, default=5e-6, help="lr") # 2e-5/image_ids: 5e-6
    parser.add_argument("--seed", type=int, default=42, help="seed")
    parser.add_argument("--num_warmup_rate", type=float, default=0.1, help="num_warmup_steps")
    parser.add_argument("--train_data_path", type=str, default='/mnt/cfs/NLP/zcl/multi_modal/data/new/blip2_data_train.json', help="train_data_path")
    parser.add_argument("--val_data_path", type=str, default='/mnt/cfs/NLP/zcl/multi_modal/data/new/blip2_data_val.json', help="val_data_path")
    parser.add_argument("--tiku_data_path", type=str, default='/mnt/cfs/NLP/zcl/multi_modal/data/new/blip2_data_val.json', help="tiku_data_path")

    parser.add_argument("--vit_model_name", type=str, default='eva_clip_g', help="vit_model_name")

    parser.add_argument("--checkpoint_step", type=int, default=1000, help="checkpoint_step")
    parser.add_argument("--resume_from_checkpoint", type=str, default=None, help="resume_from_checkpoint")

    args = parser.parse_args()

    main(args)