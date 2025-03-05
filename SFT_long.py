import os
import argparse
import math
import time
import torch
import torch.nn as nn
import torch.optim as optim
from contextlib import nullcontext
from transformers import AutoTokenizer
from torch.utils.data import DataLoader
from model import XiaoXin
from Config import LLMConfig
from dataset import SFTDataset

## cos_decay 计算当前学习率
def get_lr(current_step, total_steps, lr, warmup_iters = 0):
    return lr / 10 + 0.5 * lr * (1 + math.cos(math.pi * current_step / total_steps))

def train_epoch(epoch, wandb):
    loss_fct = nn.CrossEntropyLoss(reduction = "none")

    start_time = time.time()
    
    for step, (X, Y, loss_mask) in enumerate(train_loader):
        X = X.to(args.device)
        Y = Y.to(args.device)
        loss_mask = loss_mask.to(args.device)

        lr = get_lr(epoch * iter_per_epoch + step, args.epochs * iter_per_epoch, args.learning_rate, args.warmup_iters)

        for param_group in optimizer.param_groups:
            param_group["lr"] = lr
            
        with ctx:
            res = model(X)
            loss = loss_fct(
                res.logits.view(-1, res.logits.size(-1)),
                Y.view(-1)
            ).view(Y.size())
            loss = (loss * loss_mask).sum() / loss_mask.sum()
            loss = loss / args.accumulation_steps
        scaler.scale(loss).backward()
        # 累计一定步数进行优化器更新
        if (step + 1) % args.accumulation_steps == 0:
            scaler.unscale_(optimizer) # 解除梯度缩放
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip) # 梯度裁剪
            scaler.step(optimizer) # 执行优化步骤
            scaler.update() # 更新缩放器
            optimizer.zero_grad(set_to_none = True) # 清空梯度

        if (step + 1) % args.log_step == 0:
            spend_time = time.time() - start_time
            print(
                'Epoch: [{}/{}]({}/{}) loss:{:.3f} lr:{:.12f} epoch__Time:{}min:'.format(
                    epoch + 1,
                    args.epochs,
                    step,
                    iter_per_epoch,
                    loss.item() * args.accumulation_steps,
                    optimizer.param_groups[-1]['lr'],
                    spend_time / (step + 1) * iter_per_epoch // 60 - spend_time // 60
                )
            )

            if (wandb is not None):
                wandb.log({
                    "loss": loss.item() * args.accumulation_steps,
                    "lr": optimizer.param_groups[-1]["lr"],
                    "epoch_Time": spend_time / (step + 1) * iter_per_epoch // 60 - spend_time // 60
                })

        if (step + 1) % args.save_step == 0:
            model.eval()
            ckp = f'{args.save_dir}/SFT-long-{args.epochs}-{args.batch_size}-{args.accumulation_steps}.pth'
            state_dict = model.state_dict()
            torch.save(state_dict, ckp)
            model.train()

def init_model(llm_config):
    tokenizer = AutoTokenizer.from_pretrained("./XiaoXin_tokenizer")
    model = XiaoXin(llm_config).to(args.device)
    ckp = './results/SFT-1-84-2.pth'
    state_dict = torch.load(ckp, map_location = args.device)
    model.load_state_dict(state_dict, strict = False)
    print(f"LLM总参数量: {sum(p.numel() for p in model.parameters() if p.requires_grad) / 1e6:.3f} 百万")
    model = model.to(args.device)
    return model, tokenizer

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--save_dir', type = str, default = "results")
    parser.add_argument('--epochs', type = int, default = 1)
    parser.add_argument('--batch_size', type = int, default = 28)
    # 对比 pretrain，下调了学习率
    parser.add_argument('--learning_rate', type = float, default = 5e-4)
    parser.add_argument('--device', type = str, default = 'cuda:0' if torch.cuda.is_available() else "cpu")
    parser.add_argument('--use_wandb', type = bool, default = True)
    parser.add_argument('--dtype', type = str, default = "bfloat16")
    parser.add_argument('--wandb_project', type = str, default = "XiaoXin-SFT")
    parser.add_argument('--num_workers', type = int, default = 1)
    parser.add_argument('--accumulation_steps', type = int, default = 8)
    parser.add_argument('--grad_clip', type = float, default = 1.0)
    parser.add_argument('--warmup_iters', type = int, default = 0)
    parser.add_argument('--log_step', type = int, default = 10)
    parser.add_argument('--save_step', type = int, default = 1000)
    parser.add_argument('--max_seq_len', type = int, default = 1024)
    parser.add_argument('--data_path', type = str, default = "sft_1024.jsonl")
    # 解析命令行参数
    args = parser.parse_args()
    
    # 配置语言模型参数
    llm_config = LLMConfig(max_seq_len = args.max_seq_len)

    # 创建保存目录
    args.save_dir = os.path.join(args.save_dir)
    os.makedirs(args.save_dir, exist_ok = True)

    # 每次迭代处理的 token 数量
    tokens_per_iter = args.batch_size * args.max_seq_len

    # 配置设备类型
    device_type = args.device

    # 设置 wandb 运行名称
    args.wandb_run_name = f'XiaoXin-SFT-long-{args.epochs}-{args.accumulation_steps}-{args.batch_size}-{args.learning_rate}'

    # 根据设备类型选择自动混合精度 AMP 或者 不使用 AMP
    ctx = nullcontext() if device_type == "cpu" else torch.cuda.amp.autocast()

    # 如果使用 wandb 进行实验跟踪
    if args.use_wandb:
        import wandb
        # 初始化 wandb 项目
        wandb.login(key = "d2a599550ffed3ef565542098b0acaecbf166f6c")
        wandb.init(project = args.wandb_project, name = args.wandb_run_name)
    else:
        wandb = None

    # 初始化模型和分词器
    model, tokenizer = init_model(llm_config)

    # 加载训练数据集
    train_ds = SFTDataset(args.data_path, tokenizer, max_length = args.max_seq_len)
    
    # 创建数据加载器
    train_loader = DataLoader(
        train_ds,
        batch_size = args.batch_size,
        shuffle = False,
        num_workers = args.num_workers,
        pin_memory = True,
        drop_last = False,
    )

    scaler = torch.amp.GradScaler('cuda', enabled = args.dtype in ["float16", "bfloat16"])
    optimizer = optim.AdamW(model.parameters(), lr = args.learning_rate)

    iter_per_epoch = len(train_loader)
    for epoch in range(args.epochs):
        train_epoch(epoch, wandb)

        