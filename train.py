import torch
from torch.cuda.amp import GradScaler

from loss import cross_entropy


def train_client_worker(args_tuple):
    """
    单个客户端训练的工作函数（用于多进程并行）

    这个函数被设计为可以在独立的进程中运行，实现客户端并行训练

    Args:
        args_tuple: 包含训练参数的元组
            - state_dict: 模型状态字典
            - dataloader_idx: 数据加载器的索引
            - args: 训练配置
            - client_idx: 客户端索引
            - train_loader: 训练数据加载器字典

    Returns:
        tuple: (客户端模型状态字典, 平均损失)
    """
    state_dict, dataloader_idx, args, client_idx, train_loader = args_tuple

    # 从状态字典重建模型（需要导入模型类）
    from backbone.BaseTransformer import BASE_Transformer

    # 创建模型并加载全局模型参数
    client_model = BASE_Transformer(
        input_nc=3,
        output_nc=2,
        token_len=4,
        resnet_stages_num=4,
        with_pos="learned",
        enc_depth=1,
        dec_depth=8,
    )
    client_model.load_state_dict(state_dict)
    client_model.to(args.device)
    client_model.train()

    # 创建优化器
    optimizer = torch.optim.Adam(
        client_model.parameters(),
        lr=args.lr,
        betas=args.betas,
        eps=args.eps,
        weight_decay=args.weight_decay,
    )

    # 创建梯度缩放器用于混合精度训练
    client_scaler = GradScaler()

    # 获取当前客户端的数据加载器
    dataloader = train_loader[dataloader_idx]

    total_loss = 0.0
    num_batches = 0
    total_batches = len(dataloader) * args.num_client_epoch

    # 在客户端上进行多个epoch的本地训练
    for epoch in range(args.num_client_epoch):
        for batch_idx, (A, B, Label, _) in enumerate(dataloader):
            # 将数据移动到指定设备并确保内存连续
            A = A.contiguous().to(args.device, non_blocking=True)
            B = B.contiguous().to(args.device, non_blocking=True)
            Label = Label.contiguous().to(args.device, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)

            # 使用自动混合精度训练（AMP）
            with torch.autocast(device_type=args.device, dtype=torch.float16):
                pred = client_model(A, B)
                loss = cross_entropy(pred[0].contiguous(), Label)

            client_scaler.scale(loss).backward()
            client_scaler.step(optimizer)
            client_scaler.update()

            total_loss += loss.item()
            num_batches += 1

    # 计算平均损失
    avg_loss = total_loss / (num_batches * args.num_client_epoch)

    # 返回模型状态字典（不需要返回整个模型，只返回参数）
    return client_model.state_dict(), avg_loss
