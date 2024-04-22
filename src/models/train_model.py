import logging

import numpy as np
import torch
from model import TransformerWithPE
from torch.utils.data import DataLoader

from src.config import (BS, FEATURE_DIM, LR, NUM_EPOCHS, NUM_FEATURES,
                        NUM_HEADS, NUM_LAYERS, NUM_VIS_EXAMPLES,
                        TEST_DATA_PATH, TRAIN_DATA_PATH)
from src.visualization.visualize import visualize_prediction


def move_to_device(device: torch.device, *tensors: torch.Tensor) -> list[torch.Tensor]:
    """Move all given tensors to the given device.

    Args:
        device: device to move tensors to
        tensors: tensors to move

    Returns:
        list[torch.Tensor]: moved tensors
    """
    moved_tensors = []
    for tensor in tensors:
        if isinstance(tensor, torch.Tensor):
            moved_tensors.append(tensor.to(device))
        else:
            moved_tensors.append(tensor)
    return moved_tensors


def split_sequence(
        sequence: np.ndarray, ratio: float = 0.8
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Splits a sequence into 2 (3) parts, as is required by our transformer
    model.

    Assume our sequence length is L, we then split this into src of length N
    and tgt_y of length M, with N + M = L.
    src, the first part of the input sequence, is the input to the encoder, and we
    expect the decoder to predict tgt_y, the second part of the input sequence.
    In addition, we generate tgt, which is tgt_y but "shifted left" by one - i.e. it
    starts with the last token of src, and ends with the second-last token in tgt_y.
    This sequence will be the input to the decoder.


    Args:
        sequence: batched input sequences to split [bs, seq_len, num_features]
        ratio: split ratio, N = ratio * L

    Returns:
        tuple[torch.Tensor, torch.Tensor, torch.Tensor]: src, tgt, tgt_y
    """
    src_end = int(sequence.shape[1] * ratio)
    # [bs, src_seq_len, num_features]
    src = sequence[:, :src_end]
    # [bs, tgt_seq_len, num_features]
    tgt = sequence[:, src_end - 1: -1]
    # [bs, tgt_seq_len, num_features]
    tgt_y = sequence[:, src_end:]

    return src, tgt, tgt_y


def main() -> None:
    logger = logging.getLogger(__name__)

    logger.info('Training model')

    train_set = torch.load(TRAIN_DATA_PATH)
    test_set = torch.load(TEST_DATA_PATH)

    train_loader = DataLoader(train_set, batch_size=BS, shuffle=True)
    test_loader = DataLoader(test_set, batch_size=BS, shuffle=False)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Initialize model, optimizer and loss criterion
    model = TransformerWithPE(
        NUM_FEATURES, NUM_FEATURES, FEATURE_DIM, NUM_HEADS, NUM_LAYERS
    ).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    criterion = torch.nn.MSELoss()

    # Train loop
    for epoch in range(NUM_EPOCHS):
        epoch_loss = 0.0
        for batch in train_loader:
            optimizer.zero_grad()

            src, tgt, tgt_y = split_sequence(batch[0])
            src, tgt, tgt_y = move_to_device(device, src, tgt, tgt_y)
            # [bs, tgt_seq_len, num_features]
            pred = model(src, tgt)
            loss = criterion(pred, tgt_y)
            epoch_loss += loss.item()

            loss.backward()
            optimizer.step()

        print(
            f'Epoch [{epoch + 1}/{NUM_EPOCHS}], Loss: '
            f'{(epoch_loss / len(train_loader)):.4f}'
        )

    # Evaluate model
    model.eval()
    eval_loss = 0.0
    infer_loss = 0.0
    with torch.no_grad():
        for idx, batch in enumerate(test_loader):
            src, tgt, tgt_y = split_sequence(batch[0])
            src, tgt, tgt_y = move_to_device(device, src, tgt, tgt_y)

            # [bs, tgt_seq_len, num_features]
            pred = model(src, tgt)
            loss = criterion(pred, tgt_y)
            eval_loss += loss.item()

            # Run inference with model
            pred_infer = model.infer(src, tgt.shape[1])
            loss_infer = criterion(pred_infer, tgt_y)
            infer_loss += loss_infer.item()

            if idx < NUM_VIS_EXAMPLES:
                visualize_prediction(src, tgt, pred, pred_infer)

    avg_eval_loss = eval_loss / len(test_loader)
    avg_infer_loss = infer_loss / len(test_loader)

    print(f'Eval / Infer Loss on test set: {avg_eval_loss:.4f} / {avg_infer_loss:.4f}')


if __name__ == '__main__':
    log_fmt = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    logging.basicConfig(level=logging.INFO, format=log_fmt)

    main()
