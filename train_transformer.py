from pathlib import Path
from pyexpat import model
from typing import Optional
from torchtext.datasets import multi30k, Multi30k
from byte_pair_encoder import BytePairEncoder
from transformer import Transformer
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
import wandb
from dataclasses import asdict, dataclass
from optimizer import Scheduler



@dataclass
class TrainingConfig:
    n_epochs_training: int = 200
    detailed_log_freq: int = 100
    training_bs: int = 96
    learning_rate: float = 0.0001
    use_scheduler: bool = True
    cross_entropy_label_smoothing: float = 0.1
    bpe_language: str = 'universal'
    bpe_vocab_size: int = 1000
    bpe_max_token_len: int = 300
    bpe_use_start_token: bool = True
    bpe_use_end_token: bool = True
    bpe_use_padding_token: bool = True
    bpe_load_vocabulary_from: str = "data/universal_bpe_encoder.pkl"
    model_embedding_dim: int = 64
    model_embedding_padding_idx: int = 0
    model_ff_hidden_features: int = 256
    model_encoder_blocks: int = 4
    model_decoder_blocks: int = 4
    model_attn_heads: int = 8
    clip_gradient_norm_to: Optional[float] = None
    load_checkpoint: Optional[str] = None

CONFIGS = TrainingConfig()


def make_dataloader():
    multi30k.URL["train"] = "https://raw.githubusercontent.com/neychev/small_DL_repo/master/datasets/Multi30k/training.tar.gz"
    multi30k.URL["valid"] = "https://raw.githubusercontent.com/neychev/small_DL_repo/master/datasets/Multi30k/validation.tar.gz"
    multi30k.URL["test"] = "https://raw.githubusercontent.com/neychev/small_DL_repo/master/datasets/Multi30k/mmt16_task1_test.tar.gz"

    datasets = Multi30k()

    train = DataLoader(datasets[0], batch_size=CONFIGS.training_bs, shuffle=True)
    val = DataLoader(datasets[1], batch_size=1, shuffle=False)
    test = DataLoader(datasets[2], batch_size=1, shuffle=False)

    return train, val, test


def make_text_encoder():
    encoder = BytePairEncoder(
        language=CONFIGS.bpe_language,
        max_vocab_size=CONFIGS.bpe_vocab_size,
        use_start_token=CONFIGS.bpe_use_start_token,
        use_end_token=CONFIGS.bpe_use_end_token,
        use_padding_token=CONFIGS.bpe_use_padding_token,
        max_token_len=CONFIGS.bpe_max_token_len,
        load_vocabulary_from=CONFIGS.bpe_load_vocabulary_from,
    )

    return encoder


def make_model(load_from: Optional[str] = None):
    transformer = Transformer(
        dictionary_len=CONFIGS.bpe_vocab_size,
        embedding_dim=CONFIGS.model_embedding_dim,
        embedding_padding_idx=CONFIGS.model_embedding_padding_idx,
        ff_hidden_features=CONFIGS.model_ff_hidden_features,
        n_encoder_blocks=CONFIGS.model_encoder_blocks,
        n_decoder_blocks=CONFIGS.model_decoder_blocks,
        n_attn_heads=CONFIGS.model_attn_heads,
    )
    weight = torch.ones(1000)
    weight[0] = 0  # RESERVED
    weight[2] = 0  # PADDING
    weight[4] = 0  # UNKNOWN
    weight = weight.cuda()
    criterion = torch.nn.CrossEntropyLoss(ignore_index=CONFIGS.model_embedding_padding_idx, weight=weight, label_smoothing=CONFIGS.cross_entropy_label_smoothing)
    optimizer = torch.optim.Adam(transformer.parameters(), lr=CONFIGS.learning_rate, betas=(0.9, 0.98), eps=1e-9)
    scheduler = Scheduler(optimizer, CONFIGS.model_embedding_dim, 4000) if CONFIGS.use_scheduler else None

    start_epoch = 0

    if load_from is not None and Path(load_from).exists():
        checkpoint = torch.load(load_from)
        transformer.load_state_dict(checkpoint["model_state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        start_epoch = checkpoint["epoch"]

    if torch.cuda.is_available():
        transformer = transformer.cuda()

    return transformer, criterion, optimizer, scheduler, start_epoch


def init_wandb():
    wandb.init(
        project="transformer-small",
        config=asdict(CONFIGS),
    )


def run_training():
    init_wandb()
    train, val, _ = make_dataloader()
    text_encoder = make_text_encoder()
    transformer, criterion, optimizer, scheduler, start_epoch = make_model(CONFIGS.load_checkpoint)
    wandb.watch(transformer, log_freq=CONFIGS.detailed_log_freq)

    for epoch in range(start_epoch, CONFIGS.n_epochs_training):
        transformer.train()
        epoch_loss = 0
        for batch_idx, batch in tqdm(enumerate(train)):
            src, tgt = batch
            src = torch.from_numpy(text_encoder.encode_corpus(list(src)))
            tgt = torch.from_numpy(text_encoder.encode_corpus(list(tgt)))
            src = src.cuda()
            tgt = tgt.cuda()
            optimizer.zero_grad()
            output = transformer({ "source": src, "target": tgt})
            if batch_idx % CONFIGS.detailed_log_freq == 0:
                output_tokens = torch.argmax(output, dim=-1)
                output_tokens = output_tokens.cpu().numpy()
                tgt_tokens = tgt.cpu().numpy()
                src_tokens = src.cpu().numpy()
                samples_pred = {
                    "source_text": text_encoder.decode_sentence(src_tokens[0]),
                    "target_gt": text_encoder.decode_sentence(tgt_tokens[0]),
                    "target_pred": text_encoder.decode_sentence(output_tokens[0]),
                }
                print("\n-----------------------------------------------------")
                print("Input:", samples_pred["source_text"])
                print("Output:", samples_pred["target_pred"])
                print("Target:", samples_pred["target_gt"])
                print("-----------------------------------------------------\n")
                # wandb.log(samples_pred, commit=False)
            output = output.view(-1, output.shape[-1])
            tgt = tgt.view(-1)
            loss = criterion(output, tgt)
            loss.backward()
            if CONFIGS.clip_gradient_norm_to is not None:
                torch.nn.utils.clip_grad_norm_(transformer.parameters(), CONFIGS.clip_gradient_norm_to)
            optimizer.step()
            if scheduler is not None:
                scheduler.step()
                current_lr = scheduler.get_lr()[0]
            else:
                current_lr = CONFIGS.learning_rate
            wandb.log({"loss": loss.item(), "learning_rate": current_lr}, commit=True)
            epoch_loss += loss.item()
        epoch_loss /= batch_idx
        print(f"Epoch: {epoch + 1} | Train loss: {epoch_loss:.3f}")

        # make checkpoint
        torch.save(
            {
                "epoch": epoch,
                "model_state_dict": transformer.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "epoch_loss": epoch_loss,
            },
            f"data/transformer_epoch_{epoch}.pth",
        )

        # # validation
        # transformer.eval()
        # epoch_loss = 0
        # for batch_idx, batch in tqdm(enumerate(val)):
        #     src, tgt = batch
        #     src = torch.from_numpy(text_encoder.encode_corpus(list(src)))
        #     tgt = torch.from_numpy(text_encoder.encode_corpus(list(tgt)))
        #     src = src.cuda()
        #     tgt = tgt.cuda()
        #     output = transformer({ "source": src})
        #     output = output.view(-1, output.shape[-1])
        #     tgt = tgt.view(-1)
        #     loss = criterion(output, tgt)
        #     epoch_loss += loss.item()
        #     wandb.log({"val_loss": loss.item()}, commit=True)
        # epoch_loss /= batch_idx
        # print(f"Epoch: {epoch + 1} | Val loss: {epoch_loss:.3f}")


if __name__ == "__main__":
    run_training()
