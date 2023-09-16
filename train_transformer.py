from pathlib import Path
from torchtext.datasets import multi30k, Multi30k
from byte_pair_encoder import BytePairEncoder
from transformer import Transformer
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm


MODEL_SAVE_PATH = "data/transformer.pth"
N_EPOCHS = 200
N_BATCH_INSPECTION = 100


def make_dataloader():
    multi30k.URL["train"] = "https://raw.githubusercontent.com/neychev/small_DL_repo/master/datasets/Multi30k/training.tar.gz"
    multi30k.URL["valid"] = "https://raw.githubusercontent.com/neychev/small_DL_repo/master/datasets/Multi30k/validation.tar.gz"
    multi30k.URL["test"] = "https://raw.githubusercontent.com/neychev/small_DL_repo/master/datasets/Multi30k/mmt16_task1_test.tar.gz"

    datasets = Multi30k()

    train = DataLoader(datasets[0], batch_size=8, shuffle=True)
    val = DataLoader(datasets[1], batch_size=1, shuffle=False)
    test = DataLoader(datasets[2], batch_size=1, shuffle=False)

    return train, val, test


def make_text_encoder():
    encoder = BytePairEncoder(
        language='universal',
        max_vocab_size=1000,
        use_start_token=True,
        use_end_token=True,
        use_padding_token=True,
        max_token_len=300,
        load_vocabulary_from="data/universal_bpe_encoder.pkl",
    )

    return encoder


def make_model():
    if Path(MODEL_SAVE_PATH).exists():
        transformer = Transformer.load(MODEL_SAVE_PATH)
    else:
        transformer = Transformer(
            dictionary_len=1000,
            embedding_dim=128,
            embedding_padding_idx=0,
            ff_hidden_features=512,
            n_encoder_blocks=8,
            n_decoder_blocks=8,
            n_attn_heads=8,
        )
    if torch.cuda.is_available():
        transformer = transformer.cuda()

    return transformer


def run_training():
    train, val, _ = make_dataloader()
    text_encoder = make_text_encoder()
    transformer = make_model()
    weight = torch.ones(1000)
    weight[0] = 0
    weight[2] = 0
    weight[4] = 0
    weight = weight.cuda()
    criterion = torch.nn.CrossEntropyLoss(ignore_index=0, weight=weight)
    optimizer = torch.optim.Adam(transformer.parameters(), lr=0.0001)

    for epoch in range(N_EPOCHS):
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
            if batch_idx % N_BATCH_INSPECTION == 0:
                output_tokens = torch.argmax(output, dim=-1)
                output_tokens = output_tokens.cpu().numpy()
                tgt_tokens = tgt.cpu().numpy()
                src_tokens = src.cpu().numpy()
                print("\n-----------------------------------------------------")
                print("Input:", text_encoder.decode_corpus([src_tokens[0]]))
                print("Output:", text_encoder.decode_corpus([output_tokens[0]]))
                print("Target:", text_encoder.decode_corpus([tgt_tokens[0]]))
                print("-----------------------------------------------------\n")
            output = output.view(-1, output.shape[-1])
            tgt = tgt.view(-1)
            loss = criterion(output, tgt)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(transformer.parameters(), 1)
            optimizer.step()
            epoch_loss += loss.item()
        epoch_loss /= batch_idx
        print(f"Epoch: {epoch + 1} | Train loss: {epoch_loss:.3f}")

        # make checkpoint
        transformer.save(f"data/transformer_{epoch + 1}.pth")

        # transformer.eval()
        # epoch_loss = 0
        # for batch_idx, batch in tqdm(enumerate(val)):

        #     src, tgt = batch
        #     src = torch.from_numpy(text_encoder.encode_corpus(list(src)))
        #     tgt = torch.from_numpy(text_encoder.encode_corpus(list(tgt)))
        #     src = src.cuda()
        #     tgt = tgt.cuda()
        #     output = transformer({"source": src})
        #     if batch_idx % N_BATCH_INSPECTION == 0:
        #         output_tokens = torch.argmax(output, dim=-1)
        #         output_tokens = output_tokens.cpu().numpy()
        #         tgt_tokens = tgt.cpu().numpy()
        #         src_tokens = src.cpu().numpy()
        #         print("Input:", text_encoder.decode_corpus(src_tokens))
        #         print("Output:", text_encoder.decode_corpus(output_tokens))
        #         print("Target:", text_encoder.decode_corpus(tgt_tokens))
        #     output = output.view(-1, output.shape[-1])
        #     tgt = tgt.view(-1)
        #     loss = criterion(output, tgt)
        #     epoch_loss += loss.item()
        # epoch_loss /= batch_idx
        # print(f"Epoch: {epoch + 1} | Val loss: {epoch_loss:.3f}")


if __name__ == "__main__":
    run_training()
