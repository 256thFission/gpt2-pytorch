import torch
from torch.utils.data import Dataset, DataLoader

def tokenize_text(text_path):
    """Tokenize text and build vocabulary."""
    with open(text_path, 'r') as f:
        text = f.read()

    # Build vocabulary
    chars = sorted(set(text))
    vocab_size = len(chars)

    # Create mappings
    stoi = {ch: i for i, ch in enumerate(chars)}
    itos = {i: ch for i, ch in enumerate(chars)}

    # Tokenize
    tokens = [stoi[c] for c in text]

    return tokens, vocab_size, stoi, itos

class TextDataset(Dataset):
    def __init__(self, data, block_size):
        self.data = data
        self.block_size = block_size

    def __len__(self):
        return len(self.data) - self.block_size

    def __getitem__(self, idx):
        x = torch.tensor(self.data[idx:idx + self.block_size], dtype=torch.long)
        y = torch.tensor(self.data[idx + 1:idx + 1 + self.block_size], dtype=torch.long)
        return x, y

def get_dataloaders(text_path, block_size, batch_size, train_split=0.9):
    """Load train and test dataloaders."""
    # Tokenize
    tokens, vocab_size, stoi, itos = tokenize_text(text_path)

    # Split
    split_idx = int(len(tokens) * train_split)
    train_tokens = tokens[:split_idx]
    test_tokens = tokens[split_idx:]

    # Create datasets
    train_dataset = TextDataset(train_tokens, block_size)
    test_dataset = TextDataset(test_tokens, block_size)

    # Create dataloaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    return train_loader, test_loader, vocab_size, stoi, itos


