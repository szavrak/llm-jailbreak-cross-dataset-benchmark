"""Fine-tuned BERT/RoBERTa binary classifier for jailbreak detection."""
import copy
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import numpy as np
import time


def train_and_evaluate_bert(train_texts, train_labels, test_texts, test_labels,
                            val_texts=None, val_labels=None,
                            model_name='bert-base-uncased', lr=2e-5, epochs=5,
                            batch_size=16, max_len=512,
                            early_stopping_patience=2,
                            seed=42, device=None, class_weight=None):
    """Fine-tune a BERT/RoBERTa model and evaluate on test set.

    If val_texts/val_labels are provided, uses validation loss for early
    stopping (patience=early_stopping_patience) and restores the best model
    state. Otherwise trains for the full fixed number of epochs.

    class_weight: None -> default unweighted cross-entropy (the model's own
    loss). 'balanced' -> inverse-frequency weighted cross-entropy computed from
    the training labels (sklearn's n_samples / (n_classes * bincount) formula),
    matching the MiniLM+LR class_weight='balanced' scheme for the A1 ablation.

    Returns:
        dict with predictions, probabilities, train_time, infer_time, num_params
    """
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    torch.manual_seed(seed)
    np.random.seed(seed)

    rep_start = time.time()
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    train_enc = tokenizer(train_texts, padding=True, truncation=True,
                          max_length=max_len, return_tensors='pt')
    test_enc = tokenizer(test_texts, padding=True, truncation=True,
                         max_length=max_len, return_tensors='pt')

    train_labels_t = torch.tensor(train_labels, dtype=torch.long)
    test_labels_t = torch.tensor(test_labels, dtype=torch.long)

    train_dataset = TensorDataset(
        train_enc['input_ids'], train_enc['attention_mask'], train_labels_t)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

    val_loader = None
    if val_texts is not None and val_labels is not None and len(val_texts) > 0:
        val_enc = tokenizer(val_texts, padding=True, truncation=True,
                            max_length=max_len, return_tensors='pt')
        val_labels_t = torch.tensor(val_labels, dtype=torch.long)
        val_dataset = TensorDataset(
            val_enc['input_ids'], val_enc['attention_mask'], val_labels_t)
        val_loader = DataLoader(val_dataset, batch_size=batch_size)
    representation_time = time.time() - rep_start

    model = AutoModelForSequenceClassification.from_pretrained(
        model_name, num_labels=2)
    model.to(device)

    num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    # A1 ablation: inverse-frequency class weighting on the loss. When None, the
    # model's own (unweighted) loss is used, preserving the original behaviour.
    criterion = None
    if class_weight == 'balanced':
        counts = np.bincount(np.asarray(train_labels), minlength=2).astype(float)
        counts[counts == 0] = 1.0  # guard against degenerate splits
        weights = counts.sum() / (len(counts) * counts)
        criterion = nn.CrossEntropyLoss(
            weight=torch.tensor(weights, dtype=torch.float, device=device))
    elif class_weight is not None:
        raise ValueError(f"Unsupported class_weight: {class_weight!r}")

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)

    train_start = time.time()
    best_val_loss = float('inf')
    best_state = None
    patience_counter = 0

    for epoch in range(epochs):
        model.train()
        total_loss = 0
        for batch in train_loader:
            input_ids, attention_mask, labels = [b.to(device) for b in batch]
            optimizer.zero_grad()
            outputs = model(input_ids=input_ids, attention_mask=attention_mask,
                            labels=labels)
            loss = (criterion(outputs.logits, labels) if criterion is not None
                    else outputs.loss)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        train_loss = total_loss / len(train_loader)

        if val_loader is not None:
            model.eval()
            val_total = 0
            n_batches = 0
            with torch.no_grad():
                for batch in val_loader:
                    input_ids, attention_mask, labels = [b.to(device) for b in batch]
                    outputs = model(input_ids=input_ids,
                                    attention_mask=attention_mask,
                                    labels=labels)
                    val_loss_b = (criterion(outputs.logits, labels)
                                  if criterion is not None else outputs.loss)
                    val_total += val_loss_b.item()
                    n_batches += 1
            val_loss = val_total / max(n_batches, 1)
            print(f"  Epoch {epoch+1}/{epochs}, Train: {train_loss:.4f}, Val: {val_loss:.4f}")

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_counter = 0
                best_state = copy.deepcopy(model.state_dict())
            else:
                patience_counter += 1
                if patience_counter >= early_stopping_patience:
                    print(f"  Early stopping at epoch {epoch+1}")
                    break
        else:
            print(f"  Epoch {epoch+1}/{epochs}, Loss: {train_loss:.4f}")

    if best_state is not None:
        model.load_state_dict(best_state)

    train_time = time.time() - train_start

    model.eval()
    infer_start = time.time()
    all_preds = []
    all_probs = []

    test_dataset = TensorDataset(
        test_enc['input_ids'], test_enc['attention_mask'], test_labels_t)
    test_loader = DataLoader(test_dataset, batch_size=batch_size)

    with torch.no_grad():
        for batch in test_loader:
            input_ids, attention_mask, labels = [b.to(device) for b in batch]
            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            probs = torch.softmax(outputs.logits, dim=1)
            preds = torch.argmax(probs, dim=1)
            all_preds.extend(preds.cpu().numpy())
            all_probs.extend(probs.cpu().numpy())
    infer_time = time.time() - infer_start

    return {
        'predictions': np.array(all_preds),
        'probabilities': np.array(all_probs),
        'representation_time': representation_time,
        'train_time': train_time,
        'infer_time': infer_time,
        'num_params': num_params,
        'device': str(device),
    }
