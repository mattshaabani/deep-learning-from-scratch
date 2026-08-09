"""
src/phase3_rnn/text_generator.py

Text generation (sampling) from a trained VanillaRNN or LSTM model.

Usage:
    from src.phase3_rnn.text_generator import generate_text
    text = generate_text(model, dataset, seed_text="ROMEO:", length=300)
"""

import torch
import torch.nn.functional as F
from src.utils.config import settings
from src.utils.logger import get_logger

logger = get_logger(__name__)


def generate_text(
    model,
    dataset,
    seed_text: str = "\n",
    length: int = None,
    temperature: float = None,
    device: str = "cpu",
) -> str:
    """
    Autoregressively generate text, one character at a time.

    At each step:
        1. Feed the current sequence through the model
        2. Take the logits for the LAST position only
        3. Scale by temperature (higher = more random, lower = more greedy)
        4. Sample the next character from the resulting distribution
        5. Append it and repeat

    Temperature intuition (same softmax scaling concept from Project 1):
        temperature -> 0:   nearly argmax, deterministic, can get repetitive
        temperature = 1:    unscaled distribution
        temperature > 1:    flatter distribution, more random/creative
    """
    length = length or settings.phase3_generation.generate_length
    temperature = temperature or settings.phase3_generation.temperature

    model.eval()
    is_lstm = hasattr(model.cell, "candidate_gate")

    input_ids = dataset.encode(seed_text).unsqueeze(0).to(device)   # (1, seed_len)
    generated_ids = input_ids.clone()

    with torch.no_grad():
        # Prime the hidden state on the seed text
        if is_lstm:
            h_t, c_t = model.cell.init_hidden(1, device=device)
        else:
            h_t = model.cell.init_hidden(1, device=device)

        embedded = model.embedding(input_ids)
        for t in range(embedded.shape[1]):
            if is_lstm:
                h_t, c_t = model.cell(embedded[:, t, :], h_t, c_t)
            else:
                h_t = model.cell(embedded[:, t, :], h_t)

        # Now generate new characters one at a time
        for _ in range(length):
            logits = model.output_layer(h_t)   # (1, vocab_size)
            scaled_logits = logits / temperature
            probs = F.softmax(scaled_logits, dim=-1)

            next_id = torch.multinomial(probs, num_samples=1)   # (1, 1)
            generated_ids = torch.cat([generated_ids, next_id], dim=1)

            next_embedded = model.embedding(next_id).squeeze(1)   # (1, embedding_size)
            if is_lstm:
                h_t, c_t = model.cell(next_embedded, h_t, c_t)
            else:
                h_t = model.cell(next_embedded, h_t)

    return dataset.decode(generated_ids[0].tolist())