import torch
import torch.nn as nn
import torch.nn.functional as F


class Encoder(nn.Module):
    def __init__(self, input_dim, emb_dim, hidden_dim, num_layers=2, dropout=0.3):
        super().__init__()
        self.embedding = nn.Embedding(input_dim, emb_dim)
        self.lstm = nn.LSTM(
            emb_dim, hidden_dim, num_layers=num_layers,
            dropout=dropout, bidirectional=True, batch_first=True
        )
        self.fc = nn.Linear(hidden_dim * 2, hidden_dim)  # to reduce biLSTM → hidden_dim
        self.dropout = nn.Dropout(dropout)

    def forward(self, src, src_lengths):
        # src: [batch, src_len]
        embedded = self.dropout(self.embedding(src))  # [batch, src_len, emb_dim]

        # pack padded sequence
        packed_embedded = nn.utils.rnn.pack_padded_sequence(
            embedded, src_lengths.cpu(), batch_first=True, enforce_sorted=False
        )
        packed_outputs, (hidden, cell) = self.lstm(packed_embedded)

        outputs, _ = nn.utils.rnn.pad_packed_sequence(
            packed_outputs, batch_first=True, total_length=src.size(1)
        )
                # outputs: [batch, src_len, hidden_dim*2]

        # concat final forward + backward hidden state
        hidden_cat = torch.cat((hidden[-2], hidden[-1]), dim=1)  # [batch, hidden_dim*2]
        hidden = torch.tanh(self.fc(hidden_cat)).unsqueeze(0)  # [1, batch, hidden_dim]
        cell = torch.zeros_like(hidden)

        return outputs, hidden, cell


class Attention(nn.Module):
    def __init__(self, hidden_dim):
        super().__init__()
        self.attn = nn.Linear(hidden_dim * 3, hidden_dim)
        self.v = nn.Linear(hidden_dim, 1, bias=False)

    def forward(self, hidden, encoder_outputs, mask):
        # hidden: [1, batch, hidden_dim]
        hidden = hidden.permute(1, 0, 2)  # [batch, 1, hidden_dim]
        src_len = encoder_outputs.shape[1]

        hidden_repeat = hidden.repeat(1, src_len, 1)  # [batch, src_len, hidden_dim]
        energy = torch.tanh(self.attn(torch.cat((hidden_repeat, encoder_outputs), dim=2)))
        attention = self.v(energy).squeeze(2)  # [batch, src_len]

        attention = attention.masked_fill(mask == 0, -1e10)
        return F.softmax(attention, dim=1)


class Decoder(nn.Module):
    def __init__(self, output_dim, emb_dim, hidden_dim, dropout=0.3):
        super().__init__()
        self.output_dim = output_dim
        self.embedding = nn.Embedding(output_dim, emb_dim)
        self.lstm = nn.LSTM(hidden_dim * 2 + emb_dim, hidden_dim, batch_first=True)
        self.fc_out = nn.Linear(hidden_dim * 3 + emb_dim, output_dim)
        self.attention = Attention(hidden_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, input, hidden, cell, encoder_outputs, mask):
        # input: [batch], hidden: [1, batch, hidden_dim]
        input = input.unsqueeze(1)  # [batch, 1]
        embedded = self.dropout(self.embedding(input))  # [batch, 1, emb_dim]

        attn_weights = self.attention(hidden, encoder_outputs, mask)  # [batch, src_len]
        attn_weights = attn_weights.unsqueeze(1)  # [batch, 1, src_len]

        context = torch.bmm(attn_weights, encoder_outputs)  # [batch, 1, hidden_dim*2]

        rnn_input = torch.cat((embedded, context), dim=2)  # [batch, 1, emb_dim+hidden_dim*2]
        output, (hidden, cell) = self.lstm(rnn_input, (hidden, cell))

        # prediction
        output = output.squeeze(1)      # [batch, hidden_dim]
        context = context.squeeze(1)    # [batch, hidden_dim*2]
        embedded = embedded.squeeze(1)  # [batch, emb_dim]

        pred = self.fc_out(torch.cat((output, context, embedded), dim=1))  # [batch, output_dim]

        return pred, hidden, cell, attn_weights.squeeze(1)


class Seq2Seq(nn.Module):
    def __init__(self, encoder, decoder, device):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.device = device

    def create_mask(self, src):
        return (src != 0).to(self.device)

    def forward(self, src, src_lengths, trg, teacher_forcing_ratio=0.5):
        batch_size = src.shape[0]
        trg_len = trg.shape[1]
        trg_vocab_size = self.decoder.output_dim

        outputs = torch.zeros(batch_size, trg_len, trg_vocab_size).to(self.device)

        encoder_outputs, hidden, cell = self.encoder(src, src_lengths)
        input = trg[:, 0]  # <sos> token

        mask = self.create_mask(src)

        for t in range(1, trg_len):
            output, hidden, cell, _ = self.decoder(input, hidden, cell, encoder_outputs, mask)
            outputs[:, t] = output
            top1 = output.argmax(1)
            input = trg[:, t] if torch.rand(1).item() < teacher_forcing_ratio else top1

        return outputs
