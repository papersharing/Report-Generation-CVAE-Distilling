import torch
import torch.nn as nn
import torch.nn.utils.rnn as rnn_utils
from utils import to_var


class CDVAE(nn.Module):

    def __init__(self, vocab_size, embedding_size, hidden_size, latent_size,
                 sos_idx, eos_idx, pad_idx, unk_idx, max_sequence_length, num_layers=1, bidirectional=False):

        super().__init__()
        self.tensor = torch.cuda.FloatTensor if torch.cuda.is_available() else torch.Tensor

        self.max_sequence_length = max_sequence_length
        self.sos_idx = sos_idx
        self.eos_idx = eos_idx
        self.pad_idx = pad_idx
        self.unk_idx = unk_idx

        self.latent_size = latent_size

        self.bidirectional = bidirectional
        self.num_layers = num_layers
        self.hidden_size = hidden_size

        self.embedding = nn.Embedding(vocab_size, embedding_size)

        self.report_encoder = nn.GRU(embedding_size, hidden_size, num_layers=num_layers,
                                     bidirectional=self.bidirectional, batch_first=True)
        self.news_encoder = nn.GRU(embedding_size, hidden_size, num_layers=num_layers, bidirectional=self.bidirectional,
                                   batch_first=True)
        self.encoder_rnn = nn.GRU(embedding_size, hidden_size, num_layers=num_layers, bidirectional=self.bidirectional,
                                  batch_first=True)
        self.decoder_rnn = nn.GRU(embedding_size, hidden_size, num_layers=num_layers, bidirectional=self.bidirectional,
                                  batch_first=True)

        self.hidden_factor = (2 if bidirectional else 1) * num_layers

        self.hidden2mean = nn.Linear(hidden_size * self.hidden_factor, latent_size)
        self.hidden2logv = nn.Linear(hidden_size * self.hidden_factor, latent_size)

        self.rec_hidden2mean = nn.Linear(2 * hidden_size * self.hidden_factor, latent_size)
        self.rec_hidden2logv = nn.Linear(2 * hidden_size * self.hidden_factor, latent_size)

        self.latent2hidden = nn.Linear(latent_size, hidden_size * self.hidden_factor)
        self.outputs2vocab = nn.Linear(hidden_size * (2 if bidirectional else 1), vocab_size)

    def forward(self, input_sequence, length, N_news_input, news_length, N_report_input, report_length, output_sequence,
                output_length):

        batch_size = input_sequence.size(0)

        sorted_lengths, sorted_idx = torch.sort(length, descending=True)
        input_sequence = input_sequence[sorted_idx]

        news_sorted_lengths, news_sorted_idx = torch.sort(news_length, descending=True)
        N_news_input = N_news_input[news_sorted_idx]

        report_sorted_lengths, report_sorted_idx = torch.sort(report_length, descending=True)
        N_report_input = N_report_input[report_sorted_idx]

        output_sorted_lengths, output_sorted_idx = torch.sort(output_length, descending=True)
        output_sequence = output_sequence[output_sorted_idx]

        # ENCODER
        input_embedding = self.embedding(input_sequence)
        news_embedding = self.embedding(N_news_input)
        report_embedding = self.embedding(N_report_input)
        output_embedding = self.embedding(output_sequence)

        packed_input = rnn_utils.pack_padded_sequence(input_embedding, sorted_lengths.data.tolist(), batch_first=True)
        packed_news = rnn_utils.pack_padded_sequence(news_embedding, news_sorted_lengths.data.tolist(),
                                                     batch_first=True)
        packed_report = rnn_utils.pack_padded_sequence(report_embedding, report_sorted_lengths.data.tolist(),
                                                       batch_first=True)
        packed_output = rnn_utils.pack_padded_sequence(output_embedding, output_sorted_lengths.data.tolist(),
                                                       batch_first=True)

        _, hidden = self.encoder_rnn(packed_input)
        _, news_hidden = self.news_encoder(packed_news)
        _, report_hidden = self.report_encoder(packed_news)

        _, reversed_idx = torch.sort(sorted_idx)
        _, news_reversed_idx = torch.sort(news_sorted_idx)
        _, report_reversed_idx = torch.sort(report_sorted_idx)

        hidden = hidden[reversed_idx][output_sorted_idx]
        news_hidden = news_hidden[news_reversed_idx][output_sorted_idx]
        report_hidden = report_hidden[report_reversed_idx][output_sorted_idx]

        if self.bidirectional or self.num_layers > 1:
            # flatten hidden state
            hidden = hidden.view(batch_size, self.hidden_size * self.hidden_factor)
            news_hidden = news_hidden.view(batch_size, self.hidden_size * self.hidden_factor)
            report_hidden = report_hidden.view(batch_size, self.hidden_size * self.hidden_factor)
        else:
            hidden = hidden.squeeze()
            news_hidden = news_hidden.squeeze()
            report_hidden = report_hidden.squeeze()

        rec_hidden = torch.cat((news_hidden, report_hidden), 1)

        # REPARAMETERIZATION
        mean = self.hidden2mean(hidden)
        logv = self.hidden2logv(hidden)
        std = torch.exp(0.5 * logv)

        rec_mean = self.rec_hidden2mean(rec_hidden)
        rec_logv = self.rec_hidden2logv(rec_hidden)
        rec_std = torch.exp(0.5 * rec_logv)

        z = to_var(torch.randn([batch_size, self.latent_size]))
        z = z * rec_std + rec_mean

        # DECODER
        hidden = self.latent2hidden(z)

        if self.bidirectional or self.num_layers > 1:
            # unflatten hidden state
            hidden = hidden.view(self.hidden_factor, batch_size, self.hidden_size)
        else:
            hidden = hidden.unsqueeze(0)

        # decoder forward pass
        outputs, _ = self.decoder_rnn(packed_output, hidden)

        # process outputs
        padded_outputs = rnn_utils.pad_packed_sequence(outputs, batch_first=True)[0]
        padded_outputs = padded_outputs.contiguous()
        b, s, _ = padded_outputs.size()

        # project outputs to vocab
        logp = nn.functional.log_softmax(self.outputs2vocab(padded_outputs.view(-1, padded_outputs.size(2))), dim=-1)
        logp = logp.view(b, s, self.embedding.num_embeddings)

        return logp, mean, logv, rec_mean, rec_logv
