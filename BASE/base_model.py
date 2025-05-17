import torch
import torch.nn as nn


# Custom Model Class
class TrajectoryBaseModel(nn.Module):
    def __init__(self, venue_vocab_size, hour_vocab_size, max_length_venue_id=100,
                 d_model=128, n_head=4, num_encoder_layers=4):
        super(TrajectoryBaseModel, self).__init__()
        self.venue_embedding = nn.Embedding(venue_vocab_size, d_model)
        self.hour_embedding = nn.Embedding(hour_vocab_size, d_model)
        self.position_embedding = nn.Embedding(max_length_venue_id, d_model)
        self.transformer_encoder = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(3 * d_model, n_head, batch_first=True, dropout=0.3),
            num_encoder_layers
        )
        self.fc = nn.Linear(3 * d_model, venue_vocab_size)

    def forward(self, venue_input, hour_input):
        batch_size, seq_length = venue_input.size()

        venue_embedded = self.venue_embedding(venue_input)
        hour_embedded = self.hour_embedding(hour_input)

        # Generate position embeddings for each position in the sequence
        position_ids = torch.arange(seq_length, dtype=torch.long, device=venue_input.device)
        position_ids = position_ids.unsqueeze(0).expand(batch_size, -1)
        position_embedded = self.position_embedding(position_ids)

        # combined_input = venue_embedded + hour_embedded + position_embedded
        combined_input = torch.cat([venue_embedded, hour_embedded, position_embedded], dim=2)  # [b,l,3d]

        transformer_output = self.transformer_encoder(combined_input)
        venue_output = self.fc(transformer_output)

        return venue_output
