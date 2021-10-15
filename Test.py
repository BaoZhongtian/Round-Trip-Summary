from transformers import EncoderDecoderModel, BertTokenizer
import torch
torch.mean()
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
model = EncoderDecoderModel.from_encoder_decoder_pretrained('bert-base-uncased',
                                                            'bert-base-uncased')  # initialize Bert2Bert

# forward
input_ids = torch.tensor(tokenizer.encode("Hello, my dog is cute", add_special_tokens=True)).unsqueeze(
    0)  # Batch size 1
outputs = model(input_ids=input_ids, decoder_input_ids=input_ids)

# training
loss, outputs = model(input_ids=input_ids, decoder_input_ids=input_ids, lm_labels=input_ids)[:2]
print(loss, outputs)
# generation
generated = model.generate(input_ids, decoder_start_token_id=model.config.decoder.pad_token_id)
print(generated)
