import nlp
import torch
from transformers import LongformerTokenizer, EncoderDecoderModel

tokenizer = LongformerTokenizer.from_pretrained("allenai/longformer-base-4096")
model = EncoderDecoderModel.from_pretrained("patrickvonplaten/longformer2roberta-cnn_dailymail-fp16")
model.to("cuda")

test_dataset = nlp.load_dataset("cnn_dailymail", "3.0.0", split="test")
batch_size = 32

encoder_length = 2048
decoder_length = 128


# map data correctly
def generate_summary(batch):
    # Tokenizer will automatically set [BOS] <text> [EOS]
    # cut off at BERT max length 512
    inputs = tokenizer(batch["article"], padding="max_length", truncation=True, max_length=encoder_length, return_tensors="pt")
    input_ids = inputs.input_ids.to("cuda")
    attention_mask = inputs.attention_mask.to("cuda")
    global_attention_mask = torch.zeros_like(attention_mask)
    global_attention_mask[:, :decoder_length] = 1

    outputs = model.generate(input_ids, attention_mask=attention_mask, global_attention_mask=global_attention_mask)

    # all special tokens including will be removed
    output_str = tokenizer.batch_decode(outputs, skip_special_tokens=True)

    batch["pred"] = output_str

    return batch


results = test_dataset.map(generate_summary, batched=True, batch_size=batch_size, remove_columns=["article"])

# load rouge for validation
rouge = nlp.load_metric("rouge")

pred_str = results["pred"]
label_str = results["highlights"]

rouge_output = rouge.compute(predictions=pred_str, references=label_str, rouge_types=["rouge2"])["rouge2"].mid

print(rouge_output)