from transformers import RoFormerTokenizer, RoFormerForMaskedLM
import torch
from tqdm import tqdm
import time
from task9_scorer import mean_average_precision

def get_MAP(logits, cur_hyponym, cur_hypernyms):
    topk = torch.topk(logits.data, k=16, sorted=True, largest=True)
    topk = topk.indices[0]
    temp_mask_candidates = tokenizer.convert_ids_to_tokens(topk[-4])

    mask_candidates = temp_mask_candidates[:15]
    if cur_hyponym in temp_mask_candidates:
        index = temp_mask_candidates.index(cur_hyponym)
        for idx in range(index, 15):
            mask_candidates[idx] = temp_mask_candidates[idx + 1]

    true_table = [0 for idx in range(15)]

    for idx in range(15):
        if idx < len(cur_hypernyms):
            if mask_candidates[idx] in cur_hypernyms:
                true_table[idx] = 1

    return mean_average_precision(true_table, len(cur_hypernyms))

tokenizer = RoFormerTokenizer.from_pretrained("junnyu/roformer_chinese_base")
model = RoFormerForMaskedLM.from_pretrained("junnyu/roformer_chinese_base")
device = "cuda:0"
model = model.to(device)

dataset = "BK"

train_hyponyms = []
train_hypernyms = []
with open(f"{dataset}/train_set.txt", "r", encoding='UTF-8') as file:
    for line in tqdm(file):
        temp = line.strip().split('\t')
        train_hyponyms.append(temp[0])
        train_hypernyms.append(temp[1:])

dev_hyponyms = []
dev_hypernyms = []
with open(f"{dataset}/dev_set.txt", "r", encoding='UTF-8') as file:
    for line in tqdm(file):
        temp = line.strip().split('\t')
        dev_hyponyms.append(temp[0])
        dev_hypernyms.append(temp[1:])

# def display(model):
#     print("下位词：大学")
#     # prompt = "大学是一类[MASK]。"
#     prompt = "我最喜欢的[MASK]是大学。"
#     print("prompt:", prompt)
#     prompt = tokenizer(prompt, return_tensors="pt").to(device)
#     with torch.no_grad():
#         logits = model(**prompt).logits
#     topk = torch.topk(logits.data, k=16, sorted=True, largest=True)
#     topk = topk.indices[0]
#     # temp_mask_candidates = tokenizer.convert_ids_to_tokens(topk[-3])
#     temp_mask_candidates = tokenizer.convert_ids_to_tokens(topk[5])
#     mask_candidates = temp_mask_candidates[:15]
#     if "大学" in temp_mask_candidates:
#         index = temp_mask_candidates.index("大学")
#         for idx in range(index, 15):
#             mask_candidates[idx] = temp_mask_candidates[idx + 1]
#
#     print("候选上位词：", mask_candidates)


# print("————————————————————————————————before fine-tuning————————————————————————————————")
# # display(model)
# loss_before_finetuning = 0
# start_time = time.time()
# model.eval()
# for i in range(len(dev_hyponyms)):
#     # prompt = "大家一致同意" + dev_hyponyms[i] + "是一类[MASK]。"
#     # prompt = "我最喜欢的[MASK]是" + dev_hyponyms[i] + "。"
#     prompt = "有人说" + dev_hyponyms[i] + "是最好的[MASK]之一。"
#     for hypernym in dev_hypernyms[i]:
#         inputs = tokenizer(prompt, return_tensors="pt").to(device)
#         labels = tokenizer(prompt.replace("[MASK]", hypernym), return_tensors="pt")["input_ids"].to(device)
#         if inputs.input_ids.shape[1] != labels.data.shape[1]:
#             continue
#         labels = torch.where(inputs.input_ids == tokenizer.mask_token_id, labels, -100)
#         with torch.no_grad():
#             outputs = model(**inputs, labels=labels)
#         loss = outputs.loss
#         loss_before_finetuning += loss.item()
#
# print("dev loss: ", loss_before_finetuning)
# print("time passed：", time.time() - start_time)

optimizer = torch.optim.AdamW(model.parameters(), lr=0.00001)

lowest_loss = float("inf")
patience = 50
no_gain_epochs = 0
for epoch in range(500):
    start_time = time.time()
    print(f"————————————————————————————————epoch:{epoch+1}————————————————————————————————")
    model.train()
    for i in tqdm(range(len(train_hyponyms)), ncols=80):
        # prompt = "大家一致同意" + train_hyponyms[i] + "是一类[MASK]。"
        # prompt = "我最喜欢的[MASK]是" + train_hyponyms[i] + "。"
        prompt = train_hyponyms[i] + "是[MASK]之一。"
        inputs = tokenizer(prompt, return_tensors="pt").to(device)
        for hypernym in train_hypernyms[i]:
            labels = tokenizer(prompt.replace("[MASK]", hypernym), return_tensors="pt")["input_ids"].to(device)
            if inputs.input_ids.shape[1] != labels.data.shape[1]:
                continue
            labels = torch.where(inputs.input_ids == tokenizer.mask_token_id, labels, -100)
            outputs = model(**inputs, labels=labels)
            loss = outputs.loss
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

    model.eval()
    loss_for_epoch = 0
    total_dev_MAP = 0
    for i in tqdm(range(len(dev_hyponyms)), ncols=80):
        # prompt = "大家一致同意" + dev_hyponyms[i] + "是一类[MASK]。"
        # prompt = "我最喜欢的[MASK]是" + dev_hyponyms[i] + "。"
        prompt = dev_hyponyms[i] + "是[MASK]之一。"
        inputs = tokenizer(prompt, return_tensors="pt").to(device)
        for hypernym in dev_hypernyms[i]:
            labels = tokenizer(prompt.replace("[MASK]", hypernym), return_tensors="pt")["input_ids"].to(device)
            if inputs.input_ids.shape[1] != labels.data.shape[1]:
                continue
            labels = torch.where(inputs.input_ids == tokenizer.mask_token_id, labels, -100)
            with torch.no_grad():
                outputs = model(**inputs, labels=labels)
            loss = outputs.loss
            loss_for_epoch += loss.item()

        with torch.no_grad():
            outputs = model(**inputs)
        total_dev_MAP += get_MAP(outputs.logits, dev_hyponyms[i], dev_hypernyms[i])

    print("dev loss:", loss_for_epoch)
    print("dev MAP:", total_dev_MAP/len(dev_hyponyms))

    # early stop
    if loss_for_epoch < lowest_loss:
        lowest_loss = loss_for_epoch
        model.save_pretrained(f"zhi_yi_RoFormer_{dataset}")
        no_gain_epochs = 0
    else:
        no_gain_epochs += 1
    if no_gain_epochs >= patience:
        print("early stop.")
        break



