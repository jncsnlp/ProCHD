from transformers import RoFormerTokenizer, RoFormerForMaskedLM
import torch

from tqdm import tqdm
from task9_scorer import precision_at_k
from task9_scorer import mean_average_precision
from task9_scorer import mean_reciprocal_rank

dataset = "BK"

tokenizer = RoFormerTokenizer.from_pretrained("junnyu/roformer_chinese_base")
# model = RoFormerForMaskedLM.from_pretrained("junnyu/roformer_chinese_base")
model = RoFormerForMaskedLM.from_pretrained(f"zhi_yi_RoFormer_{dataset}")
device = "cuda:1"
model = model.to(device)

hyponyms = []
hypernyms = []


with open(f"{dataset}/test_set.txt", "r", encoding='UTF-8') as file:
    for line in tqdm(file):
        temp = line.strip().split('\t')
        hyponyms.append(temp[0])
        hypernyms.append(temp[1:])

metrics = {'MAP':0, 'MRR':0, 'p@1':0, 'p@3':0, 'p@5' :0, 'p@15':0}

for i in range(len(hyponyms)):
# for i in range(1):
    print("下位词：" + hyponyms[i] + "    上位词：" + str(hypernyms[i]))
    # template = "我最喜欢的[MASK]是" + hyponyms[i] + "。"
    # template = "大家一致同意" + hyponyms[i] + "是一类[MASK]。"
    template = hyponyms[i] + "是[MASK]之一。"
    # template = "茶叶蛋是[MASK]之一。"
    print(template)

    inputs = tokenizer(template, return_tensors="pt").to(device)
    with torch.no_grad():
        logits = model(**inputs).logits

    # temp = torch.argmax(logits, dim=-1).to(device)
    # temp = tokenizer.convert_ids_to_tokens(temp[0])
    # temp = temp[1:len(temp) - 1]
    topk = torch.topk(logits.data, k=16, sorted=True, largest=True)

    topk = topk.indices[0]
    # temp_mask_candidates = tokenizer.convert_ids_to_tokens(topk[5])    # [MASK]的位置
    # temp_mask_candidates = tokenizer.convert_ids_to_tokens(topk[-3])   # [MASK]的位置
    temp_mask_candidates = tokenizer.convert_ids_to_tokens(topk[-4])    # [MASK]的位置

    print(temp_mask_candidates)

    mask_candidates = temp_mask_candidates[:15]
    if hyponyms[i] in temp_mask_candidates:
        print("mark")
        index = temp_mask_candidates.index(hyponyms[i])
        for idx in range(index, 15):
            mask_candidates[idx] = temp_mask_candidates[idx + 1]
    print(mask_candidates)

    true_table = [0 for idx in range(15)]

    for idx in range(15):
        if idx < len(hypernyms[i]):
            if mask_candidates[idx] in hypernyms[i]:
                print("\n下位词：{}, 标准上位词：{}, 序号：{}".format(hyponyms[i], hypernyms[i], i))
                print("候选上位词：" + str(mask_candidates))
                true_table[idx] = 1
                text = template.replace("[MASK]", mask_candidates[idx])

                # text = ''.join(text)
                print(text)
                print("命中上位词: {}".format(mask_candidates[idx]))
                print()
    print("-----------------------------------------------------------------------------")
    ith_MAP = mean_average_precision(true_table, len(hypernyms[i]))
    ith_MRR = mean_reciprocal_rank(true_table)
    ith_query_precision_at_1 = precision_at_k(true_table, 1, len(hypernyms[i]))
    ith_query_precision_at_3 = precision_at_k(true_table, 3, len(hypernyms[i]))
    ith_query_precision_at_5 = precision_at_k(true_table, 5, len(hypernyms[i]))
    ith_query_precision_at_15 = precision_at_k(true_table, 15, len(hypernyms[i]))

    metrics['MAP'] += ith_MAP
    metrics['MRR'] += ith_MRR
    metrics['p@1'] += ith_query_precision_at_1
    metrics['p@3'] += ith_query_precision_at_3
    metrics['p@5'] += ith_query_precision_at_5
    metrics['p@15'] += ith_query_precision_at_15

print("MAP: {}".format(metrics['MAP'] / len(hyponyms)))
print("MRR: {}".format(metrics['MRR'] / len(hyponyms)))
print("precision @ 1: {}".format(metrics['p@1'] / len(hyponyms)))
print("precision @ 3: {}".format(metrics['p@3'] / len(hyponyms)))
print("precision @ 5: {}".format(metrics['p@5'] / len(hyponyms)))
print("precision @ 15: {}".format(metrics['p@15'] / len(hyponyms)))

