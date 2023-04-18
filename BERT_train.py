from transformers import BertTokenizer, BertForMaskedLM
import torch
from tqdm import tqdm
from task9_scorer import mean_average_precision

def get_MAP(logits, cur_hyponym, cur_hypernyms):
    topk = torch.topk(logits.data, k=16, sorted=True, largest=True)
    topk = topk.indices[0]
    first_mask_candidates = tokenizer.convert_ids_to_tokens(topk[-4])
    second_mask_candidates = tokenizer.convert_ids_to_tokens(topk[-3])
    temp_mask_candidates = []
    for idx in range(16):
        temp_mask_candidates.append(first_mask_candidates[idx] + second_mask_candidates[idx])

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

if __name__ == "__main__":
    dataset = "BK"
    device = "cuda:0"
    train_hyponyms = []
    train_hypernyms = []
    with open(f"{dataset}/train_set.txt", "r", encoding='utf-8') as file:
        for line in tqdm(file):
            temp = line.strip().split('\t')
            train_hyponyms.append(temp[0])
            train_hypernyms.append(temp[1:])

    dev_hyponyms = []
    dev_hypernyms = []
    with open(f"{dataset}/dev_set.txt", "r", encoding='utf-8') as file:
        for line in tqdm(file):
            temp = line.strip().split('\t')
            dev_hyponyms.append(temp[0])
            dev_hypernyms.append(temp[1:])

    # test_hyponyms = []
    # test_hypernyms = []
    # with open(f"{dataset}/test_set.txt", "r", encoding='utf-8') as file:
    #     for line in tqdm(file):
    #         temp = line.strip().split('\t')
    #         test_hyponyms.append(temp[0])
    #         test_hypernyms.append(temp[1:])

    tokenizer = BertTokenizer.from_pretrained("Langboat/mengzi-bert-base")
    model = BertForMaskedLM.from_pretrained("Langboat/mengzi-bert-base")
    model.cuda(device)

    # print("————————————————————————————————before fine-tuning————————————————————————————————")
    # # display(model)
    # loss_before_finetuning = 0
    # start_time = time.time()
    # model.eval()
    # for i in range(len(dev_hyponyms)):
    #     prompt = "我不觉得" + dev_hyponyms[i] + "不是一类[MASK][MASK]。"
    #     # prompt = "有人说" + dev_hyponyms[i] + "是最好的[MASK][MASK]之一。"
    #     # prompt = "你最喜欢的[MASK][MASK]是" + dev_hyponyms[i] + "吗？"
    #     inputs = tokenizer(prompt, return_tensors="pt").to("cuda")
    #     for hypernym in dev_hypernyms[i]:
    #         labels = tokenizer("我不觉得" + dev_hyponyms[i] + "不是一类" + hypernym + "。", return_tensors="pt")["input_ids"].to("cuda")
    #         # labels = tokenizer("有人说" + dev_hyponyms[i] + "是最好的" + hypernym + "之一。", return_tensors="pt")["input_ids"].to("cuda")
    #         # labels = tokenizer("你最喜欢的" + hypernym + "是" + dev_hyponyms[i] + "吗？", return_tensors="pt")["input_ids"].to("cuda")
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
        print(f"————————————————————————————————epoch:{epoch + 1}————————————————————————————————")
        model.train()
        for i in tqdm(range(len(train_hyponyms)), ncols=80):
            prompt = "我不觉得" + train_hyponyms[i] + "不是一类[MASK][MASK]。"
            # prompt = "有人说" + train_hyponyms[i] + "是最好的[MASK][MASK]之一。"
            # prompt = "你最喜欢的[MASK][MASK]是" + train_hyponyms[i] + "吗？"
            for hypernym in train_hypernyms[i]:
                inputs = tokenizer(prompt, return_tensors="pt").to(device)
                labels = tokenizer("我不觉得" + train_hyponyms[i] + "不是一类" + hypernym + "。", return_tensors="pt")["input_ids"].to(device)
                # labels = tokenizer("有人说" + train_hyponyms[i] + "是最好的" + hypernym + "之一。", return_tensors="pt")["input_ids"].to("cuda")
                # labels = tokenizer("你最喜欢的" + hypernym + "是" + train_hyponyms[i] + "吗？", return_tensors="pt")["input_ids"].to("cuda")
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
            prompt = "我不觉得" + dev_hyponyms[i] + "不是一类[MASK][MASK]。"
            # prompt = "有人说" + dev_hyponyms[i] + "是最好的[MASK][MASK]之一。"
            # prompt = "你最喜欢的[MASK][MASK]是" + dev_hyponyms[i] + "吗？"
            inputs = tokenizer(prompt, return_tensors="pt").to(device)
            for hypernym in dev_hypernyms[i]:
                labels = tokenizer("我不觉得" + dev_hyponyms[i] + "不是一类" + hypernym + "。", return_tensors="pt")["input_ids"].to(device)
                # labels = tokenizer("有人说" + dev_hyponyms[i] + "是最好的" + hypernym + "之一。", return_tensors="pt")["input_ids"].to("cuda")
                # labels = tokenizer("你最喜欢的" + hypernym + "是" + dev_hyponyms[i] + "吗？", return_tensors="pt")["input_ids"].to("cuda")
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
            model.save_pretrained(f"dont_juede_not_mengzi_BERT_{dataset}")
            no_gain_epochs = 0
        else:
            no_gain_epochs += 1
        if no_gain_epochs >= patience:
            print("early stop.")
            break
















