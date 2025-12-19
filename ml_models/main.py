import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from esp_ppq.api import espdl_quantize_torch
from esp_ppq.executor.torch import TorchExecutor
import src.painter as painter
import src.wakeDataset as waked
import src.wakeLoss as wakel
import src.wakeModel as wakem
import src.extract_mfcc as mfcc

def train_model(train_loader, test_loader, num_epochs, learning_rate=0.001):
    model = wakem.LightweightKWS(num_classes=1)
    critizen = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.AdamW(
        model.parameters(), 
        lr=0.0005,  # 降低学习率
        betas=(0.9, 0.99),  # 调整beta
        weight_decay=0.001,  # 减少权重衰减
        eps=1e-7
    )
    model.train()
    losses = []
    accuracies = []
    for epoch in range(num_epochs):
        model.train()
        total_loss = 0
        for feature,label in train_loader:
            label = label.float().view(-1, 1)
            out = model(feature)
            loss = critizen(out, label)
            optimizer.zero_grad()
            loss.backward()
            # torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            total_norm = 0
            for p in model.parameters():
                if p.grad is not None:
                    param_norm = p.grad.data.norm(2)
                    total_norm += param_norm.item() ** 2
            total_norm = total_norm ** 0.5
            # print(f"Gradient norm: {total_norm:.6f}")
            optimizer.step()
            total_loss += loss.item()
        total = 0
        correct = 0
        
        with torch.no_grad():
            model.eval()
            for feature, label in test_loader:
                label = label.view(-1, 1)
                out = model(torch.squeeze(feature))
                predicted = (torch.sigmoid(out)>0.5).float()
                total += label.size(0)
                correct += (predicted == label).sum().item()
                # print("correct:", correct)
        # print(f"Epoch [{epoch+1}/{num_epochs}], TestLoss: {total_loss / len(train_loader):.4f}, total: {total}, correct: {correct}")
        losses.append(total_loss/len(train_loader))
        accuracy = 100 * correct / total
        accuracies.append(100 * correct / total)
        # print(f"Test Accuracy: {accuracy:.2f}%")
        # if(accuracy >= 95.0):
        #     return losses, accuracies, model
    return losses, accuracies, model

def collate_fn(batch):
    # TensorDataset 迭代的时候返回的是 Tuple(x, y), 量化的时候只需要x, 不需要label y。
    batch = batch[0].to('cpu')
    return batch

def quantize_model_esp(input_shape):
    num_epochs = 9
    losses, accuracies, model= train_model(train_loader, test_loader, num_epochs, 0.001)
    
    # 原始型测试集精度
    total = 0
    correct = 0
    
    quant_loader = DataLoader(train_dataset, batch_size=20, shuffle=True)
    painter.display_traning_result(losses=losses, accuracies=accuracies)
    quant_ppq_graph = espdl_quantize_torch(
        model=model,
        espdl_export_file='xiaoa.espdl',
        calib_dataloader=quant_loader,
        calib_steps=32,
        collate_fn=collate_fn,  # 使用自定义的collate_fn
          # 修正：使用dummy_input而不是input
        input_shape=[[1, 13, input_shape[1]]],
        target="esp32p4",
        num_of_bits=8,
        device='cpu',
        calibration = 'percentile',  # 使用百分位校准
        percentile=99.9,  # 设置百分位值W
        per_channel=True,  # 启用逐通道量化
        # fp32_ops=['/classifier/classifier.2/MatMul','/classifier/classifier.0/MatMul'],  # 只保护输出层
        export_test_values=True,
        # quantization_type= 'channel_wise_symmetric',  # 使用对称量化
        #fp32_ops=['/classifier/classifier.6/MatMul','/classifier/classifier.4/Gemm', '/classifier/classifier.2/Gemm'],  # 问题最严重的层# 移除其他可选参数
    )
    
    # 量化模型测试集精度
    executor = TorchExecutor(graph=quant_ppq_graph, device='cpu')
    total = 0
    correct = 0
    model.eval()
    with torch.no_grad():
        for feature, label in test_loader:
            label = label.view(-1, 1)
            out = model(feature)
            #print("out:", out)
            predicted = (torch.sigmoid(out)>0.5).float()
            # print(predicted)
            total += label.size(0)
            correct += (predicted == label).sum().item()
    accuracy = 100 * correct / total
    print(f"Test Accuracy: {accuracy:.2f}%")
    for features, labels in test_loader:
        labels = labels.view(-1, 1)
        int8_features = [mfcc.normalize_mfcc((tensor).round())*16 for tensor in features]
        int8_features = torch.stack(int8_features)
        outputs = executor(int8_features)
        # print(outputs[0])
        predicted = (torch.sigmoid(outputs[0])>0.5).float()
        total += labels.size(0)
        correct += (predicted == labels).sum().item()
    accuracy = 100 * correct / total
    print(f"Executor Accuracy: {accuracy:.2f}%")

    return model, accuracy,executor, quant_ppq_graph

pos_features, pos_labels= mfcc.extract_features('./audio_data/xiaoa', label=1, add_noise_to_pad=True, normalize_method='cmvn', augment_audio=True)
neg_features, neg_labels= mfcc.extract_features('./audio_data/others', label=0,augment_audio=True, normalize_method='cmvn')

train_features = pos_features[:int(len(pos_features)*0.7)]+neg_features[:int(len(neg_features)*0.7)]
train_labels = pos_labels[:int(len(pos_labels)*0.7)]+neg_labels[:int(len(neg_labels)*0.7)]
test_features = pos_features[int(len(pos_features)*0.3):]+neg_features[int(len(neg_features)*0.3):]
test_labels = pos_labels[int(len(pos_labels)*0.3):]+neg_labels[int(len(neg_labels)*0.3):]

train_dataset = waked.AudioDataset(train_features, train_labels)
test_dataset = waked.AudioDataset(test_features, test_labels)
train_loader = DataLoader(train_dataset, batch_size=200, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=100, shuffle=True)
accuracy = 0.0
while accuracy<95.0:
    model, accuracy, executor, quant_ppq_graph = quantize_model_esp(train_features[0].shape)
    
    