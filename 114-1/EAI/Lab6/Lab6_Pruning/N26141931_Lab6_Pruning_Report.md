# Lab 6 - Transformer Pruning Report
  
1. **請說明 get_real_idx 實作部分是怎麼做的** 10%
    #### 取得上一層保留下來的真實索引
        prev_idx = idxs[i-1]
        
    #### 構造一個包含 "Fused Token" 索引的 tensor (用 0 代表左上角)
    #### 因上一層輸出最後token是Fused Token，如果這一層選到了它，我們將其映射回0
        B = prev_idx.shape[0]
        zeros = torch.zeros((B, 1), device=prev_idx.device, dtype=prev_idx.dtype)
        
    #### 將上一層的真實索引與Fused Token 索引(0)拼接,source_idx 形狀: [B, prev_left_tokens + 1]
        source_idx = torch.cat((prev_idx, zeros), dim=1)
        
    #### 利用這一層idx從source_idx中取值，得到相對於原始圖片真實索引
        idxs[i] = torch.gather(source_idx, dim=1, index=idxs[i])


2. **實際在哪些層做了 pruning ?** 10%
    只有當 Block 被初始化時傳入的 keep_rate 小於1.0時才會剪枝，而在第4層、第7層和第10層keep rate皆小於1.0，過需做剪枝
    
    
3. **如果沒有 get_real_idx 可視化結果會長怎樣，為什麼 ?** 10%
get_real_idx 函式目的在於做逐層的Inverse Mapping，主要是將前層相對索引映射回原始256個Image Patch中的絕對索引。若沒有**get_real_idx**則可視化結果會是完全錯誤和毫無任何實質意義
    

4. **分析視覺化的圖，這些變化代表著什麼 ?** 10%
視覺化圖片展示EViT Token Pruning在Layer 4、Layer 7及Layer 10對影像Token的選擇結果，圖片中black block代表被Pruning 的 Patch，而保留下來的image block則代表model認為對於image classification來說是重要的Token
    
    