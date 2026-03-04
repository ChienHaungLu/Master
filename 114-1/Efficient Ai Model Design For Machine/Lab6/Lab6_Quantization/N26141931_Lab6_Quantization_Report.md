# Lab 6 - Transformer Quantization Report
  
1. **What's the difference between SmoothQuant and the method in Lab3?** 10%

    SmoothQuant和在Lab 3學習的基本量化方法的主要區別在於如何處理模型中的Activation和outliers，這是成功量化的最大的挑戰。
    在Lab6使用smoothing factor $\mathbf{s}$ 將outliers縮小,使得平滑後的激活 ($\hat{\mathbf{X}}$)不再有極端離群值，以利於進行硬體的逐個Token INT8量化，另外在靜態量化部分使得硬體上可以透過最快的INT8速度執行，簡單來說Lab6相對於Lab3實現高效能、低記憶體且幾乎無準確度損失的INT8部署，這降低了部署和運行LLM的硬體門檻

2. **When applying SmoothQuant, **where** do activation values get divided by the smooth factor?** 10%
    
    具體位置是在Layer Normalization之後並在Linear layer(矩陣乘法)運算之前，這麼做是為了使其分佈更均勻，便於後續的量化
    
    
3. **How is the smooth factor being calculated?** 10%
    
    Smooth因子主要透過這個公式計算(j是通到數、alpha是平滑參數):$$
\mathbf{s}_j = \frac{\max(|\mathbf{X}_j|)^\alpha}{\max(|\mathbf{W}_j|)^{1-\alpha}}$$
    

4. **What's the difference between ViT-S and CNN models when doing quantization?** 10%

    CNN 因卷積運算具備權重共享與Local Receptive Field，使Activate distribution相對集中且穩定，因此在8-bit量化下誤差實作難度低；相反地，由於ViT LayerNorm與Attention所導致的嚴重離群值與極端不均勻分佈，使量化範圍被少數大值撐大，導致資訊大量流失，使準確率對量化極度敏感，故須透過SmoothQuant等技術才能勉強使效能維持一定水準

5. **What's your observation on the visualization of weight and activation values distribution?** 10%
    
    Activation Distribution:
    當在Smoothing之前其動態範圍大，導致離群值多而難以量化，而在Smoothing之後始動態範圍小讓分佈平滑易於量化。

    Weights Distribution:
    在Smoothing之前動態範圍小使分佈平滑易於量化，而在smoothing後動態範圍變大使分佈仍平滑，但這需要更精準的Per-Channel Quantization來處理其更大的範圍