# 建立虛擬環境
1. python -m venv venv
2. Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser
3. .\venv\Scripts\activate
4. pip install -r requirements.txt



Note: 如果Keras不相容可能需要更新版本: pip install --upgrade keras tensorflow



model: 是一個pre-trained model，此外
best_model.keras是存已學習好的最佳模型用以載入