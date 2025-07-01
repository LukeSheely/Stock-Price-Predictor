#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np #arrays


# In[2]:


import pandas as pd #dataframes


# In[3]:


import matplotlib.pyplot as plt #visualizations


# In[4]:


import yfinance as yf #yahoo finance


# In[ ]:





# In[5]:


import torch #training and building


# In[6]:


import torch.nn as nn #training and building


# In[7]:


import torch.optim as optim #training and building


# In[ ]:





# In[8]:


from sklearn.preprocessing import StandardScaler #scale


# In[9]:


from sklearn.metrics import root_mean_squared_error #evaluate


# In[ ]:





# In[10]:


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


# In[11]:


ticker = input("Input a stock ticker of your choice, for example MSFT for Microsoft \n  Ticker: ")


# In[12]:


df = yf.download(ticker, start='2021-01-01', end='2025-06-30')


# In[13]:


scaler = StandardScaler()

df['Close'] = scaler.fit_transform(df['Close'])


# In[14]:


seq_length = 30
data = []

for i in range(len(df) - seq_length):
    data.append(df.Close[i:i+seq_length])

data = np.array(data)

train_size = int(0.8 * len(data))

X_train = torch.from_numpy(data[:train_size, :-1, :]).type(torch.Tensor).to(device)
y_train = torch.from_numpy(data[:train_size, -1, :]).type(torch.Tensor).to(device)
X_test = torch.from_numpy(data[train_size:, :-1, :]).type(torch.Tensor).to(device)
y_test = torch.from_numpy(data[train_size:, -1, :]).type(torch.Tensor).to(device)


# In[15]:


class PredictionModel(nn.Module):

    def __init__(self, input_dim, hidden_dim, num_layers, output_dim):
        super(PredictionModel, self).__init__()

        self.num_layers = num_layers
        self.hidden_dim = hidden_dim

        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_dim, device=device)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_dim, device=device)

        out, (hn, cn) = self.lstm(x, (h0.detach(), c0.detach()))
        out = self.fc(out[:, -1, :])

        return out


# In[16]:


model = PredictionModel(input_dim=1, hidden_dim=32, num_layers=2, output_dim=1).to(device)


# In[17]:


criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.01)


# In[18]:


num_epochs = 200

for i in range(num_epochs):
    y_train_pred = model(X_train)

    loss = criterion(y_train_pred, y_train)

    if i % 25 == 0:
        print(i, loss.item())

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()


# In[19]:


model.eval()

y_test_pred = model(X_test)

y_train_pred = scaler.inverse_transform(y_train_pred.detach().cpu().numpy())
y_train = scaler.inverse_transform(y_train.detach().cpu().numpy())
y_test_pred = scaler.inverse_transform(y_test_pred.detach().cpu().numpy())
y_test = scaler.inverse_transform(y_test.detach().cpu().numpy())


# In[20]:


train_rmse = root_mean_squared_error(y_train[:, 0], y_train_pred[:, 0])
test_rmse = root_mean_squared_error(y_test[:, 0], y_test_pred[:, 0])


# In[21]:


fig = plt.figure(figsize=(12,10))

gs = fig.add_gridspec(4, 1)

ax1 = fig.add_subplot(gs[:3, 0])
ax1.plot(df.iloc[-len(y_test):].index, y_test, color = 'blue', label = 'Actual Price')
ax1.plot(df.iloc[-len(y_test):].index, y_test_pred, color = 'green', label = 'Predicted Price')
ax1.legend()
plt.title(f"{ticker} Stock Price Prediction")
plt.xlabel('Date')
plt.ylabel('Price')

ax2 = fig.add_subplot(gs[3, 0])
ax2.axhline(test_rmse, color = 'blue', linestyle='--', label='RMSE')
ax2.plot(df[-len(y_test):].index, abs(y_test - y_test_pred), 'r', label = 'Prediction Error')
ax2.legend()
plt.title('Prediction Error')
plt.xlabel('Date')
plt.ylabel('Error')

plt.tight_layout()
plt.show()


# In[ ]:




