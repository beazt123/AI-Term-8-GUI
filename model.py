import torch
import torch.nn as nn
import torch.nn.functional as F

class FF(nn.Module): # Please change the name to your own network
  def __init__(self, input_size):
    super(FF, self).__init__()

    self.fc1 = nn.Linear(input_size, 2048)
    self.fc2 = nn.Linear(2048, 512)
    self.fc3 = nn.Linear(512, 128)
    self.fc4 = nn.Linear(128, 1)
    self.dropout = nn.Dropout(0.3)
    # self.nonlinearity = nn.ReLU()
    self.batchnorm1 = nn.BatchNorm1d(2048)
    self.batchnorm2 = nn.BatchNorm1d(512)
    self.batchnorm3 = nn.BatchNorm1d(128)

  def forward(self, x):

    x = self.dropout(F.relu(self.batchnorm1(self.fc1(x))))
    x = self.dropout(F.relu(self.batchnorm2(self.fc2(x))))
    x = self.dropout(F.relu(self.batchnorm3(self.fc3(x))))
    x = F.relu(self.fc4(x))

    return x

input_size = 18
model = FF(input_size)
checkpoint = torch.load("model_20210809_032813_35.pt", map_location="cpu")
model.load_state_dict(checkpoint['model_state_dict'])
model.eval()

# "#followers",
# "#friends",
# "#retweets",
# "#favorites",
# "weekend",
# "entity_count",
# "hashtag_count",
# "mention_count",
# "url_count",
# "tlen",
# "ratio_fav_#followers",
# "time_importance",
# "sentiment_ppn",
# "sine_hour",
# "cosine_hour",
# "sine_day",
# "cosine_day",
# "sine_day_of_week",
# "cosine_day_of_week"