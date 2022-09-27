import numpy as np
import torch
from torch import nn
import pandas as pd

import matplotlib.pyplot as plt

"""
Github: Yonv1943 Zen4 Jia1 hao2
https://github.com/Yonv1943/DL_RL_Zoo/blob/master/RNN
The source of training data 
https://github.com/L1aoXingyu/
code-of-learn-deep-learning-with-pytorch/blob/master/
chapter5_RNN/time-series/lstm-time-series.ipynb
"""

# 对数据中的每一行进行操作


def create_dataset(dataset, look_back):
    #这里的look_back与timestep相同
    # print("data_set: ", dataset.shape)
    dataX, dataY = [], []
    for i in range(dataset.shape[1]-look_back-1):
        a = dataset[:, i:(i+look_back)]
        dataX.append(a)
        dataY.append(dataset[:, i + look_back])
    # print("dataX shape: ", np.array(dataX).shape, "\tdataY shape: ", np.array(dataY).shape)
    return np.array(dataX).reshape(150, 30), np.array(dataY).reshape(-1, 1)



def load_data3(data, prefix):
    # passengers number of international airline , 1949-01 ~ 1960-12 per month
    parking_id = set(data[data["PREFIX"] == prefix]["PARKING_ID"])
    all = np.zeros(shape=(1, 181))
    t = pd.DataFrame()
    time_list = ["03.25", "03.26", "03.29", "03.30", "03.31", "04.01", "04.02"]
    for i in time_list:
        for j in parking_id:
            temp = data[(data["PREFIX"] == prefix) &
                        (data["PARKING_ID"] == j)][i]
            temp = np.array(temp.T)
        #print(all)
            all = np.row_stack((all, temp))

        print("all: ", all.shape)
        # 删除之前的第一行的全零行
        all = np.delete(all, 0, axis=0)
        # print("all: ", all.shape)
        all = np.mean(all, axis=0)
        # print("all: ", all.shape)
        t[i] = all

    return t


inp_dim = 30
out_dim = 1
data = pd.read_csv(
    r"D:\学科\比赛\计算机设计大赛\代码\爬虫\上海停车\处理\final车场信息_上海停车_总车位_allMerged_timeAggregated_type1.csv")
data = load_data3(data, "pd")
data = np.array(data.T)
print(data.shape)
data_x, data_y = [], []
for i in range(data.shape[0]):
    t = create_dataset(data[i:i+1, :], inp_dim)
    # print(t)
    data_x.append(t[0])
    data_y.append(t[1])

data_x = np.array(data_x).reshape(-1, 30)
data_y = np.array(data_y).reshape(-1, 1)

train_size = int(len(data_x) * 0.75)
train_x = data_x[:train_size].copy()
train_y = data_y[:train_size].copy()
print(train_y.shape)

def run_train_gru():
    inp_dim = 3
    out_dim = 1
    batch_size = 12 * 4

    '''load data'''
    data = load_data()
    data_x = data[:-1, :]
    data_y = data[+1:, 0]
    assert data_x.shape[1] == inp_dim

    train_size = int(len(data_x) * 0.75)

    train_x = data_x[:train_size]
    train_y = data_y[:train_size]
    train_x = train_x.reshape((train_size, inp_dim))
    train_y = train_y.reshape((train_size, out_dim))

    '''build model'''
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    net = RegGRU(inp_dim, out_dim, mod_dim=12, mid_layers=2).to(device)
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(net.parameters(), lr=1e-2)

    '''train'''
    var_x = torch.tensor(train_x, dtype=torch.float32, device=device)
    var_y = torch.tensor(train_y, dtype=torch.float32, device=device)

    batch_var_x = list()
    batch_var_y = list()

    for i in range(batch_size):
        j = train_size - i
        batch_var_x.append(var_x[j:])
        batch_var_y.append(var_y[j:])

    from torch.nn.utils.rnn import pad_sequence
    batch_var_x = pad_sequence(batch_var_x)
    batch_var_y = pad_sequence(batch_var_y)

    with torch.no_grad():
        weights = np.tanh(np.arange(len(train_y)) * (np.e / len(train_y)))
        weights = torch.tensor(weights, dtype=torch.float32, device=device)

    for e in range(256):
        out = net(batch_var_x)

        # loss = criterion(out, batch_var_y)
        loss = (out - batch_var_y) ** 2 * weights
        loss = loss.mean()

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if e % 100 == 0:
            print('Epoch: {}, Loss: {:.5f}'.format(e, loss.item()))

    '''eval'''
    net = net.eval()

    test_x = data_x.copy()
    test_x[train_size:, 0] = 0
    test_x = test_x[:, np.newaxis, :]
    test_x = torch.tensor(test_x, dtype=torch.float32, device=device)
    for i in range(train_size, len(data) - 2):
        test_y = net(test_x[:i])
        test_x[i + 1, 0, 0] = test_y[-1]
    pred_y = test_x[1:, 0, 0]
    pred_y = pred_y.cpu().data.numpy()

    diff_y = pred_y[train_size:] - data_y[train_size:-1]
    l1_loss = np.mean(np.abs(diff_y))
    l2_loss = np.mean(diff_y ** 2)
    print("L1: {:.3f}    L2: {:.3f}".format(l1_loss, l2_loss))
    plt.plot(pred_y, 'r', label='pred')
    plt.plot(data_y, 'b', label='real')
    plt.legend(loc='best')
    plt.pause(4)


def run_train_lstm(epoch=384, batch_size=48, learning_rate=1e-2):
    inp_dim = 30
    out_dim = 1
    mid_dim = 8
    mid_layers = 1
    # batch_size = 12 * 4
    mod_dir = '.'

    '''load data'''
    # data = load_data2(r"D:\学科\比赛\计算机设计大赛\代码\爬虫\上海停车\处理\final车场信息_上海停车_总车位_allMerged_timeAggregated_type1.csv",
    #   "pd")  # axis1: number, year, month

    # data_x, data_y = np.apply_along_axis(create_dataset, axis=1, arr=data, look_back=inp_dim)
    # data = pd.read_csv(
    #     r"D:\学科\比赛\计算机设计大赛\代码\爬虫\上海停车\处理\final车场信息_上海停车_总车位_allMerged_timeAggregated_type1.csv")
    # data = load_data3(data, "pd")
    # data = np.array(data.T)
    # print(data.shape)
    # data_x, data_y = [], []
    # for i in range(data.shape[0]):
    #     t = create_dataset(data[i:i+1, :], inp_dim)
    #     # print(t)
    #     data_x.append(t[0])
    #     data_y.append(t[1])

    # data_x = np.array(data_x).reshape(-1, 30)
    # data_y = np.array(data_y).reshape(-1, 1)

    # train_size = int(len(data_x) * 0.75)
    # train_x = data_x[:train_size].copy()
    # train_y = data_y[:train_size].copy()
    # print(train_y.shape)

    # train_x = train_x.reshape((-1, 1, inp_dim))
    # train_y = train_y.reshape((-1, 1, out_dim))
    assert train_x.shape[0] == train_y.shape[0]

    '''build model'''
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    net = RegLSTM(inp_dim, out_dim, mid_dim, mid_layers).to(device)
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(net.parameters(), lr=learning_rate)

    '''train'''
    var_x = torch.tensor(train_x, dtype=torch.float32, device=device)
    var_y = torch.tensor(train_y, dtype=torch.float32, device=device)

    batch_var_x = list()
    batch_var_y = list()

    for i in range(batch_size):
        j = train_size - i
        batch_var_x.append(var_x[j:])
        batch_var_y.append(var_y[j:])

    from torch.nn.utils.rnn import pad_sequence
    batch_var_x = pad_sequence(batch_var_x)
    batch_var_y = pad_sequence(batch_var_y)

    with torch.no_grad():
        weights = np.tanh(np.arange(len(train_y)) * (np.e / len(train_y)))
        weights = torch.tensor(weights, dtype=torch.float32, device=device)

    print(batch_var_x.shape)
    print("Training Start")
    for e in range(384):
        out = net(batch_var_x)

        # loss = criterion(out, batch_var_y)
        loss = (out - batch_var_y) ** 2 * weights
        loss = loss.mean()

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if e % 64 == 0:
            print('Epoch: {:4}, Loss: {:.5f}'.format(e, loss.item()))

    prefix = "epo_{epo}_bat_{bat}_lr_{lr}".format(
        epo=epoch, bat=batch_size, lr=learning_rate)
    torch.save(net.state_dict(), '{}/net_LSTM1_'.format(mod_dir) +
               prefix + '.pth')
    print("Save in:", '{}/net_LSTM1_'.format(mod_dir) +
          prefix + '.pth')

    '''eval'''
    net.load_state_dict(torch.load('{}/net_LSTM1_'.format(mod_dir) +
                                   prefix + '.pth',
                                   map_location=lambda storage, loc: storage))
    net = net.eval()

    test_x = data_x[train_size:].copy()
    pred_y = []
    test_x = torch.tensor(test_x, dtype=torch.float32, device=device)

    '''simple way but no elegant'''
    # for i in range(train_size, len(data) - 2):
    #     test_y = net(test_x[:i])
    #     test_x[i, 0, 0] = test_y[-1]

    '''elegant way but slightly complicated'''
    eval_size = 1
    zero_ten = torch.zeros((mid_layers, eval_size, mid_dim),
                           dtype=torch.float32, device=device)
    # # hc 相当于lstm的参数，记录了状态
    # # 每次传递hc参数用作标记状态
    # test_y, hc = net.output_y_hc(test_x[:train_size], (zero_ten, zero_ten))
    # test_x[train_size + 1, 0, 0] = test_y[-1]
    # for i in range(train_size + 1, len(data) - 2):
    #     test_y, hc = net.output_y_hc(test_x[i:i + 1], hc)
    #     test_x[i + 1, 0, 0] = test_y[-1]
    # pred_y = test_x[1:, 0, 0]
    # pred_y = pred_y.cpu().data.numpy()

    print("eval test_x.shape: ", test_x.shape)
    hc = (zero_ten, zero_ten)
    for i in range(test_x.shape[0]):
        t = test_x[i]
        # print("t.shape ", t.shape)
        # test_y, hc = net.output_y_hc(
        #     t.reshape((-1, 1, inp_dim)), hc)
        pred_y.append(net(t.reshape((-1, 1, inp_dim))))

    # pred_y = np.array(pred_y).reshape(-1, 1)
    pred_y = [i.item() for i in pred_y]
    pred_y = np.array(pred_y).reshape(-1, 1)
    print("pred_y.shape: ", pred_y.shape,
          "data_y[train_size: ].shape: ", data_y[train_size:].shape)
    diff_y = pred_y - data_y[train_size:]
    diff_y = np.array(diff_y)
    print((diff_y))
    # diff_y = diff_y.cpu().data().numpy()
    l1_loss = np.mean(np.abs(diff_y))
    l2_loss = np.mean(diff_y ** 2)
    print("L1: {:.6f}    L2: {:.6f}".format(l1_loss, l2_loss))

    plt.plot(pred_y[:], 'r', label='pred')
    plt.plot(data_y[train_size:], 'b', label='real', alpha=0.3)
    # plt.plot([train_size, train_size], [-1, 2],
    #          color='k', label='train | pred')
    plt.legend(loc='best')
    plt.savefig('lstm_reg_LSTM1_' +
                prefix+ '.png')
    plt.pause(4)
    return l1_loss

def run_origin():
    inp_dim = 30
    out_dim = 1
    mod_dir = '.'

    '''load data'''
    # data = load_data2(r"D:\学科\比赛\计算机设计大赛\代码\爬虫\上海停车\处理\final车场信息_上海停车_总车位_allMerged_timeAggregated_type1.csv",
    #                   "pd")  # axis1: number, year, month

    # # data_x, data_y = np.apply_along_axis(create_dataset, axis=1, arr=data, look_back=inp_dim)
    # data_x, data_y = [], []
    # for i in range(data.shape[0]):
    #     t = create_dataset(data[i:i+1, :], inp_dim)
    #     # print(t)
    #     data_x.append(t[0])
    #     data_y.append(t[1])

    data = pd.read_csv(
        r"D:\学科\比赛\计算机设计大赛\代码\爬虫\上海停车\处理\final车场信息_上海停车_总车位_allMerged_timeAggregated_type1.csv")
    data = load_data3(data, "pd")
    data = np.array(data.T)
    print(data.shape)
    data_x, data_y = [], []
    for i in range(data.shape[0]):
        t = create_dataset(data[i:i+1, :], inp_dim)
        # print(t)
        data_x.append(t[0])
        data_y.append(t[1])

    data_x = np.array(data_x).reshape(-1, 30)
    data_y = np.array(data_y).reshape(-1, 1)
    print(data_x.shape, data_y.shape)
    assert data_x.shape[1] == 30
    # assert data_y.shape == (1, 1)

    # print(data_x, data_y)
    train_size = int(len(data_x) * 0.75)
    train_x = data_x[:train_size].copy()
    train_y = data_y[:train_size].copy()
    print(train_y.shape)

    train_x = train_x.reshape((-1, 1, inp_dim))
    train_y = train_y.reshape((-1, 1, out_dim))
    assert train_x.shape[0] == train_y.shape[0]

    '''build model'''
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    net = RegLSTM(inp_dim, out_dim, mid_dim=4, mid_layers=2).to(device)
    criterion = nn.SmoothL1Loss()
    optimizer = torch.optim.Adam(net.parameters(), lr=1e-2)

    '''train'''
    var_x = torch.tensor(train_x, dtype=torch.float32, device=device)
    var_y = torch.tensor(train_y, dtype=torch.float32, device=device)
    print('var_x.size():', var_x.size())
    print('var_y.size():', var_y.size())

    # for e in range(512):
    #     out = net(var_x)
    #     loss = criterion(out, var_y)

    #     optimizer.zero_grad()
    #     loss.backward()
    #     optimizer.step()

    #     if (e + 1) % 100 == 0:  # 每 100 次输出结果
    #         print('Epoch: {}, Loss: {:.5f}'.format(e + 1, loss.item()))

    # torch.save(net.state_dict(), '{}/net.pth'.format(mod_dir))

    '''eval'''
    net.load_state_dict(torch.load(
        '{}/net_origin1.pth'.format(mod_dir), map_location=lambda storage, loc: storage))
    net = net.eval()  # 转换成测试模式

    """
    inappropriate way of seq prediction: 
    use all real data to predict the number of next month
    """
    # test_x = data_x.reshape((-1, 1, inp_dim))
    # var_data = torch.tensor(test_x, dtype=torch.float32, device=device)
    # eval_y = net(var_data)  # 测试集的预测结果
    # pred_y = eval_y.view(-1).cpu().data.numpy()

    # plt.plot(pred_y[1:], 'r', label='pred inappr', alpha=0.3)
    # plt.plot(data_y, 'b', label='real', alpha=0.3)
    # plt.plot([train_size, train_size], [-1, 2], label='train | pred')

    """
    appropriate way of seq prediction: 
    use real+pred data to predict the number of next 3 years.
    """
    # train_size = 1
    test_x = data_x[train_size:]
    pre_y = []
    # test_x[train_size:] = 0  # delete the data of next 3 years.
    test_x = torch.tensor(test_x, dtype=torch.float32, device=device)
    for i in range(len(test_x)):
        t = test_x[i]
        # print(t)
        pre_y.append(net(t.reshape((-1, 1, inp_dim))))
        # test_x = np.append(test_x.detach().numpy().reshape((1, -1)), test_y.detach().numpy().reshape((1, -1)))
        # test_x=torch.tensor(test_x, dtype=torch.float32, device=device).reshape(1, -1)
        # print(test_x)

    # pred_y = test_x.cpu().data.numpy()
    # pred_y = pred_y[:, 0, 0]
    plt.plot(data_y[train_size:train_size+200].reshape(-1), 'r', label='real')
    plt.plot(pre_y[:200], 'g', label='pred appr')

    plt.legend(loc='best')
    plt.savefig('lstm_origin.png')
    plt.pause(4)


class RegLSTM(nn.Module):
    def __init__(self, inp_dim, out_dim, mid_dim, mid_layers):
        super(RegLSTM, self).__init__()

        self.rnn = nn.LSTM(inp_dim, mid_dim, mid_layers)  # rnn
        self.reg = nn.Sequential(
            nn.Linear(mid_dim, mid_dim),
            nn.Tanh(),
            nn.Linear(mid_dim, out_dim),
        )  # regression

    def forward(self, x):
        y = self.rnn(x)[0]  # y, (h, c) = self.rnn(x)
        # print(y.shape)
        seq_len, batch_size, hid_dim = y.shape
        y = y.view(-1, hid_dim)
        y = self.reg(y)
        y = y.view(seq_len, batch_size, -1)
        return y

    """
    PyCharm Crtl+click nn.LSTM() jump to code of PyTorch:
    Examples::
        >>> rnn = nn.LSTM(10, 20, 2)
        >>> input = torch.randn(5, 3, 10)
        >>> h0 = torch.randn(2, 3, 20)
        >>> c0 = torch.randn(2, 3, 20)
        >>> output, (hn, cn) = rnn(input, (h0, c0))
    """

    def output_y_hc(self, x, hc):
        y, hc = self.rnn(x, hc)  # y, (h, c) = self.rnn(x)

        seq_len, batch_size, hid_dim = y.size()
        y = y.view(-1, hid_dim)
        y = self.reg(y)
        y = y.view(seq_len, batch_size, -1)
        return y, hc


class RegGRU(nn.Module):
    def __init__(self, inp_dim, out_dim, mod_dim, mid_layers):
        super(RegGRU, self).__init__()

        self.rnn = nn.GRU(inp_dim, mod_dim, mid_layers)
        self.reg = nn.Linear(mod_dim, out_dim)

    def forward(self, x):
        x, h = self.rnn(x)  # (seq, batch, hidden)

        seq_len, batch_size, hid_dim = x.shape
        x = x.view(-1, hid_dim)
        x = self.reg(x)
        x = x.view(seq_len, batch_size, -1)
        return x

    def output_y_h(self, x, h):
        y, h = self.rnn(x, h)

        seq_len, batch_size, hid_dim = y.size()
        y = y.view(-1, hid_dim)
        y = self.reg(y)
        y = y.view(seq_len, batch_size, -1)
        return y, h


def load_data():
    # passengers number of international airline , 1949-01 ~ 1960-12 per month
    seq_number = np.array(
        [112., 118., 132., 129., 121., 135., 148., 148., 136., 119., 104.,
         118., 115., 126., 141., 135., 125., 149., 170., 170., 158., 133.,
         114., 140., 145., 150., 178., 163., 172., 178., 199., 199., 184.,
         162., 146., 166., 171., 180., 193., 181., 183., 218., 230., 242.,
         209., 191., 172., 194., 196., 196., 236., 235., 229., 243., 264.,
         272., 237., 211., 180., 201., 204., 188., 235., 227., 234., 264.,
         302., 293., 259., 229., 203., 229., 242., 233., 267., 269., 270.,
         315., 364., 347., 312., 274., 237., 278., 284., 277., 317., 313.,
         318., 374., 413., 405., 355., 306., 271., 306., 315., 301., 356.,
         348., 355., 422., 465., 467., 404., 347., 305., 336., 340., 318.,
         362., 348., 363., 435., 491., 505., 404., 359., 310., 337., 360.,
         342., 406., 396., 420., 472., 548., 559., 463., 407., 362., 405.,
         417., 391., 419., 461., 472., 535., 622., 606., 508., 461., 390.,
         432.], dtype=np.float32)
    # assert seq_number.shape == (144, )
    # plt.plot(seq_number)
    # plt.ion()
    # plt.pause(1)
    seq_number = seq_number[:, np.newaxis]

    # print(repr(seq))
    # 1949~1960, 12 years, 12*12==144 month
    seq_year = np.arange(12)
    seq_month = np.arange(12)
    seq_year_month = np.transpose(
        [np.repeat(seq_year, len(seq_month)),
         np.tile(seq_month, len(seq_year))],
    )  # Cartesian Product

    print("seq_year_month: ", seq_year_month)
    seq = np.concatenate((seq_number, seq_year_month), axis=1)
    print("seq: ", seq)

    # normalization
    seq = (seq - seq.mean(axis=0)) / seq.std(axis=0)
    return seq




def load_data2(path, prefix):
    data = pd.read_csv(path)
    parking_id = set(data[data["PREFIX"] == prefix]["PARKING_ID"])
    all = np.zeros(shape=(1, 181))
    # time_list = ["03.25", "03.26", "03.29", "03.30", "03.31", "04.01", "04.02"]
    time_list = ["03.25", "03.26"]
    for i in parking_id:
        temp = data[(
            data["PARKING_ID"] == i)][time_list]
        temp = np.array(temp.T)
        #print(all)
        all = np.row_stack((all, temp))
        # 删除之前的第一行的全零行
        all = np.delete(all, 0, axis=0)
    return all

    # # assert seq_number.shape == (144, )
    # # plt.plot(seq_number)
    # # plt.ion()
    # # plt.pause(1)
    # seq_number = seq_number[:, np.newaxis]

    # # print(repr(seq))
    # # 1949~1960, 12 years, 12*12==144 month
    # seq_year = np.arange(12)
    # seq_month = np.arange(12)
    # seq_year_month = np.transpose(
    #     [np.repeat(seq_year, len(seq_month)),
    #      np.tile(seq_month, len(seq_year))],
    # )  # Cartesian Product

    # print("seq_year_month: ", seq_year_month)
    # seq = np.concatenate((seq_number, seq_year_month), axis=1)
    # print("seq: ", seq)

    # # normalization
    # seq = (seq - seq.mean(axis=0)) / seq.std(axis=0)
    # return seq



if __name__ == '__main__':
    run_train_lstm()
    # run_train_gru()
    # run_origin()
