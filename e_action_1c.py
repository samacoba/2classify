#coding:utf-8
#１クラス検出　実行用スクリプト
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

import numpy as np

import time
from PIL import Image
import json

#表示幅の変更
from IPython.core.display import display, HTML
display(HTML("<style>.container { width:98% !important; }</style>"))

#標準スクリプト実行
with open('e_data.py', 'r', encoding='utf-8') as f:    
    exec(f.read()) #e_data.pyをスクリプト実行
with open('e_model.py', 'r', encoding='utf-8') as f:
    exec(f.read()) #e_model.pyをスクリプト実行

class Empty_structure:#構造体のように使う空のクラス 辞書型の[""]が書きずらく.でアクセスするため
    pass

Gv = Empty_structure()

#グローバル変数
Gv.B_out_ch = 3
Gv.BatchSize = 100
Gv.Recon_rate = 0.01
Gv.Peek_threshold = 0.2
Gv.Cir_color = ['C','M']
Gv.check_near = False #画像⇒点位置の抽出時(get_posi_from_img)に点同士が近いかチェック
Gv.disp_m = False #中間部分を表示

#モデル読み込み、初期設定
def run_set_model():

    global Md#モデル
    global Op#オプティマイザ  
    global Iteration
    Iteration = 0

    Md ={}
    Op ={}
   
    Md['g_ab_c'] = Conv()
    Md['g_ab_d'] = DeConv(out_ch = Gv.B_out_ch)

    Md['g_ba_c'] = Conv()
    Md['g_ba_d'] = DeConv(out_ch = 3)


    Md['d_a'] = Discriminator()
    Md['d_b'] = Discriminator()

    for key in Md.keys():
        Md[key] = Md[key].to('cuda')
        Op[key] = optim.Adam(Md[key].parameters(),lr=2e-4, betas=(0.5, 0.999), weight_decay = 1e-4)
 

    

#トレーニングとテスト
def run_train(N_iteration = 1000, test_interval = 20):

    global Iteration
    st_time = time.time()    

    for loop in range(N_iteration): 

        run_disco_update()         
        Iteration = Iteration + 1

        if (loop % test_interval) ==  (test_interval - 1):#インターバルの最後にテスト
            run_test()

    end_time = time.time()
    learn_time = '学習時間：' + str(int(end_time -st_time)) + '秒'

    print(len(DataA['posi']), learn_time, sep = '\t')


#DiscoGAN UPdate
def run_disco_update():

    # read data    
    DataN = get_data_N_rand(DataA, N_pic = Gv.BatchSize, imgH = 64, imgW = 64, keys =['x'])
    x_a = (torch.tensor(DataN['x'])).to('cuda')
    
    DataN = get_data_N_rand(DataB, N_pic = Gv.BatchSize, imgH = 64, imgW = 64, keys =['x'])        
    x_b = (torch.tensor(DataN['x'])).to('cuda')


    # conversion
    x_ab = Md['g_ab_d'](Md['g_ab_c'](x_a))
    x_ba = Md['g_ba_d'](Md['g_ba_c'](x_b))

    # reconversion
    x_aba = Md['g_ba_d'](Md['g_ba_c'](x_ab))
    x_bab = Md['g_ab_d'](Md['g_ab_c'](x_ba))

    # reconstruction loss
    recon_loss_a = F.mse_loss(x_a, x_aba)
    recon_loss_b = F.mse_loss(x_b, x_bab)

    # discriminate
    y_a_real, feats_a_real = Md['d_a'](x_a)
    y_a_fake, feats_a_fake = Md['d_a'](x_ba)

    y_b_real, feats_b_real = Md['d_b'](x_b)
    y_b_fake, feats_b_fake = Md['d_b'](x_ab)

    # GAN loss
    gan_loss_dis_a, gan_loss_gen_a = compute_loss_gan(y_a_real, y_a_fake)
    feat_loss_a = compute_loss_feat(feats_a_real, feats_a_fake)

    gan_loss_dis_b, gan_loss_gen_b  = compute_loss_gan(y_b_real, y_b_fake)
    feat_loss_b = compute_loss_feat(feats_b_real, feats_b_fake)
   

    # compute loss
    total_loss_gen_a = (1.-Gv.Recon_rate)*(0.1*gan_loss_gen_b + 0.9*feat_loss_b) + Gv.Recon_rate * recon_loss_a
    total_loss_gen_b = (1.-Gv.Recon_rate)*(0.1*gan_loss_gen_a + 0.9*feat_loss_a) + Gv.Recon_rate * recon_loss_b

    gen_loss = total_loss_gen_a + total_loss_gen_b 
    dis_loss = gan_loss_dis_a + gan_loss_dis_b 

    if Iteration % 3 == 0:
        Md['d_a'].zero_grad()
        Md['d_b'].zero_grad()
        dis_loss.backward()
        Op['d_a'].step()
        Op['d_b'].step()
    else:
        Md['g_ab_c'].zero_grad()
        Md['g_ab_d'].zero_grad()
        Md['g_ba_c'].zero_grad()
        Md['g_ba_d'].zero_grad()
        gen_loss.backward()
        Op['g_ab_c'].step()
        Op['g_ab_d'].step()
        Op['g_ba_c'].step()
        Op['g_ba_d'].step()

def compute_loss_gan(y_real, y_fake):
    batchsize = y_real.shape[0]
    loss_dis = 0.5 * torch.sum(F.softplus(-y_real) + F.softplus(y_fake))
    loss_gen = torch.sum(F.softplus(-y_fake))
    return loss_dis / batchsize, loss_gen / batchsize

def compute_loss_feat(feats_real, feats_fake):
    losses = 0
    for feat_real, feat_fake in zip(feats_real, feats_fake):
        feat_real_mean = torch.sum(feat_real, 0) / feat_real.shape[0]
        feat_fake_mean = torch.sum(feat_fake, 0) / feat_fake.shape[0]
        l2 = (feat_real_mean - feat_fake_mean) ** 2
        loss = torch.sum(l2) / (l2.shape[0]*l2.shape[1]*l2.shape[2])
        losses += loss
    return losses


#＜テスト部分スクリプト＞            
def run_test():

    x_a = (torch.tensor(DataA['x'])).to('cuda')
    x_ax = Md['g_ab_c'](x_a)
    x_ab = Md['g_ab_d'](x_ax)
    DataA['y'] = x_ab.to('cpu').detach().numpy()

    #中間部分を可視化
    if Gv.disp_m:
        DataA['m'] = np.mean(x_ax.to('cpu').detach().numpy(),axis=1, keepdims = True) * 100
    

    #BtoAを計算            
    # x_b = (torch.tensor(DataB['x'][0:1])).to('cuda')
    # x_ba = Md['g_ba_d'](Md['g_ba_c'](x_b))
    # DataB['y'] = x_ba.to('cpu').detach().numpy()

    #ピーク位置検出
    DataA['posi'] = get_posi_from_img(DataA['y'], threshold = Gv.Peek_threshold )

    #円重ね描き
    DataA['x_y_circle'] = add_cross_from_posi(DataA['x'], DataA['posi'], color = Gv.Cir_color[0])


    #bokeh描画
    if Gv.disp_m:
        imgs = [DataA['x_y_circle'][0], DataA['y'][0], DataA['m'][0]]
    else :
        imgs = [DataA['x_y_circle'][0], DataA['y'][0]]
    infos = ['iteration = ' + str(Iteration), 'count = ' + str(len(DataA['posi']))]
    bokeh_update_img(imgs = imgs , infos = infos)
    





