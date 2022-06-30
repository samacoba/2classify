#coding:utf-8
#２クラス検出実行用スクリプト
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

    
    Md['g_ab_c1'] = Conv()
    Md['g_ab_c2'] = Conv()
    Md['g_ab_d'] = DeConv(out_ch = Gv.B_out_ch)

    Md['g_ba_c'] = Conv()
    Md['g_ba_d1'] = DeConv(out_ch = 3)
    Md['g_ba_d2'] = DeConv(out_ch = 3)

    Md['d_a'] = Discriminator(in_ch = 3)
    Md['d_b'] = Discriminator(in_ch = 6)


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

    print(len(DataA['posi1']), len(DataA['posi2']), learn_time, sep = '\t')


#DiscoGAN UPdate
def run_disco_update():

    # read data    
    DataN = get_data_N_rand(DataA, N_pic = Gv.BatchSize, imgH = 64, imgW = 64, keys =['x'])
    x_a = (torch.tensor(DataN['x'])).to('cuda') 
    
    DataN = get_data_N_rand(DataB, N_pic = Gv.BatchSize, imgH = 64, imgW = 64, keys =['x'])        
    x_b = (torch.tensor(DataN['x'])).to('cuda')


    # conversion
    x_ax_1 = Md['g_ab_c1'](x_a)
    x_ax_2 = Md['g_ab_c2'](x_a)
    x_ab_1 = Md['g_ab_d'](x_ax_1)
    x_ab_2 = Md['g_ab_d'](x_ax_2)

    x_ab = torch.cat((x_ab_1, x_ab_2), dim=1)#3ch,3ch⇒6ch

    x_b_1, x_b_2 = torch.chunk(x_b, 2, dim=1)#6ch⇒3ch,3ch

    x_ba_1 = Md['g_ba_d1'](Md['g_ba_c'](x_b_1))
    x_ba_2 = Md['g_ba_d2'](Md['g_ba_c'](x_b_2))

    x_ba = x_ba_1 + x_ba_2

    # reconversion
    x_aba_1 = Md['g_ba_d1'](Md['g_ba_c'](x_ab_1))
    x_aba_2 = Md['g_ba_d2'](Md['g_ba_c'](x_ab_2))
    x_aba = x_aba_1 + x_aba_2

    x_bab_1 = Md['g_ab_d'](Md['g_ab_c1'](x_ba))
    x_bab_2 = Md['g_ab_d'](Md['g_ab_c2'](x_ba))
    x_bab = torch.cat((x_bab_1, x_bab_2), dim=1)#3ch,3ch⇒6ch

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
        Md['g_ab_c1'].zero_grad()
        Md['g_ab_c2'].zero_grad()
        Md['g_ab_d'].zero_grad()
        Md['g_ba_c'].zero_grad()
        Md['g_ba_d1'].zero_grad()
        Md['g_ba_d2'].zero_grad()

        gen_loss.backward()
        Op['g_ab_c1'].step()
        Op['g_ab_c2'].step()
        Op['g_ab_d'].step()
        Op['g_ba_c'].step()
        Op['g_ba_d1'].step()
        Op['g_ba_d2'].step()
 

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
    x_ax_1 = Md['g_ab_c1'](x_a)
    x_ax_2 = Md['g_ab_c2'](x_a)
    x_ab_1 = Md['g_ab_d'](x_ax_1)
    x_ab_2 = Md['g_ab_d'](x_ax_2)
    x_ab = torch.cat((x_ab_1, x_ab_2), dim=1)#3ch,3ch⇒6ch

    #numpy化
    DataA['y'] = x_ab.to('cpu').detach().numpy()
    DataA['y1'] = x_ab_1.to('cpu').detach().numpy()
    DataA['y2'] = x_ab_2.to('cpu').detach().numpy()

    if Gv.disp_m:
        DataA['m1'] = np.mean(x_ax_1.to('cpu').detach().numpy(), axis=1, keepdims= True)*100
        DataA['m2'] = np.mean(x_ax_2.to('cpu').detach().numpy(), axis=1, keepdims= True)*100

    #ピーク位置検出
    DataA['posi1'] = get_posi_from_img(DataA['y1'], threshold = Gv.Peek_threshold )
    DataA['posi2'] = get_posi_from_img(DataA['y2'], threshold = Gv.Peek_threshold )

    #円重ね描き
    DataA['x_y_circle'] = add_cross_from_posi(DataA['x'], DataA['posi1'], color = Gv.Cir_color[0])
    DataA['x_y_circle'] = add_cross_from_posi(DataA['x_y_circle'], DataA['posi2'], color = Gv.Cir_color[1])

    #bokeh描画
    if Gv.disp_m:
        imgs = [DataA['x_y_circle'][0], DataA['y1'][0], DataA['y2'][0], DataA['m1'][0], DataA['m2'][0]]
    else:
        imgs = [DataA['x_y_circle'][0], DataA['y1'][0], DataA['y2'][0]]
    infos = ['iteration = ' + str(Iteration), 'countA = ' + str(len(DataA['posi1'])), 'countB = ' + str(len(DataA['posi2']))]
    bokeh_update_img(imgs = imgs , infos = infos)
    





