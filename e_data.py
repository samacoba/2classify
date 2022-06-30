# coding: utf-8
# データ用スクリプト
# 外部依存がなく、関数内で完結する関数が中心
import torch
import torch.nn.functional as F

from skimage import draw
import numpy as np
import math

from bokeh.plotting import figure
from bokeh.layouts import gridplot
from bokeh.io import push_notebook, output_notebook

from PIL import Image

### Bokeh関連 ###

def get_bokeh_view(imgs, plot_w =512, plot_h =512):# 
    """ imgs = [[1/3ch][H][W], ……] (0-255 fp)(np)　
        return : Bv (Bokehのviewを扱うグローバル辞書オブジェクト) """
    
    _, imgH, imgW = imgs[0].shape   

    Bv = {}
    Bv['p'] = []
     
    for i in range(len(imgs)):
        
        Bv['p'].append({})
        if i == 0:#1枚目
            Bv['p'][i]['fig'] = figure(title = '-', x_range=[0, imgW], y_range=[imgH, 0])#y_range=[imgH, 0]によりy軸を反転
        else :#2枚以上の場合
            Bv['p'][i]['fig'] = figure(title = '-', x_range=Bv['p'][0]['fig'].x_range, y_range=Bv['p'][0]['fig'].y_range)

        v_img = bokeh_conv_view_img(imgs[i])#[1/3ch][H][W] ⇒ bokehのイメージに変換

        Bv['p'][i]['img']= Bv['p'][i]['fig'].image_rgba(image=[v_img],x=[0], y=[imgH], dw=[imgW], dh=[imgH])#反転軸のためy=[imgH]

    gplots = [Bv['p'][i]['fig'] for i in range( len(imgs))] 
    Bv['gp'] = gridplot( [gplots], plot_width = plot_w, plot_height = plot_h)

    output_notebook()
    from bokeh.io import show
    Bv['handle'] = show(Bv['gp'], notebook_handle=True)

    return Bv


def bokeh_conv_view_img(img):
    """ [1/3ch][H][W] 0-255⇒ bokehのイメージに変換 """

    ch, imgW, imgH= img.shape
    if ch == 1:#1chの場合 ⇒ 3ch
        img = np.broadcast_to(img, (3, imgH, imgW))
    img = np.clip(img, 0, 255).transpose((1, 2, 0))
    img_plt = np.empty((imgH,imgW), dtype="uint32")
    view = img_plt.view(dtype=np.uint8).reshape((imgH, imgW, 4))
    view[:, :, 0:3] = np.flipud(img[:, :, 0:3])#上下反転あり
    view[:, :, 3] = 255    
    return img_plt

def bokeh_update_img(imgs, infos  = []):
    """ アップデート表示 imgs = [[1/3ch][H][W], ……] (0-1 or 0-255 fp)(np) """

    for i in range(len(imgs)):
        v_img = bokeh_conv_view_img(imgs[i])
        Bv['p'][i]['img'].data_source.data['image'] = [v_img]
    
    for i in range(len(infos)):
        Bv['p'][i]['fig'].title.text= infos[i]

    push_notebook(handle = Bv['handle'])


def bokeh_save_img(fname = 'out.png'):
    """bokeh画像を保存"""
    from bokeh.io import export_png
    export_png(Bv['gp'], filename = fname)  



### 画像ファイル関係 ###

def load_img(fpath = 'imgA1.png'): 
    """ 画像ファイルパス ⇒ [1][ch][imgH][imgW] (0-255 fp32) を返却 """   
    img = np.asarray(Image.open(fpath))
    img = img.astype("float32").transpose((2, 0, 1))
    img = img.reshape((1,*img.shape))
    return img

def save_png_img(img, fpath = 'result.png'):
    """PNG保存 img =[1][1 or 3ch][imgH][imgW] (0-255 fp32) ,fpath = 画像ファイルパス"""
    
    if img.shape[1] == 1:#1ch
        img = img[0][0]        
    elif img.shape[1] == 3:#3ch
        img = img[0].transpose(1,2,0)
    else :
        print('入力chエラー')
        return

    img = (np.clip(img,0,255)).astype(np.uint8)
    img = Image.fromarray(img)
    img.save(fpath, format ='PNG')


def load_resize_img(fpath = 'imgA1.png', opath = 'resize.png', out_size = 512):
    """ 画像ファイルパス ⇒ センターリサイズ ⇒ 一時保存 ⇒[1][ch][imgH][imgW] (0-1 or 0-255, fp32) を返却 """

    #画像をRGB化、センター加工、リサイズ
    def conv_RBG_squre_resize(img, out_size = 512):#imgはpil Image    
        img = img.convert('RGB')    
        w,h = img.size
        if w > h :
            box = ((w-h)//2, 0, (w-h)//2 + h, h)
        elif h > w :
            box = (0, (h-w)//2, w, (h-w)//2 + w)     
        else :
            box = (0, 0, w, h)        
        img = img.resize((out_size, out_size), box = box)    
        return img

    img = Image.open(fpath)
    img = conv_RBG_squre_resize(img, out_size = out_size)#RGB化、center切り取り、リサイズ
    img.save(opath, format ='PNG')#一時保存(imgはpilのままなので)
    img = load_img(fpath = opath )#画像ファイルを[1][ch][imgH][imgW]で読み出し
    return img


### データ加工関連 ###

      
def get_data_N_rand(DataO, N_pic =1, imgH = 256, imgW = 256, keys =['x','t_core']):
    """ ランダムでデータの切り出し 入力：Data ⇒ return: 切り出し後の新たなDataを返却 """
    Data={}
    
    #切り出したデータの保存先 dim=[N][ch][imgH][imgW] ,float32       
    for key in keys:
        Data[key] = np.zeros((N_pic, DataO[key].shape[1], imgH, imgW), dtype = "float32")
    
    #切り出し限界を設定
    xlim = DataO[keys[0]].shape[3] - imgW + 1
    ylim = DataO[keys[0]].shape[2] - imgH + 1

    im_num =np.random.randint(0, DataO[keys[0]].shape[0], size=N_pic)#複数枚の内、切り取る写真の番号
    rotNo = np.random.randint(4, size=N_pic) #回転No
    flipNo = np.random.randint(2, size=N_pic) #フリップNo
    cutx = np.random.randint(0, xlim, size=N_pic)
    cuty = np.random.randint(0, ylim, size=N_pic)

    #np配列をもらって左右上下の反転・90、180、270°の回転した配列を返す
    def rand_rot(img, rotNo, flipNo):#img[ch][H][W]
        img = np.rot90(img, k=rotNo, axes=(1,2))
        if flipNo:
            img = np.flip(img, axis=2)
        return img 

    for i in range(0, N_pic):          
        for key in keys:
            Data[key][i] = rand_rot((DataO[key][im_num[i]][:, cuty[i]:cuty[i]+imgH, cutx[i]:cutx[i]+imgW]), rotNo[i], flipNo[i])
    
    return Data 
             

def get_local_max_point(in_imgs, threshold = 0.2): 
    """ 極大値を取得、in_imgs = [N_pic][1][imgH][imgW] np, fp32 (0-1/0-255), threshold は0-1にて範囲指定 
        ⇒return :[N_pic][1][imgH][imgW] np, fp32 (0-1)"""

    threshold =  255 * threshold
    m_imgs = torch.tensor(in_imgs)
    m_imgs = F.max_pool2d(m_imgs, kernel_size=9 ,stride=1, padding=4)
    m_imgs = m_imgs.numpy()
    p_array = (in_imgs == m_imgs) #極大値判定（True or False）の配列
    out_imgs = in_imgs * p_array
    
    out_imgs[out_imgs >= threshold] = 255
    out_imgs[out_imgs < threshold] = 0
    
    return out_imgs/255 #pointは0/1のみで返却


def conv_point_to_posi(img_p):
    """ 点画像 [1][1][imgH][imgW] ⇒　点位置 posi[N][y,x] """  

    posi = np.where(img_p[0][0] == 1)    
    posi = np.asarray(posi)#1列アレイが２本のリストとして出てくる
    posi = posi.transpose(1,0).tolist()    
    
    return posi


def conv_posi_to_point(posi, imgH =512, imgW =512, ch = 1):
    """ 点位置 posi[N][y,x] ⇒　点画像 [1][1][imgH][imgW]  0/1 """ 


    img_p = np.zeros((1, 1, imgH, imgW), dtype= "float32")
    for p in posi:
        try:
            img_p[0][0][p[0]][p[1]] = 1
        except:
            print('範囲外入力エラー？')

    return img_p #[1][1ch][H][W]


def conv_point_to_circle(in_imgs):
    """ 点画像 ⇒ circleを描画、in_imgs = [N_pic][1][imgH][imgW] np, fp32 (0-1)
        ⇒return: [N_pic][1][imgH][imgW] np, fp32 0-255"""
    cir = np.zeros((1,1,15,15), dtype= "float32")
    rr, cc = draw.circle_perimeter(7,7,5)
    cir[0][0][rr, cc] = 255
    cir = torch.tensor(cir)
    in_imgs = torch.tensor(in_imgs)
    out_imgs = F.conv_transpose2d(in_imgs, cir, bias=None, stride = 1, padding = 7)
    return out_imgs.numpy()


def conv_point_to_core(in_imgs, sig=3.0, max_xy = 15, c_xy= 7):
    """ 点画像 ⇒ コアを描画、in_imgs = [1][1][imgH][imgW] np, fp32 (0-1)
        ⇒return: [N_pic][1][imgH][imgW] np, fp32 (0-1)"""
    sig2=sig*sig
    core=np.zeros((max_xy, max_xy), dtype = "float32")
    for px in range(0, max_xy):
        for py in range(0, max_xy):
            r2 = float((px-c_xy)*(px-c_xy)+(py-c_xy)*(py-c_xy))
            core[py][px] = math.exp(-r2/sig2)*1
    core = core.reshape((1, 1, core.shape[0],core.shape[1]))
    
    core = torch.tensor(core)
    in_imgs = torch.tensor(in_imgs)
    out_imgs = F.conv_transpose2d(in_imgs, core, bias=None, stride = 1, padding = c_xy)

    return out_imgs.numpy()


def check_near_points(in_imgs):
    """点同士が近いかチェック ⇒ 点の間が3マス以下場合、ランダムにピークを選択した後の点画像を返す 
        点画像 in_imgs = [1][1][imgH][imgW] np, fp32 (0/1) """

    box_size = 5 
    box = np.ones((1, 1, box_size, box_size), dtype = "float32")
    box = torch.tensor(box)

    for i in range(5):
        in_imgs = torch.tensor(in_imgs)
        out_imgs = F.conv_transpose2d(in_imgs, box, bias=None, stride = 1, padding = 2)#5×5ベタ塗のDeconvolution
        in_imgs = in_imgs.numpy()
        out_imgs = out_imgs.numpy()        
        #print(out_imgs.max())
        
        if out_imgs.max() > 1:#5×5のDeconvolutionでオーバラップ(点の間が3マス以下）がある場合
            print('check_near_points overlap')
            delta = np.random.rand(*in_imgs.shape).astype(np.float32)#小さな乱数を足して
            in_imgs = get_local_max_point(in_imgs*255 + delta, threshold = 0.5)#どちらかのピークをランダムに選択した点画像を取得
        else:
            return in_imgs#オーバラップがない場合、そのまま返却
    
    print('check_near_points Err 5 times')
    return in_imgs


def get_posi_from_img(img, threshold = 0.2):
    """画像 img = [1][3 or 1][imgH][imgW] ⇒ 点位置 posi = [N][y,x] """  
    img_p = get_local_max_point(np.mean(img, axis = 1, keepdims = True), threshold = threshold)
    if Gv.check_near == True:
        img_p = check_near_points(img_p)#点同士が近いかチェック
    posi = conv_point_to_posi(img_p)
    return posi


def add_circle_from_posi(img, posi, color = 'R'): 
    """ 円が追加される画像 img = [1][3][imgH][imgW] ,点位置 posi = [N][y,x],
    　  color = 'R','G','B','C','M','Y','W',その他は黒,'N'は何もしない
        ⇒ return 円が追加された画像 [1][3][imgH][imgW]"""

    if color == 'N':#Nの時はそのまま返却
        return img

    imgH, imgW = img[0][0].shape    
    img = img.copy()

    for i in range(len(posi)):
        
            #円の塗り替えるピクセルのインデックスを取得
            rr, cc = draw.circle_perimeter(posi[i][0],posi[i][1], radius = 5, shape = (imgH, imgW)) 
            
            if color in('R', 'M' ,'Y', 'W'):
                img[0][0][rr, cc] = 255 #赤〇追加 
            else:
                img[0][0][rr, cc] = 0
            if color in('G', 'Y', 'C', 'W'):    
                img[0][1][rr, cc] = 255 #緑〇追加
            else:
                img[0][1][rr, cc] = 0             
            if color in('B', 'C', 'M', 'W') :    
                img[0][2][rr, cc] = 255 #青〇追加
            else:
                img[0][2][rr, cc] = 0 
                        
    return img

def add_cross_from_posi(img, posi, color = 'R', size = 5): 
    """ ×が追加される画像 img = [1][3][imgH][imgW] ,点位置 posi = [N][y,x],
    　  color = 'R','G','B','C','M','Y','W',その他は黒,'N'は何もしない
        ⇒ return ×が追加された画像 [1][3][imgH][imgW]"""

    if color == 'N':#Nの時はそのまま返却
        return img

    imgH, imgW = img[0][0].shape    
    img = img.copy()

    for i in range(len(posi)):
        
            #×の塗り替えるピクセルのインデックスを取得 
            
            cy = posi[i][0]
            cx = posi[i][1]            
            rr1, cc1 = draw.line(cy-size, cx-size, cy+size, cx+size) 
            rr2, cc2 = draw.line(cy-size, cx+size, cy+size, cx-size)            
            rr = np.concatenate([rr1,rr2])
            cc = np.concatenate([cc1,cc2])
 
            #範囲内のみ取り出す
            safe = (0 <= rr) & (rr < imgH) & (0 <= cc) & (cc < imgW)    
            rr =rr[safe]
            cc =cc[safe]
            
            if color in('R', 'M' ,'Y', 'W'):
                img[0][0][rr, cc] = 255 #赤〇追加 
            else:
                img[0][0][rr, cc] = 0
            if color in('G', 'Y', 'C', 'W'):    
                img[0][1][rr, cc] = 255 #緑〇追加
            else:
                img[0][1][rr, cc] = 0             
            if color in('B', 'C', 'M', 'W') :    
                img[0][2][rr, cc] = 255 #青〇追加
            else:
                img[0][2][rr, cc] = 0 
                        
    return img


    
### 評価関連 ###

def calc_cell_posi_judge(y_posi, t_posi, d_lim =5.0, return_reason = False):
    """出力位置と正解位置から、セル判定を計算して、出力側のOKとNGの位置 と 判定理由の数を返却
    input ⇒ 出力位置：y_posi[N][y,x] 、正解位置：t_posi[N][y,x]、判定距離：d_lim
    return ⇒ 出力の判定位置：j_posi['OK'/'NG'][N][y,x]  j_posi['OK'] は セル判定OKの数と一致
              (判定理由:reason['OK'/'zero'/'two'/'out']の4つ場合の数)"""
        
    def get_index_nn(y ,t):#y[y,x], t[N][y,x]を入力して、最小の距離dとそのときのtのインデックスを取得
        L = t - y
        L2 =  np.power(L, 2)
        L2s = L2.sum(axis = 1)
        d = np.sqrt(L2s)
        min_index = d.argmin()
        min_d = d[min_index]
        return min_index, min_d 
    
    y_arr = np.asarray(y_posi, dtype = np.float32)
    t_arr = np.asarray(t_posi, dtype = np.float32)
           
    cell= []
    j_posi = {'OK':[], 'NG':[]}#判定有の出力位置
    
    for i in range(len(t_arr)):#cellを作成
        cell.append({'t':t_posi[i], 'y':[], 'max_min_d':0.0})

    for i in range(len(y_arr)):#nnによってそれぞれのyを一番近いt(cell)にリストアップ
        min_index, min_d =  get_index_nn(y_arr[i] ,t_arr) 
        cell[min_index]['y'].append(y_posi[i])
        if cell[min_index]['max_min_d'] < min_d:
            cell[min_index]['max_min_d'] =  min_d            
        
    reason ={'OK':0, 'zero':0, 'two':0, 'out':0}
    
    for i in range(len(cell)):#cellの判定 
        
        if cell[i]['max_min_d'] <= d_lim and len(cell[i]['y'])  == 1:
            cell[i]['judge'] = True
            j_posi['OK'].extend(cell[i]['y'])
            reason['OK'] += 1# 1個かつ範囲内
            
        elif len(cell[i]['y'])  == 0:
            cell[i]['judge'] = False
            j_posi['NG'].extend(cell[i]['y'])
            reason['zero'] += 1# 0個
            
        elif len(cell[i]['y'])  >= 2:
            cell[i]['judge'] = False
            j_posi['NG'].extend(cell[i]['y'])
            reason['two'] += 1# 2個以上
            
        else: 
            cell[i]['judge'] = False
            j_posi['NG'].extend(cell[i]['y'])
            reason['out'] += 1# 1個かつ範囲外
    
    if return_reason :
        return j_posi, reason
    else:
        return j_posi


def calc_cell_posi_judge_2c(y_posi, t_posi, d_lim =5.0):
    """2クラスの出力位置と正解位置から、セル判定を計算して、合計の正解が多い方の出力側のOKとNGの位置を返却
    input ⇒ 出力位置：y_posi[2c][N][y,x] 、正解位置：t_posi[2c][N][y,x]、判定距離：d_lim
    return ⇒ 出力の判定位置：j_posi[2c]['OK'or'NG][N][y,x]  j_posi['OK'] は セル判定OKの数と一致
    ※j_posiはt基準で[0,1]の順で返却されるため、yとは逆の順になることがある
    """
    
    j_posi_00 = calc_cell_posi_judge(y_posi[0], t_posi[0], d_lim = d_lim)
    j_posi_11 = calc_cell_posi_judge(y_posi[1], t_posi[1], d_lim = d_lim)
    j_posi_01 = calc_cell_posi_judge(y_posi[0], t_posi[1], d_lim = d_lim)
    j_posi_10 = calc_cell_posi_judge(y_posi[1], t_posi[0], d_lim = d_lim)
    
    if (len(j_posi_00['OK'])+len(j_posi_11['OK'])) >= (len(j_posi_01['OK'])+len(j_posi_10['OK'])) :
        
        return [j_posi_00, j_posi_11]
    
    else:
        
        return [j_posi_10, j_posi_01]


### サンプル生成 ###

def get_rand_core(N_pic=1, imgH=512, imgW=512, p_num = 100, ch = 3, sig=3.0, max_xy = 15, c_xy= 7, return_posi = False):
    """ ランダムガウス玉とその位置を取得　⇒ return: 球画像 img[N_pic][ch][imgH][imgW] np, fp32(0-1/0-255), (位置 posi[N][y,x])"""
    
    threshold = 1-float(p_num)/(imgW*imgH)#p_numの粒子数になるように閾値を設定
    img_p = np.random.rand(N_pic*imgW*imgH)
    img_p[img_p < threshold] = 0
    img_p[img_p >= threshold] = 1

    img_p = img_p.reshape((N_pic,1, imgH, imgW)).astype("float32")

    #点⇒球に変換
    img = conv_point_to_core(img_p, sig = sig, max_xy = max_xy, c_xy = c_xy) *255
    
    if ch ==3 :#ch1⇒ch3
        img = np.broadcast_to(img, (N_pic, 3, imgH, imgW))
    
    #点⇒posiに変換 
    posi = conv_point_to_posi(img_p) 
    
    if return_posi :
        return img, posi
    else:
        return img


def get_rand_core_6ch(N_pic=1, p_num1 = 50 , p_num2 =50):
    """ 6chのランダムガウス玉を取得　⇒ return: 球画像 img[N_pic][6ch][imgH][imgW] np, fp32(0-1/0-255)""" 
    x_012 = get_rand_core(N_pic = N_pic, p_num = p_num1 ,ch = 3 )
    x_345 = get_rand_core(N_pic = N_pic, p_num = p_num2 ,ch = 3 )
    xb_stack = np.concatenate((x_012, x_345), axis=1)
    return xb_stack


def get_rand_RB_mix_ball(p_num = 50 ,sig=5.0 , ratio = 0.7, set_green = False, return_posi = False): 
    """ ランダムで取得した赤玉と青玉をratioで混合、球の画像と球の位置を取得
        return: 球画像 img[1][ch][imgH][imgW] , (赤位置 posi_R[N][y,x] ,青球位置 posi_B[N][y,x])"""
    x_red_o ,t_posi_R= get_rand_core(N_pic=1, p_num = p_num ,ch =1,sig=sig ,return_posi = True)
    x_blue_o, t_posi_B= get_rand_core(N_pic=1, p_num = p_num ,ch =1,sig=sig ,return_posi = True)

    t_posi =[t_posi_R, t_posi_B]
    
    x_red = x_red_o * ratio + x_blue_o * (1 - ratio)
    x_blue = x_red_o * (1 - ratio) + x_blue_o * ratio

    if set_green:#greenのチャンネルに青赤両方の球セット
        x_green = x_red_o + x_blue_o
    else:       #greenのチャンネルはゼロ
        x_green = np.zeros_like(x_red_o)
        
    xa_stack = np.concatenate((x_red,x_green,x_blue), axis=1)

    if return_posi:
        return xa_stack, t_posi
    else:
        return xa_stack


def get_rand_grid_ball(p_num = 50 ,sig = 5.0,  return_posi = False): 
    """ ランダム配置のd1間隔とd2間隔のグリッド球の画像と球の位置を取得
        return: 球画像 img[1][ch][imgH][imgW] , (d1球位置 posi_d1[N][y,x] ,d2球位置 posi_d2[N][y,x])"""
        
    ball_d1, t_posi_d1 = get_rand_core(N_pic=1, p_num = p_num ,ch =1,sig=sig,return_posi = True)
    ball_d2, t_posi_d2 = get_rand_core(N_pic=1, p_num = p_num ,ch =1,sig=sig,return_posi = True)
    
    t_posi =[t_posi_d1, t_posi_d2]
    
    grid_d1 = np.asarray([[1.5, 0.5, 1.5, 0.5],[0.5, 1.5, 0.5, 1.5],[1.5, 0.5, 1.5, 0.5],[0.5, 1.5, 0.5, 1.5]])
    grid_d1 = np.tile(grid_d1, (128,128))#128回タイル状に並べる
    grid_d1 = grid_d1.reshape((1, 1, 512, 512))

    x_grid_d1 = ball_d1 * grid_d1    
    
    grid_d2 = np.asarray([[1.5, 0.5, 0.5, 1.5],[0.5, 1.5, 1.5, 0.5],[0.5, 1.5, 1.5, 0.5],[1.5, 0.5, 0.5, 1.5]])
    grid_d2 = np.tile(grid_d2,(128,128))#128回タイル状に並べる
    grid_d2 = grid_d2.reshape((1, 1, 512, 512))
    
    x_grid_d2 = ball_d2 * grid_d2  

    x_grid_d1 = np.broadcast_to(x_grid_d1, (1, 3, 512, 512)).astype(np.float32)
    x_grid_d2 = np.broadcast_to(x_grid_d2, (1, 3, 512, 512)).astype(np.float32)
    
    xa_stack = x_grid_d1 + x_grid_d2
    
    if return_posi:
        return xa_stack, t_posi
    else:
        return xa_stack
