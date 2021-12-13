# -*- coding: utf-8 -*-
import datetime
import json
import cv2
import os
import sys
sys.path.append("../..")
sys.path.append("..")

import pymysql
from PyQt5 import QtCore, QtGui, QtWidgets
from PyQt5.QtWidgets import QMessageBox

#导入前台页面
from ui_and_pys.Ui_Main import Ui_MainWindow
from ui_and_pys.Ui_open_succ import open_succ_window
from ui_and_pys.Ui_select_item import select_item_window
from ui_and_pys.Ui_select_item_results import select_item_results_window
#导入预测模块
from realtime import realtime_det_recog,Predictor
from yolox.exp import get_exp
from yolox.data.datasets import COCO_CLASSES
import torch
import time
#数据库连接参数
host = 'localhost'
port = 3306
user = 'root'
password = '123456'
db_name = 'container'

#商品配置
dct_id_to_name = {
    0:"芬达",
    1:"果粒橙",
    2:"劲凉",
    3:"冰红茶",
    4:"无糖可乐",
    5:"丝滑拿铁",
    6:"雪碧"
}

class mywindow(QtWidgets.QMainWindow, Ui_MainWindow):
    def __init__(self):
        super(mywindow, self).__init__()
        self.setupUi(self)

        # setting main window geometry
        desktop_geometry = QtWidgets.QApplication.desktop()  # 获取屏幕大小
        main_window_width = desktop_geometry.width()  # 屏幕的宽
        main_window_height = desktop_geometry.height()  # 屏幕的高
        rect = self.geometry()  # 获取窗口界面大小
        window_width = rect.width()  # 窗口界面的宽
        window_height = rect.height()  # 窗口界面的高
        x = (main_window_width - window_width) // 2  # 计算窗口左上角点横坐标
        y = (main_window_height - window_height) // 2  # 计算窗口左上角点纵坐标
        self.setGeometry(x, y, window_width, window_height)  # 设置窗口界面在屏幕上的位置

        #开柜按钮跳转
        self.open_succ_window = open_succ_window() #登陆成功
        self.open_btn.clicked.connect(self.open_clicked)# 槽函数连接

        #跳转至商品选择页面
        self.timer = QtCore.QTimer(self)
        self.timer.setSingleShot(True)
        self.select_item_window = select_item_window()
        self.select_item_window.pushButton.clicked.connect(self.close_door)#关门
        self.select_item_results_window=select_item_results_window()
        self.timer2 = QtCore.QTimer(self)
        self.timer2.setSingleShot(True)
        self.timer3 = QtCore.QTimer(self)
        self.timer3.setSingleShot(True)
        self.timer4 = QtCore.QTimer(self)
        self.timer4.setSingleShot(True)
        self.timer0 = QtCore.QTimer(self)
        self.timer0.setSingleShot(True)

    #开柜按钮跳转
    def open_clicked(self):
        self.open_succ_window.show()
        self.timer.start(3000)
        self.timer.timeout.connect(self.tbc_select_items)

    def call_webcam(self):
        exp = get_exp(None, 'yolox-s')
        file_name = os.path.join(exp.output_dir, exp.exp_name)
        os.makedirs(file_name, exist_ok=True)
        vis_folder = None
        vis_folder = os.path.join(file_name, "vis_res")
        os.makedirs(vis_folder, exist_ok=True)
        exp.test_conf = 0.25
        exp.nmsthre = 0.45
        exp.test_size = (640,640)
        model = exp.get_model()
        model.cuda()
        model.eval()
        ckpt_file = "yolox_s.pth"
        ckpt = torch.load(ckpt_file, map_location="cpu")
        model.load_state_dict(ckpt["model"])
        trt_file = None
        decoder = None
        predictor = Predictor(
            model, exp, COCO_CLASSES, trt_file, decoder,
            'gpu', None, None,'assets/simple_model' #在此处修改模型
        )
        current_time = time.localtime()
        return realtime_det_recog(predictor, vis_folder, current_time, save_result=False)


    #选择商品
    def tbc_select_items(self):
        self.open_succ_window.close()
        self.select_item_window.show()
        print('selected started')
        self.op_time=datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        self.item_results_map=self.call_webcam()
        # self.item_results_map={0:0,1:1,2:1,3:0,4:0,5:0,6:1} #test case
        
    def close_door(self):
        #关门
        cv2.destroyAllWindows()
        self.select_item_window.close()
        print('选购完成')
        st=''

        sump=0
        prices = [3.8, 4.5, 3.5, 3.5, 3.8, 6.5, 3.5]
        for (i, j) ,p in zip(self.item_results_map.items(),prices):
            #i:真实名，j数量，p价格
            st += dct_id_to_name[i]
            st += '\t'
            st += str(j)
            sump+=j*p
            st += '\n'
        st=st+'总价为'+str(sump)+'元'
        '''
        导入DB
        '''
        order_time=datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        #落盘
        detail_str=json.dumps(self.item_results_map,ensure_ascii=False)
        self.close_door_write_db(order_time,detail_str)
        #输出选购结果
        self.select_item_results_window.textBrowser.setText(st)
        self.select_item_results_window.show()

    def close_door_write_db(self,order_time,detail):
        #关门时存数据库的操作
        conn = pymysql.connect(host=host, port=port, user=user, password=password, db=db_name, charset='utf8')
        cursor = conn.cursor()
        
        sql=f"INSERT INTO orders(order_time, detail) VALUES ('{order_time}', '{detail}');"
        print(sql)
        cursor.execute(sql)
        conn.commit()
        cursor.close()
        conn.close()
        return 'Insert Success'

import sys
if __name__ == '__main__':
    app = QtWidgets.QApplication(sys.argv)
    window = mywindow()
    window.show()
    sys.exit(app.exec_())