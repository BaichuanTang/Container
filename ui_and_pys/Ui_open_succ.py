# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file 'e:\Object Detection\YOLOX\ui_and_pys\open_succ.ui'
#
# Created by: PyQt5 UI code generator 5.15.4
#
# WARNING: Any manual changes made to this file will be lost when pyuic5 is
# run again.  Do not edit this file unless you know what you are doing.


from PyQt5 import QtCore, QtGui, QtWidgets


class Ui_MainWindow(object):
    def setupUi(self, MainWindow):
        MainWindow.setObjectName("MainWindow")
        MainWindow.resize(1600, 900)
        self.centralwidget = QtWidgets.QWidget(MainWindow)
        self.centralwidget.setObjectName("centralwidget")
        self.label = QtWidgets.QLabel(self.centralwidget)
        self.label.setGeometry(QtCore.QRect(330, 100, 861, 211))
        font = QtGui.QFont()
        font.setFamily("黑体")
        font.setPointSize(48)
        self.label.setFont(font)
        self.label.setObjectName("label")
        self.logined_3 = QtWidgets.QLabel(self.centralwidget)
        self.logined_3.setGeometry(QtCore.QRect(350, 390, 891, 161))
        font = QtGui.QFont()
        font.setFamily("楷体")
        font.setPointSize(22)
        self.logined_3.setFont(font)
        self.logined_3.setObjectName("logined_3")
        # MainWindow.setCentralWidget(self.centralwidget)
        self.menubar = QtWidgets.QMenuBar(MainWindow)
        self.menubar.setGeometry(QtCore.QRect(0, 0, 1600, 38))
        self.menubar.setObjectName("menubar")
        # MainWindow.setMenuBar(self.menubar)
        self.statusbar = QtWidgets.QStatusBar(MainWindow)
        self.statusbar.setObjectName("statusbar")
        # MainWindow.setStatusBar(self.statusbar)

        self.retranslateUi(MainWindow)
        QtCore.QMetaObject.connectSlotsByName(MainWindow)

    def retranslateUi(self, MainWindow):
        _translate = QtCore.QCoreApplication.translate
        MainWindow.setWindowTitle(_translate("MainWindow", "MainWindow"))
        self.label.setText(_translate("MainWindow", " 欢迎选购商品"))
        self.logined_3.setText(_translate("MainWindow", "两秒后自动进入购买界面……"))

class open_succ_window(QtWidgets.QWidget, Ui_MainWindow):  # 创建子UI类

    def __init__(self):
        super(open_succ_window, self).__init__()
        self.setupUi(self)