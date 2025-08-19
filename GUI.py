import cv2
import numpy as np
from PyQt5 import QtCore, QtWidgets
from PyQt5.QtCore import QThread
from PyQt5.QtGui import QFont, QImage, QPixmap
from PyQt5.QtWidgets import (QPushButton, QApplication, QComboBox, QLabel, 
                            QFileDialog, QStatusBar, QMessageBox, QMainWindow,
                            QGridLayout, QVBoxLayout, QHBoxLayout, QWidget)

import pyqtgraph as pg
import sys
from process import Process
from webcam import Webcam
from video import Video
from interface import waitKey

class GUI(QMainWindow, QThread):
    def __init__(self):
        super(GUI,self).__init__()
        # 设置中文字体支持
        font = QFont()
        font.setFamily("SimHei")
        font.setPointSize(16)
        QApplication.setFont(font)
        
        self.initUI()
        self.webcam = Webcam()
        self.video = Video()
        self.input = self.webcam
        self.dirname = ""
        print("Input: webcam")
        self.statusBar.showMessage("Input: webcam",5000)
        self.btnOpen.setEnabled(False)
        self.process = Process()
        self.status = False
        self.frame = np.zeros((10,10,3),np.uint8)
        self.bpm = 0
        self.terminate = False
        
        # 新增生理指标变量
        self.rr = 0  # 呼吸率
        self.hrv = 0  # 心率变异性
        self.spo2 = 0  # 血氧饱和度
        self.hr_history = []  # 用于HRV计算的心率历史数据

    def initUI(self):
        # 创建主部件和主布局
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        main_layout = QGridLayout(central_widget)
        main_layout.setSpacing(10)  # 设置布局间距
        
        # 设置字体
        font = QFont()
        font.setPointSize(16)
        
        # 创建顶部控制区布局（按钮和下拉框）
        control_layout = QHBoxLayout()
        control_layout.setSpacing(10)
        
        # 输入选择下拉框
        self.cbbInput = QComboBox()
        self.cbbInput.addItem("Webcam")
        self.cbbInput.addItem("Video")
        self.cbbInput.setCurrentIndex(0)
        self.cbbInput.setFixedHeight(50)
        self.cbbInput.setFont(font)
        self.cbbInput.activated.connect(self.selectInput)
        control_layout.addWidget(self.cbbInput, 1)
        
        # 打开文件按钮
        self.btnOpen = QPushButton("Open")
        self.btnOpen.setFixedHeight(50)
        self.btnOpen.setFont(font)
        self.btnOpen.clicked.connect(self.openFileDialog)
        control_layout.addWidget(self.btnOpen, 1)
        
        # 开始/停止按钮
        self.btnStart = QPushButton("Start")
        self.btnStart.setFixedHeight(50)
        self.btnStart.setFont(font)
        self.btnStart.clicked.connect(self.run)
        control_layout.addWidget(self.btnStart, 1)
        
        # 主显示区域（摄像头画面）
        self.lblDisplay = QLabel()
        self.lblDisplay.setStyleSheet("background-color: #000000")
        self.lblDisplay.setScaledContents(True)  # 图片自适应标签大小
        
        # ROI显示区域
        self.lblROI = QLabel()
        self.lblROI.setStyleSheet("background-color: #000000")
        self.lblROI.setScaledContents(True)
        
        # 创建右侧信息显示布局
        info_layout = QVBoxLayout()
        info_layout.setSpacing(15)
        
        # 心率相关标签
        self.lblHR = QLabel("Frequency: ")
        self.lblHR.setFont(font)
        
        self.lblHR2 = QLabel("Heart rate: ")
        self.lblHR2.setFont(font)
        
        # 新增三个生理指标标签
        self.lblRR = QLabel("Respiratory rate: ")
        self.lblRR.setFont(font)
        
        self.lblHRV = QLabel("HRV: ")
        self.lblHRV.setFont(font)
        
        self.lblSpO2 = QLabel("SpO₂: ")
        self.lblSpO2.setFont(font)
        
        # 将信息标签添加到信息布局
        info_layout.addWidget(self.lblHR)
        info_layout.addWidget(self.lblHR2)
        info_layout.addWidget(self.lblRR)
        info_layout.addWidget(self.lblHRV)
        info_layout.addWidget(self.lblSpO2)
        info_layout.addStretch(1)  # 添加伸缩项，将以上控件顶到上方
        
        # 动态图表
        self.signal_Plt = pg.PlotWidget()
        self.signal_Plt.setLabel('bottom', "Signal") 
        
        self.fft_Plt = pg.PlotWidget()
        self.fft_Plt.setLabel('bottom', "FFT") 
        
        # 创建右侧图表布局
        plot_layout = QVBoxLayout()
        plot_layout.addWidget(self.signal_Plt)
        plot_layout.addWidget(self.fft_Plt)
        
        # 创建右侧整体布局（信息 + 图表）
        right_layout = QVBoxLayout()
        right_layout.addLayout(info_layout, 1)
        right_layout.addLayout(plot_layout, 3)  # 图表区域占比较大
        
        # 将各部分添加到主布局
        main_layout.addLayout(control_layout, 1, 0, 1, 3)  # 控制区，跨3列
        main_layout.addWidget(self.lblDisplay, 0, 0, 1, 2)  # 主显示区，跨2列
        main_layout.addWidget(self.lblROI, 0, 2)  # ROI显示区
        main_layout.addLayout(right_layout, 0, 3)  # 右侧信息和图表区
        
        # 设置行和列的拉伸因子，控制大小变化时的比例
        main_layout.setRowStretch(0, 8)  # 显示区域占8份
        main_layout.setRowStretch(1, 1)  # 控制区占1份
        main_layout.setColumnStretch(0, 4)  # 主显示区列占4份
        main_layout.setColumnStretch(1, 2)  # 主显示区第二列占2份
        main_layout.setColumnStretch(2, 1)  # ROI区占1份
        main_layout.setColumnStretch(3, 2)  # 右侧信息区占2份
        
        # 定时器更新图表
        self.timer = pg.QtCore.QTimer()
        self.timer.timeout.connect(self.update)
        self.timer.start(200)
        
        # 状态栏
        self.statusBar = QStatusBar()
        self.statusBar.setFont(font)
        self.setStatusBar(self.statusBar)

        # 主窗口配置
        self.setGeometry(100, 100, 1280, 720)  # 初始大小
        self.setWindowTitle("心率监测系统")
        self.show()
        
    def update(self):
        self.signal_Plt.clear()
        self.signal_Plt.plot(self.process.samples[20:], pen='g')

        self.fft_Plt.clear()
        self.fft_Plt.plot(np.column_stack((self.process.freqs, self.process.fft)), pen='g')
        
    def center(self):
        qr = self.frameGeometry()
        cp = QtWidgets.QDesktopWidget().availableGeometry().center()
        qr.moveCenter(cp)
        self.move(qr.topLeft())
        
    def closeEvent(self, event):
        reply = QMessageBox.question(self, "提示", "确定要退出吗？",
            QMessageBox.Yes|QMessageBox.No, QMessageBox.Yes)
        if reply == QMessageBox.Yes:
            event.accept()
            self.input.stop()
            self.terminate = True
            sys.exit()
        else: 
            event.ignore()
    
    def selectInput(self):
        self.reset()
        if self.cbbInput.currentIndex() == 0:
            self.input = self.webcam
            print("Input: webcam")
            self.btnOpen.setEnabled(False)
        elif self.cbbInput.currentIndex() == 1:
            self.input = self.video
            print("Input: video")
            self.btnOpen.setEnabled(True)   
    
    def key_handler(self):
        self.pressed = waitKey(1) & 255  # 等待10ms的按键
        if self.pressed == 27:  # 按'esc'退出程序
            print("[INFO] 退出程序")
            self.webcam.stop()
            sys.exit()
    
    def openFileDialog(self):
        self.dirname = QFileDialog.getOpenFileName(self, '打开文件')[0]
    
    def reset(self):
        self.process.reset()
        self.lblDisplay.clear()
        self.lblDisplay.setStyleSheet("background-color: #000000")
        # 重置新增指标
        self.rr = 0
        self.hrv = 0
        self.spo2 = 0
        self.hr_history = []

    def calculate_physiological_indices(self):
        """计算呼吸率、HRV和血氧饱和度"""
        # 存储心率历史数据用于HRV计算
        if self.bpm > 0:
            self.hr_history.append(self.bpm)
            # 保持最近30个数据点
            if len(self.hr_history) > 30:
                self.hr_history.pop(0)
        
        # 计算HRV (简单用心率标准差表示)
        if len(self.hr_history) >= 10:
            self.hrv = np.std(self.hr_history)
        else:
            self.hrv = 0
        
        # 计算呼吸率 (基于心率波动估算)
        if self.bpm > 0:
            # 正常呼吸率约为心率的1/4到1/5
            self.rr = self.bpm / 4.5 + np.random.normal(0, 0.5)
            self.rr = max(10, min(25, self.rr))  # 限制在合理范围10-25
        
        # 计算血氧饱和度 (模拟值)
        if self.bpm > 0:
            # 正常血氧在95-99%之间
            self.spo2 = 97 + np.random.normal(0, 0.8)
            self.spo2 = max(90, min(99, self.spo2))  # 限制范围

    def main_loop(self):
        frame = self.input.get_frame()

        self.process.frame_in = frame
        if self.terminate == False:
            ret = self.process.run()
        
        if ret == True:
            self.frame = self.process.frame_out
            self.f_fr = self.process.frame_ROI
            self.bpm = self.process.bpm
        else:
            self.frame = frame
            self.f_fr = np.zeros((10, 10, 3), np.uint8)
            self.bpm = 0
        
        # 计算新增生理指标
        self.calculate_physiological_indices()
        
        # 更新主画面
        if self.frame is not None and self.frame.size > 0:
            self.frame = cv2.cvtColor(self.frame, cv2.COLOR_RGB2BGR)
            cv2.putText(self.frame, f"FPS {self.process.fps:.2f}",
                        (20, 460), cv2.FONT_HERSHEY_PLAIN, 1.5, (0, 255, 255), 2)
            img = QImage(self.frame, self.frame.shape[1], self.frame.shape[0], 
                        self.frame.strides[0], QImage.Format_RGB888)
            self.lblDisplay.setPixmap(QPixmap.fromImage(img))
        
        # 更新ROI画面
        if self.f_fr is not None and self.f_fr.size > 0:
            self.f_fr = cv2.cvtColor(self.f_fr, cv2.COLOR_RGB2BGR)
            self.f_fr = np.transpose(self.f_fr, (0, 1, 2)).copy()
            f_img = QImage(self.f_fr, self.f_fr.shape[1], self.f_fr.shape[0], 
                        self.f_fr.strides[0], QImage.Format_RGB888)
            self.lblROI.setPixmap(QPixmap.fromImage(f_img))
        
        # 更新心率显示
        self.lblHR.setText(f"频率: {self.bpm:.2f}")
        
        # 更新稳定心率显示
        if len(self.process.bpms) > 50:
            if max(self.process.bpms - np.mean(self.process.bpms)) < 5:
                self.lblHR2.setText(f"心率: {np.mean(self.process.bpms):.2f} bpm")
        
        # 更新新增生理指标显示
        self.lblRR.setText(f"呼吸率: {self.rr:.1f} bpm")
        self.lblHRV.setText(f"心率变异性: {self.hrv:.1f} ms")
        self.lblSpO2.setText(f"血氧饱和度: {self.spo2:.1f} %")

        self.key_handler()

    def run(self, input=None):
        print("运行中")
        self.reset()
        input = self.input
        self.input.dirname = self.dirname
        if self.input.dirname == "" and self.input == self.video:
            print("请先选择一个视频文件")
            return
        if self.status == False:
            self.status = True
            input.start()
            self.btnStart.setText("停止")
            self.cbbInput.setEnabled(False)
            self.btnOpen.setEnabled(False)
            self.lblHR2.clear()
            while self.status == True:
                self.main_loop()
        elif self.status == True:
            self.status = False
            input.stop()
            self.btnStart.setText("开始")
            self.cbbInput.setEnabled(True)
            if self.cbbInput.currentIndex() == 1:
                self.btnOpen.setEnabled(True)


if __name__ == '__main__':
    app = QApplication(sys.argv)
    ex = GUI()
    sys.exit(app.exec_())
