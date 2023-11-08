# Imports required modules
# Numpy is used for scientific computing. Here, it obtains the zero crossing rate
# Scipy.signal is a module used for signal processing. Here, it helps in obtaining the audio signal before processing
# soundfile is used to read and write audio files. Libraries enable audio signal processing such as resampling
# PyQt5 combines Qt5, a application developing module along with the audio signal processing module to get a GUI
# matplotlib is used for plotting.
# Librosa helps in audio processing, loading of signal and printing waveforms
 
import sys
import soundfile as sf
import librosa
import numpy as np
import matplotlib.pyplot as plt
from PyQt5.QtWidgets import QApplication, QMainWindow, QFileDialog, QLabel, QPushButton, QMessageBox
from PyQt5.QtCore import Qt

# MainWindow function is used for the main page of the GUI. Components such as buttons, text are added to it
class MainWindow(QMainWindow):
    def __init__(self):
        # Defines and gives a title to the GUI
        super().__init__()
        self.setWindowTitle("Speech Endpoint Detection")
        self.setGeometry(200, 200, 2000, 2000)

        self.filename = None
        self.fs = 16000
        self.audio = None
        self.peaks = None

        # Displays path of the file
        self.label = QLabel(self)
        self.label.setGeometry(50, 50, 300, 50)
        self.label.setAlignment(Qt.AlignCenter)
        self.label.setText("No audio file selected")

        # Enables a button which can select the audio file
        self.select_button = QPushButton(self)
        self.select_button.setGeometry(50, 150, 300, 50)
        self.select_button.setText("Select audio file")
        self.select_button.clicked.connect(self.select_file)

        # Enables a button that plots the input waveform
        self.plot_button = QPushButton(self)
        self.plot_button.setGeometry(50, 250, 300, 50)
        self.plot_button.setText("Plot audio waveform")
        self.plot_button.clicked.connect(self.plot_audio)

        # Enables a button that plots the speech endpoints
        self.plot_button = QPushButton(self)
        self.plot_button.setGeometry(50, 350, 300, 50)
        self.plot_button.setText("Plot speech endpoints")
        self.plot_button.clicked.connect(self.plot_endpoints)

        # Enables a button that prints speech endpoints
        self.plot_button = QPushButton(self)
        self.plot_button.setGeometry(50, 450, 300, 50)
        self.plot_button.setText("Print speech endpoints")
        self.plot_button.clicked.connect(self.print_endpoints)

    # Defines the function that enables selecting of file from the device. Uses Dialog box to get the filename
    def select_file(self):
        options = QFileDialog.Options()
        options |= QFileDialog.DontUseNativeDialog
        self.filename, _ = QFileDialog.getOpenFileName(self,"Select audio file", "","All Files (*);;WAV Files (*.wav)", options=options)
        if self.filename:
            self.label.setText(self.filename)
            self.audio, _ = librosa.load(self.filename, sr=self.fs, mono=True)

    # Plots the selected audio signal as waveform
    def plot_audio(self):
        if self.audio is None:
            return
        a_file, s_rate = sf.read(self.filename)
        lrg=a_file[0]
        for i in range(len(a_file)):
            if(np.absolute(a_file[i])>lrg):
                lrg = np.absolute(a_file[i])
        for i in range(len(a_file)):
            a_file[i]=a_file[i]/lrg
        mean_sig = np.mean(a_file)
        for i in a_file:
            i=i-mean_sig
        plt.figure()
        librosa.display.waveshow(a_file, sr=s_rate)

    # Plots the endpoints of the selected audio signal as waveform
    def plot_endpoints(self):
        if self.audio is None:
            return
        a_file, s_rate = sf.read(self.filename)
        lrg=a_file[0]
        for i in range(len(a_file)):
            if(np.absolute(a_file[i])>lrg):
                lrg = np.absolute(a_file[i])
        for i in range(len(a_file)):
            a_file[i]=a_file[i]/lrg
        mean_sig = np.mean(a_file)
        for i in a_file:
            i=i-mean_sig

        a_signal = a_file
        b_size = int(len(a_file)*0.03)
        signal_mean = np.mean(a_file)
        abs_deviation = np.abs(a_file - mean_sig)
        mad = np.mean(abs_deviation)
        th = 7*mad
        for i in range(1,len(a_file)):
            if(i<len(a_file)-b_size):
                window = a_file[i:i+b_size]
                zcr = librosa.feature.zero_crossing_rate(window)
                zcr = np.mean(zcr)
                energy = 0.0
                for k in window:
                    energy+=k*k
                if(zcr>th or energy<0.5):
                    for j in range(i,i+b_size):
                        a_signal[j]=0
            else:
                temp = [0]*(b_size-len(a_file)+i)
                t1 = a_file[i:len(a_file)]
                window = np.concatenate((t1,temp), axis = None)
                zcr = librosa.feature.zero_crossing_rate(window)
                zcr = np.mean(zcr)
                energy = 0.0
                for k in window:
                    energy+=k*k
                if(zcr>th or energy<0.5):
                    for j in range(i,len(a_file)):
                        a_signal[j]=0
                        
        plt.figure()
        librosa.display.waveshow(a_signal, sr=s_rate)
        plt.show()

    # Prints the endpoints of the selected audio signal
    def print_endpoints(self):
        if self.audio is None:
            return
        a_file, s_rate = sf.read(self.filename)
        lrg=a_file[0]
        for i in range(len(a_file)):
            if(np.absolute(a_file[i])>lrg):
                lrg = np.absolute(a_file[i])
        for i in range(len(a_file)):
            a_file[i]=a_file[i]/lrg
        mean_sig = np.mean(a_file)
        for i in a_file:
            i=i-mean_sig

        a_signal = a_file
        b_size = int(len(a_file)*0.03)
        signal_mean = np.mean(a_file)
        abs_deviation = np.abs(a_file - mean_sig)
        mad = np.mean(abs_deviation)
        th = 7*mad
        for i in range(1,len(a_file)):
            if(i<len(a_file)-b_size):
                window = a_file[i:i+b_size]
                zcr = librosa.feature.zero_crossing_rate(window)
                zcr = np.mean(zcr)
                energy = 0.0
                for k in window:
                    energy+=k*k
                if(zcr>th or energy<0.5):
                    for j in range(i,i+b_size):
                        a_signal[j]=0
            else:
                temp = [0]*(b_size-len(a_file)+i)
                t1 = a_file[i:len(a_file)]
                window = np.concatenate((t1,temp), axis = None)
                zcr = librosa.feature.zero_crossing_rate(window)
                zcr = np.mean(zcr)
                energy = 0.0
                for k in window:
                    energy+=k*k
                if(zcr>th or energy<0.5):
                    for j in range(i,len(a_file)):
                        a_signal[j]=0
        
        strt = []
        end = []
        sp=False
        for i in range(0,len(a_signal),5):
            if(a_signal[i]!=0 and sp == False):
                sp=True
                strt.append(i/s_rate)
            elif(a_signal[i]==0 and sp==True):
                sp=False
                end.append((i-1)/s_rate)
            else:
                continue
        a=""
        for i in range(len(strt)):
            a += "Start: "+ str(strt[i])+ "\t End: "+ str(end[i]) +"\n"
            
        msg = QMessageBox()
        msg.setGeometry(100,100,200,200)
        msg.setWindowTitle("Speech Endpoints")
        msg.setText(a)
        x = msg.exec_()          
            
# Executes the mainloop and runs GUI
if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec_())
