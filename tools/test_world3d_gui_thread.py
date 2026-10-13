"""Qt integration: worker completion/error callbacks must stay on GUI thread."""
import os,sys,time
os.environ.setdefault('QT_QPA_PLATFORM','offscreen')
sys.path.insert(0,os.path.dirname(os.path.dirname(__file__)))
from PySide6 import QtCore,QtWidgets
from rw.gui_world3d import World3DPage
app=QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
page=World3DPage(); received=[]
page._spawn(lambda: 123,lambda result:received.append((result,QtCore.QThread.currentThread()==page.thread())))
deadline=time.monotonic()+10
while time.monotonic()<deadline and (not received or page._th3.isRunning()):
    app.processEvents();time.sleep(.005)
assert received==[(123,True)],received
print('PASS: result delivered on GUI thread')
page.close()
# Failure receiver is a QObject slot and must restore UI on the same thread.
class ObservedPage(World3DPage):
    def _sim_err(self,tb):
        received.append(('error',QtCore.QThread.currentThread()==self.thread(),'division by zero' in tb))
        super()._sim_err(tb)
page=ObservedPage()
page._spawn(lambda:1/0,lambda _:None)
deadline=time.monotonic()+10
while time.monotonic()<deadline and (len(received)<2 or page._th3.isRunning()):
    app.processEvents();time.sleep(.005)
assert received[-1]==('error',True,True),received
assert page.btn.isEnabled()
print('PASS: worker failure restored UI on GUI thread')
page.close()
