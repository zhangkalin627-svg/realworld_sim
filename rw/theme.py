"""浅色 / 深色 / 跟随系统 主题（v0.17，用户建议）。
Qt 控件：QPalette + 样式表（覆盖参数区/表格/弹窗/禁用态）；matplotlib 图表：rcParams（新图）+ 已建画布重着色（切换即生效）。
选择用 QSettings 记住，下次启动自动恢复。"""
from PySide6 import QtCore, QtGui, QtWidgets

_ORG = ("RealWorldSim", "RealWorldSim")


def current():
    return QtCore.QSettings(_ORG[0], _ORG[1]).value("theme", "system")


def _system_dark():
    try:
        return QtGui.QGuiApplication.styleHints().colorScheme() == QtCore.Qt.ColorScheme.Dark
    except Exception:
        return False


def _palette(dark):
    if not dark:
        return QtWidgets.QApplication.style().standardPalette()
    p = QtGui.QPalette()
    win, base, txt, dis = QtGui.QColor(43, 45, 48), QtGui.QColor(30, 31, 34), QtGui.QColor(232, 232, 232), QtGui.QColor(128, 128, 128)
    hl = QtGui.QColor(42, 130, 218)
    for role in (QtGui.QPalette.Window, QtGui.QPalette.Button):
        p.setColor(role, win)
    for role in (QtGui.QPalette.Base, QtGui.QPalette.AlternateBase):
        p.setColor(role, base)
    for role in (QtGui.QPalette.WindowText, QtGui.QPalette.Text, QtGui.QPalette.ButtonText, QtGui.QPalette.ToolTipText):
        p.setColor(role, txt)
    p.setColor(QtGui.QPalette.ToolTipBase, base)
    p.setColor(QtGui.QPalette.Highlight, hl)
    p.setColor(QtGui.QPalette.HighlightedText, QtGui.QColor(255, 255, 255))
    p.setColor(QtGui.QPalette.Link, QtGui.QColor(110, 175, 255))
    p.setColor(QtGui.QPalette.PlaceholderText, dis)
    for role in (QtGui.QPalette.WindowText, QtGui.QPalette.Text, QtGui.QPalette.ButtonText):
        p.setColor(QtGui.QPalette.Disabled, role, dis)
    return p


_QSS_DARK = """
QToolTip { color:#e8e8e8; background:#1e1f22; border:1px solid #555; }
QGroupBox { border:1px solid #4a4d52; border-radius:6px; margin-top:10px; }
QGroupBox::title { subcontrol-origin:margin; left:8px; padding:0 3px; }
QLineEdit, QSpinBox, QDoubleSpinBox, QComboBox, QTextEdit, QTextBrowser, QPlainTextEdit, QTableWidget {
  background:#1e1f22; color:#e8e8e8; border:1px solid #4a4d52; border-radius:4px; selection-background-color:#2a82da; }
QTabBar::tab { background:#34363a; color:#ccc; padding:6px 14px; }
QTabBar::tab:selected { background:#2a82da; color:#fff; }
QHeaderView::section { background:#34363a; color:#ddd; border:1px solid #2b2d30; padding:3px; }
QPushButton { background:#3a3d42; color:#e8e8e8; border:1px solid #4a4d52; border-radius:4px; padding:5px 10px; }
QPushButton:hover { background:#45484e; }
QPushButton:disabled { color:#888; background:#2b2d30; }
QProgressBar { border:1px solid #4a4d52; border-radius:4px; text-align:center; background:#1e1f22; }
QProgressBar::chunk { background:#2a82da; }
QMenuBar, QMenu { background:#2b2d30; color:#e8e8e8; }
QMenu::item:selected, QMenuBar::item:selected { background:#2a82da; }
QScrollBar:vertical { background:#2b2d30; width:12px; } QScrollBar::handle:vertical { background:#555; min-height:20px; border-radius:6px; }
"""

_MPL = {
    "dark": {"figure.facecolor": "#2b2d30", "axes.facecolor": "#1e1f22", "axes.edgecolor": "#aaaaaa",
             "axes.labelcolor": "#e8e8e8", "text.color": "#e8e8e8", "xtick.color": "#cccccc",
             "ytick.color": "#cccccc", "grid.color": "#555555", "savefig.facecolor": "#2b2d30",
             "legend.facecolor": "#2b2d30", "legend.edgecolor": "#555555"},
    "light": {"figure.facecolor": "white", "axes.facecolor": "white", "axes.edgecolor": "black",
              "axes.labelcolor": "black", "text.color": "black", "xtick.color": "black",
              "ytick.color": "black", "grid.color": "#b0b0b0", "savefig.facecolor": "white",
              "legend.facecolor": "inherit", "legend.edgecolor": "0.8"},
}


def is_dark(mode=None):
    mode = mode or current()
    return mode == "dark" or (mode == "system" and _system_dark())


def apply(app, mode):
    """应用主题（立即生效，含已建 matplotlib 画布）并记住选择。mode: system/light/dark"""
    QtCore.QSettings(_ORG[0], _ORG[1]).setValue("theme", mode)
    dark = is_dark(mode)
    app.setPalette(_palette(dark))
    app.setStyleSheet(_QSS_DARK if dark else "")
    import matplotlib
    matplotlib.rcParams.update(_MPL["dark" if dark else "light"])
    restyle_canvases(dark)


def restyle_canvases(dark):
    """把已建好的 FigureCanvasQTAgg 画布重着色（新建的图吃 rcParams，不用管）。"""
    import matplotlib
    matplotlib.rcParams.update(_MPL["dark" if dark else "light"])
    from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg
    fg, bg, tx = ("#2b2d30", "#1e1f22", "#e8e8e8") if dark else ("white", "white", "black")
    tick = "#cccccc" if dark else "black"
    for w in QtWidgets.QApplication.allWidgets():
        if not isinstance(w, FigureCanvasQTAgg):
            continue
        fig = w.figure
        fig.set_facecolor(fg)
        for ax in fig.axes:
            ax.set_facecolor(bg)
            ax.title.set_color(tx); ax.xaxis.label.set_color(tx); ax.yaxis.label.set_color(tx)
            ax.tick_params(colors=tick)
            for s in ax.spines.values():
                s.set_color(tick)
            leg = ax.get_legend()
            if leg is not None:
                leg.get_frame().set_facecolor(fg); leg.get_frame().set_edgecolor(tick)
                for t in leg.get_texts():
                    t.set_color(tx)
        w.draw_idle()
