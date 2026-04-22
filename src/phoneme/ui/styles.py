"""Centralized QSS."""

APP_QSS = """
QMainWindow, QWidget#Central {
    background: #0e1116;
    color: #e6e6e6;
}
QLabel { color: #e6e6e6; }

QLabel#WordLabel {
    font-family: -apple-system, 'SF Pro Display', 'Helvetica Neue', Arial;
    font-size: 48px;
    font-weight: 600;
    letter-spacing: 1px;
    padding: 12px 16px;
}

QLabel#LiveCaption {
    font-family: -apple-system, 'SF Mono', Menlo, monospace;
    font-size: 18px;
    color: #9aa3b2;
    padding: 6px 16px;
}

QFrame#ChipStrip {
    background: #161a22;
    border: 1px solid #232a36;
    border-radius: 14px;
    padding: 12px;
}

QPushButton {
    background: #1e2530;
    color: #e6e6e6;
    border: 1px solid #2b3442;
    border-radius: 8px;
    padding: 6px 12px;
}
QPushButton:hover { background: #263142; }
QPushButton:pressed { background: #161a22; }
QPushButton:checked {
    background: #2a5fdc;
    border-color: #3d74f2;
}

QStatusBar { background: #0b0e13; color: #9aa3b2; }

QComboBox, QSpinBox, QDoubleSpinBox, QLineEdit {
    background: #161a22;
    border: 1px solid #2b3442;
    border-radius: 6px;
    padding: 4px 8px;
    color: #e6e6e6;
}

QGroupBox {
    border: 1px solid #2b3442;
    border-radius: 8px;
    margin-top: 12px;
    padding-top: 12px;
    color: #cfd6e3;
}
QGroupBox::title {
    subcontrol-origin: margin;
    left: 10px;
    padding: 0 4px;
}

#HoldHint {
    color: #6a7687;
    font-family: -apple-system, 'SF Mono', Menlo, monospace;
}

#HoldHint[active="true"] {
    color: #3dc77b;
}

QProgressBar {
    background: #0b0e13;
    border: 1px solid #232a36;
    border-radius: 4px;
    text-align: center;
    color: #e6e6e6;
}
QProgressBar::chunk {
    background: #3dc77b;
    border-radius: 3px;
}
"""
