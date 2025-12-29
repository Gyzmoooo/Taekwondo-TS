import sys
from PyQt6.QtWidgets import QApplication, QLabel, QMainWindow
from PyQt6.QtGui import QPixmap
from PyQt6.QtCore import Qt

class SpriteViewer(QLabel):
    def __init__(self, image_path, parent=None):
        super().__init__(parent)
        pixmap = QPixmap(image_path)
        self.setPixmap(pixmap)
        self.setFixedSize(pixmap.size())

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Sprites Separati con PyQt6")

        # Ottieni le dimensioni dello schermo principale
        screen_size = app.primaryScreen().size()
        
        # Imposta la geometria della finestra a schermo intero
        self.setGeometry(0, 0, screen_size.width(), screen_size.height())

        # ---
        # Crea il primo sfondo, lo ridimensiona e lo rende visibile
        initial_background_pixmap = QPixmap('sfondo1.png')
        if not initial_background_pixmap.isNull():
            initial_background_pixmap = initial_background_pixmap.scaled(
                screen_size,
                Qt.AspectRatioMode.IgnoreAspectRatio,  # Ignora l'aspect ratio
                Qt.TransformationMode.SmoothTransformation
            )
        self.initial_background = QLabel(self)
        self.initial_background.setPixmap(initial_background_pixmap)
        self.initial_background.setGeometry(0, 0, screen_size.width(), screen_size.height())

        # ---
        # Crea il secondo sfondo ('tamplate.png'), lo ridimensiona e lo nasconde
        template_background_pixmap = QPixmap('tamplate.png')
        if not template_background_pixmap.isNull():
            template_background_pixmap = template_background_pixmap.scaled(
                screen_size,
                Qt.AspectRatioMode.IgnoreAspectRatio,
                Qt.TransformationMode.SmoothTransformation
            )
        self.template_background = QLabel(self)
        self.template_background.setPixmap(template_background_pixmap)
        self.template_background.setGeometry(0, 0, screen_size.width(), screen_size.height())
        self.template_background.hide()
        
        # ---
        # Crea gli altri sprite, che hanno una dimensione fissa, e li nasconde
        self.sfondo_calci = SpriteViewer('sfondo-calci.png', self)
        self.sfondo_calci.move(243, 243)
        self.sfondo_calci.hide()

        self.kicks_recognition = SpriteViewer('kicks-recognition.png', self)
        self.kicks_recognition.move(243, 25)
        self.kicks_recognition.hide()

        self.logo = SpriteViewer('logo.png', self)
        self.logo.move(1500, 800)
        self.logo.hide()

        
        self.bandal = SpriteViewer('bandal_chagi.png', self)
        self.bandal.move(550, 710)
        self.bandal.hide()

        self.chiki = SpriteViewer('chiki.png', self)
        self.chiki.move(550, 280)
        self.chiki.hide()

        self.cut = SpriteViewer('cut.png', self)
        self.cut.move(550, 500)
        self.cut.hide()

        self.dx = SpriteViewer('dx.png', self)
        self.dx.move(1000, 300)
        self.dx.hide()

        self.sx = SpriteViewer('sx.png', self)
        self.sx.move(1000, 525)
        self.sx.hide()
        
        
        self.setFocusPolicy(Qt.FocusPolicy.StrongFocus)

    def keyPressEvent(self, event):
        if event.key() == Qt.Key.Key_Space:
            self.initial_background.hide()
            self.template_background.show()
            self.sfondo_calci.show()
            self.kicks_recognition.show()
            self.logo.show()
            
            self.bandal.show()
            self.chiki.show()
            self.cut.show()
            self.dx.show()
            self.sx.show()
            

if __name__ == '__main__':
    app = QApplication(sys.argv)
    window = MainWindow()
    window.showFullScreen()
    sys.exit(app.exec())