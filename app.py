import sys
from typing import NoReturn, Optional

import numpy as np
import pandas as pd

# from backend import CrossAnalysisSolver
from PyQt5.QtGui import QFont
from PyQt5.QtWidgets import (
    QApplication,
    QFileDialog,
    QFrame,
    QHeaderView,
    QLabel,
    QLineEdit,
    QPushButton,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
    QWidget,
)


class App(QWidget):
    def __init__(self) -> None:
        super().__init__()
        self.init_ui()

    def init_ui(self) -> NoReturn:
        font_bold = QFont()
        font_bold.setBold(True)

        up = QFrame(self)
        up.setFrameShape(QFrame.StyledPanel)

        label_input_data = QLabel("Вхідні дані", up)
        label_input_data.move(2, 10)
        self.input_data = QLineEdit(up)
        self.input_data.setText("./data/tesla_stock.csv")
        self.input_data.setFixedWidth(200)
        self.input_data.move(100, 7)

        open_input_data = QPushButton("...", up)
        open_input_data.setCheckable(True)
        open_input_data.move(310, 7)
        open_input_data.clicked[bool].connect(self.open_input_data_dialog)

        label_target_column = QLabel("Цільова змінна", up)
        label_target_column.move(2, 50)
        self.input_target_column = QLineEdit(up)
        self.input_target_column.setText("Close")
        self.input_target_column.setFixedWidth(100)
        self.input_target_column.move(150, 45)

        label_index_column = QLabel("Колонка індексів", up)
        label_index_column.move(2, 90)
        self.index_column = QLineEdit(up)
        self.index_column.setText("0")
        self.index_column.setFixedWidth(30)
        self.index_column.move(170, 86)

        execute_button = QPushButton("Виконати", up)
        execute_button.move(450, 130)
        execute_button.clicked.connect(self.execute)

        # value = QLabel("Інтегральний коефіціент достовірності", up)
        # value.move(2, 180)
        # self.value = QLineEdit(up)
        # # self.value.setText("10000")
        # self.value.setFixedWidth(100)
        # self.value.move(365, 177)

        table_frame = QFrame(self)
        table_frame.setFrameShape(QFrame.StyledPanel)

        self.table_widget = QTableWidget(table_frame)
        self.table_widget.resize(1800, 525)
        self.table_widget.setRowCount(8)
        self.table_widget.setColumnCount(10)
        self.table_widget.move(0, 0)
        header = self.table_widget.horizontalHeader()
        for index in range(10):
            header.setSectionResizeMode(index, QHeaderView.ResizeToContents)
        for index in range(8):
            self.table_widget.setVerticalHeaderItem(index, QTableWidgetItem(f"e{index + 1}"))
        self.table_widget.setHorizontalHeaderItem(0, QTableWidgetItem(f"Середні оцінки \n ймовірностей експертів"))
        self.table_widget.setHorizontalHeaderItem(1, QTableWidgetItem(f"Відкалібровані \n ймовірності"))
        for index in range(8):
            self.table_widget.setHorizontalHeaderItem(index + 2, QTableWidgetItem(f"Сценарій {index + 1}"))

        vertical_layout = QVBoxLayout()
        vertical_layout.addWidget(up)
        vertical_layout.addWidget(table_frame)
        self.setLayout(vertical_layout)

        self.setGeometry(150, 150, 1200, 600)
        self.setWindowTitle("Метод перехресного впливу")
        self.show()

    def open_input_data_dialog(self) -> NoReturn:
        options = QFileDialog.Options()
        options |= QFileDialog.DontUseNativeDialog
        target_path = str(QFileDialog.getOpenFileName(self, "Виберіть файл", options=options)[0])
        if target_path:
            self.input_data.setText(target_path)

    @staticmethod
    def text_to_int(text: str) -> Optional[int]:
        try:
            return int(text)
        except ValueError:
            return None

    def execute(self) -> NoReturn:
        try:
            data: pd.DataFrame = pd.read_csv(
                self.input_data.text(), index_col=App.text_to_int(self.index_column.text())
            )
            target_variable: pd.Series = data[self.input_target_column.text()]
            print(target_variable)
        except FileNotFoundError:
            print("Неправильний шлях до файлу")
        except KeyError:
            print("Такої цільової змінної не існує")
        except IndexError:
            print("Такої колонки індексів не існує")

        # solver = CrossAnalysisSolver(
        #     probs_path=self.input_data.text(), cond_probs_path=self.input_cond_prob_file.text(),
        # )
        # try:
        #     final_table, value = solver.solve(number_of_executions=int(self.number_of_executions.text()))

        #     for i in range(final_table.shape[0]):  # pylint: disable=E1136
        #         for j in range(2):
        #             self.table_widget.setItem(i, j, QTableWidgetItem("%.3f" % final_table[i][j]))
        #     for i in range(final_table.shape[0]):  # pylint: disable=E1136
        #         for j in range(2, final_table.shape[1]):  # pylint: disable=E1136
        #             main_part = "%.3f" % final_table[i][j]
        #             if i != j - 2:
        #                 delta = final_table[i][j] - final_table[i][1]
        #                 sign = "+" if delta > 0 else "-"
        #                 main_part += f" ({sign} {np.abs(delta):.3f})"
        #             self.table_widget.setItem(i, j, QTableWidgetItem(main_part))

        #         self.value.setText(f"{value:.3f}")
        # except ValueError:
        #     print("Кількість ітерацій має бути цілим значенням")


def main() -> NoReturn:
    pyqt_application = QApplication(sys.argv)
    _ = App()
    sys.exit(pyqt_application.exec_())


if __name__ == "__main__":
    main()
