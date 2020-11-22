import sys
from typing import Any, NoReturn, Optional

import pandas as pd
from matplotlib import pyplot as plt
from PyQt5.QtGui import QFont
from PyQt5.QtWidgets import (
    QApplication,
    QComboBox,
    QFileDialog,
    QFrame,
    QLabel,
    QLineEdit,
    QPushButton,
    QTextEdit,
    QVBoxLayout,
    QWidget,
)

from models.model import BestFilterFinder


class App(QWidget):
    def __init__(self) -> None:
        super().__init__()
        self.model_name = "LinearRegression"
        self.metric_name = "MAE"
        self.validation_percent_int: float = 5
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
        self.input_target_column.move(145, 45)

        label_index_column = QLabel("Колонка індексів", up)
        label_index_column.move(2, 90)
        self.index_column = QLineEdit(up)
        self.index_column.setText("0")
        self.index_column.setFixedWidth(30)
        self.index_column.move(160, 86)

        label_p = QLabel("P", up)
        label_p.move(2, 135)
        self.p = QLineEdit(up)
        self.p.setText("10")
        self.p.setFixedWidth(30)
        self.p.move(20, 130)

        label_q = QLabel("Q", up)
        label_q.move(60, 135)
        self.q = QLineEdit(up)
        self.q.setFixedWidth(30)
        self.q.move(80, 130)

        validation_percent_label = QLabel("Відсоток даних для валідації", up)
        validation_percent_label.move(2, 180)
        self.validation_percent = QLineEdit(up)
        self.validation_percent.setText("5")
        self.validation_percent.setFixedWidth(40)
        self.validation_percent.move(255, 175)

        model_label = QLabel("Модель", up)
        model_label.move(450, 10)
        self.model_name_box = QComboBox(up)
        self.model_name_box.addItems(
            [
                "LinearRegression",
                "RidgeRegression",
                "LassoRegression",
                "SVM",
                "XGBoostRegression",
                "RandomForestRegression",
            ]
        )
        self.model_name_box.move(530, 7)
        self.model_name_box.activated[str].connect(self.model_name_handler)

        metric_label = QLabel("Метрика для оптимізації", up)
        metric_label.move(450, 50)
        self.metric_name_box = QComboBox(up)
        self.metric_name_box.addItems(["MAE", "MSE", "R2"])
        self.metric_name_box.move(670, 47)
        self.metric_name_box.activated[str].connect(self.metric_name_handler)

        self.execute_button = QPushButton("Виконати", up)
        self.execute_button.move(550, 230)
        self.execute_button.clicked.connect(self.execute)

        table_frame = QFrame(self)
        table_frame.setFrameShape(QFrame.StyledPanel)

        self.output = QTextEdit(table_frame)
        self.output.setReadOnly(True)
        self.output.setLineWrapMode(QTextEdit.NoWrap)
        self.output.setFixedWidth(1160)
        self.output.setMinimumHeight(300)
        self.output.setMaximumHeight(1000)
        self.output.move(5, 5)

        vertical_layout = QVBoxLayout()
        vertical_layout.addWidget(up)
        vertical_layout.addWidget(table_frame)
        self.setLayout(vertical_layout)

        self.setGeometry(150, 150, 1200, 600)
        self.setWindowTitle("Порівняння методів фільтрації")
        self.show()

    def open_input_data_dialog(self) -> NoReturn:
        options = QFileDialog.Options()
        options |= QFileDialog.DontUseNativeDialog
        target_path = str(QFileDialog.getOpenFileName(self, "Виберіть файл", options=options)[0])
        if target_path:
            self.input_data.setText(target_path)

    @staticmethod
    def text_to_int(text: str, default: Optional[Any] = None) -> Optional[Any]:
        try:
            return int(text)
        except ValueError:
            return default

    def model_name_handler(self, value: str) -> NoReturn:
        self.model_name = value

    def metric_name_handler(self, value: str) -> NoReturn:
        self.metric_name = value

    @staticmethod
    def percent_handler(value: str) -> float:
        int_value = int(value)
        if 0 < int_value <= 1:
            return int_value
        elif 0 < int_value <= 100:
            return int_value * 0.01
        else:
            raise ValueError("Такий відсоток не підтримується")

    def execute(self) -> None:
        try:
            data: pd.DataFrame = pd.read_csv(
                self.input_data.text(), index_col=App.text_to_int(self.index_column.text())
            )
        except FileNotFoundError:
            print("Неправильний шлях до файлу")
            return
        except IndexError:
            print("Такої колонки індексів не існує")
            return

        try:
            variable: pd.Series = data[self.input_target_column.text()]
        except KeyError:
            print("Такої цільової змінної не існує")
            return

        try:
            self.validation_percent_int = App.percent_handler(self.validation_percent.text())
        except ValueError:
            print("Такий відсоток не підтримується")

        model = BestFilterFinder(
            model_name=self.model_name,
            metric_name=self.metric_name.lower(),
            validation_percent=self.validation_percent_int,
        )
        text = ""
        y_test, ma_filter, ma_predict, ma_params, ma_metrics = model.grid_search_moving_average(
            variable=variable.copy(), q=App.text_to_int(self.q.text()), p=App.text_to_int(self.p.text(), default=1)
        )
        text += (
            f"Moving Average: {ma_metrics} з параметрами q = {ma_params.q} та вікном = {ma_params.moving_average}\n"
        )

        _, exp_ma_filter, exp_ma_predict, exp_ma_params, exp_ma_metrics = model.grid_search_exp_moving_average(
            variable=variable.copy(), q=App.text_to_int(self.q.text()), p=App.text_to_int(self.p.text(), default=1)
        )

        text += f"Exponential Moving Average: {exp_ma_metrics} з параметрами q = {exp_ma_params.q} та alpha = {exp_ma_params.alpha}"
        self.output.setText(text)
        try:
            y_test.index = pd.to_datetime(y_test.index)
        except ValueError:
            pass
        fig, axs = plt.subplots(2, 1, figsize=(16, 16))
        axs[0].plot(y_test.index, y_test, label="Правильні значення")
        axs[0].plot(y_test.index, ma_filter, label=f"Moving average з вікном = {ma_params.moving_average}")
        axs[0].plot(y_test.index, exp_ma_filter, label=f"Exponential Moving average з alpha = {exp_ma_params.alpha}")
        axs[0].set_title("Візуалізація найкращих фільтрів")
        axs[0].legend()

        axs[1].plot(y_test.index, y_test, label="Правильні значення")
        axs[1].plot(
            y_test.index,
            ma_predict,
            label=f"{self.model_name} та Moving average з вікном = {ma_params.moving_average}",
        )
        axs[1].plot(
            y_test.index,
            exp_ma_predict,
            label=f"{self.model_name} та Exponential Moving average з alpha = {exp_ma_params.alpha}",
        )
        axs[1].set_title("Передбачення моделей, які використовують найкращі фільтри")
        axs[1].legend()
        plt.show()


def main() -> NoReturn:
    pyqt_application = QApplication(sys.argv)
    _ = App()
    sys.exit(pyqt_application.exec_())


if __name__ == "__main__":
    main()
