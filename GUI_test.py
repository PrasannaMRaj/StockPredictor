from PyQt5.QtWidgets import QApplication, QMainWindow, QLabel, QGridLayout, QWidget, QPushButton, QTableWidget, QTableWidgetItem, QInputDialog

def open_calculator():
    value, ok = QInputDialog.getDouble(
        window, # parent widget
        'Tax Calculator', # window title
        'Yearly Income:', # entry label
        min=0.0,
        max=1000000.0,
    )
    if not ok:
        return
    yearly_income.setText('Yearly Income: ${:,.2f}'.format(value))
    if value <= 9700:
        rate = 10.0
    elif value <= 39475:
        rate = 12.0
    elif value <= 84200:
        rate = 22.0
    elif value <= 160725:
        rate = 24.0
    elif value <= 204100:
        rate = 32.0
    else:
        rate=50
    tax_rate.setText('Highest Marginal Tax Rate: {}%'.format(rate))





app = QApplication([])
window = QMainWindow()

screen = QWidget()
layout = QGridLayout()
screen.setLayout(layout)

window.setCentralWidget(screen)
window.setWindowTitle('Freelancer Utility')
window.setMinimumSize(500, 200)
window.show()


yearly_income = QLabel()
yearly_income.setText('Yearly Income: $0.00')
layout.addWidget(yearly_income, 0, 0)

tax_rate = QLabel()
tax_rate.setText('Highest Marginal Tax Rate: 0%')
layout.addWidget(tax_rate, 0, 1)

button = QPushButton()
button.setText('Calculator')
button.clicked.connect(open_calculator)
layout.addWidget(button, 1, 0, 1, 2)

columns = ('Week', 'Hours Worked', 'Hourly Rate', 'Earned Income')

table_data = [
    [7, 40.0, 100.0],
    [8, 37.5, 85.0],
    [9, 65, 150.0],
]

table = QTableWidget()
table.setColumnCount(len(columns))
table.setHorizontalHeaderLabels(columns)

table.setRowCount(len(table_data))

for row_index, row in enumerate(table_data):
    # Set each column value in the table
    for column_index, value in enumerate(row):
        item = QTableWidgetItem(str(value))
        table.setItem(row_index, column_index, item)
    # Calculate the total and add it as another column
    table.setItem(row_index, 3, QTableWidgetItem(str(row[1] * row[2])))

layout.addWidget(table, 2, 0, 1, 2)


app.exec_()

