import sys

class DualOutput:
    def __init__(self, filename):
        self.terminal = sys.stdout
        self.log = open(filename, "a")  # Открываем файл в режиме добавления

    def write(self, message):
        self.terminal.write(message)  # Выводим в консоль
        self.log.write(message)  # Пишем в файл

    def flush(self):
        self.terminal.flush()
        self.log.flush()

def enable_dual_output(filename):
    sys.stdout = DualOutput(filename)

