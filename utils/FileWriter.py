from utils.Tools import *

class FileWriter():
    def __init__(self, path, filename):
        self.path = path
        self.filename = filename
        self.handler = None

        # 文件已经存在时直接追加
        if(os.path.isfile(genPath(path, filename))):
            self.handler = open(genPath(path, filename), "a+")
        # 文件不存在时创目录，创文件，并写入
        else:
            if(not os.path.exists(path)):
                os.makedirs(path)
            self.handler = open(genPath(path, filename), "w+")

    def writeLine(self, string):
        self.handler.write("{}\n".format(string))

    def destory(self):
        self.handler.close()
