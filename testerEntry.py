from utils.FileWriter import FileWriter

if __name__ == "__main__":
    path = "C:/temp"
    filename = "tester.txt"

    # 第一次新建文件并且写入
    fileWriter = FileWriter(path=path, filename=filename)
    fileWriter.writeLine("this a test file")
    fileWriter.writeLine("first writing")
    fileWriter.destory()

    # 第二次找到已有文件并追加
    fileWriter = FileWriter(path=path, filename=filename)
    fileWriter.writeLine("")
    fileWriter.writeLine("this is a test file")
    fileWriter.writeLine("second writing")
    fileWriter.destory()