# excel文件转化成csv
import pandas as pd

file = 'D:\\picture processing\\training_data\\All.xlsx'
outfile = 'D:\\picture processing\\training_data\\All_csv.csv'


def xlsx_to_csv_pd():
   data_xls = pd.read_excel(file, index_col=0)

   data_xls.to_csv(outfile, encoding='utf-8')

if __name__ == '__main__':
  xlsx_to_csv_pd()
print("\n转化完成！！！\nCSV文件所处位置：" + str(outfile))

# import csv
# import xlwt
# import pandas as pd

# def csv_to_xlsx(csvfile):
#     with open(csvfile, encoding='utf-8') as fc:
#         r_csv = csv.reader(fc)
#         workbook = xlwt.Workbook()
#         sheet = workbook.add_sheet('sheet1')  # 创建一个sheet表格
#         i = 0
#         j = 0
#         for line in r_csv:
#             for v in line:
#                 sheet.write(i, j, v)
#                 j = j + 1
#             i = i + 1
#         workbook.save(outfile)  # 保存Excel
#
#
# file = 'E:\\pythondata\\ql\\test1.csv'  # 待转化的源文件
# outfile = 'E:\\pythondata\\ql\\test1_out.xlsx'  # 转化后的excel所处的位置与文件名
#
# if __name__ == '__main__':
#     csv_to_xlsx(file)
#     print("转化完成！！！\nExcel文件所处位置：" + str(outfile))