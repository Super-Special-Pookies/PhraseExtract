import xlrd

with xlrd.open_workbook(r'FinancialDatasets.xlsx') as workbook:
    sheet = workbook.sheet_by_index(0) # sheet索引从0开始

    for rown in range(1, sheet.nrows):
        context = sheet.cell_value(rown, 2)

        with open('data/%d.txt'%rown, 'w') as f:
            f.write(context)
