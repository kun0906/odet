from representation.report.paper_plots import parse_csv

in_file = 'examples/representation/out/20210910/res_20210910.csv'
data = parse_csv(in_file)
print(data)

