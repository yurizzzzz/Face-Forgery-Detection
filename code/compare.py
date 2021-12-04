import csv
import sys
import os

fzwFile = open("./predict3.csv", "r")
wxFile = open("./submission.csv", "r")
reader1 = csv.reader(fzwFile)
reader2 = csv.reader(wxFile)

label_list1 = []
label_list2 = []

for item1 in reader1:
    key = item1[0]
    value = item1[1]
    if value != 'label':
        value = int(value)
        label_list1.append(value)

# for item2 in reader2:
#     key = item2[0]
#     value = item2[1]
#     if value != 'label':
#         value = int(value)
#         label_list2.append(value)
for item2 in reader2:
    key = item2[-1][:-2]
    value = item2[-1][-1]
    if value != 'l':
        value = int(value)
        label_list2.append(value)

dif_list = []
for i in range(8076):
    if label_list1[i] != label_list2[i]:
        dif_list.append(i)


f = open('difference.csv', 'w', encoding='utf-8', newline="")
csv_write = csv.writer(f)
csv_write.writerow(["filename", "fzw", "wx"])
print(dif_list)
for i in range(len(dif_list)):
    filename = 'test_' + str(dif_list[i]) + '.jpg'
    file_label1 = label_list1[dif_list[i]]
    file_label2 = label_list2[dif_list[i]]
    csv_write.writerow([filename, str(file_label1), str(file_label2)])
