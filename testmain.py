list1 = [0, 1, 2]
list2 = [3, 4, 5]

list3 = [a for val in zip(list1, list2) for a in val]
print(list3)