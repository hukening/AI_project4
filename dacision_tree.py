from pandas import DataFrame
import pandas as pd
import sys
import math


class tree(object):
    def __init__(self, name):
        self.name = name
        self.children = []
        self.value = []


class Decision_tree(object):
    attributes_list=[]
    # print(self.data)
    def __init__(self):
        ''',filename'''
        self.data = pd.read_csv('iris.data.discrete.txt', header=None)

    def importance(self, attributes, examples):  # attributes is the list of attribute
        info = self.get_info(examples)
        print("dddd",info)
        max = 0
        final=0
        for i in attributes:
            info_a = self.get_info_attribute(examples, i)
            information_gain = info - info_a
            if information_gain > max:
                max = information_gain
                final = i
        print(max)
        return final

    def get_info_attribute(self, examples, attribute):
        attribute_list = []
        # print(attribute)
        # print(examples[attribute])
        for i in examples[attribute]:
            attribute_list.append(i)
        count = 0
        single_attr_dict = {}
        count_attr_dict = {}
        in_dict_list = []
        in_dict = {}
        value_set = set()
        value_list = []
        in_dict_count = 0
        for value in attribute_list:
            in_dict = {}
            if value not in single_attr_dict:
                single_attr_dict[value] = in_dict

                if examples[examples.columns.size - 1][count] not in in_dict:
                    in_dict[examples[examples.columns.size - 1][count]] = 1
                else:
                    in_dict[examples[examples.columns.size - 1][count]] += 1
            else:
                if examples[examples.columns.size - 1][count] not in single_attr_dict[value]:
                    single_attr_dict[value][examples[examples.columns.size - 1][count]] = 1
                else:
                    single_attr_dict[value][examples[examples.columns.size - 1][count]] += 1

            count += 1
        for value in attribute_list:
            value_set.add(value)
            if value not in count_attr_dict:
                count_attr_dict[value] = 1
            else:
                count_attr_dict[value] += 1
        value_list = list(value_set)
        info_a = 0
        for i in value_list:
            cnt = 0
            for value in single_attr_dict[i].values():
                cnt += value
            inf = 0
            for j in single_attr_dict[i].keys():
                # print(single_attr_dict[i][j]/cnt)
                inf += -(single_attr_dict[i][j] / cnt) * math.log(single_attr_dict[i][j] / cnt, 2)
            info_a += cnt / examples.iloc[:, 0].size * inf

        return info_a

    def get_info(self, examples):
        class_list = examples[examples.columns.size - 1]
        class_dict = {}
        info = 0
        for i in class_list:
            if i not in class_dict:
                class_dict[i] = 1
            else:
                class_dict[i] += 1  # get the classification dict

        for i in class_dict:
            info += -(class_dict[i] / examples.iloc[:, 0].size * math.log(class_dict[i] / examples.iloc[:, 0].size, 2))
        return info

    def plurality_value(self, examples):
        value_dict = {}
        for i in examples[examples.columns.size - 1]:
            if i not in value_dict:
                value_dict[i] = 1
            else:
                value_dict[i] += 1
        maxvalue = 0
        for key, value in value_dict.items():
            if value >= maxvalue:
                maxvalue = value
                final = key
        dtree2 = tree(final)
        return dtree2

    def have_same_classification(self, examples):
        count = 0
        for i in range(examples.iloc[:, 0].size - 1):
            if examples.loc[i,examples.columns.size-1]==examples.loc[i+1,examples.columns.size-1]:
                count += 1
            else:
                break
        if count == examples.iloc[:, 0].size-1 :
            return True
        else:
            return False

    def decision_tree(self, examples, attributes, parents_examples):
        tmp_example=examples
        drop_num=len(self.attributes_list)-1
        tmp_example=tmp_example.drop(columns=[drop_num])
        if tmp_example.empty:
            return self.plurality_value(parents_examples)
        elif self.have_same_classification(examples):
            dtree1 = tree(examples.loc[0,examples.columns.size-1])
            return dtree1

        elif len(attributes)-1 == 0:
            return self.plurality_value(examples)
        else:
            at=attributes.copy()
            at.remove(attributes[len(attributes)-1])
            attr = self.importance(at, examples)
            print(attr)
            dtree = tree(attr)
            value_dict = {}
            columns_list=[]
            for i in range(len(self.attributes_list)):
                columns_list.append(i)
           # columns_list.append(len(columns_list))
            exs = pd.DataFrame(columns=columns_list)
            for value in self.data[attr]:
                if value not in value_dict:
                    value_dict[value] = 1
                else:
                    value_dict[value] += 1
            for key in value_dict.keys():
                exs = pd.DataFrame(columns=columns_list)
                index = 0
                for i in range((examples.iloc[:, 0].size)):
                    if key == examples[attr][i]:
                        row=examples.loc[i]
                        exs.loc[index]=row
                        index += 1
                tmp_att = attributes.copy()
                tmp_att.remove(attr)
                subtree = self.decision_tree(exs, tmp_att, examples)
                dtree.value.append(key)
                dtree.children.append(subtree)
        return dtree

    def run(self):

        self.attributes_list = self.data.columns.values.tolist()
        # attributes_list.remove(len(attributes_list) - 1)
        t = self.decision_tree(self.data, self.attributes_list, None)
        return t


def print_tree(t):
    print("name:", t.name, "value:", t.value)
    for i in t.children:
        print_tree(i)


def bfs(t):
    res = []
    l = [t]
    res.append(l)
    while (len(l) != 0):
        new_l = []
        for i in l:
            new_l += i.children
        res.append(new_l)
        l = new_l
    return res


def d_print(t):
    res = bfs(t)
    for bros in res:
        for bro in bros:
            print('{', "name:", bro.name, "value:", bro.value, '}', end='')
        print("\n")


def main():
    # argv: python dacision_tree.py iris.data.txt 0.3 0.2 0.4 0.6  ->argv[2] is test condition
    # input=sys.argv[1]

    dc = Decision_tree()
    d_print(dc.run())


if __name__ == "__main__":
    main()
