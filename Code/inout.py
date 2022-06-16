import matplotlib.pyplot as plt
import numpy as np
import pickle


def save_list(file_name, list_to_save):
    open_file = open(file_name, "wb")
    pickle.dump(list_to_save, open_file)
    open_file.close()


def load_list(file_name):
    open_file = open(file_name, "rb")
    loaded_list = pickle.load(open_file)
    open_file.close()
    return loaded_list


def imshow(img):
    img = img / 2 + 0.5  # unnormalize
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    plt.show()


def to_latex_heatmap(n_classes, classes_name, matrix):
    # "\newcommand\items{4}   %Number of classes
    # \arrayrulecolor{black} %Table line colors
    # \noindent\begin{tabular}{c*{\items}{|E}|}

    # this output

    # \end{tabular}"

    basic_string = "\multicolumn{1}{c}{} &" + "\n"

    for i in range(n_classes):
        basic_string += "\multicolumn{1}{c}{" + str(i + 1) + "} "
        if i != n_classes - 1:
            basic_string += "& \n"

    basic_string += "\\\ \hhline{~ *\items{ | -} |}" + "\n"

    for i in range(n_classes):
        basic_string += str(i + 1)

        for j in range(n_classes):
            basic_string += "& " + f"{matrix[i][j]:.1f}"

        basic_string += " \\\ \hhline{~*\items{|-}|}" + "\n"

    print(basic_string)

    #
    # A  & 100   & 0  & 10  & 0   \\ \hhline{~*\items{|-}|}
    # B  & 10   & 80  & 10  & 0 \\ \hhline{~*\items{|-}|}
    # C  & 30   & 0   & 70  & 0 \\ \hhline{~*\items{|-}|}
    # D  & 30   & 0   & 70  & 0 \\ \hhline{~*\items{|-}|}
    # \end{tabular}"


def readpgm(name):
    with open(name) as f:
        lines = f.readlines()
    # This ignores commented lines
    for l in list(lines):
        if l[0] == '#':
            lines.remove(l)
    # here,it makes sure it is ASCII format (P2)
    assert lines[0].strip() == 'P2'
    # Converts data to a list of integers
    data = []
    for line in lines[1:]:
        data.extend([int(c) for c in line.split()])
    return (np.array(data[3:]), (data[1], data[0]), data[2])

