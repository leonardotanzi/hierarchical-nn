import plotly.express as px
import pandas as pd
import numpy as np
from utils import load_list


# plot the top-3 superclasse a class was assigned
def plot_graph_top3superclasses(y_pred, labels_fine, classes, superclasses):
    df2 = px.data.medals_long()
    full_df = pd.DataFrame()

    samples_per_classes = 100

    for j in range(len(classes)):
        super_names = []
        values, counts = np.unique(y_pred[j*samples_per_classes:j*samples_per_classes+samples_per_classes], return_counts=True)
        for n in values:
            super_names.append(superclasses[n])
        df = pd.DataFrame({"class": classes[labels_fine[j*samples_per_classes]], "superclass": super_names, "counts": counts})
        df = df.sort_values(by=['counts'], ascending=False, ignore_index=True)[0:3]

        full_df = full_df.append(df)

    fig = px.bar(full_df, x="class", y="counts", color="superclass", text_auto=True, width=2500)
    fig.show()


def plot_graph(y_pred, labels_fine, classes):
    # it predicts 20 values for superclasses
    # for each class, i want to visualize how many times it was classified in the correct superclass
    correct_superclasses = [0 for i in range(len(classes))]

    for i, label in enumerate(labels_fine):
        if label//5 == y_pred[i]:
            correct_superclasses[i // 100] += 1

    df = pd.DataFrame({"Classes": classes, "CorrectSuperclass": correct_superclasses})
    fig = px.bar(df, x="Classes", y="CorrectSuperclass", color="CorrectSuperclass", text_auto=".2s", width=2000)
    fig.update_traces(textfont_size=12, textangle=0, textposition="outside", cliponaxis=False)
    fig.show()


def plot_box(labels_fine, classes, superclasses):
    # it predicts 20 values for superclasses
    # for each class, i want to visualize how many times it was classified in the correct superclass

    y_pred_justcoarse = load_list("JustCoarse.pkl")
    y_pred_hloss = load_list("HLoss.pkl")

    correct_superclasses_jc = [0 for i in range(len(classes))]

    for i, label in enumerate(labels_fine):
        if label//5 == y_pred_justcoarse[i]:
            correct_superclasses_jc[i // 100] += 1

    correct_superclasses_hl = [0 for i in range(len(classes))]

    for i, label in enumerate(labels_fine):
        if label//5 == y_pred_hloss[i]:
            correct_superclasses_hl[i // 100] += 1

    superclasses = [superclasses[i//5] for i in range(len(classes))]

    jc = ["JustCoarse" for i in range(len(classes))]
    df1 = pd.DataFrame({"Classes": superclasses, "CorrectSuperclass": correct_superclasses_jc, "Loss":jc})

    hl = ["HLoss" for i in range(len(classes))]
    df2 = pd.DataFrame({"Classes": superclasses, "CorrectSuperclass": correct_superclasses_hl, "Loss": hl})

    df = pd.concat([df1, df2])

    fig = px.box(df, x="Classes", y="CorrectSuperclass", color="Loss")
    fig.update_traces(quartilemethod="linear")  # or "inclusive", or "linear" by default
    fig.show()
