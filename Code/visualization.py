import plotly.express as px
import pandas as pd
import numpy as np
from inout import load_list


# plot the top-3 superclasse a class was assigned
def plot_graph_top3superclasses(y_pred, labels_fine, classes, superclasses):
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

    y_pred_justcoarse = load_list("pkl//JustCoarse.pkl")
    y_pred_hloss = load_list("pkl//HLoss.pkl")

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


def plot_variance(labels_fine, classes, superclasses):
    y_pred_justcoarse = load_list("pkl//JustCoarse.pkl")
    y_pred_hloss = load_list("pkl//HLoss.pkl")

    correct_superclasses_jc = [0 for i in range(len(classes))]

    for i, label in enumerate(labels_fine):
        if label // 5 == y_pred_justcoarse[i]:
            correct_superclasses_jc[i // 100] += 1

    variance_superclasses_jc = [0 for i in range(len(superclasses))]
    for i in range(len(superclasses)):
        variance_superclasses_jc[i] = np.var(correct_superclasses_jc[i*5:i*5+5])

    correct_superclasses_hl = [0 for i in range(len(classes))]

    for i, label in enumerate(labels_fine):
        if label // 5 == y_pred_hloss[i]:
            correct_superclasses_hl[i // 100] += 1

    variance_superclasses_hl = [0 for i in range(len(superclasses))]
    for i in range(len(superclasses)):
        variance_superclasses_hl[i] = np.var(correct_superclasses_hl[i * 5:i * 5 + 5])

    # superclasses = [superclasses[i // 5] for i in range(len(classes))]

    jc = ["JustCoarse" for i in range(len(superclasses))]
    df1 = pd.DataFrame({"Classes": superclasses, "Variance": variance_superclasses_jc, "Loss": jc})

    hl = ["HLoss" for i in range(len(superclasses))]
    df2 = pd.DataFrame({"Classes": superclasses, "Variance": variance_superclasses_hl, "Loss": hl})

    df = pd.concat([df1, df2])

    fig = px.scatter(df, x="Classes", y="Variance", color="Loss", symbol="Loss")
    fig.update_traces(marker_size=10)
    fig.show()


def show_tables_acc():
    import plotly.graph_objects as go

    # CIFAR
    # nsamples = [3.5, 7, 14, 28, 56]
    # # 1, MC
    # resnet_fine_MC = [1 - i for i in [0.20, 0.28, 0.37, 0.39, 0.45]]
    # resnet_hier_MC = [1 - i for i in [0.20, 0.29, 0.38, 0.47, 0.48]]
    # inception_fine_MC = [1 - i for i in [0.16, 0.22, 0.27, 0.31, 0.39]]
    # inception_hier_MC = [1 - i for i in [0.18, 0.25, 0.32, 0.47, 0.50]]
    # vit_fine_MC = [1 - i for i in [0.04, 0.06, 0.06, 0.12, 0.15]]
    # vit_hier_MC = [1 - i for i in [0.03, 0.07, 0.11, 0.13, 0.20]]
    # # 2, HMC
    # resnet_fine_HMC = [0.67, 0.62, 0.56, 0.56, 0.51]
    # resnet_hier_HMC = [0.65, 0.57, 0.51, 0.47, 0.47]
    # inception_fine_HMC = [0.68, 0.65, 0.63, 0.60, 0.56]
    # inception_hier_HMC = [0.64, 0.61, 0.55, 0.47, 0.46]
    # vit_fine_HMC = [0.86, 0.85, 0.84, 0.78, 0.77]
    # vit_hier_HMC = [0.84, 0.80, 0.76, 0.75, 0.67]

    # imagenet
    nsamples = [9.5, 19, 37, 74, 148]
    # 1, MC
    resnet_fine_MC = [1 - i for i in [0.04, 0.06, 0.11, 0.14, 0.19]]
    resnet_hier_MC = [1 - i for i in [0.05, 0.08, 0.14, 0.23, 0.36]]
    inception_fine_MC = [1 - i for i in [0.03, 0.05, 0.13, 0.20, 0.30]]
    inception_hier_MC = [1 - i for i in [0.20, 0.31, 0.33, 0.36, 0.47]]
    vit_fine_MC = [1 - i for i in [0.02, 0.03, 0.05, 0.09, 0.11]]
    vit_hier_MC = [1 - i for i in [0.06, 0.08, 0.14, 0.31, 0.30]]
    # 2, HMC
    resnet_fine_HMC = [0.10, 0.11, 0.09, 0.10, 0.09]
    resnet_hier_HMC = [0.08, 0.08, 0.06, 0.06, 0.04]
    inception_fine_HMC = [0.11, 0.08, 0.07, 0.06, 0.06]
    inception_hier_HMC = [0.06, 0.05, 0.04, 0.04, 0.04]
    vit_fine_HMC = [0.28, 0.17, 0.14, 0.12, 0.12]
    vit_hier_HMC = [0.08, 0.07, 0.07, 0.05, 0.05]

    # aircraft
    # nsamples = [3.5, 7.5, 15, 30, 60]
    # # 1, CE
    # resnet_fine_MC = [1 - i for i in [0.10, 0.22, 0.42, 0.46, 0.61]]
    # resnet_hier_MC = [1 - i for i in [0.10, 0.25, 0.31, 0.55, 0.76]]
    # inception_fine_MC = [1 - i for i in [0.16, 0.28, 0.40, 0.53, 0.63]]
    # inception_hier_MC = [1 - i for i in [0.18, 0.32, 0.53, 0.74, 0.78]]
    # vit_fine_MC = [1 - i for i in [0.03, 0.03, 0.03, 0.03, 0.04]]
    # vit_hier_MC = [1 - i for i in [0.05, 0.05, 0.05, 0.13, 0.19]]
    # # 2, HMC
    # resnet_fine_HMC = [0.88, 0.81, 0.69, 0.67, 0.61]
    # resnet_hier_HMC = [0.86, 0.75, 0.73, 0.62, 0.53]
    # inception_fine_HMC = [0.86, 0.76, 0.68, 0.63, 0.60]
    # inception_hier_HMC = [0.78, 0.71, 0.61, 0.53, 0.53]
    # vit_fine_HMC = [0.96, 0.97, 0.97, 0.93, 0.93]
    # vit_hier_HMC = [0.95, 0.95, 0.95, 0.88, 0.84]

    fig = go.Figure()
    # Create and style traces
    fig.add_trace(go.Scatter(x=nsamples, y=resnet_fine_HMC, name='ResNet Baseline',
                             line=dict(color='firebrick', width=4)))

    fig.add_trace(go.Scatter(x=nsamples, y=resnet_hier_HMC, name='ResNet Hierarchy',
                             line=dict(color='royalblue', width=4)))

    fig.add_trace(go.Scatter(x=nsamples, y=inception_fine_HMC, name='Inception Baseline',
                             line=dict(color='firebrick', width=4, dash='dash')))

    fig.add_trace(go.Scatter(x=nsamples, y=inception_hier_HMC, name='Inception Hierarchy',
                             line=dict(color='royalblue', width=4, dash='dash')))

    fig.add_trace(go.Scatter(x=nsamples, y=vit_fine_HMC, name='ViT Baseline',
                             line=dict(color='firebrick', width=4, dash='dot')))

    fig.add_trace(go.Scatter(x=nsamples, y=vit_hier_HMC, name='ViT Hierarchy',
                             line=dict(color='royalblue', width=4, dash='dot')))

    # Edit the layout
    fig.update_layout(font=dict(
            size=15,
            color="black"
        ),
        xaxis_title='Number of samples per class',
        yaxis_title='Hierarchical Misclassification Cost (MC)', plot_bgcolor='lavenderblush')

    # fig.update_layout(legend=dict(
    #     yanchor="bottom",
    #     y=0.01,
    #     xanchor="left",
    #     x=0.01
    # ), yaxis_range=[0,1])

    fig.update_layout(legend=dict(
        # orientation="h",
        # itemwidth=100,
        font=dict(
            size=22,
            color="black"
        ),
        yanchor="top",
        y=1-0.01,
        xanchor="right",
        x=1-0.01
    ), yaxis_range=[0, 1])

    fig.show()


def show_tables_WD():
    import plotly.graph_objects as go

    # aircraft
    weight_decays = ["10", "1", "0.1", "0.01", "0.001", "0.0001"]
    # 1, CE
    red_8_mc = [1 - i for i in [0.54, 0.56, 0.53, 0.53, 0.52, 0.54]]
    red_16_mc = [1 - i for i in [0.26, 0.45, 0.46, 0.45, 0.45, 0.43]]
    red_32_mc = [1 - i for i in [0.35, 0.42, 0.36, 0.33, 0.29, 0.29]]
    red_64_mc = [1 - i for i in [0.30, 0.25, 0.21, 0.28, 0.23, 0.24]]
    red_128_mc = [1 - i for i in [0.15, 0.10, 0.12, 0.13, 0.12, 0.14]]

    # 2, HMC
    red_8_hmc = [0.44, 0.43, 0.45, 0.45, 0.48, 0.46]
    red_16_hmc = [0.55, 0.49, 0.49, 0.49, 0.50, 0.51]
    red_32_hmc = [0.43, 0.50, 0.55, 0.57, 0.60, 0.61]
    red_64_hmc = [0.58, 0.63, 0.67, 0.61, 0.62, 0.63]
    red_128_hmc = [0.70, 0.80, 0.77, 0.74, 0.75, 0.73]

    fig = go.Figure()
    # Create and style traces
    fig.add_trace(go.Scatter(x=weight_decays, y=red_8_hmc, name='Reduction factor 8',
                             line=dict(color='firebrick', width=4)))

    fig.add_trace(go.Scatter(x=weight_decays, y=red_16_hmc, name='Reduction factor 16',
                             line=dict(color='royalblue', width=4)))

    fig.add_trace(go.Scatter(x=weight_decays, y=red_32_hmc, name='Reduction factor 32',
                             line=dict(color='green', width=4)))

    fig.add_trace(go.Scatter(x=weight_decays, y=red_64_hmc, name='Reduction factor 64',
                             line=dict(color='black', width=4)))

    fig.add_trace(go.Scatter(x=weight_decays, y=red_128_hmc, name='Reduction factor 128',
                             line=dict(color='gray', width=4)))


    # Edit the layout
    fig.update_layout(xaxis_title='Lambda hyperparameter value',
                      yaxis_title='Hierarchical Misclassification Cost (HMC)')#, plot_bgcolor='lavenderblush')

    fig.update_layout(legend=dict(
        # orientation="h",
        # itemwidth=100,
        font=dict(
            size=20,
            color="black"
        ),
        yanchor="bottom",
        y=0.01,
        xanchor="left",
        x=0.01
    ), yaxis_range=[0,1])

    fig.show()


if __name__ == "__main__":
    show_tables_acc()