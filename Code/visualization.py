import plotly.express as px
import pandas as pd
import numpy as np
from inout import load_list


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


def show_tables_acc_newp_updated():
    import plotly.graph_objects as go

    # CIFAR
    nsamples = [3.5, 7, 14, 28, 56, 102, 204, 408]
    # 1, MC
    resnet_fine_MC = [0.8, 0.72, 0.63, 0.61, 0.55, 0.48, 0.43, 0.36]
    resnet_hier_MC = [0.8, 0.71, 0.62, 0.53, 0.52, 0.42, 0.31, 0.31]
    inception_fine_MC = [0.84, 0.78, 0.73, 0.69, 0.61, 0.4, 0.35, 0.31]
    inception_hier_MC = [0.82, 0.75, 0.67, 0.53, 0.5, 0.36, 0.29, 0.24]
    vit_fine_MC = [0.96, 0.94, 0.94, 0.88, 0.85, 0.67, 0.67, 0.43]
    vit_hier_MC = [0.97, 0.92, 0.89, 0.87, 0.8, 0.37, 0.31, 0.29]
    # 2, HMC
    resnet_fine_HMC = [0.21, 0.19, 0.17, 0.17, 0.16, 0.15, 0.14, 0.12]
    resnet_hier_HMC = [0.20, 0.18, 0.16, 0.15, 0.15, 0.13, 0.11, 0.11]
    inception_fine_HMC = [0.21, 0.20, 0.20, 0.19, 0.17, 0.13, 0.13, 0.12]
    inception_hier_HMC = [0.20, 0.19, 0.17, 0.15, 0.14, 0.12, 0.11, 0.11]
    vit_fine_HMC = [0.27, 0.27, 0.26, 0.24, 0.24, 0.18, 0.19, 0.14]
    vit_hier_HMC = [0.26, 0.25, 0.24, 0.24,  0.21, 0.12, 0.12, 0.11]

    #imagenet
    # nsamples = [4.7, 9.5, 19, 37, 74, 148, 296, 592]
    # # 1, MC
    # resnet_fine_MC = [0.98, 0.96, 0.94, 0.89, 0.86, 0.81, 0.77, 0.79]
    # resnet_hier_MC = [0.96, 0.95, 0.92, 0.86, 0.77, 0.64, 0.669, 0.57, 0.5]
    # inception_fine_MC = [0.99, 0.97, 0.95, 0.87, 0.8, 0.7, 0.63, 0.59, 0.57]
    # inception_hier_MC = [0.82, 0.8, 0.69, 0.66, 0.64, 0.53, 0.44, 0.38, 0.32]
    # vit_fine_MC = [0.99, 0.98, 0.97, 0.95, 0.91, 0.89, 0.86, 0.85, 0.83]
    # vit_hier_MC = [0.97, 0.94, 0.92, 0.86, 0.69, 0.7, 0.64, 0.47, 0.4]
    # # 2, HMC
    # resnet_fine_HMC = [0.03, 0.03, 0.03, 0.02, 0.03, 0.02, 0.02, 0.02, 0.03]
    # resnet_hier_HMC = [0.02,0.03,0.03,0.02,0.02,0.01,0.01,0.01,0.01]
    # inception_fine_HMC = [0.11,0.04,0.03,0.02,0.02,0.02,0.02,0.02,0.02]
    # inception_hier_HMC = [0.02,0.02,0.02,0.01,0.01,0.01,0.01,0.01,0.01]
    # vit_fine_HMC = [0.10,0.09,0.05,0.04,0.04,0.04,0.03,0.03,0.03]
    # vit_hier_HMC = [0.03,0.03,0.02,0.02,0.02,0.02,0.02,0.01,0.01]

    # aircraft
    # nsamples = [3.5, 7.5, 15, 30, 60]
    # # 1, CE
    # resnet_fine_MC = [0.9, 0.78, 0.58, 0.54, 0.39]
    # resnet_hier_MC = [0.9, 0.75, 0.69, 0.44, 0.24]
    # inception_fine_MC = [0.84, 0.72, 0.6, 0.47, 0.37]
    # inception_hier_MC = [0.82, 0.67, 0.47, 0.26, 0.21]
    # vit_fine_MC = [0.97, 0.97, 0.97, 0.97, 0.96]
    # vit_hier_MC = [0.95, 0.95, 0.95, 0.87, 0.81]
    # # 2, HMC
    # resnet_fine_HMC = [0.28,0.26,0.22,0.21,0.20]
    # resnet_hier_HMC = [0.28,0.24,0.23,0.20,0.17]
    # inception_fine_HMC = [0.28,0.24,0.22,0.20,0.19]
    # inception_hier_HMC = [0.25,0.23,0.20,0.17,0.17]
    # vit_fine_HMC = [0.31,0.31,0.31,0.30,0.30]
    # vit_hier_HMC = [0.30,0.30,0.30,0.28,0.27]

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
            size=28,
            color="black"
        ),
        xaxis_title='Number of samples per class',
        yaxis_title='Hierarchical Misclassification Cost (HMC)', plot_bgcolor='lavenderblush')

    # fig.update_layout(legend=dict(
    #     font = dict(
    #         size=28,
    #         color="black"
    #     ),
    #     yanchor="bottom",
    #     y=0.01,
    #     xanchor="left",
    #     x=0.01
    # ), yaxis_range=[0,1])

    fig.update_layout(legend=dict(
        # orientation="h",
        # itemwidth=100,
        font=dict(
            size=28,
            color="black"
        ),
        yanchor="top",
        y=1-0.01,
        xanchor="right",
        x=1-0.01
    ), yaxis_range=[0, 1])

    fig.show()


def show_tables_acc_newp():
    import plotly.graph_objects as go

    # CIFAR
    # nsamples = [3.5, 7, 14, 28, 56, 102, 204, 408]
    # # 1, MC
    # resnet_fine_MC = [0.8, 0.72, 0.63, 0.61, 0.55, 0.48, 0.43, 0.36]
    # resnet_hier_MC = [0.8, 0.71, 0.62, 0.53, 0.52, 0.42, 0.31, 0.31]
    # inception_fine_MC = [0.84, 0.78, 0.73, 0.69, 0.61, 0.4, 0.35, 0.31]
    # inception_hier_MC = [0.82, 0.75, 0.67, 0.53, 0.5, 0.36, 0.29, 0.24]
    # vit_fine_MC = [0.96, 0.94, 0.94, 0.88, 0.85, 0.67, 0.67, 0.43]
    # vit_hier_MC = [0.97, 0.92, 0.89, 0.87, 0.8, 0.37, 0.31, 0.29]
    # # 2, HMC
    # resnet_fine_HMC = [0.67, 0.62, 0.56, 0.56, 0.51, 0.47, 0.45, 0.40]
    # resnet_hier_HMC = [0.65, 0.57, 0.51, 0.47, 0.47, 0.42, 0.37, 0.37]
    # inception_fine_HMC = [0.68, 0.65, 0.63, 0.60, 0.56, 0.43, 0.41, 0.39]
    # inception_hier_HMC = [0.64, 0.61, 0.55, 0.47, 0.46, 0.40, 0.36, 0.35]
    # vit_fine_HMC = [0.86, 0.85, 0.84, 0.78, 0.77, 0.59, 0.62, 0.45]
    # vit_hier_HMC = [0.84, 0.80, 0.76, 0.75, 0.67, 0.40, 0.38, 0.37]

    #imagenet
    nsamples = [4.7, 9.5, 19, 37, 74, 148, 296, 592]
    # 1, MC
    resnet_fine_MC = [1 - i for i in [0.02, 0.04, 0.06, 0.11, 0.14, 0.19, 0.23, 0.21]]
    resnet_hier_MC = [1 - i for i in [0.04, 0.05, 0.08, 0.14, 0.23, 0.36, 0.33, 0.43, 0.50]]
    inception_fine_MC = [1 - i for i in [0.01, 0.03, 0.05, 0.13, 0.20, 0.30, 0.37, 0.41, 0.43]]
    inception_hier_MC = [1 - i for i in [0.18, 0.20, 0.31, 0.33, 0.36, 0.47, 0.55, 0.62, 0.67]]
    vit_fine_MC = [1 - i for i in [0.01, 0.02, 0.03, 0.05, 0.09, 0.11, 0.14, 0.15, 0.17]]
    vit_hier_MC = [1 - i for i in [0.03, 0.06, 0.08, 0.14, 0.31, 0.30, 0.36, 0.53, 0.60]]
    # 2, HMC
    resnet_fine_HMC = [0.10, 0.10, 0.11, 0.09, 0.10, 0.09, 0.09, 0.09, 0.10]
    resnet_hier_HMC = [0.07, 0.08, 0.08, 0.06, 0.06, 0.04, 0.04, 0.04, 0.04]
    inception_fine_HMC = [0.33, 0.11, 0.08, 0.07, 0.06, 0.06, 0.05, 0.05, 0.05]
    inception_hier_HMC = [0.06, 0.06, 0.05, 0.04, 0.04, 0.04, 0.03, 0.02, 0.02]
    vit_fine_HMC = [0.32, 0.28, 0.17, 0.14, 0.12, 0.12, 0.10, 0.10, 0.09]
    vit_hier_HMC = [0.09, 0.08, 0.07, 0.07, 0.05, 0.05, 0.05, 0.03, 0.03]

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
    fig.add_trace(go.Scatter(x=nsamples, y=resnet_fine_MC, name='ResNet Baseline',
                             line=dict(color='firebrick', width=4)))

    fig.add_trace(go.Scatter(x=nsamples, y=resnet_hier_MC, name='ResNet Hierarchy',
                             line=dict(color='royalblue', width=4)))

    fig.add_trace(go.Scatter(x=nsamples, y=inception_fine_MC, name='Inception Baseline',
                             line=dict(color='firebrick', width=4, dash='dash')))

    fig.add_trace(go.Scatter(x=nsamples, y=inception_hier_MC, name='Inception Hierarchy',
                             line=dict(color='royalblue', width=4, dash='dash')))

    fig.add_trace(go.Scatter(x=nsamples, y=vit_fine_MC, name='ViT Baseline',
                             line=dict(color='firebrick', width=4, dash='dot')))

    fig.add_trace(go.Scatter(x=nsamples, y=vit_hier_MC, name='ViT Hierarchy',
                             line=dict(color='royalblue', width=4, dash='dot')))

    # Edit the layout
    fig.update_layout(font=dict(
            size=28,
            color="black"
        ),
        xaxis_title='Number of samples per class',
        yaxis_title='Misclassification Cost (MC)') #, plot_bgcolor='lavenderblush')

    fig.update_layout(legend=dict(
        font = dict(
            size=28,
            color="black"
        ),
        yanchor="bottom",
        y=0.01,
        xanchor="left",
        x=0.01
    ), yaxis_range=[0,1])

    # fig.update_layout(legend=dict(
    #     # orientation="h",
    #     # itemwidth=100,
    #     font=dict(
    #         size=28,
    #         color="black"
    #     ),
    #     yanchor="top",
    #     y=1-0.01,
    #     xanchor="right",
    #     x=1-0.01
    # ), yaxis_range=[0, 1])

    fig.show()

#-----------------------------------------------------------------------------------------------------------

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
    fig.add_trace(go.Scatter(x=weight_decays, y=red_8_mc, name='56 samples',
                             line=dict(color='firebrick', width=4)))

    fig.add_trace(go.Scatter(x=weight_decays, y=red_16_mc, name='28 samples',
                             line=dict(color='royalblue', width=4)))

    fig.add_trace(go.Scatter(x=weight_decays, y=red_32_mc, name='14 samples',
                             line=dict(color='green', width=4)))

    fig.add_trace(go.Scatter(x=weight_decays, y=red_64_mc, name='7 samples',
                             line=dict(color='black', width=4)))

    fig.add_trace(go.Scatter(x=weight_decays, y=red_128_mc, name='3.5 samples',
                             line=dict(color='gray', width=4)))


    # Edit the layout
    fig.update_layout(font=dict(size=15,color="black"),
        xaxis_title='Lambda hyperparameter value',
        yaxis_title='Misclassification Cost (MC)')
        #plot_bgcolor='lavenderblush')

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


def show_tables_tdbu():
    import plotly.graph_objects as go

    # aircraft
    nsamples = ["3.5", "7.5", "15", "30", "60"]
    # 1, CE
    both_MC = [0.85, 0.75, 0.59, 0.53, 0.50]
    td_MC = [0.81, 0.70, 0.60, 0.55, 0.53]
    bu_MC = [0.81, 0.73, 0.62, 0.54, 0.52]

    both_HMC = [0.80, 0.68, 0.51, 0.34, 0.15]
    td_HMC = [0.80, 0.70, 0.56, 0.33, 0.16]
    bu_HMC = [0.80, 0.70, 0.51, 0.34, 0.15]

    # both_MC = [0.85, 0.75, 0.57, 0.34, 0.16]
    # td_MC = [0.76, 0.69, 0.56, 0.33, 0.16]
    # bu_MC = [0.81, 0.73, 0.51, 0.34, 0.15]
    #
    # both_HMC = [0.80, 0.68, 0.59, 0.53, 0.50]
    # td_HMC = [0.80, 0.70, 0.60, 0.55, 0.53]
    # bu_HMC = [0.80, 0.70, 0.62, 0.54, 0.52]

    fig = go.Figure()
    # Create and style traces
    fig.add_trace(go.Scatter(x=nsamples, y=both_MC, name='Both MC',
                             line=dict(color='firebrick', width=4)))

    fig.add_trace(go.Scatter(x=nsamples, y=td_MC, name='Top-Down MC',
                             line=dict(color='royalblue', width=4)))

    fig.add_trace(go.Scatter(x=nsamples, y=bu_MC, name='Bottom-Up MC',
                             line=dict(color='orange', width=4)))

    fig.add_trace(go.Scatter(x=nsamples, y=both_HMC, name='Both HMC',
                             line=dict(color='firebrick', width=4, dash='dot')))

    fig.add_trace(go.Scatter(x=nsamples, y=td_HMC, name='Top-Down HMC',
                             line=dict(color='royalblue', width=4, dash='dot')))

    fig.add_trace(go.Scatter(x=nsamples, y=bu_HMC, name='Bottom-Up HMC',
                             line=dict(color='orange', width=4, dash='dot')))

    # Edit the layout
    fig.update_layout(font=dict(
            size=15,
            color="black"
        ),
        xaxis_title='Number of samples per class',
        # yaxis_title='Hierarchical Misclassification Cost (MC)',
        plot_bgcolor='lavenderblush')

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
        yanchor="bottom",
        y=0.01,
        xanchor="left",
        x=0.01
    ), yaxis_range=[0, 1])

    fig.show()


def show_tables_frozen():
    import plotly.graph_objects as go

    # aircraft
    nsamples = ["3.5", "7", "14", "26", "56"]
    # 1, CE
    frozen_mc = [0.81, 0.70, 0.62, 0.57, 0.53]
    unfrozen_mc = [0.80, 0.72, 0.62, 0.54, 0.47]

    frozen_hmc = [0.68, 0.60, 0.54, 0.51, 0.51]
    unfrozen_hmc = [0.69, 0.61, 0.55, 0.49, 0.45]

    fig = go.Figure()
    # Create and style traces
    fig.add_trace(go.Scatter(x=nsamples, y=frozen_mc, name='Frozen MC',
                             line=dict(color='firebrick', width=4)))

    fig.add_trace(go.Scatter(x=nsamples, y=unfrozen_mc, name='Unfrozen MC',
                             line=dict(color='royalblue', width=4)))

    fig.add_trace(go.Scatter(x=nsamples, y=frozen_hmc, name='Frozen HMC',
                             line=dict(color='firebrick', width=4, dash='dot')))

    fig.add_trace(go.Scatter(x=nsamples, y=unfrozen_hmc, name='Unfrozen HMC',
                             line=dict(color='royalblue', width=4, dash='dot')))

    # Edit the layout
    fig.update_layout(font=dict(
            size=15,
            color="black"
        ),
        xaxis_title='Number of samples per class',
        # yaxis_title='Hierarchical Misclassification Cost (MC)',
        plot_bgcolor='lavenderblush')

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
        yanchor="bottom",
        y=0.01,
        xanchor="left",
        x=0.01
    ), yaxis_range=[0, 1])

    fig.show()


def ablation():
    import plotly.graph_objects as go

    # CIFAR
    nsamples = [3.5, 7, 14, 28, 56, 102, 204, 408]
    # 1, MC
    inception_fine_MC = [1 - i for i in [0.16, 0.22, 0.27, 0.31, 0.39, 0.60, 0.65, 0.68]]
    inception_loss_MC = [1 - i for i in [0.19, 0.26, 0.39, 0.48, 0.58, 0.64, 0.68, 0.72]]
    inception_reg_MC = [1 - i for i in [0.22, 0.35, 0.41, 0.45, 0.60, 0.63, 0.69, 0.74]]
    inception_hier_MC = [1 - i for i in [0.18, 0.25, 0.32, 0.47, 0.50, 0.64, 0.71, 0.76]]

    # 2, HMC
    inception_fine_HMC = [0.68, 0.65, 0.63, 0.60, 0.56, 0.43, 0.41, 0.39]
    inception_loss_HMC = [0.64, 0.60, 0.53, 0.47, 0.42, 0.40, 0.38, 0.37]
    inception_reg_HMC = [0.67, 0.56, 0.53, 0.50, 0.43, 0.42, 0.39, 0.36]
    inception_hier_HMC = [0.64, 0.61, 0.55, 0.47, 0.46, 0.40, 0.36, 0.35]

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
        yaxis_title='Hierarchical Misclassification Cost (HMC)', plot_bgcolor='lavenderblush')

    # fig.update_layout(legend=dict(
    #     font = dict(
    #         size=22,
    #         color="black"
    #     ),
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


if __name__ == "__main__":
    show_tables_acc_newp_updated()