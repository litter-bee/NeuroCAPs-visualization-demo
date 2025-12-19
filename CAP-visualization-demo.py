from neurocaps.extraction import TimeseriesExtractor
from neurocaps.analysis import (CAP, merge_dicts, standardize, change_dtype)
import numpy as np, joblib, neurocaps

import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import seaborn as sns


import pickle
from matplotlib.colors import LinearSegmentedColormap

from neurocaps.analysis import transition_matrix

def func_merge_3():
    '''
    两组数据生成cap分析
    1、读取数据
    2、换成aal图谱
    :return:
    '''
    with open('ses1_subject_timeseries.pkl', 'rb') as f:
        subject_timeseries_1 = pickle.load(f)
    with open('ses2_subject_timeseries.pkl', 'rb') as f:
        subject_timeseries_2 = pickle.load(f)

    all_dicts = merge_dicts(subject_timeseries_list=[subject_timeseries_1, subject_timeseries_2],
                            return_merged_dict=True,
                            return_reduced_dicts=True)

    # print(all_dicts["dict_0"].keys())
    # print(all_dicts["dict_1"].keys())
    # print(all_dicts["merged"].keys())


    parcel_approach = {"AAL": {"version": "SPM12"}}
    cap_analysis = CAP(parcel_approach=parcel_approach)

    # cap_analysis = CAP(parcel_approach={"Schaefer": {"n_rois": 400, "resolution_mm": 1, "yeo_networks": 7}})

    cap_analysis.get_caps(subject_timeseries=all_dicts["dict_0"], n_clusters=3)


    output = cap_analysis.calculate_metrics(subject_timeseries=all_dicts["dict_0"], return_df=True, runs=[1])
    # output = cap_analysis.calculate_metrics(subject_timeseries=all_dicts["dict_0"], return_df=True, runs=[1, 2])

    v1 = output["temporal_fraction"]
    # print(v1)

    output = cap_analysis.calculate_metrics(subject_timeseries=all_dicts["dict_1"],
                                            return_df=True,
                                            runs=[1],
                                            continuous_runs=True)

    v2 = output["persistence"]
    # print(v2)

    output = cap_analysis.calculate_metrics(subject_timeseries=all_dicts["dict_1"], return_df=True, runs=[1, 2])
    v3 = output["transition_frequency"]
    # print(v3)

    corr_df = cap_analysis.caps2corr(annot=True, figsize=(5, 4), xticklabels_size=12, yticklabels_size=12,
                                     return_df=True,
                                     vmin=-1, vmax=1, annot_kws={"size": 13}, cbarlabels_size=13)
    v4 = corr_df["All Subjects"]
    # print(v4)

    from matplotlib.colors import LinearSegmentedColormap

    colors = ["#1bfffe", "#00ccff", "#0099ff", "#0066ff", "#0033ff",
              "#c4c4c4", "#ff6666", "#ff3333", "#FF0000", "#ffcc00",
              "#FFFF00"]

    cap_analysis.caps2surf(cmap="cold_hot", layout="row", size=(500, 100))
    custom_cmap = LinearSegmentedColormap.from_list("custom_cold_hot", colors, N=256)
    cap_analysis.caps2surf(cmap=custom_cmap, size=(500, 100), layout="row", surface="veryinflated")




def func_get_data():
    with open('ses1_subject_timeseries.pkl', 'rb') as f:
        subject_timeseries_1 = pickle.load(f)
    with open('ses2_subject_timeseries.pkl', 'rb') as f:
        subject_timeseries_2 = pickle.load(f)
    all_dicts = merge_dicts(subject_timeseries_list=[subject_timeseries_1, subject_timeseries_2],
                            return_merged_dict=True,
                            return_reduced_dicts=True)

    # 计算cap ,矩阵，持续时间等
    parcel_approach = {"AAL": {"version": "SPM12"}}
    cap_analysis = CAP(parcel_approach=parcel_approach)

    # cap_analysis = get_cap_analysis()
    cap_analysis.get_caps(subject_timeseries=all_dicts["merged"], n_clusters=4)
    return all_dicts, cap_analysis


def func_demo1(all_dicts):

    # 计算cap ,矩阵，持续时间等
    parcel_approach = {"AAL": {"version": "SPM12"}}
    cap_analysis = CAP(parcel_approach=parcel_approach)
    cap_analysis.get_caps(subject_timeseries=all_dicts["dict_0"], n_clusters=3)
    output = cap_analysis.calculate_metrics(subject_timeseries=all_dicts["dict_0"],
                                            return_df=True,
                                            runs=[1])
    v1 = output["temporal_fraction"]
    print(v1)

    output = cap_analysis.calculate_metrics(subject_timeseries=all_dicts["dict_0"],
                                            return_df=True,
                                            runs=[1],
                                            continuous_runs=True)

    v2 = output["persistence"]
    print(v2)

    output = cap_analysis.calculate_metrics(subject_timeseries=all_dicts["dict_0"],
                                            return_df=True,
                                            runs=[1])
    v3 = output["transition_frequency"]
    print(v3)

    corr_df = cap_analysis.caps2corr(annot=True, figsize=(5, 4), xticklabels_size=12, yticklabels_size=12,
                                     return_df=True,
                                     vmin=-1, vmax=1,
                                     annot_kws={"size": 13},
                                     cbarlabels_size=13)
    v4 = corr_df["All Subjects"]
    print(v4)


def func_surface(all_dicts):

    # 计算cap ,矩阵，持续时间等
    parcel_approach = {"AAL": {"version": "SPM12"}}
    cap_analysis = CAP(parcel_approach=parcel_approach)
    cap_analysis.get_caps(subject_timeseries=all_dicts["dict_0"], n_clusters=4)

    colors = ["#1bfffe", "#00ccff", "#0099ff", "#0066ff", "#0033ff",
              "#c4c4c4", "#ff6666", "#ff3333", "#FF0000", "#ffcc00",
              "#FFFF00"]

    # cap_analysis.caps2surf(cmap="cold_hot", layout="row", size=(1000, 200))

    custom_cmap = LinearSegmentedColormap.from_list("custom_cold_hot", colors, N=256)

    cap_analysis.caps2surf(cmap=custom_cmap, size=(500, 100), layout="row", surface="veryinflated")

    radialaxis = {"showline": True,
                  "linewidth": 2,
                  "linecolor": "rgba(0, 0, 0, 0.25)",
                  "gridcolor": "rgba(0, 0, 0, 0.25)",
                  "ticks": "outside",
                  "tickfont": {"size": 14, "color": "black"},
                  "range": [0, 0.6],
                  "tickvals": [0.1, "", "", 0.4, "", "", 0.6]}

    legend = {"yanchor": "top",
              "y": 0.99,
              "x": 0.99,
              "title_font_family": "Times New Roman",
              "font": {"size": 12, "color": "black"}}

    colors = {"High Amplitude": "black", "Low Amplitude": "orange"}

    cap_analysis.caps2radar(radialaxis=radialaxis,
                            fill="toself",
                            height=400,
                            width=600,
                            color_discrete_map=colors,
                            legend=legend)


def func_cicle(all_dict, cap_analysis):
    radialaxis = {"showline": True,
                  "linewidth": 2,
                  "linecolor": "rgba(0, 0, 0, 0.25)",
                  "gridcolor": "rgba(0, 0, 0, 0.25)",
                  "ticks": "outside",
                  "tickfont": {"size": 14, "color": "black"},
                  "range": [0, 0.6],
                  "tickvals": [0.1, "", "", 0.4, "", "", 0.6]}

    legend = {"yanchor": "top",
              "y": 0.99,
              "x": 0.99,
              "title_font_family": "Times New Roman",
              "font": {"size": 11, "color": "black"}}

    colors = {"High Amplitude": "black", "Low Amplitude": "orange"}


    cap_analysis.caps2radar(radialaxis=radialaxis,
                            fill="toself",
                            height=600,
                            width=900,
                            color_discrete_map=colors,
                            legend=legend,

                            output_dir='./'
                            )

def func_transition_probability(all_dicts):
    # cap_analysis.get_caps(
    #     subject_timeseries=all_dicts['dict_0'],
    #     cluster_selection_method="silhouette",
    #     standardize=True,
    #     show_figs=True,
    #     n_clusters=range(2, 6),
    # )

    # cap_analysis.get_caps(subject_timeseries=all_dicts["dict_0"], n_clusters=3)

    outputs = cap_analysis.calculate_metrics(
        subject_timeseries=all_dicts["dict_0"],
        return_df=True,
        metrics=["transition_probability"],
        continuous_runs=True,
    )


    kwargs = {
        "cmap": "Blues",
        "fmt": ".3f",
        "annot": True,
        "vmin": 0,
        "vmax": 1,
        "xticklabels_size": 10,
        "yticklabels_size": 10,
        "cbarlabels_size": 10,
    }

    trans_outputs = transition_matrix(
        trans_dict=outputs["transition_probability"], show_figs=True, return_df=True, **kwargs
    )

    print(trans_outputs["All Subjects"])


def now_getpic(cap_analysis):
    kwargs = {

        # "cmap": "coolwarm",
        # "xticklabels_size": 10,
        # "yticklabels_size": 10,
        # "xlabel_rotation": 90,
        # "cbarlabels_size": 10,
    }


    cap_analysis.caps2plot(
        yticklabels_size=10,
        wspace=0.1,
        subplots=False,
        visual_scope="regions",
        xlabel_rotation=90,
        xticklabels_size=10,
        hspace=0.6,
        tight_layout=False,
        figsize=(8, 8),
        output_dir='./',
        **kwargs
    )


if __name__ == '__main__':

    all_dict, cap_analysis = func_get_data()
    # func_demo1(all_dict)

    #1、 图片：cap 表层空间映射
    # func_surface(all_dict)

    #2、 图片：余弦相似度分析
    # func_cicle(all_dict, cap_analysis)

    # 3、数据：状态转移概率
    func_transition_probability(all_dict)

    # 4、状态转移概率

    # 5、fc
    # func_fc_matrix(all_dict)





