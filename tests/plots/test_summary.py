import matplotlib
import numpy as np
matplotlib.use('Agg')
import shap_domino
import warnings


def test_random_summary():
    """ Just make sure the summary_plot function doesn't crash.
    """

    shap_domino.summary_plot(np.random.randn(20, 5), show=False)

def test_random_summary_with_data():
    """ Just make sure the summary_plot function doesn't crash with data.
    """

    shap_domino.summary_plot(np.random.randn(20, 5), np.random.randn(20, 5), show=False)

def test_random_multi_class_summary():
    shap_domino.summary_plot([np.random.randn(20, 5) for i in range(3)], np.random.randn(20, 5), show=False)

def test_random_summary_bar_with_data():
    shap_domino.summary_plot(np.random.randn(20, 5), np.random.randn(20, 5), plot_type="bar", show=False)

def test_random_summary_dot_with_data():
    shap_domino.summary_plot(np.random.randn(20, 5), np.random.randn(20, 5), plot_type="dot", show=False)

def test_random_summary_violin_with_data():
    shap_domino.summary_plot(np.random.randn(20, 5), np.random.randn(20, 5), plot_type="violin", show=False)

def test_random_summary_layered_violin_with_data():
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        shap_domino.summary_plot(np.random.randn(20, 5), np.random.randn(20, 5), plot_type="layered_violin", show=False)

def test_random_summary_with_log_scale():
    shap_domino.summary_plot(np.random.randn(20, 5), use_log_scale=True, show=False)
