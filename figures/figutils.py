from statannotations.PValueFormat import PValueFormat
import numpy as np
import pandas as pd

# constants

cm = 1 / 2.54

# palettes


# functions

def set_figure_style():

    import matplotlib as mpl
    import matplotlib.font_manager as fm
    import seaborn as sns

    # Font settings — use Arial if available, otherwise DejaVu Sans
    available = {f.name for f in fm.fontManager.ttflist}
    mpl.rcParams['font.family'] = 'Arial' if 'Arial' in available else 'DejaVu Sans'
    mpl.rcParams["font.size"] = 6
    
    mpl.rcParams['xtick.labelsize'] = 6
    mpl.rcParams['ytick.labelsize'] = 6
    
    mpl.rcParams['axes.labelsize'] = 8
    mpl.rcParams['axes.titlesize'] = 8
    
    mpl.rcParams['legend.fontsize'] = 6
    mpl.rcParams['legend.title_fontsize'] = 8

    mpl.rcParams['axes.linewidth'] = 1
    mpl.rcParams['figure.figsize'] = (2.5, 2.5)
    mpl.rcParams['axes.spines.top'] = False
    mpl.rcParams['axes.spines.right'] = False
    
    mpl.rcParams['figure.dpi'] = 150
    mpl.rcParams['savefig.dpi'] = 300
    
    sns.set_context(rc={'figure.figsize': (2.5, 2.5)})
    
    
def stat_cor(g, regplot=True, correlation='spearman'):
    from scipy import stats
    import seaborn as sns
    import numpy as np

    x = g._x_var
    y = g._y_var
    row_var = g._row_var
    col_var = g._col_var

    for key, ax in g.axes_dict.items():
        # handle 1D vs 2D facets
        if row_var and col_var:
            row_val, col_val = key
            subset = g.data[
                (g.data[row_var] == row_val) &
                (g.data[col_var] == col_val)
            ]
        elif col_var:
            col_val = key
            subset = g.data[g.data[col_var] == col_val]
        elif row_var:
            row_val = key
            subset = g.data[g.data[row_var] == row_val]
        else:
            subset = g.data

        subset = subset.dropna(subset=[x, y])
        
        if len(subset) > 1:
            if regplot:
                sns.regplot(
                    data=subset, x=x, y=y,
                    scatter=False, ax=ax,
                    line_kws={'linestyle': '--', 'color': 'black', 'linewidth': 1}
                )

            if correlation == 'spearman':
                try:
                    r, p = stats.spearmanr(subset[x], subset[y])
                except Exception:
                    r, p = np.nan, np.nan
                    
            if correlation == 'pearson':
                try:
                    r, p = stats.pearsonr(subset[x], subset[y])
                except Exception:
                    r, p = np.nan, np.nan

            ax.text(
                0.05, 0.95,
                f"R = {r:.2f}\np = {p:.1e}\nn = {len(subset)}",
                transform=ax.transAxes, ha='left', va='top',
                fontsize=6, fontfamily='Arial'
            )
            
            
class MyPvalueFormat(PValueFormat):
    def format_data(self, result):
        # Use scientific notation for very small p-values, fixed for others
        fmt = ".3f" if result.pvalue >= 0.001 else ".1e"
        return f"p = {result.pvalue:{fmt}}{result.significance_suffix}"