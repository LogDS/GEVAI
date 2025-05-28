import matplotlib.font_manager as fm
import numpy as np
import pandas as pd
from plotnine import ggplot, aes, scale_y_log10, labs, geom_point, element_text, element_blank, theme, \
    scale_color_brewer, element_rect, \
    theme_light, facet_wrap, guides, guide_legend

if __name__ == '__main__':
    font_regular = fm.FontProperties(fname='./fonts/Satoshi-Medium.ttf', size=11)
    font_bold = fm.FontProperties(fname='./fonts/Satoshi-Bold.ttf', size=11)
    font_title = fm.FontProperties(fname='./fonts/Satoshi-Bold.ttf', size=13)

    csv_name = "../../../../results/may01/benchmark_5ex_post_explainers.csv" # "apr29/benchmark_5ex_post_explainers.csv"
    cols = list(pd.read_csv(csv_name, nrows=1))
    excluded_columns = ["Hypothesis", "Ad hoc model", "Load configuration", "Load dataset"]
    data = pd.read_csv(csv_name, quotechar="'", usecols=[i for i in cols if i not in excluded_columns])

    ad_hoc_types_column = "Ad hoc type"
    other_columns = [col for col in data.columns if col != ad_hoc_types_column and col != 'Ad hoc model']

    data_melted = data.melt(id_vars=[ad_hoc_types_column], value_vars=other_columns, var_name='phase',
                            value_name='seconds')

    data_mean_std = data_melted.groupby([ad_hoc_types_column, 'phase']).agg(
        mean_seconds=('seconds', np.mean),
        std_seconds=('seconds', np.std),
        count_seconds=('seconds', 'count')
    ).reset_index()

    # Calculate minimum seconds for each phase
    min_seconds_df = data_mean_std.groupby('phase')['mean_seconds'].min().reset_index()
    min_seconds_df = min_seconds_df.rename(columns={'mean_seconds': 'min_seconds'})

    # Merge back to get the corresponding 'Ad hoc type'
    min_data = pd.merge(min_seconds_df, data_mean_std, left_on=['phase', 'min_seconds'],
                        right_on=['phase', 'mean_seconds'], how='left')

    plot = (
            ggplot(data_melted, aes(x=ad_hoc_types_column, y='seconds', fill='phase'))
            + geom_point(  # show the mean
                data=data_mean_std,
                mapping=aes(x=ad_hoc_types_column, y='mean_seconds', shape='phase'),
                size=4
            )
            + scale_y_log10(minor_breaks=[],
                            breaks=[10 ** x for x in range(-5, 7)],
                            labels=lambda l: ["{:.0e}".format(v).replace("+0", "+").replace("-0", "-") for v
                                              in l])
            + labs(
                title='Ad Hoc Types vs. GEVAI Phases',
                x=ad_hoc_types_column,
                y='Time (seconds, log scale)',
                fill='Phase',
                shape='Phase',
            )
                + theme_light()
                + theme(
                    plot_background=element_rect(fill='white', color="white"),
                    text=element_text(fontproperties=font_regular),
                    legend_text=element_text(ha='left', fontproperties=font_regular),
                    axis_title_x=element_text(fontproperties=font_bold),
                    axis_title_y=element_text(fontproperties=font_bold),
                    legend_title=element_text(ha='left', fontproperties=font_bold),
                    plot_title=element_text(ha='center', fontproperties=font_title),
                    panel_border=element_blank(),
                    legend_position='bottom',
                )
        + scale_color_brewer(type='qual', palette='Dark2')
        + guides(color=guide_legend(nrow=2), shape=guide_legend(nrow=2))
    )

    plot.save("ex_post_comparison.png", dpi=1200, width=8.5, height=5)

    plot += facet_wrap('~phase', scales='free_x')  # Create a separate plot for each 'phase'
    plot.save("ex_post_comparison_facet.png", dpi=1200, width=8.5, height=5)
