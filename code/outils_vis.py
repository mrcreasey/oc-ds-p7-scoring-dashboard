"""
General plotting routines
"""

from math import nan
import matplotlib.pyplot as plt

settings = dict(
    SAVE_IMAGES=True,
    IMAGE_FOLDER='images'
)

# - pour enregistrer les graphiques, define **`SAVE_IMAGES = True`**


def set_option(key, value):
    """Set option key,value"""
    settings[key] = value


def get_option(key):
    """Get option key,value"""
    return settings.get(key, nan)


def sanitize(fig_name: str) -> str:
    """Enlever les caractères interdits dans les filenames ou filepaths"""
    return fig_name.replace(' ', '_').replace(':', '-').replace(
        '.', '-').replace('\n', '').replace('/', '_').replace('>', 'gt.').replace('<', 'lt.')


def to_png(fig_name=None) -> None:
    """
    Register the current plot figure as an image in a file.
    Must call plt.show() or show image (by calling to_png() as last row in python cell)
    to apply the call 'bbox_inches=tight', to be sure to include the whole title / legend
    in the plot area.
    """

    def get_title() -> str:
        """find current plot title (or suptitle if more than one plot)"""
        # pylint: disable=protected-access
        if plt.gcf()._suptitle is None:  # noqa
            return plt.gca().get_title()
        else:
            # pylint: disable=protected-access
            return plt.gcf()._suptitle.get_text()  # noqa

    if settings.get('SAVE_IMAGES') is True:
        if fig_name is None:
            fig_name = get_title()
        elif len(fig_name) < 9:
            fig_name = f'{fig_name}_{get_title()}'
        fig_name = sanitize(fig_name)
        print(f'"{fig_name}.png"')
        plt.gcf().savefig(
            f"{settings.get('IMAGE_FOLDER')}/{fig_name}.png", bbox_inches='tight')
