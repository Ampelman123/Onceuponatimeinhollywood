import matplotlib.pyplot as plt
from tueplots import bundles, axes, cycler
from tueplots.constants.color import rgb


def set_style(column="full", nrows=1, ncols=1):
    """
    Sets the matplotlib style using tueplots for scientific publication quality.
    Adjusts standard params for PDF output.
    """
    # Use a standard scientific journal bundle (e.g., NeurIPS or ICML)
    # This sets font sizes, family, and figure proportions.
    plt.rcParams.update(
        bundles.icml2024(usetex=False, column=column, nrows=nrows, ncols=ncols)
    )

    # Styles already set by bundle

    # Ensure PDF compatibility
    plt.rcParams["pdf.fonttype"] = 42  # TrueType
    plt.rcParams["ps.fonttype"] = 42

    return plt.rcParams


# Standard Colors for Consistency (Paul Tol Muted approximate mapping for semantic meaning if needed)
# Or manually defined compatible colors if specific semantics (like Profit/Loss) are required.
# Muted: [rose, indigo, sand, green, cyan, wine, teal, olive, purple, pale grey, black]

COLORS = {
    "increase": rgb.tue_green,  # Standard matplotlib green, reasonable match for 'good'
    "decrease": rgb.tue_red,  # Standard matplotlib red, reasonable match for 'bad'
    "revenue": rgb.tue_darkblue,  # Indigo (Paul Tol)
    "budget": rgb.tue_red,  # Wine (Paul Tol)
    "franchise": rgb.tue_blue,  # Teal (Paul Tol)
    "neutral": rgb.tue_gray,  # Pale Grey
    "text_main": "#333333",
    "text_light": "#666666",
}
