from sys import argv
from util.plotting import plot_results

if __name__ == "__main__":
    language = "en"
    if len(argv) > 1:
        language = argv[1]

    plot_results(language)
