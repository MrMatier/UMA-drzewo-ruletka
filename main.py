from tests import run_all_tests
from wykresy import make_plots

def main():
    results = run_all_tests()

    print()
    print("wyniki testów:")
    print(results.round(4))

    print()
    print("zapisano wyniki do results/results.csv")
    make_plots()

if __name__ == "__main__":
    main()