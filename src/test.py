from gsu.Sentiment import Sentiment


def main():
    s = Sentiment()
    print(s.Analyse("This movie was awesome! The acting was great, plot was wonderful, and there were "
                    "pythons...so yea!"))

    print(s.Analyse("I am happy and awesome"))

    print(s.Analyse("This movie was awesome"))

if __name__ == '__main__':
    main()
