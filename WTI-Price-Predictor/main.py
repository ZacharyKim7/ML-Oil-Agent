from data_helpers import data_tabulate as data

def main():
    df = data.get_data_from_csv()

    print(df.isna().any(axis=1).sum())

    return 1

if __name__ == "__main__":
    main()
