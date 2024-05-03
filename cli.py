import argparse

def main():
    parser = argparse.ArgumentParser(description='Process some integers.')
    parser.add_argument('--input', type=str, help='Input file path')
    parser.add_argument('--output', type=str, help='Output file path')

    args = parser.parse_args()

    print(f"Input file: {args.input}")
    print(f"Output file: {args.output}")

if __name__ == "__main__":
    main()