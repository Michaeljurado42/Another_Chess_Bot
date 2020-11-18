import subprocess
import argparse

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Allows you to play against a bot. Useful for testing and debugging.')
    parser.add_argument('first_path', help='Path to first bot source file.')
    parser.add_argument('second_path', help='Path to second bot source file.')

    args = parser.parse_args()

    first_path = args.first_path
    second_path = args.second_path

    subprocess.call(["python", first_path, second_path])
