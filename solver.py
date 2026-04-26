import itertools


def get_feedback(secret, guess):
    """Returns (bulls, cows) feedback for a guess."""
    bulls = sum(s == g for s, g in zip(secret, guess))
    cows = sum(min(secret.count(x), guess.count(x)) for x in set(guess)) - bulls
    return bulls, cows


def solve():
    # Generate all possible 4-digit combinations (0000 - 9999)
    possible_numbers = [''.join(p) for p in itertools.product('0123456789', repeat=4)]

    print("--- 4-Digit Number Predictor ---")

    while len(possible_numbers) > 1:
        # Strategy: Pick the first available number as the next guess
        current_guess = possible_numbers[0]
        print(f"\nTry this number: {current_guess}")

        # Get feedback from the actual game you are playing
        try:
            b = int(input("How many BULLS (correct digit, correct spot)? "))
            c = int(input("How many COWS (correct digit, wrong spot)? "))
        except ValueError:
            print("Please enter a valid number.")
            continue

        if b == 4:
            print(f"Success! The number is {current_guess}")
            break

        # ELIMINATION STEP: Keep only numbers that would give the SAME feedback
        possible_numbers = [
            num for num in possible_numbers
            if get_feedback(num, current_guess) == (b, c)
        ]

        print(f"Remaining possibilities: {len(possible_numbers)}")


if __name__ == "__main__":
    solve()