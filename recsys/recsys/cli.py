#!/usr/bin/env python3
import argparse
from recsys.data_loader import load_ratings
from recsys.recommended_top import get_top_n_products
from recsys.recommend_user import recommend_for_user

def main():
    parser = argparse.ArgumentParser(description="Recommendation Engine CLI")
    parser.add_argument("csv", help="Path to ratings.csv")

    subparsers = parser.add_subparsers(dest="cmd")

    # Top N Command
    t = subparsers.add_parser("top", help="Get top-rated products")
    t.add_argument("--days", type=int, default=10000)
    t.add_argument("--n", type=int, default=10)

    # User Recommendation Command
    u = subparsers.add_parser("user", help="Get user product recommendations")
    u.add_argument("--user_id", type=int, required=True)
    u.add_argument("--n", type=int, default=10)

    args = parser.parse_args()
    df = load_ratings(args.csv)

    if args.cmd == "top":
        result = get_top_n_products(df, args.days, args.n)
        print(result)

    elif args.cmd == "user":
        result = recommend_for_user(df, args.user_id, args.n)
        print(result)

    else:
        print("Use either 'top' or 'user'.")

if __name__ == "__main__":
    main()
