"""Offline-first command line entry point."""
import argparse
from .adapters import AuthorizedFeedAdapter
from .storage import connect, upsert_product, approved_total

def main():
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="command", required=True)
    imp = sub.add_parser("import-products")
    imp.add_argument("--platform", required=True, choices=("shopee", "lazada"))
    imp.add_argument("--csv", required=True)
    sub.add_parser("approved-total")
    sub.add_parser("serve")
    args = parser.parse_args()
    if args.command == "serve":
        from .web import serve
        serve()
    elif args.command == "approved-total":
        with connect() as db:
            print(approved_total(db))
    else:
        count = 0
        with connect() as db:
            for p in AuthorizedFeedAdapter(args.platform, args.csv).products():
                upsert_product(db, p.platform, p.product_id, p.title,
                               p.price, p.commission_rate, p.affiliate_url, p.provenance)
                count += 1
        print(f"Imported {count} products")
if __name__ == "__main__":
    main()
