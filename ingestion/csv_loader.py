import pandas as pd
from pathlib import Path
from langchain.schema import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter

DATA_DIR = Path(__file__).parent.parent / "data"
CSV_FILE = DATA_DIR / "financial_summary.csv"

CHUNK_SIZE = 1000
CHUNK_OVERLAP = 200


def load_csv():
    """Load financial_summary.csv and convert rows to Document objects."""
    df = pd.read_csv(CSV_FILE)

    docs = []
    for _, row in df.iterrows():
        # represent each row as a readable text block
        text = (
            f"Company: {row['company']}\n"
            f"Year: {row['year']}\n"
            f"Net Income: {row['net_income_millions']} million USD\n"
            f"Total Revenue: {row['total_revenue_millions']} million USD\n"
            f"Provision for Credit Losses: {row['provision_for_credit_losses_millions']} million USD\n"
            f"Net Charge-Off Rate: {row['net_charge_off_rate_pct']}%"
        )
        doc = Document(
            page_content=text,
            metadata={"source": "financial_summary.csv", "company": row["company"], "year": int(row["year"])},
        )
        docs.append(doc)

    # split in case rows are long (mostly won't be for this CSV)
    splitter = RecursiveCharacterTextSplitter(chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP)
    chunks = splitter.split_documents(docs)
    print(f"  financial_summary.csv: {len(df)} rows → {len(chunks)} chunks")
    return chunks


if __name__ == "__main__":
    chunks = load_csv()
    print(f"\nTotal chunks: {len(chunks)}")
    for c in chunks:
        print(f"\n{c.page_content}\n---")
