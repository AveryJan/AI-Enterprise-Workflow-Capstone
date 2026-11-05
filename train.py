from src.data import ingest
from src.model import train
if __name__ == "__main__":
    print(train(ingest("data/demo_revenue.csv"), "artifacts"))
