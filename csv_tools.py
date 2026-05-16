import pandas as pd

#
from langchain_core.tools import tool

@tool
def get_csv_summary(filename: str) -> dict:
    """
    Read a CSV file and return summary statistics.
    """
    try:
        df = pd.read_csv(filename)

        return {
            "rows": len(df),
            "columns": list(df.columns),
            "dtypes": df.dtypes.astype(str).to_dict(),
            "summary": df.describe(include="all").fillna("").to_dict()
        }

    except Exception as e:
        return {"error": str(e)}