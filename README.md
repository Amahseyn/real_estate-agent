# Real Estate Agent Application

## Overview
This project is a real estate agent application built using Python. It includes functionality for managing real estate data and interacting with a database. The application uses Streamlit for the user interface.

## Project Structure
```
real_estate-agent/
├── app.py                  # Main entry point for the application
├── requirements.txt        # Python dependencies
├── chromadbgpt/            # Contains application logic
│   ├── app_simple.py       # Simplified app logic
│   ├── app.py              # Main app logic
│   ├── load_data_simple.py # Simplified data loading
│   └── load_data.py        # Data loading logic
├── data/                   # Contains data files
│   └── realtor-data.csv    # Sample real estate data
├── real_estate_chroma_db/  # Database files
│   ├── chroma.sqlite3      # SQLite database
│   └── ...                 # Additional database files
├── tests/                  # Test files
│   └── __pycache__/        # Cached test files
```

## Prerequisites
- Python 3.12
- Conda environment (optional)

## Installation
1. Clone the repository:
   ```bash
   git clone <repository-url>
   cd real_estate-agent
   ```

2. Create and activate a virtual environment (optional):
   ```bash
   conda create -n agent python=3.12
   conda activate agent
   ```

3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

## Usage
To run the application, use the following command:
```bash
streamlit run chromadbgpt/app.py
```

## Data
The `data/realtor-data.csv` file contains sample real estate data used by the application. You can replace this file with your own data.

## Database
The application uses an SQLite database located in the `real_estate_chroma_db/` directory. Ensure the database files are present before running the application.

## Testing
Tests are located in the `tests/` directory. To run the tests, use:
```bash
pytest
```

## Contributing
Feel free to submit issues or pull requests if you have suggestions or improvements.

## License
This project is licensed under the MIT License. See the LICENSE file for details.