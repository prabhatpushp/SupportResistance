# Support and Resistance Levels Identification

## Introduction
This project provides functionality to identify and visualize support and resistance levels in stock price data. It uses a Deterministic Finite Automaton (DFA) approach to analyze price movements and determine key levels that traders often use to make decisions.

## Features
- **Support and Resistance Identification**: Automatically identifies support and resistance levels based on historical price data.
- **Data Visualization**: Visualizes stock price data along with identified levels using candlestick charts.
- **Customizable Parameters**: Allows customization of parameters for identifying levels and plotting.

## Tech Stack
- **Python**: The implementation is written in Python.
- **Pandas**: Used for data manipulation and analysis.
- **NumPy**: Utilized for numerical operations and calculations.
- **Matplotlib**: For plotting and visualizing data.
- **mplfinance**: A library for visualizing financial data in candlestick format.

## Installation
1. Clone the repository:
   ```bash
   git clone <YOUR_GIT_URL>
   cd SupportResistance
   ```
2. Set up a virtual environment (optional but recommended):
   ```bash
   python -m venv .venv
   source .venv/bin/activate  # On Windows use .venv\Scripts\activate
   ```
3. Install the required dependencies:
   ```bash
   pip install pandas numpy matplotlib mplfinance scipy pytz
   ```

## Usage
To use the SupportResistance functionality, you can create an instance of the `SupportResistance` class and analyze stock price data. Here is an example:
```python
import pandas as pd
from SupportResistance import SupportResistance

# Load your dataset
# df = pd.read_csv('path_to_your_data.csv')

# Create an instance of SupportResistance
support_resistance = SupportResistance(df)

# Identify levels
levels = support_resistance.identify_levels()

# Analyze and plot levels
support_resistance.analyze()
```

## Development
- The main implementation is in `main.py`. You can modify or extend the functionality as needed.
- Ensure to test your changes thoroughly to maintain the integrity of the analysis.

## Contributing
Contributions are welcome! If you'd like to contribute, please follow these steps:
1. Fork the repository.
2. Create a new branch for your feature or bug fix.
3. Make your changes and commit them.
4. Push your changes and create a pull request.

## License
This project is licensed under the MIT License. Feel free to use this project for personal or commercial purposes. 