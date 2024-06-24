# ML Lottery Predictions

This project aims to predict lottery numbers using machine learning techniques. The project involves parsing lottery data, creating initial models, and making predictions.

## Description

### Predictions

- **predictions/predictions_2024-05-31.csv**: Contains the prediction results generated on 2024-05-31.

### PDF-Parsing

- **PDF-Parsing/c4l-parse.py**: Script to parse lottery data from text files.
- **PDF-Parsing/data/c4l_fl-06_29_2024-02-20-2017.txt**: Raw lottery data in text format.
- **PDF-Parsing/data/c4l_fl-06_29_2024-02-20-2017.pdf**: Original PDF file containing lottery data.

### Initial Model Creation

- **Initial-Model-Creation/Cash4Life-Florida/initial_model_generation_c4l.py**: Script for creating initial machine learning models.
- **Initial-Model-Creation/Cash4Life-Florida/data/parsed_c4l_fl-06_29_2024-02-20-2017.csv**: Parsed and processed lottery data.

### Predictions

- **Predictions/Cash4Life-Florida/make_predictions.py**: Script to make predictions using the trained models.

## Setup Python Environment

To set up the Python environment for this project, follow these steps:

1. **Clone the Repository**

    ```sh
    git clone https://github.com/your-username/ML_Lottery_predictions.git
    cd ML_Lottery_predictions
    ```

2. **Create a Virtual Environment**

    Create a virtual environment to manage project dependencies:

    ```sh
    python -m venv venv
    ```

3. **Activate the Virtual Environment**

    - On Windows:

      ```sh
      venv\Scripts\activate
      ```

    - On macOS and Linux:

      ```sh
      source venv/bin/activate
      ```

4. **Install Dependencies**

    Install the required Python packages using `pip`:

    ```sh
    pip install -r requirements.txt
    ```

    If `requirements.txt` doesn't exist yet, you can create it with the necessary dependencies:

    ```sh
    pip install pandas numpy tensorflow scikit-learn
    pip freeze > requirements.txt
    ```

## How to Use

### Parsing Lottery Data

1. Run `PDF-Parsing/c4l-parse.py` to parse the lottery data from the text file.

    ```sh
    python PDF-Parsing/c4l-parse.py
    ```

### Creating Initial Models

1. Run `Initial-Model-Creation/Cash4Life-Florida/initial_model_generation_c4l.py` to generate initial machine learning models.

    ```sh
    python Initial-Model-Creation/Cash4Life-Florida/initial_model_generation_c4l.py
    ```

### Making Predictions

1. Run `Predictions/Cash4Life-Florida/make_predictions.py` to make predictions using the trained models.

    ```sh
    python Predictions/Cash4Life-Florida/make_predictions.py
    ```

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for more details.

## Contributing

Contributions are welcome! Please open an issue or submit a pull request.

## Contact

For any inquiries or feedback, please contact Evan McCall at e.mccallvr@gmail.com.

