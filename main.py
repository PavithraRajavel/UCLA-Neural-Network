import logging
from src.utils import setup_logging
from src.data_preprocessing import load_data, preprocess_data
from src.eda import plot_distributions, plot_correlations
from src.model import split_data, train_model, evaluate_model


def main():
    setup_logging()

    # Load and preprocess the data
    data = load_data('data/admission.csv')
    data = preprocess_data(data)

    # Perform EDA
    plot_distributions(data)
    plot_correlations(data)

    # Split the data
    X_train, X_test, y_train, y_test = split_data(data, 'Admit_Chance')

    # Train the model
    model = train_model(X_train, y_train)

    # Evaluate the model
    accuracy = evaluate_model(model, X_test, y_test)
    logging.info(f'Model accuracy: {accuracy}')


if __name__ == '__main__':
    main()
