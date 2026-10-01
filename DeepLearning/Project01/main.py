import csv
from pathlib import Path

import tensorflow as tf
from matplotlib import pyplot as plt

from model import DenseModel
from util import (
    confidence_interval,
    load_fashion_mnist,
    plot_confusion_matrix,
    visualize_predictions,
)


def main():
    train, val, test = load_fashion_mnist()
    x_train, y_train = train
    x_val, y_val = val
    x_test, y_test = test

    architectures = {'one_hidden_layer': (128,), 'two_hidden_layers': (128, 64)}

    # Each configuration is (architecture name, hidden units, learning rate, dropout).
    model_configs = [
        ('one_hidden_layer', (128,), 0.001, 0.0),
        ('one_hidden_layer', (128,), 0.001, 0.3),
        ('one_hidden_layer', (128,), 0.0001, 0.0),
        ('one_hidden_layer', (128,), 0.0001, 0.3),
        ('two_hidden_layers', (128, 64), 0.001, 0.0),
        ('two_hidden_layers', (128, 64), 0.001, 0.3),
        ('two_hidden_layers', (128, 64), 0.0001, 0.0),
        ('two_hidden_layers', (128, 64), 0.0001, 0.3),
    ]
    results = []
    best_result = None
    best_weights = None

    # Two architectures x two learning rates x dropout off/on = eight runs.
    for name, units, learning_rate, dropout_rate in model_configs:
        # Start each run with a fresh model and optimizer.
        tf.keras.backend.clear_session()
        tf.keras.utils.set_random_seed(42)
        print(f'\n{name}: learning_rate={learning_rate}, dropout={dropout_rate}')
        model = DenseModel(
            hidden_layers=units, learning_rate=learning_rate, dropout_rate=dropout_rate
        )
        model.build((None, 28, 28))
        model.fit(
            x_train, y_train,
            validation_data=(x_val, y_val),
            epochs=10,
            batch_size=64,
            verbose=2,
        )

        # Select settings using validation data only.
        val_loss, val_accuracy = model.evaluate(x_val, y_val, verbose=0)
        results.append({
            'architecture': name,
            'learning_rate': learning_rate,
            'dropout_rate': dropout_rate,
            'epochs': 10,
            'batch_size': 64,
            'val_loss': val_loss,
            'val_accuracy': val_accuracy,
        })
        if best_result is None or val_loss < best_result['val_loss']:
            best_result = results[-1]
            best_weights = model.get_weights()
        print(f'Validation loss: {val_loss:.4f}, accuracy: {val_accuracy:.4f}')

    #save results to a CSV file
    results_path = Path(__file__).with_name('results.csv')
    with results_path.open('w', newline='') as file:
        writer = csv.DictWriter(file, fieldnames=results[0].keys())
        writer.writeheader()
        writer.writerows(results)
    print(f'\nSaved all eight validation results to {results_path}')

    #Evaluate the selected model on the test set once.
    tf.keras.backend.clear_session()
    best_model = DenseModel(
        hidden_layers=architectures[best_result['architecture']],
        learning_rate=best_result['learning_rate'],
        dropout_rate=best_result['dropout_rate'],
    )

    best_model.build((None, 28, 28))
    best_model.set_weights(best_weights)
    test_loss, test_accuracy = best_model.evaluate(x_test, y_test, verbose=0)
    test_predictions = best_model.predict(x_test, verbose=0)

    #create a directory to save images
    images_dir = Path(__file__).with_name('images')
    images_dir.mkdir(exist_ok=True)

    #generate confusion matrix
    confusion_display = plot_confusion_matrix(
        y_test, test_predictions,
        title='Fashion-MNIST Test Confusion Matrix',
        save_path=images_dir / 'confusion_matrix.png',
    )
    plt.close(confusion_display.figure_)


    # 95% confidence interval on the test-set generalization error rate.
    error_rate, error_ci_lower, error_ci_upper = confidence_interval(y_test, test_predictions)

    final_result = dict(
        best_result,
        test_loss=test_loss,
        test_accuracy=test_accuracy,
        test_error_rate=error_rate,
        test_error_ci_lower=error_ci_lower,
        test_error_ci_upper=error_ci_upper,
    )

    #save final test results to a CSV file
    with results_path.with_name('test_results.csv').open('w', newline='') as file:
        writer = csv.DictWriter(file, fieldnames=final_result.keys())
        writer.writeheader()
        writer.writerow(final_result)
    print(f'Selected configuration: {best_result}')
    print(f'Final test loss: {test_loss:.4f}, accuracy: {test_accuracy:.4f}')
    print(
        f'Test error rate: {error_rate:.4f} '
        f'(95% CI: [{error_ci_lower:.4f}, {error_ci_upper:.4f}])'
    )
    print(f'Confusion matrix saved to {images_dir / "confusion_matrix.png"}')


    visualize_predictions(best_model, x_test, y_test, test_predictions, images_dir)


if __name__ == '__main__':
    main()
