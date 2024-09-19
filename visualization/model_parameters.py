from os import path
import pickle


def main():
    model_file = path.join('..', 'models', 'model_hc.pkl')
    scaler_file = path.join('..', 'models', 'scaler.pkl')

    with open(model_file, 'rb') as f:
        ols_results = pickle.load(f)

    with open(scaler_file, 'rb') as f:
        scaler = pickle.load(f)

    print(ols_results.params)
    print(scaler.mean_)
    print(scaler.var_)
    print(scaler.scale_)


if __name__ == "__main__":
    main()
