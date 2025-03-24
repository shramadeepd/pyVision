import argparse
from pipe import FullPipeline

def main():
    parser = argparse.ArgumentParser(description='Run FullPipeline with specified parameters.')
    parser.add_argument('--model', type=str, required=True, help='Path to the model file')
    parser.add_argument('--LS_ID', type=int, nargs='+', required=True, help='List of LS_ID parameters (at least one required)')
    parser.add_argument('--epochs', type=int, required=True, help='Number of epochs')
    parser.add_argument('--batch_size', type=int, required=True, help='Batch size')

    args = parser.parse_args()

    if not args.LS_ID:
        parser.error("The --LS_ID parameter requires at least one value.")

    pipeline = FullPipeline(model=args.model, LS_ID=args.LS_ID, epochs=args.epochs, batch_size=args.batch_size)
    pipeline.run()

if __name__ == '__main__':
    main()

#python /home/yupcha/Desktop/project/pipeline/test.py --model <path_to_model> --LS_ID <LS_ID_value1> <LS_ID_value2> ... --epochs <number_of_epochs> --batch_size <batch_size_value>
