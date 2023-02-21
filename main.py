# main.py
#   variational auto encoder
# by: Noah Syrkis

# imports
from src import Model, train, load_data, get_args, plot
from multiprocessing import Pool
import torch
from torch import optim, nn
from torch.utils.data import DataLoader


# main
def main():
    args = get_args()

    if args.train:
        device = "cuda" if torch.cuda.is_available() else 'cpu'
        model = Model(args)
        model.to(device)
        args = get_args()
        optimizer = optim.Adam(model.parameters(), lr=args.lr)
        data = load_data()
        loader = DataLoader(data, batch_size=args.batch_size, shuffle=True)
        model = train(model, loader, optimizer, args.epochs, device)
        model = model.to('cpu')
        torch.save(model.state_dict(), f'models/model_dim_{args.latent_dim}.pth')


    if args.generate:
        device = "cuda" if torch.cuda.is_available() else 'cpu'
        model = Model(args)
        model.to(device)
        args = get_args()
        model.load_state_dict(torch.load(f'models/model_dim_{args.latent_dim}.pth'))
        model.eval()
        z = torch.randn(9 ** 2, args.latent_dim)
        imgs = model.decode(z).reshape(-1, 28, 28).detach().numpy()
        plot(imgs, imgs.shape[0], args)


    if args.parallel:
        latent_dims = [2, 5, 10, 20, 50, 100]
        def train_cpu(latent_dim):
            model = Model(args)
            args = get_args()
            args.latent_dim = latent_dim
            optimizer = optim.Adam(model.parameters(), lr=args.lr)
            data = load_data()
            loader = DataLoader(data, batch_size=args.batch_size, shuffle=True)
            model = train(model, loader, optimizer, args.epochs, 'cpu')
            model = model.to('cpu')
            torch.save(model.state_dict(), f'models/model_dim_{args.latent_dim}.pth')
        with Pool(6) as p:
            p.map(train_cpu, latent_dims)
        

# run main
if __name__ == '__main__':
    main()

