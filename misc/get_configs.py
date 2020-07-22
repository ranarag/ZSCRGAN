import argparse
def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--batch_size', type=int, default=32, help='batch size used during training')
    parser.add_argument('--gf_dim', type=int, default=2048, help='hidden layer dimension of generator')
    parser.add_argument('--df_dim', type=int, default=1024, help='hidden layer dimension of discriminator')
    parser.add_argument('--embed_dim', type=int, default=1024,  help='dimension of mu and sigma')
    parser.add_argument('--CSEM_lr', type=float, default=0.002, help='learning rate of CSEM')
    parser.add_argument('--generator_lr', type=float, default=0.0002, help='learning rate of generator')
    parser.add_argument('--discriminator_lr', type=float, default=0.0002, help='learning rate of discriminator')
    parser.add_argument('--epochs', type=int, default=5000, help='number of epochs')
    parser.add_argument('--kl_div_coefficient', type=float, default=1.0, help='coefficient of the kl-divergence loss')
    parser.add_argument('--mm_reg_coeff', type=float, default=2.0, help='coefficient of the max-margin regularize')
    parser.add_argument('--z_dim', type=int, default=256, help='dimension of noise vector')
    parser.add_argument('--clip_val', type=float, default=0.01, help='clipping values of the discriminator in WGAN')
    parser.add_argument('--dataset', type=str, help='training dataset folder name')
    args = parser.parse_args()
    return args
