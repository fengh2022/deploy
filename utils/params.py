import argparse


def init_param():
    parser = argparse.ArgumentParser('CatDogClassification')
    parser.add_argument('--debug', action='store_true')
    parser.add_argument('--model_name', type=str, default='VGG')

    parser.add_argument('--device', type=str, default='cuda:0')
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--epochs', type=int, default=20)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--img_h', type=int, default=256)
    parser.add_argument('--img_w', type=int, default=256)


    parser.add_argument('--root_dir', type=str, default='../kagglecatsanddogs_5340/PetImages/')

    parser.add_argument('--num_workers', type=int, default=0)
    parser.add_argument('--verbose_step', type=int, default=100)
    parser.add_argument('--export_dir', type=str, default='./ckpts/')

    parser.add_argument('--calibration_dir', type=str, default='../kagglecatsanddogs_5340/calibration_data/')
    parser.add_argument('--quantify_mode', type=str, default='int8', help='none/fp16/int8')


    return parser.parse_args()


arg = init_param()
