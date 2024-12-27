import os
import matplotlib.pyplot as plt
import torch
from src.colorization.Zhang import eccv16, siggraph17, load_img, preprocess_img, postprocess_tens

def create_colorized_pictures(model='eccv16', img_path='../../../resources/hand-pictures/try/IMG20241123170713.jpg',
                              use_gpu=False, save_prefix='saved'):

    if model == 'eccv16':
        # Initialize the ECCV16 model
        colorizer_eccv16 = eccv16(pretrained=True).eval()
        if use_gpu:
            colorizer_eccv16.cuda()

        # Load and preprocess the image
        img = load_img(img_path)
        (tens_l_orig, tens_l_rs) = preprocess_img(img, HW=(256, 256))
        if use_gpu:
            tens_l_rs = tens_l_rs.cuda()

        # Colorize using ECCV16 model
        img_bw = postprocess_tens(tens_l_orig, torch.cat((0 * tens_l_orig, 0 * tens_l_orig), dim=1))
        out_img_eccv16 = postprocess_tens(tens_l_orig, colorizer_eccv16(tens_l_rs).cpu())


        output_path = f'../../resources/stylized-pictures/eccv16/{save_prefix}_eccv16.png'
        if not os.path.exists(os.path.dirname(output_path)):
            os.makedirs(os.path.dirname(output_path))
        plt.imsave(output_path, out_img_eccv16)
        # print(f"Saved colorized image at {output_path}")

    else:
        # Initialize the Siggraph17 model
        colorizer_siggraph17 = siggraph17(pretrained=True).eval()
        if use_gpu:
            colorizer_siggraph17.cuda()

        # Load and preprocess the image
        img = load_img(img_path)
        (tens_l_orig, tens_l_rs) = preprocess_img(img, HW=(256, 256))
        if use_gpu:
            tens_l_rs = tens_l_rs.cuda()

        # Colorize using Siggraph17 model
        img_bw = postprocess_tens(tens_l_orig, torch.cat((0 * tens_l_orig, 0 * tens_l_orig), dim=1))
        out_img_siggraph17 = postprocess_tens(tens_l_orig, colorizer_siggraph17(tens_l_rs).cpu())

        # # Construct the output path (absolute path)
        output_path = f'../../resources/stylized-pictures/siggraph17/{save_prefix}_siggraph17.png'
        if not os.path.exists(os.path.dirname(output_path)):
            os.makedirs(os.path.dirname(output_path))
        plt.imsave(output_path, out_img_siggraph17)
        # print(f"Saved colorized image at {output_path}")


