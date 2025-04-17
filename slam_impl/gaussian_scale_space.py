import torch
import torch.nn.functional as F
import numpy as np

import pyscsp
# pyscsp : Scale-Space Toolbox for Python
# Tony Lindeberg: https://github.com/tonylindeberg/pyscsp
#
# To obain the same result as pytorch conv2d (with padding mode 'reflect')
# use padding mode 'mirror' in pyscsp.discscsp.discgaussconv, for method correlate1d


import scipy

from PIL import Image
import matplotlib.pyplot as plt

import time



def image_conv_gaussian (image : torch.Tensor, sigma=1, epsilon=0.01) -> torch.Tensor:
    """ Convolve an image (rgd, or gray) by Discrete Gassian kernel from Tony Lindeberg
    see. https://github.com/tonylindeberg/pyscsp    
    The implementation takes advantage of the fact that Gaussian kernel is separable.
    see. https://en.wikipedia.org/wiki/Separable_filter
    """
    ndim = image.ndim
    channels = 1
    if ndim >= 3:
        channels = image.shape[-3]

    # Gaussian kernel form pyscsp
    sep1Dfilter = pyscsp.discscsp.make1Ddiscgaussfilter(
        sigma=sigma,
        epsilon=epsilon,
        D=2
    )
    kernel_1d = torch.from_numpy(sep1Dfilter).type(image.dtype).to(image.device)
    padding = len(kernel_1d) // 2

    # make 4D input
    # input size (mini_batch,  in_channels,  iH,  iW)
    if ndim == 2:
        inputs = image.unsqueeze(0).unsqueeze(0)
    if ndim == 3:
        inputs = image.unsqueeze(0)
    if ndim == 4:
        inputs = image

    # pytorch padding_mode: 'zeros', 'reflect', 'replicate' or 'circular'. Default: 'zeros'
    # In pytorch, circular, replicate and reflection padding are implemented for the last 2 dimensions of a 3D or 4D input tensor
    # add padding. use 'reflect' in pytorch means 'mirror' in scipy/pyscsp
    p2d = (padding, padding, padding, padding)
    img = torch.nn.functional.pad(inputs, pad=p2d, mode='reflect')

    # Convolve along columns and rows. input 4D tensor
    # kernel size (out_channels,  in_channels/groups,  kH,  kW)
    # groups=channels, to convolve each color channel separately
    img = F.conv2d(img, weight=kernel_1d.view(1, 1, -1, 1).repeat(channels, 1, 1, 1), padding=0, bias=None, groups=channels)
    img = F.conv2d(img, weight=kernel_1d.view(1, 1, 1, -1).repeat(channels, 1, 1, 1), padding=0, bias=None, groups=channels)

    if ndim == 2:
        return img.squeeze_(0).squeeze_(0)
    if ndim == 3:
        return img.squeeze_(0)
    if ndim == 4:
        return img





def image_conv_gaussian_naive (image : torch.Tensor, sigma=1, epsilon=0.01) -> torch.Tensor:
    """ Convolve an image (rgd, or gray) by Discrete Gassian kernel from Tony Lindeberg
    see. https://github.com/tonylindeberg/pyscsp    
    This implementation is less efficient than exploiting the separable kernel.
    """
    channels = 1
    ndim = image.ndim
    if ndim >= 3:
        channels = image.shape[-3]

    sep1Dfilter = pyscsp.discscsp.make1Ddiscgaussfilter(
        sigma=sigma,
        epsilon=epsilon,
        D=2
    )
    sep2Dfilter = np.outer(sep1Dfilter, sep1Dfilter)

    kernel_2d = torch.from_numpy(sep2Dfilter).type(image.dtype).to(image.device)
    padding = len(kernel_2d) // 2

    # kernel size (out_channels,  in_channels/groups,  kH,  kW)
    kernel_2d = kernel_2d.view(1, 1, kernel_2d.shape[0], kernel_2d.shape[1])
    kernel_2d = kernel_2d.repeat(channels, 1, 1, 1)

    # input size (mini_batch,  in_channels,  iH,  iW)
    if ndim == 2:
        inputs = image.unsqueeze(0).unsqueeze(0)
    if ndim == 3:
        inputs = image.unsqueeze(0)
    if ndim == 4:
        inputs = image

    # pytorch padding_mode: 'zeros', 'reflect', 'replicate' or 'circular'. Default: 'zeros'
    # In pytorch, circular, replicate and reflection padding are implemented for the last 2 dimensions of a 3D or 4D input tensor
    # add padding. use 'reflect' in pytorch means 'mirror' in scipy/pyscsp
    p2d = (padding, padding, padding, padding)
    inputs = torch.nn.functional.pad(inputs, pad=p2d, mode='reflect', value=None)

    # groups=channels, to convolve each color channel separately
    img = F.conv2d(inputs, kernel_2d, padding=0, bias=None, groups=channels)

    if ndim == 2:
        return img.squeeze_(0).squeeze_(0)
    if ndim == 3:
        return img.squeeze_(0)
    if ndim == 4:
        return img





if __name__ == "__main__":



    image_name = "/hdd/tandt/truck/images/000007.jpg"


    img = Image.open(image_name)
    img_np = np.array(img)


    print("\n## image information ##")

    print(f"img.size = {img.size}")
    print(f"img_np.shape = {img_np.shape}")

    coldala = plt.imread(image_name)
    image = torch.from_numpy(coldala.transpose(2, 0, 1)).type(torch.float64)

    print(f"numpy image shape: {coldala.shape}")
    print(f"tensor image shape: {image.shape}")


    print("\n## pyscsp.discscsp.make1Ddiscgaussfilter ##")

    for D in np.arange(1, 7):
        sep1Dfilter = pyscsp.discscsp.make1Ddiscgaussfilter(
            sigma=4,
            epsilon=0.01,
            D=D
        )
        print(f"sep1Dfilter (D = {D}) = \n {sep1Dfilter}")



    print("\n## pyscsp.discscsp.discgaussconv ##")

    smoothpic = pyscsp.discscsp.discgaussconv(
        inpic = coldala,
        sigma = 4,
        epsilon = 0.01
    )
    smoothpic_c1 = pyscsp.discscsp.discgaussconv(
        inpic = coldala[:,:,0],
        sigma = 4,
        epsilon = 0.01
    )
    smoothpic_c2 = pyscsp.discscsp.discgaussconv(
        inpic = coldala[:,:,1],
        sigma = 4,
        epsilon = 0.01
    )
    smoothpic_c3 = pyscsp.discscsp.discgaussconv(
        inpic = coldala[:,:,2],
        sigma = 4,
        epsilon = 0.01
    )
    print(f"test pyscsp.discscsp.discgaussconv. error in channel 1: {np.linalg.norm(smoothpic[:,:,0] - smoothpic_c1)}")
    print(f"test pyscsp.discscsp.discgaussconv. error in channel 2: {np.linalg.norm(smoothpic[:,:,1] - smoothpic_c2)}")
    print(f"test pyscsp.discscsp.discgaussconv. error in channel 3: {np.linalg.norm(smoothpic[:,:,2] - smoothpic_c3)}")




    print("\n## 2D conv and 1D conv by separability ##")


    conv_img = image_conv_gaussian_naive(image, sigma=4, epsilon = 0.01)
    conv_img_sep = image_conv_gaussian(image, sigma=4, epsilon = 0.01)

    print(f"test 1D (of separable kernel) and 2D convolution difference.\n\terror in total: {torch.norm(conv_img - conv_img_sep)}")
    print(f"\terror in channel 1: {torch.norm(conv_img[0] - conv_img_sep[0])}")
    print(f"\terror in channel 2: {torch.norm(conv_img[1] - conv_img_sep[1])}")
    print(f"\terror in channel 3: {torch.norm(conv_img[2] - conv_img_sep[2])}")



    conv_img1 = image_conv_gaussian_naive(image[0], sigma=4, epsilon = 0.01)
    conv_img2 = image_conv_gaussian_naive(image[1], sigma=4, epsilon = 0.01)
    conv_img3 = image_conv_gaussian_naive(image[2], sigma=4, epsilon = 0.01)

    # print(f"test 2D convolution. error in channel 1: {torch.norm(conv_img[0] - conv_img1)}")
    # print(f"test 2D convolution. error in channel 2: {torch.norm(conv_img[1] - conv_img2)}")
    # print(f"test 2D convolution. error in channel 3: {torch.norm(conv_img[2] - conv_img3)}")


    conv_img_c1 = image_conv_gaussian(image[0], sigma=4, epsilon = 0.01)
    conv_img_c2 = image_conv_gaussian(image[1], sigma=4, epsilon = 0.01)
    conv_img_c3 = image_conv_gaussian(image[2], sigma=4, epsilon = 0.01)


    print(f"test 1D (of separable kernel) and 2D convolution difference. error in channel 1: {torch.norm(conv_img_c1 - conv_img1)}")
    print(f"test 1D (of separable kernel) and 2D convolution difference. error in channel 2: {torch.norm(conv_img_c2 - conv_img2)}")
    print(f"test 1D (of separable kernel) and 2D convolution difference. error in channel 3: {torch.norm(conv_img_c3 - conv_img3)}")




    print("\n## pyscsp and 1D (of separable kernel) ##")



    smoothpic_tensor = torch.from_numpy(smoothpic.transpose(2, 0, 1)).type(torch.float64)

    print(f"test 1D (of separable kernel) convolution and pyscsp.discscsp.discgaussconv difference.\n\terror in total: {torch.norm(smoothpic_tensor - conv_img_sep)}")
    print(f"\terror in channel 1: {torch.norm(smoothpic_tensor[0] - conv_img_sep[0])}")
    print(f"\terror in channel 2: {torch.norm(smoothpic_tensor[1] - conv_img_sep[1])}")
    print(f"\terror in channel 3: {torch.norm(smoothpic_tensor[2] - conv_img_sep[2])}")
    


    smoothpic_c1_tensor = torch.from_numpy(smoothpic_c1).type(torch.float64)
    smoothpic_c2_tensor = torch.from_numpy(smoothpic_c2).type(torch.float64)
    smoothpic_c3_tensor = torch.from_numpy(smoothpic_c3).type(torch.float64)

    print(f"torch conv 1D (of separable kernel) and pyscsp.discscsp.discgaussconv. error in channel 1: {torch.norm(smoothpic_c1_tensor - conv_img_c1)}")
    print(f"torch conv 1D (of separable kernel) and pyscsp.discscsp.discgaussconv. error in channel 2: {torch.norm(smoothpic_c2_tensor - conv_img_c2)}")
    print(f"torch conv 1D (of separable kernel) and pyscsp.discscsp.discgaussconv. error in channel 3: {torch.norm(smoothpic_c3_tensor - conv_img_c3)}")


    print(f"torch conv 2D and pyscsp.discscsp.discgaussconv. error in channel 1: {torch.norm(smoothpic_c1_tensor - conv_img1)}")
    print(f"torch conv 2D and pyscsp.discscsp.discgaussconv. error in channel 2: {torch.norm(smoothpic_c2_tensor - conv_img2)}")
    print(f"torch conv 2D and pyscsp.discscsp.discgaussconv. error in channel 3: {torch.norm(smoothpic_c3_tensor - conv_img3)}")


    diff1 = torch.abs(smoothpic_c1_tensor - conv_img1).cpu().numpy()
    diff2 = torch.abs(smoothpic_c2_tensor - conv_img2).cpu().numpy()
    diff3 = torch.abs(smoothpic_c3_tensor - conv_img3).cpu().numpy()

    plt.imshow(diff1, cmap="gray")
    plt.colorbar()
    plt.show()
    plt.imshow(diff2, cmap="gray")
    plt.colorbar()
    plt.show()
    plt.imshow(diff3, cmap="gray")
    plt.colorbar()
    plt.show()




    # simple 2D data

    print("\n## simple 2D data ##")


    data_buffer = np.linspace(1, 63, num=63, dtype=np.float64)
    data_2d = data_buffer.reshape((9, 7))
    data_2d_tensor = torch.from_numpy(data_2d).type(torch.float64)

    pytorch_conv_2 = image_conv_gaussian_naive(data_2d_tensor, sigma=1, epsilon = 0.1)
    pytorch_conv_11 = image_conv_gaussian(data_2d_tensor, sigma=1, epsilon = 0.1)


    pyscsp_conv = pyscsp.discscsp.discgaussconv(
        inpic = data_2d,
        sigma = 1,
        epsilon = 0.1
    )
    pyscsp_conv_tensor = torch.from_numpy(pyscsp_conv).type(torch.float64)


    print(f"data_2d_tensor = \n {data_2d_tensor}")
    print(f"pytorch conv_2 = \n {pytorch_conv_2}")
    print(f"pytorch conv_11 = \n {pytorch_conv_11}")
    print(f"pyscsp conv = \n {pyscsp_conv_tensor}")
    print(f"difference = \n\t conv 1D (of separable kernel) - pyscsp_conv: {np.linalg.norm(pyscsp_conv_tensor - pytorch_conv_11)} \n\t\t {np.abs(pyscsp_conv_tensor - pytorch_conv_11)}")    







    # simple 1D data


    print("\n## simple 1D data ##")


    # for test
    def conv_gaussian_1d (input : torch.Tensor, sigma=1, epsilon=0.1) -> torch.Tensor:
        # this method returns a np.array
        sep1Dfilter = pyscsp.discscsp.make1Ddiscgaussfilter(
            sigma=sigma,
            epsilon=epsilon,
            D=1
        )
        kernel_1d = torch.from_numpy(sep1Dfilter).type(input.dtype).to(input.device)
        padding = len(kernel_1d) // 2

        # make 3D
        img = input.unsqueeze(0).unsqueeze_(0)

        p1d = (padding, padding)
        img= torch.nn.functional.pad(img, pad=p1d, mode='reflect', value=0.0)
        print(f"torch.nn.functional.pad: \n{img}")

        img = F.conv1d(img, weight=kernel_1d.view(1, 1, -1), padding=0)
        print(f"torch.nn.functional.cov1d: \n{img}")

        return img.squeeze_(0).squeeze_(0)



    data_buffer = np.linspace(1, 10, 10)
    data_buffer_torch = torch.from_numpy(data_buffer).type(torch.float64)


    conv1d = conv_gaussian_1d (data_buffer_torch, sigma=1, epsilon=0.1)
    print(f"pytorch_conv1d [reflect padding] = \n{conv1d.cpu().numpy()}")


    sep1Dfilter = pyscsp.discscsp.make1Ddiscgaussfilter(
        sigma=1,
        epsilon=0.1,
        D=1
    )
    scipy_conv1d = scipy.ndimage.correlate1d(data_buffer, weights=sep1Dfilter, axis=0, output=None, mode='mirror', cval=0.0, origin = 0)
    print(f"scipy_conv1d [mirror padding] = \n{scipy_conv1d}")


    print(f"\tscipy_conv1d [mirror padding] - pytorch_conv1d [reflect padding]:  {np.linalg.norm(conv1d.cpu().numpy() - scipy_conv1d)}")


    pyscsp_conv1d = pyscsp.discscsp.discgaussconv(
        inpic = data_buffer,
        sigma = 1,
        epsilon = 0.1
    )
    print(f"pyscsp_conv1d [reflect padding]= \n{pyscsp_conv1d}")

    print(f"\tpyscsp_conv1d [mirror padding] - pytorch_conv1d [reflect padding]:  {np.linalg.norm(conv1d.cpu().numpy() - pyscsp_conv1d)}")


    print(f"\nConclusion: Scipy mirror paddings == pyTorch reflect paddings.")







    print("\n## minibatch input 4D tensor ##")

    image_name1 = "/hdd/tandt/truck/images/000002.jpg"
    image_name2 = "/hdd/tandt/truck/images/000012.jpg"
    image_name3 = "/hdd/tandt/truck/images/000022.jpg"
    image_name4 = "/hdd/tandt/truck/images/000032.jpg"
    image1 = torch.from_numpy(plt.imread(image_name1).transpose(2, 0, 1)).type(torch.float64)
    image2 = torch.from_numpy(plt.imread(image_name2).transpose(2, 0, 1)).type(torch.float64)
    image3 = torch.from_numpy(plt.imread(image_name3).transpose(2, 0, 1)).type(torch.float64)
    image4 = torch.from_numpy(plt.imread(image_name4).transpose(2, 0, 1)).type(torch.float64)

    images = torch.cat(( image1.unsqueeze(0), image2.unsqueeze(0), image3.unsqueeze(0), image4.unsqueeze(0)), 0)

    print(f"Batch processing: \n\timages.shape = {images.shape}")




    t0 = time.time()

    conv_imgs = image_conv_gaussian_naive(images.to("cuda"), sigma=4, epsilon = 0.01)
    torch.cuda.current_stream().synchronize()

    t1 = time.time()
    
    conv_imgs_sep = image_conv_gaussian(images.to("cuda"), sigma=4, epsilon = 0.01)
    torch.cuda.current_stream().synchronize()

    t2 = time.time()

    print(f"\ttime on Cuda: image_conv_gaussian_naive:           = {t1 - t0}")
    print(f"\ttime on Cuda: image_conv_gaussian:                 = {t2 - t1}   which is {(t1-t0)/(t2-t1)} times faster")



    t0 = time.time()

    conv_imgs = image_conv_gaussian_naive(images.cpu(), sigma=4, epsilon = 0.01)
    torch.cuda.current_stream().synchronize()

    t1 = time.time()
    
    conv_imgs_sep = image_conv_gaussian(images.cpu(), sigma=4, epsilon = 0.01)
    torch.cuda.current_stream().synchronize()

    t2 = time.time()

    print(f"\ttime on CPU:  image_conv_gaussian_naive:           = {t1 - t0}")
    print(f"\ttime on CPU:  image_conv_gaussian:                 = {t2 - t1}   which is {(t1-t0)/(t2-t1)} times faster")




    print(f"difference after convolution: \n\tconv_imgs - conv_imgs_sep = {torch.norm(conv_imgs - conv_imgs_sep)}")

    conv_img1 = image_conv_gaussian_naive(images[0], sigma=4, epsilon = 0.01)
    conv_img2 = image_conv_gaussian_naive(images[1], sigma=4, epsilon = 0.01)
    conv_img3 = image_conv_gaussian_naive(images[2], sigma=4, epsilon = 0.01)
    conv_img4 = image_conv_gaussian_naive(images[3], sigma=4, epsilon = 0.01)

    print(f"\terror in image1:  {torch.norm(conv_imgs_sep[0] - conv_img1)}")
    print(f"\terror in image2:  {torch.norm(conv_imgs_sep[1] - conv_img2)}")
    print(f"\terror in image3:  {torch.norm(conv_imgs_sep[2] - conv_img3)}")
    print(f"\terror in image4:  {torch.norm(conv_imgs_sep[3] - conv_img4)}")   




