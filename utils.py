import os
import numpy
from PIL import Image, ImageFilter
import matplotlib.cm as mpl_color_map
import torch.nn as nn
import torch.nn.functional as F
import torch
import copy

def preprocess_image(pil_im, resize_im=True):
    """
        Processes image for CNNs
    Args:
        PIL_img (PIL_img): PIL Image or numpy array to process
        resize_im (bool): Resize to 224 or not
    returns:
        im_as_var (torch variable): Variable that contains processed float tensor
    """
    # mean and std list for channels (Imagenet)
    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]

    #ensure or transform incoming image to PIL image
    if type(pil_im) != Image.Image:
        try:
            pil_im = Image.fromarray(pil_im)
        except Exception as e:
            print("could not transform PIL_img to a PIL Image object. Please check input.")

    # Resize image
    if resize_im:
        pil_im = pil_im.resize((224, 224), Image.ANTIALIAS)

    im_as_arr = numpy.float32(pil_im)
    im_as_arr = im_as_arr.transpose(2, 0, 1)  # Convert array to D,W,H
    # Normalize the channels
    for channel, _ in enumerate(im_as_arr):
        im_as_arr[channel] /= 255
        im_as_arr[channel] -= mean[channel]
        im_as_arr[channel] /= std[channel]
    # Convert to float tensor
    im_as_ten = torch.from_numpy(im_as_arr).float()
    # Add one more channel to the beginning. Tensor shape = 1,3,224,224
    im_as_ten.unsqueeze_(0)

    return im_as_ten

def save_class_activation_images(org_img, activation_map, file_name):
    """
        Saves cam activation map and activation map on the original image
    Args:
        org_img (PIL img): Original image
        activation_map (numpy arr): Activation map (grayscale) 0-255
        file_name (str): File name of the exported image
    """
    if not os.path.exists('results'):
        os.makedirs('results')


    # Grayscale activation map
    heatmap, heatmap_on_image = apply_colormap_on_image(org_img, activation_map, 'hsv')
    # Save colored heatmap
    path_to_file = os.path.join('results', file_name+'_Heatmap.png')
    save_image(heatmap, path_to_file)
    # Save heatmap on iamge
    path_to_file = os.path.join('results', file_name+'_On_Image.png')
    save_image(heatmap_on_image, path_to_file)
    # SAve grayscale heatmap
    path_to_file = os.path.join('results', file_name+'_Grayscale.png')
    save_image(activation_map, path_to_file)

def apply_colormap_on_image(org_im, activation, colormap_name):
    """
        Apply heatmap on image
    Args:
        org_img (PIL img): Original image
        activation_map (numpy arr): Activation map (grayscale) 0-255
        colormap_name (str): Name of the colormap
    """
    if numpy.max(activation) <= 1:
        activation = (activation * 255).astype(numpy.uint8)

    # Get colormap
    color_map = mpl_color_map.get_cmap(colormap_name)
    no_trans_heatmap = color_map(activation)
    # Change alpha channel in colormap to make sure original image is displayed
    heatmap = copy.copy(no_trans_heatmap)
    heatmap[:, :, 3] = 0.4
    heatmap = Image.fromarray((heatmap*255).astype(numpy.uint8))
    no_trans_heatmap = Image.fromarray((no_trans_heatmap*255).astype(numpy.uint8))

    # Apply heatmap on iamge
    heatmap_on_image = Image.new("RGBA", org_im.size)
    heatmap_on_image = Image.alpha_composite(heatmap_on_image, org_im.convert('RGBA'))
    heatmap_on_image = Image.alpha_composite(heatmap_on_image, heatmap)
    return no_trans_heatmap, heatmap_on_image

def save_image(image, path):
    """
        Saves a numpy matrix or PIL image as an image
    Args:
        im_as_arr (Numpy array): Matrix of shape DxWxH
        path (str): Path to the image
    """
    if isinstance(image, (numpy.ndarray, numpy.generic)):
        image = format_numpy_output(image)
        image = image.astype(numpy.uint8)
        image = Image.fromarray(image)
    image.save(path)

def format_numpy_output(image):
    # Phase/Case 1: The np arr only has 2 dimensions
    # Result: Add a dimension at the beginning
    if len(image.shape) == 2:
        image = numpy.expand_dims(image, axis=0)
    # Phase/Case 2: Np arr has only 1 channel (assuming first dim is channel)
    # Result: Repeat first channel and convert 1xWxH to 3xWxH
    if image.shape[0] == 1:
        image = numpy.repeat(image, 3, axis=0)
    # Phase/Case 3: Np arr is of shape 3xWxH
    # Result: Convert it to WxHx3 in order to make it saveable by PIL
    if image.shape[0] == 3:
        image = image.transpose(1, 2, 0)
    # Phase/Case 4: NP arr is normalized between 0-1
    # Result: Multiply with 255 and change type to make it saveable by PIL

    image = image / numpy.max(image)
    if numpy.max(image) <= 1:
        image = (image*255).astype(numpy.uint8)

    return image

# the format of input data is AAAAABBBBBCCCC, the format of output data should be ABCABCABCABCABC
def parse_format(data):
    new_data = torch.zeros((data.shape[0] * data.shape[1], data.shape[2], data.shape[3], data.shape[4]))
    count = 0
    for i in range(data.shape[1]):
        for j in range(data.shape[0]):
            new_data[count, ...] = data[j, i, ...]
            count = count + 1

    return new_data

class Adapter(nn.Module):
    def __init__(self, input_channels):
        super(Adapter, self).__init__()
        self.conv = nn.Sequential(
                nn.Conv2d(input_channels, 64, kernel_size = 3, padding = 1),
                nn.BatchNorm2d(64, momentum=1, affine=True),
                nn.ReLU(),
                nn.Conv2d(64, 32, kernel_size = 3, padding = 1),
                nn.BatchNorm2d(32, momentum=1, affine=True),
                )
        self.downsample = nn.Sequential(
                nn.Conv2d(input_channels, 32, kernel_size = 3, padding = 1),
                nn.BatchNorm2d(32))
        self.relu = nn.ReLU()

    def forward(self, x):
        out = self.conv(x)
        out = out + self.downsample(x)

        return self.relu(out)

class Attention(nn.Module):
    def __init__(self, temperature = 1, use_adapter = False, use_softmax = False):
        super(Attention, self).__init__()
        self.use_adapter = use_adapter
        self.softmax = nn.Softmax(dim = 1)
        self.log_softmax = nn.LogSoftmax(dim = 1)
        self.temperature = temperature
        self.use_softmax = use_softmax

        self.adapter = Adapter(640)
        self.conv1 = nn.Conv2d(2, 1, 3, padding=1, bias=False)
                
    def _make_attention(self, x, adapter_num = None, out_shape = None):
        #out = x.mean(1)
        #out = x.abs().mean(1)
        if out_shape is not None:
            x = F.interpolate(x, out_shape)
        out = x.pow(2).mean(1, keepdim = True)

        #avg_out = torch.mean(x, dim=1, keepdim=True)
        #max_out, _ = torch.max(x, dim=1, keepdim=True)
        #x = torch.cat([avg_out, max_out], dim=1)
        #out = self.conv1(x)

        shape = out.squeeze(1).shape

        out = out.view(shape[0], -1)
        out = out / self.temperature

        #if self.use_softmax:
        #    out = self.softmax(out)

        out = F.normalize(out)
        out = out.view(shape[0], 1, shape[1], shape[2])
        #out = torch.sigmoid(out)

        return out

    def forward_attention(self, model, x, use_reshape = True, output_size = None, if_student = False):
        #x4, x3, x2, x1 = model(x)
        x4 = model(x)

        if output_size is not None:
            x = F.interpolate(x, output_size)

        if self.use_adapter:
            x4 = self.adapter(x4)
        #if not if_student:
            #at4 = self._make_attention(at4, out_shape = (5, 5))
            #at3 = self._make_attention(at3, out_shape = (10, 10))
            #at2 = self._make_attention(at2, out_shape = (21, 21))
            #at1 = self._make_attention(at1, out_shape = (42, 42))

        x4 = self._make_attention(x4, out_shape=(5, 5))
        #x3 = self._make_attention(x3)
        #x2 = self._make_attention(x2)
        #x1 = self._make_attention(x1)

        return x4
        #return x4, x3, x2, x1

    def forward(self, base_model, student_model, x):
        #base_out4, base_out3, base_out2, base_out1 = self.forward_attention(base_model, x, )
        base_out4 = self.forward_attention(base_model, x)
        self.use_adapter = False
        x = F.interpolate(x, (84, 84))
        #student_out4, student_out3, student_out2, student_out1 = self.forward_attention(student_model, x, if_student = True)
        student_out4 = self.forward_attention(student_model, x, if_student = True)

        return (base_out4, student_out4)
        #return (base_out4, student_out4), (base_out3, student_out3), (base_out2, student_out2), (base_out1, student_out1)
