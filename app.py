############################
#
#   Imports
#
############################
import timm 
import torch
from skimage import io 
from src.gradcams import GradCam
import numpy as np 
import cv2 
import gradio  as gr 
from PIL import Image



############################
#
#   model
#
############################
model:torch.nn.Module = timm.create_model("vit_small_patch16_224",pretrained=True) # num_classes=10
model.eval()

############################
#
#   utility functions
#
############################
def prepare_input(image:np.array)->torch.Tensor:
    image = image.copy()   # (H,W,C)
    mean  = np.array([0.5,.5,.5])
    stds = np.array([.5,.5,.5])
    image -= mean 
    image /= stds 

    image = np.ascontiguousarray(np.transpose(image,(2,0,1)))  # transpose the image to match model's input format (C,H,W)
    image = image[np.newaxis,...]  # (bs, C, H, W)
    return torch.tensor(image,requires_grad=True)


def gen_cam(image, mask):
    # create a heatmap from the Grad-CAM mask 
    heatmap = cv2.applyColorMap(np.uint8(255*mask), cv2.COLORMAP_JET)
    heatmap = np.float32(heatmap)/255.
    # superimpose the heatmap on the original image
    cam =  (.5*heatmap) + (.5*image.squeeze(0).permute(1,2,0).detach().cpu().numpy())
    # normalize
    cam = cam/ np.max(cam)
    return np.uint8(255*cam)



def attn_viz(image,number:int=2):
    image = np.float32(cv2.resize(image,(224,224) )) / 255
    image = prepare_input(image)

    target_layer = model.blocks[number]
    grad_cam = GradCam(model=model,target=target_layer)
    mask = grad_cam(image)
    result = gen_cam(image=image,mask=mask)
    return Image.fromarray(result)


# Create a Gradio TabbedInterface with two tabs
with gr.Blocks(
            title="AttnViz",
    ) as demo:
    with gr.Tab("Image Processing"):
        # Create an image input and a number input
        image_input = gr.Image(label="Input Image",type='numpy')
        number_input = gr.Number(label="Number",minimum=0,maximum=11,show_label=True)
        # Create an image output
        image_output = gr.Image(label="Output Image")
        # Set up the event listener for the image processing function
        process_button = gr.Button("Process Image")
        process_button.click(attn_viz, inputs=[image_input, number_input], outputs=image_output)
        
        gr.Examples(
            examples=[
                ["samples/mr_bean.png", 1],
                ["samples/sectional-sofa.png", 8],
            ],
            inputs=[image_input, number_input],
        )


    with gr.Tab("README"):
        # Add a simple text description in the About tab
        with open("README.md", "r+") as file: readme_content = file.read()
        gr.Markdown(readme_content[140:])



if __name__=='__main__':
    demo.launch(show_error=True,share=False,)
