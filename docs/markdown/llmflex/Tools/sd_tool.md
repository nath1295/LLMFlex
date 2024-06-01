Module llmflex.Tools.sd_tool
============================

Classes
-------

`StableDiffusionTool(base_url: str)`
:   Generating images using a Stable Diffusion model.
        
    
    Initialising the tool.
    
    Args:
        base_url (str): Base url to the Automatic1111 API.

    ### Ancestors (in MRO)

    * llmflex.Tools.tool_utils.BaseTool

    ### Methods

    `text2img(self, prompt: str, negative_prompt: Optional[str] = None, num_imgs: int = 1, cfg_scale: float = 10.0, img_orientation: Literal['square', 'portrait', 'horizontal'] = 'square', steps: int = 20) ‑> List[str]`
    :   Generating images from text.
        
        Args:
            prompt (str): Text prompt to the stable diffusion model. It should contain the detailed description of the desired image.
            negative_prompt (Optional[str], optional): Description for objects that are not suppose to be in the image, for example, if trees are not supposed to be in the image, the negative prompt should be "trees". Defaults to None.
            num_imgs (int, optional): Number of images to be generated. Defaults to 1.
            cfg_scale (float, optional): A scale for how close the image generation should stick with the prompt, the higher the scale, the more likely the description will be accurate. Defaults to 10.0.
            img_orientation (Literal[&#39;square&#39;, &#39;portrait&#39;, &#39;horizontal&#39;], optional): The shape of the image. Defaults to 'square'.
            steps (int, optional): Number of steps the stable diffusion model will go through to reach the final image. Defaults to 20.
        
        Returns:
            List[str]: List of file paths of the generated images.