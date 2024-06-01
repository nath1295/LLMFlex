from .tool_utils import BaseTool
from typing import Any, List, Optional, Literal, Dict


class StableDiffusionTool(BaseTool):
    """Generating images using a Stable Diffusion model.
    """
    def __init__(self, base_url: str) -> None:
        """Initialising the tool.

        Args:
            base_url (str): Base url to the Automatic1111 API.
        """
        import os
        from ..utils import get_config
        self.base_url = base_url
        self.img_dir = os.path.join(get_config('package_home'), 'sd_tool_resource', 'imgs')
        os.makedirs(self.img_dir, exist_ok=True)
        super().__init__()

    def text2img(self, prompt: str, negative_prompt: Optional[str] = None, num_imgs: int = 1, 
                 cfg_scale: float = 10.0, img_orientation: Literal['square', 'portrait', 'horizontal'] = 'square', steps: int = 20) -> List[str]:
        """Generating images from text.

        Args:
            prompt (str): Text prompt to the stable diffusion model. It should contain the detailed description of the desired image.
            negative_prompt (Optional[str], optional): Description for objects that are not suppose to be in the image, for example, if trees are not supposed to be in the image, the negative prompt should be "trees". Defaults to None.
            num_imgs (int, optional): Number of images to be generated. Defaults to 1.
            cfg_scale (float, optional): A scale for how close the image generation should stick with the prompt, the higher the scale, the more likely the description will be accurate. Defaults to 10.0.
            img_orientation (Literal[&#39;square&#39;, &#39;portrait&#39;, &#39;horizontal&#39;], optional): The shape of the image. Defaults to 'square'.
            steps (int, optional): Number of steps the stable diffusion model will go through to reach the final image. Defaults to 20.

        Returns:
            List[str]: List of file paths of the generated images.
        """
        import requests
        import base64
        import os
        from ..utils import current_time
        width = 512 if img_orientation in ['square', 'protrait'] else 768
        height = 512 if img_orientation in ['square', 'horizontal'] else 768
        payload = dict(
            prompt=prompt,
            negative_prompt='' if negative_prompt is None else negative_prompt,
            batch_size=num_imgs,
            cfg_scale=cfg_scale,
            steps=steps,
            width=width,
            height=height
        )
        response = requests.post(url=f'{self.base_url}/sdapi/v1/txt2img', json=payload)
        r = response.json()

        timestamp = current_time()
        img_dirs = []
        for i, img in enumerate(r['images']):
            img_dir = os.path.join(self.img_dir, f'{timestamp}_{i}.png')
            img_dirs.append(img_dir)
            with open(img_dir, 'wb') as f:
                f.write(base64.b64decode(img))
        return img_dirs

    def __call__(self, prompt: str, negative_prompt: Optional[str] = None, num_imgs: int = 1, 
                 img_orientation: Literal['square', 'portrait', 'horizontal'] = 'square') -> Dict[str, List[str]]:
        """Generating images from text.

        Args:
            prompt (str): Text prompt to the stable diffusion model. It should contain the detailed description of the desired image.
            negative_prompt (Optional[str], optional): Description for objects that are not suppose to be in the image, for example, if trees are not supposed to be in the image, the negative prompt should be "trees". Defaults to None.
            num_imgs (int, optional): Number of images to be generated. Defaults to 1.
            img_orientation (Literal[&#39;square&#39;, &#39;portrait&#39;, &#39;horizontal&#39;], optional): The shape of the image. Defaults to 'square'.

        Returns:
            Dict[str, List[str]]: Image directories.
        """
        img_dirs = self.text2img(prompt=prompt, negative_prompt=negative_prompt, num_imgs=num_imgs, img_orientation=img_orientation)
        footnote = [f'![Image {i}]({img_dir})' for i, img_dir in enumerate(img_dirs)]
        return dict(images=img_dirs)

