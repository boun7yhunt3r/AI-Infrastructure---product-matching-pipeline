import numpy as np
import tritonclient.http as httpclient
from typing import Tuple
from PIL import Image
import io

class CLIPModel:
    def __init__(self, triton_url: str):
        self.triton_client = httpclient.InferenceServerClient(url=triton_url)
    
    def preprocess_image(self, image_bytes: bytes) -> np.ndarray:
        """Convert image bytes to numpy array for CLIP input"""
        image = Image.open(io.BytesIO(image_bytes))

        image = image.resize((224, 224))
        image_array = np.array(image).astype(np.float32)
        image_array = image_array / 255.0
        image_array = (image_array - np.array([0.48145466, 0.4578275, 0.40821073])) / np.array([0.26862954, 0.26130258, 0.27577711])
        return image_array.transpose(2, 0, 1)  # CHW format
    
    def get_image_embedding(self, image_bytes: bytes) -> np.ndarray:
        """Get image embedding from Triton server"""
        processed_image = self.preprocess_image(image_bytes)
        

        if processed_image.dtype != np.float32:
            processed_image = processed_image.astype(np.float32)
        

        inputs = [httpclient.InferInput("INPUT__0", processed_image.shape, "FP32")]
        inputs[0].set_data_from_numpy(processed_image)
        

        outputs = [httpclient.InferRequestedOutput("OUTPUT__0")]
        response = self.triton_client.infer(model_name="clip_vision", inputs=inputs, outputs=outputs)
        

        embedding = response.as_numpy("OUTPUT__0")[0]
        return embedding.astype(np.float32)  # Force FP32 output
    
    def get_text_embedding(self, text: str) -> np.ndarray:
        """Get text embedding from Triton server"""

        text_array = np.array([text], dtype=object)
        
        inputs = [httpclient.InferInput("INPUT__0", text_array.shape, "BYTES")]
        inputs[0].set_data_from_numpy(text_array)
        
        outputs = [httpclient.InferRequestedOutput("OUTPUT__0")]
        response = self.triton_client.infer(model_name="clip_text", inputs=inputs, outputs=outputs)
        
        return response.as_numpy("OUTPUT__0")[0]