from pydantic import BaseModel, Field

class ImageRequest(BaseModel):
    image_base64: str = Field(
        ..., 
        title="Base64 Image String", 
        description="Image encoded in Base64 format",
        min_length=1,  # Asegura que no esté vacía
        max_length=5_000_000  # Limita el tamaño máximo de la cadena (en este caso ~5MB)
    )
