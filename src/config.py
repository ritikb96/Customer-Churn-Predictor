from pydantic import BaseModel

class ModelNameConfig(BaseModel):
    """
    Model Configuration
    """

    modelName: str = "random_forest"