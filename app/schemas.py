from pydantic import BaseModel, Field

class NewsItem(BaseModel):
    content: str = Field(
        ...,
        min_length=20,
        max_length=2000,
        description="Full news text or headline + summary."
    )
