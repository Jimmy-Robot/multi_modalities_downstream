from pydantic import BaseModel, Field

class RetrievalConfig(BaseModel):

    clip_type: str = "ViCLIP"
    """
    Indicates the clip type, available types=["ViCLIP"]
    """

    tokenizer_path: str = None
    """
    Indicates the tokenizer path. It can be a directory path or file path,
    depends on the specific clip usage.
    """

    clip_model_config_path: str = None
    """
    Indicates the clip model config path, it depends on the specific clip usage.
    """

    searcher_type: str = "ANN_BF"
    """
    Indicates the searcher type, available types=["ANN_BF"]
    """

    resume_path: str = None
    
    video_dir: str = None

    retrieval_text_file: str = None

    retrieval_result_file_path: str = None

    serialize_path: str = None