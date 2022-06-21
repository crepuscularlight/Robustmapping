from .semantickitti import SemanticKitti

def get_dataset():
    return {"SemanticKitti":SemanticKitti}