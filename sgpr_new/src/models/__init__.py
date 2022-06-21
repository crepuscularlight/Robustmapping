from .SGPR_baseline import SGPR_Baseline
from .SGPR_geo_baseline import SGPR_Geo_Baseline
from .SGPR_geo_attention import SGPR_Geo_Attention
def get_model():
    return {"SGPR_Baseline":SGPR_Baseline,
            "SGPR_Geo_Baseline":SGPR_Geo_Baseline,
            "SGPR_Geo_Attention": SGPR_Geo_Attention
            }