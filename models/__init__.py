from models.fusion_models.pvt_model import DepthFusionPVT
from models.fusion_models.swin_model import DepthFusionSwin
from models.icra import FusionModel

__models__ = {
    "depth_fusion_pvt": DepthFusionPVT,
    "depth_fusion_swin": DepthFusionSwin,
    "depth_fusion_icra": FusionModel
}
