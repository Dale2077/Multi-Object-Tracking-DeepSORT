# Mikel Broström 🔥 BoxMOT 🧾 AGPL-3.0 license

__version__ = '16.0.6'

from boxmot.reid.core import ReID

from boxmot.trackers.deepsort.deepsort import DeepSort
from boxmot.trackers.tracker_zoo import create_tracker, get_tracker_config

TRACKERS = [
    "deepsort",
]

__all__ = (
    "__version__",
    "DeepSort",
    "create_tracker",
    "get_tracker_config",
)
