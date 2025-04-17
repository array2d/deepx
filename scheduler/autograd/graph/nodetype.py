from enum import IntEnum, EnumMeta
from typing import Dict, Any

 
class NodeType(IntEnum ):
    DATA = 0
    OP = 1
    CONTROL_FLOW = 2

