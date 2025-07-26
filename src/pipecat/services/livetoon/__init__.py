#
# Copyright (c) 2024–2025, Livetoon Corporation
#
# SPDX-License-Identifier: MIT
#

import sys

from pipecat.services import DeprecatedModuleProxy

from .tts import *

sys.modules[__name__] = DeprecatedModuleProxy(globals(), "livetoon", "livetoon.tts")
