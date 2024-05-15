# Copyright (c) 2023-2024, RTE (https://www.rte-france.com)
# See AUTHORS.txt
# SPDX-License-Identifier: Apache-2.0 (see LICENSE)
# This file is part of SL Alert Agent, an agent module that learns to send alerts through supervised learning when the agent is prone to failure.

import subprocess
import numpy as np

def main():
    #n_processes = 2
    #ss = np.random.SeedSequence(1234)
    #seeds = ss.spawn(n_processes) # 
    seeds = [i for i in range(10, 20)]
    for s in seeds:
        # start subprocess
        subprocess.Popen(["python", "data_collector.py", f"{s}"])

if __name__ == "__main__":
    main()
