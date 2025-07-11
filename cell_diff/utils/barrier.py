# -*- coding: utf-8 -*-
import os
import subprocess
import sys

# usage: touch READY && python barrier.py $NUM_NODES $RANK


def check(num_workers, rank):
    print(f"Sync, rank: {rank} is waiting")
    abs_path = os.path.abspath(os.path.dirname(__file__))
    while True:
        all_ready = True
        for other_rank in range(num_workers):
            if other_rank == rank:
                continue
            other_ready = bool(
                int(
                    subprocess.check_output(
                        f"ssh worker-{other_rank} 'ls {abs_path}/READY | wc -l'",
                        shell=True,
                    )
                    .decode()
                    .strip()
                )
            )
            if not other_ready:
                all_ready = False
                break
        if all_ready:
            break


if __name__ == "__main__":
    check(int(sys.argv[1]), int(sys.argv[2]))
