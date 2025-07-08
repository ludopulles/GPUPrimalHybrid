#!/usr/bin/env python
# -*- coding: utf-8 -*-
####
#
#   Copyright (C) 2018-2021 Team G6K
#
#   This file is part of G6K. G6K is free software:
#   you can redistribute it and/or modify it under the terms of the
#   GNU General Public License as published by the Free Software Foundation,
#   either version 2 of the License, or (at your option) any later version.
#
#   G6K is distributed in the hope that it will be useful,
#   but WITHOUT ANY WARRANTY; without even the implied warranty of
#   MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
#   GNU General Public License for more details.
#
#   You should have received a copy of the GNU General Public License
#   along with G6K. If not, see <http://www.gnu.org/licenses/>.
#
####


"""
Full Sieve Command Line Client
"""

from __future__ import absolute_import
import pickle as pickler
import logging
from collections import OrderedDict

from g6k.algorithms.workout import workout, pump
from g6k.siever import Siever
from g6k.utils.cli import parse_args, run_all, pop_prefixed_params
from g6k.utils.stats import SieveTreeTracer
from g6k.utils.util import load_svpchallenge_and_randomize, db_stats
import six


def hkz_kernel(A,n):
    # Pool.map only supports a single parameter

    params = {"pump__down_sieve": True, "pump__down_stop": 9999, "saturation_ratio":.8, "pump__prefer_left_insert":10, "workout__dim4free_min":0,"workout__dim4free_dec":15}
    reserved_n = n
    params = params.new(reserved_n=reserved_n, otf_lift=False)
    pump_params = pop_prefixed_params("pump", params)
    workout_params = pop_prefixed_params("workout", params)
    challenge_seed = params.pop("challenge_seed")

    A, _ = load_svpchallenge_and_randomize(n, s=challenge_seed, seed=seed)

    g6k = Siever(A, params, seed=seed)
    tracer = SieveTreeTracer(g6k, root_label=("hkz", n), start_clocks=True)

    # runs a workout woth pump-down down until the end
    workout(g6k, tracer, 0, n, pump_params=pump_params, verbose=verbose, **workout_params)
    #Just making sure
    pump(g6k, tracer, 15, n-15, 0, **pump_params)
    g6k.lll(0, n)
    return tracer.exit()


def hkz():
    """
    Attempt HKZ reduction. 
    """
    description = hkz.__doc__

    args, all_params = parse_args(description,
                                  challenge_seed=0,
                                  pump__down_sieve=True,
                                  pump__down_stop=9999,
                                  saturation_ratio=.8,
                                  pump__prefer_left_insert=10,
                                  workout__dim4free_min=0, 
                                  workout__dim4free_dec=15
                                  )

    stats = run_all(hkz_kernel, list(all_params.values()),
                    lower_bound=args.lower_bound,
                    upper_bound=args.upper_bound,
                    step_size=args.step_size,
                    trials=args.trials,
                    workers=args.workers,
                    seed=args.seed
                    )

    inverse_all_params = OrderedDict([(v, k) for (k, v) in all_params.items()])

    for (n, params) in stats:
        stat = stats[(n, params)]
        if stat[0] is None:
            logging.info("Trace disabled")
            continue

        if len(stat) > 0:
            cputime = sum([float(node["cputime"]) for node in stat])/len(stat)
            walltime = sum([float(node["walltime"]) for node in stat])/len(stat)
            flast = sum([float(node["flast"]) for node in stat])/len(stat)
            avr_db, max_db = db_stats(stat)
            fmt = "%48s :: m: %1d, n: %2d, cputime :%7.4fs, walltime :%7.4fs, flast : %2.2f, avr_max db: 2^%2.2f, max_max db: 2^%2.2f" # noqa
            logging.info(fmt % (inverse_all_params[params], params.threads, n, cputime, walltime, flast, avr_db, max_db))
        else:
            logging.info("Trace disabled")

    if args.pickle:
        pickler.dump(stats, open("hkz-asvp-%d-%d-%d-%d.sobj" %
                                 (args.lower_bound, args.upper_bound, args.step_size, args.trials), "wb"))



if __name__ == '__main__':
    hkz()