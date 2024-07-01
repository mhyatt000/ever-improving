
SIMPLER_PERF = {    # SIMPLER simulated eval performance --> extract via: SIMPLER_PERF[task][policy]
    "google_robot_pick_coke_can": {
        "rt-2-x": 0.787,
        "rt-1-converged": 0.857,
        "rt-1-15pct": 0.710,
        "rt-1-x": 0.567,
        "rt-1-begin": 0.027,
        "octo-base": 0.170,
    },
    "google_robot_move_near": {
        "rt-2-x": 0.779,
        "rt-1-converged": 0.442,
        "rt-1-15pct": 0.354,
        "rt-1-x": 0.317,
        "rt-1-begin": 0.050,
        "octo-base": 0.042,
    },
    "google_robot_open_drawer": {
        "rt-2-x": 0.157,
        "rt-1-converged": 0.601,
        "rt-1-15pct": 0.463,
        "rt-1-x": 0.296,
        "rt-1-begin": 0.000,
        "octo-base": 0.009,
    },
    "google_robot_close_drawer": {
        "rt-2-x": 0.343,
        "rt-1-converged": 0.861,
        "rt-1-15pct": 0.667,
        "rt-1-x": 0.891,
        "rt-1-begin": 0.278,
        "octo-base": 0.444,
    },
    "google_robot_place_apple_in_closed_top_drawer": {
        "rt-2-x": 0.037,
        "rt-1-converged": 0.065,
        "rt-1-15pct": 0.130,
        "rt-1-x": 0.213,
        "rt-1-begin": 0.000,
        "octo-base": 0.000,
    },
    "widowx_spoon_on_towel": {
        "rt-1-x": 0.000,
        "octo-base": 0.125,
        "octo-small": 0.472,
    },
    "widowx_carrot_on_plate": {
        "rt-1-x": 0.042,
        "octo-base": 0.083,
        "octo-small": 0.097,
    },
    "widowx_stack_cube": {
        "rt-1-x": 0.000,
        "octo-base": 0.000,
        "octo-small": 0.042,
    },
    "widowx_put_eggplant_in_basket": {
        "rt-1-x": 0.000,
        "octo-base": 0.431,
        "octo-small": 0.569,
    },
}


from pprint import pprint
perf = {k:{a:b for a,b in v.items() if 'rt' in a} for k,v in SIMPLER_PERF.items() if "google_robot" in k}
pprint(perf)
