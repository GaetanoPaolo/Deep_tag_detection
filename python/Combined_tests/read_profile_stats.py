import pstats
with open('multi_eval_T265_threshold_step_resolution.txt', 'w') as f:
    p = pstats.Stats('multi_eval_T265_threshold_step_resolution.dat', stream = f)
    p.sort_stats('time').print_stats()
    p.sort_stats('calls').print_stats()
