import pstats
with open('detect_parallel_hom_pose_est_below_3m_130.txt', 'w') as f:
    p = pstats.Stats('detect_parallel_hom_pose_est_below_3m_130.dat', stream = f)
    p.sort_stats('time').print_stats()
    p.sort_stats('calls').print_stats()
