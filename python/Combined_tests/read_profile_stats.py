import pstats
with open('cuda_total_pose_est_below_3m_160.txt', 'w') as f:
    p = pstats.Stats('cuda_total_pose_est_below_3m_160.dat', stream = f)
    p.sort_stats('time').print_stats()
    p.sort_stats('calls').print_stats()
