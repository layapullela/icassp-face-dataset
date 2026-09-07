# Overlapping tuning-pool ablation (test set)

Hypothesis: at k=8/12 each fold's tuning objective sees too few train matrices (1-2 fixed partition groups x 24 sequences) for SSC-Block-TV-Col21's 4 hyperparameters, hurting noise robustness relative to OSC (2 params) and TKSS (3 params). `overlap` retunes each fold on 4 randomly drawn, possibly overlapping k-sized subsets of that fold's held-in people (96 matrices) instead; the reported train/test matrices themselves are unchanged, so this isolates the effect of a larger tuning pool.

| Method | k | sigma | ARI orig | ARI overlap | delta ARI | NMI orig | NMI overlap | delta NMI | ACC orig | ACC overlap | delta ACC |
|---|---|---|---|---|---|---|---|---|---|---|---|
| SSC-Block-TV-Col21 | 8 | 0 | 0.793 | 0.778 | -0.014 | 0.898 | 0.890 | -0.008 | 0.846 | 0.826 | -0.020 |
| SSC-Block-TV-Col21 | 8 | 0.25 | 0.747 | 0.759 | +0.012 | 0.873 | 0.879 | +0.006 | 0.808 | 0.816 | +0.007 |
| SSC-Block-TV-Col21 | 8 | 0.5 | 0.695 | 0.681 | -0.014 | 0.843 | 0.837 | -0.006 | 0.771 | 0.754 | -0.016 |
| SSC-Block-TV-Col21 | 12 | 0 | 0.791 | 0.792 | +0.001 | 0.914 | 0.915 | +0.001 | 0.823 | 0.820 | -0.003 |
| SSC-Block-TV-Col21 | 12 | 0.25 | 0.657 | 0.738 | +0.082 | 0.837 | 0.886 | +0.048 | 0.695 | 0.782 | +0.087 |
| SSC-Block-TV-Col21 | 12 | 0.5 | 0.575 | 0.618 | +0.044 | 0.788 | 0.809 | +0.021 | 0.635 | 0.689 | +0.054 |
| OSC | 8 | 0 | 0.739 | 0.735 | -0.004 | 0.874 | 0.872 | -0.001 | 0.797 | 0.792 | -0.005 |
| OSC | 8 | 0.25 | 0.719 | 0.721 | +0.002 | 0.864 | 0.865 | +0.001 | 0.779 | 0.780 | +0.001 |
| OSC | 8 | 0.5 | 0.681 | 0.672 | -0.009 | 0.838 | 0.834 | -0.003 | 0.749 | 0.742 | -0.006 |
| OSC | 12 | 0 | 0.732 | 0.733 | +0.001 | 0.889 | 0.890 | +0.000 | 0.771 | 0.771 | +0.000 |
| OSC | 12 | 0.25 | 0.716 | 0.720 | +0.005 | 0.881 | 0.882 | +0.001 | 0.758 | 0.764 | +0.006 |
| OSC | 12 | 0.5 | 0.645 | 0.647 | +0.002 | 0.845 | 0.846 | +0.001 | 0.702 | 0.703 | +0.002 |
| TKSS | 8 | 0 | 0.693 | 0.688 | -0.005 | 0.809 | 0.806 | -0.003 | 0.816 | 0.813 | -0.003 |
| TKSS | 8 | 0.25 | 0.698 | 0.697 | -0.002 | 0.814 | 0.812 | -0.002 | 0.818 | 0.814 | -0.003 |
| TKSS | 8 | 0.5 | 0.696 | 0.697 | +0.001 | 0.811 | 0.811 | -0.000 | 0.812 | 0.809 | -0.003 |
| TKSS | 12 | 0 | 0.682 | 0.682 | -0.000 | 0.827 | 0.828 | +0.001 | 0.790 | 0.790 | -0.000 |
| TKSS | 12 | 0.25 | 0.682 | 0.682 | -0.000 | 0.828 | 0.829 | +0.001 | 0.790 | 0.789 | -0.001 |
| TKSS | 12 | 0.5 | 0.694 | 0.694 | +0.001 | 0.836 | 0.836 | +0.000 | 0.797 | 0.797 | -0.000 |
