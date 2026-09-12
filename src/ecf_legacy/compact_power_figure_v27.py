"""Re-render the same power-reconstruction series with compact panel heights."""
from plot_revision_v27 import formatting,save
import plot_process_figures as p
formatting()
def compact(fig,name):
    fig.set_size_inches(7.25,4.05)
    save(fig,name)
p.save=compact
p.power_retraction()
