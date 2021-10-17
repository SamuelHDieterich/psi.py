'''

Test file with qmtools package.

'''

# LIBRARIES
import numpy as np
import plotly.graph_objects as go

from qmtools import Psi
from qmtools import graph as qmtg

# CONSTANTS
A0 = 0.529          # Bohr radius
N_POINTS = int(1E5) # NUmber of points (calc != plot)
EPSILON = 1E-2      # Threshold
RMAX = 5            # Maximum radius

# PSI
p1 = Psi(n=2,l=1,ml=0,ms=-1/2)
p2 = Psi(n=2,l=1,ml=-1,ms=1/2) 
p = np.sqrt(2/3)*p1 + np.sqrt(1/3)*p2
print(p)

# DATA
x,y,z,value = qmtg.plot_data(p,RMAX,N_POINTS,A0,xyz=True,epsilon=EPSILON,normalize=True,norm_value=True)

# GRAPH
fig = go.Figure(                                # Create graph
        data=[
            go.Scatter3d(
                x=x,
                y=y,
                z=z,
                mode='markers',
                marker=dict(
                    size=15*np.sqrt(value),
                    color=value,
                    opacity=0.7,
                    line=dict(
                        width=0
                    )
                ),
                # Information displayed by hovering the cursor over the point
                hovertemplate="<b>x</b> = %{x:.3f}<br><b>y</b> = %{y:.3f}<br><b>z</b> = %{z:.3f}<br><b>Valor</b> = %{marker.color:.3f}<extra></extra>"
            )
        ]
    )
fig.update_layout(template='plotly_dark')   # Dark mode template
fig.show()                                  # Show figure
