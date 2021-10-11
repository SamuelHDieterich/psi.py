# BIBLIOTECAS
import numpy as np
import plotly.graph_objects as go

from quantum_psi import Psi
from quantum_psi import graph as pgraph

# CONSTANTES
A0 = 0.529          # Raio de Bohr
N_POINTS = int(1E5) # Número de pontos (cálculo != plot)
EPSILON = 1E-3
RMAX = 10

# PSI
p1 = Psi(n=2,l=1,ml=0,ms=-1/2)
p2 = Psi(n=2,l=1,ml=-1,ms=1/2)
p = np.sqrt(2/3)*p1 - np.sqrt(1/3)*p2
print(p)

# DADOS
x,y,z,value = pgraph.plot_data(p,RMAX,N_POINTS,A0,xyz=True,epsilon=EPSILON,normalize=True)

# GRÁFICO
fig = go.Figure(
        data=[
            go.Scatter3d(                       # Gráfico de dispersão 3D
                x=x,
                y=y,
                z=z,
                mode='markers',                 # Usar marcadores
                marker=dict(                    # Configuração do "marker"
                    size=15*np.sqrt(value),     # Tamanho
                    color=value,                # Cor
                    opacity=0.7,                # Opacidade
                    line=dict(                  # Configuração da "borda"
                        width=0                 # Espessura zero (sem borda)
                    )
                ),
                # Informações apresentadas ao passar o cursor em cima do ponto
                hovertemplate="<b>x</b> = %{x:.3f}<br><b>y</b> = %{y:.3f}<br><b>z</b> = %{z:.3f}<br><b>Valor</b> = %{marker.color:.3f}<extra></extra>"
            )
        ]
    )
fig.update_layout(template='plotly_dark')   # Template "dark mode"
fig.show()                                  # Mostra a figura
