# BIBLIOTECAS
from __future__ import annotations
from typing import Sequence, Tuple, Union
import numpy as np

# FUNÇÕES
def sphere_points(rmax:float,n_points:int) -> Tuple[Union[float,Sequence[float]],Union[float,Sequence[float]],Union[float,Sequence[float]]]:
    return rmax*np.cbrt(np.random.random_sample(n_points)), np.random.random_sample(n_points)*(2*np.pi), np.arccos(np.random.random_sample(n_points)*2-1)

def sph2rec(r:Union[float,Sequence[float]],theta:Union[float,Sequence[float]],phi:Union[float,Sequence[float]]) -> Tuple[Union[float,Sequence[float]],Union[float,Sequence[float]],Union[float,Sequence[float]]]:
    return r*np.cos(theta)*np.sin(phi), r*np.sin(theta)*np.sin(phi), r*np.cos(phi)

def gen_data(self,rmax:Union[int,float],npoints:int,a0:float=0.529,xyz:bool=True) -> Tuple[Union[float,Sequence[float]],Union[float,Sequence[float]],Union[float,Sequence[float]]]:
    r,theta,phi = sphere_points(rmax,npoints)
    value = self.wave_function_prob(a0)(r,theta,phi)
    if xyz:
        x,y,z = sph2rec(r,theta,phi)
        return x,y,z,value
    else:
        return r,theta,phi,value

def clean_data(value:Union[float,Sequence[float]],epsilon:float=1E-3,normalize:bool=True,*coord:list[Union[float,Sequence[float]]]) -> list[Union[float,Sequence[float]]]:
    if normalize:
        value = value/np.max(value)
    zeros = np.where(value<epsilon)[0]
    return [np.delete(value,zeros)]+[np.delete(i,zeros) for i in coord]

def plot_data(self,rmax:Union[int,float],npoints:int,a0:float=0.529,xyz:bool=True,epsilon:float=1E-3,normalize:bool=True) -> tuple[Union[float,Sequence[float]],Union[float,Sequence[float]],Union[float,Sequence[float]],Union[float,Sequence[float]]]:
    *coords,value = gen_data(self,rmax,npoints,a0,xyz)
    value,*coords = clean_data(value,epsilon,normalize,*coords)
    return *coords,value