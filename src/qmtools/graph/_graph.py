'''

Set of auxiliary functions for creating graphs using the qmtools package.

'''

# LIBRARIES
from typing import Dict, List, Sequence, Tuple, Union

import numpy as np

from .._core import Psi


# FUNCTIONS
def sphere_points(rmax:Union[int,float],n_points:int) -> Tuple[Union[float,Sequence[float]],Union[float,Sequence[float]],Union[float,Sequence[float]]]:

    '''
    sphere_points
    -------------

    Generates N points randomly distributed in spherical coordinates given a maximum radius.

    Note: Points are generated from NumPy's pseudo random generator. If you want to configure a seed, use the command:
    >>> import numpy as np
    >>> seed = 938792   # Any integer number
    >>> np.random.seed(seed)


    Parameters
    ----------

    rmax : int | float
        Maximum radius for generated points.
    
    n_points : int
        Number of points to be generated.
    

    @rmax: Maximum radius.
    @n_points: Number of points.


    Returns
    -------

    points : Tuple[ float | Sequence[float], float | Sequence[float], float | Sequence[float] ]
        Points randomly distributed in spherical coordinates (r, theta, phi).


    Example
    -------
    >>> import qmtools.graph as qmtg
    >>> r,theta,phi = qmtg.sphere_points(1,100)
    '''

    return rmax*np.cbrt(np.random.random_sample(n_points)), np.random.random_sample(n_points)*(2*np.pi), np.arccos(np.random.random_sample(n_points)*2-1)

def sph2rec(r:Union[float,Sequence[float]],theta:Union[float,Sequence[float]],phi:Union[float,Sequence[float]]) -> Tuple[Union[float,Sequence[float]],Union[float,Sequence[float]],Union[float,Sequence[float]]]:

    '''
    sph2rec
    -------

    Converts points in spherical coordinates to rectangular coordinates.

    (r, theta, phi) --> (x, y, z)


    Parameters
    ----------

    r : float | Sequence[float]
        Radius, 0 < r < inf.

    theta : float | Sequence[float]
        Azimuth coordinate, 0 < theta < 2π.

    phi : float | Sequence[float]
        Polar coordinate, 0 < phi < π.


    @r: Radius
    @theta: Azimuth coordinate
    @phi: Polar coordinate


    Returns
    -------

    points_rec : Tuple[ float | Sequence[float], float | Sequence[float], float | Sequence[float] ]
        Equivalent points in rectangular coordinates.

    
    Example
    -------
    >>> import qmtools.graph as qmtg
    >>> r,theta,phi = qmtg.sphere_points(1,100)
    >>> x,y,z = qmtg.sph2rec(r,theta,phi)
    '''

    return r*np.cos(theta)*np.sin(phi), r*np.sin(theta)*np.sin(phi), r*np.cos(phi)

def gen_data(psi:Psi,rmax:Union[int,float],n_points:int,a0:Union[int,float]=0.529,xyz:bool=True) -> Tuple[Union[float,Sequence[float]],Union[float,Sequence[float]],Union[float,Sequence[float]]]:

    '''
    gen_data
    --------

    Function for generating data from the probability density of the hydrogen wavefunction of a given Psi object.


    Parameters
    ----------

    psi : Psi
        Psi object.
    
    rmax : int | float
        Maximum radius for generated points.
    
    n_points : int
        Number of points to be generated.

    a0 : int | float = 0.529
        (optional) Bohr radius.

    xyz : bool = True
        (Optional) If True, converts points to rectangular coordinates, while False remains in spherical coordinates.

    
    @psi: Psi object.
    @rmax: Maximum radius.
    @n_points: Number of points.
    @a0: (Optional) Bohr radius.
    @xyz: (Optional) Converts to rectangular coordinates.


    Returns
    -------

    (coord1, coord2, coord3, value) : Tuple[ float | Sequence[float], float | Sequence[float], float | Sequence[float], float | Sequence[float] ]
        Tuple with coordinate data (spherical or rectangular) and calculated data.
    

    Example
    -------
    >>> from qmtools import Psi
    >>> import qmtools.graph as qmtg
    >>> p = Psi(2,1,0,1/2)
    >>> x,y,z,value = qmtg.gen_data(p,8,100000)
    '''

    # Generates points in spherical coordinates
    r,theta,phi = sphere_points(rmax,n_points)
    # Calculates the result in the probability density function of the hydrogen wavefunction
    value = psi.wavefunction_prob(a0)(r,theta,phi)
    # Convert to rectangular coordinates?
    if xyz:
        x,y,z = sph2rec(r,theta,phi)
        return x,y,z,value
    else:
        return r,theta,phi,value

def clean_data(value:Union[float,Sequence[float]],coord:List[Union[float,Sequence[float]]],epsilon:float=1E-3,normalize:bool=True,norm_value:bool=True) -> List[Union[float,Sequence[float]]]:

    '''
    clean_data
    ----------

    Clears a dataset by eliminating values below a certain threshold (epsilon). This function is especially useful for removing unimportant points on a plot, saving resources, especially if the point size, in case it is a scatter plot, is proportional to <value>.


    Parameters
    ----------

    value : float | Sequence[float]
        Set of data that will be used for comparison (result of a function, for example). If a given value here is below the threshold (epsilon), this point will be disregarded/deleted.
    
    coord : list[ float | Sequence[float] ]
        Coordinates related to the <value> dataset, i.e., if a point in <value> is deleted, the same will happen here (in the same position).

    epsilon : float = 1E-3
        (Optional) Value to be used in the comparison. Values below epsilon will be disregarded/deleted.

    normalize : bool = True
        (Optional) If True, normalize <value> before applying the operation.

    norm_value : bool = True
        (Optional) If True, returns the normalized <value>. Otherwise, it returns in the original format. It will only take effect if the `normalize` parameter is also true.


    @value: Data for comparison (normally, a result of a function).
    @coord: Coordinates related to <value>.
    @epsilon: (Optional) Value to be used in the comparison.
    @normalize: (Optional) Normalize <value> before applying the operation.
    @norm_value: (optional) Returns the normalized <value>.


    Returns
    -------

    [new_value, *new_coord] : list[ float | Sequence[float] ]
        Returns in a list the new dataset, <value> and <*coords>, removing values below a certain threshold.

    
    Example
    -------
    >>> from qmtools import Psi
    >>> import qmtools.graph as qmtg
    >>> p = Psi(2,1,0,1/2)
    >>> x,y,z,value = qmtg.gen_data(p,8,100000)
    >>> value,x,y,z = qmtg.clean_data(value,[x,y,z])
    '''

    # Copy of <value>
    new_value = value
    if normalize:
        # Normalize <new_value>
        new_value = value/np.max(value)
    # Find the "zeros"
    zeros = np.where(new_value<epsilon)[0]
    if norm_value:
        # Returns with the value of <new_value> which, in principle, is the normalized version
        return [np.delete(new_value,zeros)]+[np.delete(i,zeros) for i in coord]
    else:
        # Returns with <value>, no normalization
        return [np.delete(value,zeros)]+[np.delete(i,zeros) for i in coord]

def plot_data(psi:Psi,rmax:Union[int,float],n_points:int,a0:float=0.529,xyz:bool=True,epsilon:float=1E-3,normalize:bool=True,norm_value:bool=True) -> Tuple[Union[float,Sequence[float]],Union[float,Sequence[float]],Union[float,Sequence[float]],Union[float,Sequence[float]]]:

    '''
    plot_data
    ---------

    Script to automate the process of generating data for plotting: generates the data (gen_data) and "cleans" it (clean_data).

    Parameters
    ----------

    psi : Psi
        Psi object.
    
    rmax : int | float
        Maximum radius for generated points.
    
    n_points : int
        Number of points to be generated.

    a0 : int | float = 0.529
        (optional) Bohr radius.

    xyz : bool = True
        (Optional) If True, converts points to rectangular coordinates, while False remains in spherical coordinates.

    epsilon : float = 1E-3
        (Optional) Value to be used in the comparison. Values below epsilon will be disregarded/deleted.

    normalize : bool = True
        (Optional) If True, normalize <value> before applying the operation.

    norm_value : bool = True
        (Optional) If True, returns the normalized <value>. Otherwise, it returns in the original format. It will only take effect if the `normalize` parameter is also true.


    @psi: Psi object.
    @rmax: Maximum radius.
    @n_points: Number of points.
    @a0: (Optional) Bohr radius.
    @xyz: (Optional) Converts to rectangular coordinates.
    @epsilon: (Optional) Value to be used in the comparison.
    @normalize: (Optional) Normalize <value> before applying the operation.
    @norm_value: (optional) Returns the normalized <value>.


    Returns
    -------

    (coord1, coord2, coord3, value) : tuple[ float | Sequence[float], float | Sequence[float], float | Sequence[float], float | Sequence[float] ]
        Returns a tuple with the clean data generated.

    
    Example
    -------
    >>> from qmtools import Psi
    >>> import qmtools.graph as qmtg
    >>> p = Psi(2,1,0,1/2)
    >>> x,y,z,value = qmtg.gen_data(p,8,100000)
    >>> value,x,y,z = qmtg.clean_data(value)
    '''

    # Generate the data
    *coords,value = gen_data(psi,rmax,n_points,a0,xyz)
    # Clean the data
    value,*coords = clean_data(value,coords,epsilon,normalize,norm_value)
    # Returns coordinates and resulting values
    return coords[0],coords[1],coords[2],value
