'''

Main package of the qmtools with essential tools for the class Psi.

'''

# LIBRARIES
from typing import Callable, Dict, Sequence, Union

import attr
import numpy as np
from scipy.integrate import quad
from scipy.special import factorial, genlaguerre, sph_harm


# MAIN CLASS (in current version :) )
@attr.s(frozen=True,repr=False,auto_attribs=True)
class Psi:

    '''
    Psi
    ---

    Class responsible for creating objects with representation and properties similar to the 'ket' of Dirac's notation.

    Note: This object's initialization parameters allow you to pass a sequence (list, tuple, etc), however, this practice is not recommended. The reason for this is to handle operations between Psi functions, for example: `psi = psi1 + psi2`. In this example, a new object will be created with the properties of `psi1` and `psi2`.


    Parameters
    ----------

    n : int | Sequence[int]
        Principal quantum number.
    
    l : int | Sequence[int]
        Azimuthal quantum number.
    
    ml : int | Sequence[int]
        Magnetic quantum number.
    
    ms : int | float | Sequence[int] | Sequence[float]
        Spin quantum number.
    
    const : int | float | Sequence[float] = 1
        (Optional) Constant that multiples the 'ket'. It is not recommended to manually change this parameter, because it works for adding and subtracting of Psi objects and normalizing functions automatically.
    

    @n: Principal quantum number.
    @l: Azimuthal quantum number.
    @ml: Magnetic quantum number.
    @ms: Spin quantum number.
    @const: (Optional) Constant that multiples the 'ket'.


    Example
    -------
    >>> from qmtools import Psi
    >>> p = Psi(1,0,0,0)    # 1s
    >>> print(p)
    1.000·│ n = 1, l = 0, ml = 0, ms = 0 ⧽
    '''

    # Principal quantum number
    n:      Union[int,Sequence[int]]                        = attr.ib()
    # Azimuthal quantum number             
    l:      Union[int,Sequence[int]]                        = attr.ib()
    # Magnetic quantum number        
    ml:     Union[int,Sequence[int]]                        = attr.ib()
    # Spin quantum number             
    ms:     Union[int,float,Sequence[int],Sequence[float]]  = attr.ib()
    # Constant    
    const:  Union[int,float,Sequence[float]]                = attr.ib(default=1)

    # VALIDATORS
    @n.validator
    def __check_n(instance,atribute,value):
        if np.any(value <= 0):
            raise ValueError("'n' value must be greater than zero.")

    @l.validator
    def __check_l(instance,atribute,value):
        if np.any(value >= instance.n):
            raise ValueError(f"'l' value, {value}, must be less than 'n' value, {instance.n}.")
        elif np.any(value < 0):
            raise ValueError("'l' value must be greater or equal to zero.")

    @ml.validator
    def __check_ml(instance,atribute,value):
        if np.any(np.abs(value)>instance.l):
            raise ValueError("'ml' value must be from between '-l' and 'l'.")

    # DUNDER METHODS
    def __repr__(self) -> str:

        '''
        __repr__
        --------

        Defines what is shown when for the user by the Psi object, i.e., the `print()` method.


        Example
        -------
        >>> p = Psi(3,2,1,1/2)
        >>> print(p)
        1.000·│ n = 3, l = 2, ml = 1, ms = 0.5 ⧽
        '''

        # ket's style format
        str_ket = "{:.3f}·│ n = {}, l = {}, ml = {}, ms = {} ⧽"
        # If <const> is iterable
        if hasattr(self.const,'__iter__'):
            output = '' # Inicialise the output variable
            # Loop throughout the items of the size of <const>
            for i in range(len(self.const)):
                if self.const[i] < 0:   # If <const> is negative
                    operator = " - " if i != 0 else '- '
                elif self.const[i] > 0: # If <const> is positive
                    operator = ' + ' if i != 0 else '' 
                else:   # In case it is zero or indeterminate, disregard this part
                    continue
                # Output text will be what it already had plus the operator with the values inside the 'ket'
                output = output + operator + str_ket.format(np.abs(self.const[i]),self.n[i],self.l[i],self.ml[i],self.ms[i])
        # In case <const> is a single value
        else:
            # Simple output
            output = str_ket.format(self.const,self.n,self.l,self.ml,self.ms)
        # Returns the text resulting from the above operations
        return output

    def __mul__(self,other:Union[int,float,complex]):

        '''
        __mul__
        -------

        Define a operação de multiplicação com um objeto Psi. Apenas será válido operações entre um número (inteiro ou real) vezes o objeto Psi.

        A operação `__rmul__` apresenta a mesma propriedade.
        

        Example
        -------
        >>> p = Psi(3,2,1,1/2)
        >>> a = 2   # inteiro
        >>> b = 1/3 # real
        >>> p1 = a*p
        >>> p2 = p*b
        '''

        # Check if <other> is a integer, real or complex number
        if isinstance(other,(int,float,complex)):
            # Makes a copy of the object just multiplying <other> in <const>
            return Psi(self.n,self.l,self.ml,self.ms,self.const*other)
        # Error message
        else:
            raise TypeError(f"{__class__} does not accept multiplications with type {type(other)}.")
    
    def __rmul__(self,other:Union[int,float,complex]):

        '''
        __rmul__
        --------

        Same operation defined in `__mul__`.
        '''

        return self.__mul__(other)
    
    def __truediv__(self,other:Union[int,float,complex]):

        '''
        __truediv__
        -----------

        Defines the real division operation with a Psi object. Only operations between a number (integer, real or complex) dividing the Psi object will be allowed.


        Example
        -------
        >>> p1 = Psi(2,1,-1,0)
        >>> p = p1/2
        '''

        # Check if <other> is a integer, real or complex number
        if isinstance(other,(int,float,complex)):
            # Makes a copy of the object just dividing <other> in <const>
            return Psi(self.n,self.l,self.ml,self.ms,self.const/other)
        # Error message
        else:
            raise TypeError(f"{self.__class__} does not accept division with type {type(other)}.")
    
    def __add__(self,other):

        '''
        __add__
        -------

        Defines the sum operation that is only allowed between two Psi objects. In this case, a new object will be created with the attributes of the objects added in a list in the corresponding attributes.


        Example
        -------
        >>> p1 = Psi(2,1,-1,1/2)/2
        >>> p2 = Psi(2,1,0,-1/2)/2
        >>> p = p1 + p2
        >>> print(p)
        0.500·│ n = 2, l = 1, ml = -1, ms = 0.5 ⧽ + 0.500·│ n = 2, l = 1, ml = 0, ms = -0.5 ⧽
        '''

        # Check if <other> is a Psi object
        if isinstance(other, Psi):  
            # Creates a new Psi object with an 1D array in the atributes      
            return Psi(
                n       = np.append(self.n,     other.n),
                l       = np.append(self.l,     other.l),
                ml      = np.append(self.ml,    other.ml),
                ms      = np.append(self.ms,    other.ms),
                const   = np.append(self.const, other.const),
            )
        else:
            raise TypeError(f"{self.__class__} does not accept sum with type {type(other)}.")
    
    def __sub__(self,other):

        '''
        __sub__
        -------

        Defines the subtraction operation that is only allowed between two Psi objects. In this case, a new object will be created with the attributes of the subtraction objects listed in the corresponding attributes. Unlike `__add__`, in this case the <const> attribute of <other> will be multiplied by -1 since it is in a subtraction.


        Example
        -------
        >>> p1 = Psi(2,1,-1,1/2)/2
        >>> p2 = Psi(2,1,0,-1/2)/2
        >>> p = p1 - p2
        >>> print(p)
        0.500·│ n = 2, l = 1, ml = -1, ms = 0.5 ⧽ - 0.500·│ n = 2, l = 1, ml = 0, ms = -0.5 ⧽
        '''

        # Check if <other> is a Psi object
        if isinstance(other, Psi):  
            # Creates a new Psi object with an 1D array in the atributes      
            return Psi(
                n       = np.append(self.n,     other.n),
                l       = np.append(self.l,     other.l),
                ml      = np.append(self.ml,    other.ml),
                ms      = np.append(self.ms,    other.ms),
                const   = np.append(self.const, -other.const),
            )
        else:
            raise TypeError(f"{self.__class__} does not accept subtraction with type {type(other)}.")

    # METHODS
    def normalize(self):

        '''
        normalize
        ---------

        Normalizes the Psi object, that is, guarantees that the sum of the multiplicative constants (<const>) squared results in 1. Normally, this operation will only be useful if 2 or more Psi objects have been added/subtracted and it is not guaranteed that these are normalized. This is a method used internally for other functions that depend on this normalization.


        Returns
        -------

        normalized_psi : Psi
            New Psi object with normalized <const> attribute.

        
        Example
        -------
        >>> from qmtools import Psi
        >>> p1 = Psi(2,1,-1,1/2)
        >>> p2 = Psi(2,1,0,-1/2)
        >>> p = p1 + p2             # Operatio without normalization
        >>> print(p)                # Result without normalization
        1.000·│ n = 2, l = 1, ml = -1, ms = 0.5 ⧽ + 1.000·│ n = 2, l = 1, ml = 0, ms = -0.5 ⧽
        >>> print(p.normalize())    # Normalized
        0.500·│ n = 2, l = 1, ml = -1, ms = 0.5 ⧽ + 0.500·│ n = 2, l = 1, ml = 0, ms = -0.5 ⧽
        '''

        # Returns a copy of the object doing the operation: const = const/sum(const^2)
        return Psi(self.n,self.l,self.ml,self.ms,self.const/np.sum(self.const**2))

    def radial_wave(self,a0:Union[int,float]=0.529) -> Callable[[Union[float,Sequence[float]]], Union[float,Sequence[float]]]:

        '''
        radial_wave
        -----------

        Returns the radial wave function, R_nl(r), for the Psi object.


        Parameters
        ----------

        a0 : int | float = 0.529
            (Optional) Bohr radius.

        
        @a0: (Optional) Bohr radius.


        Returns
        -------

        R_nl : Callable[ [ float | Sequence[float] ], float | Sequence[float] ]
            Radial wave function for the Psi object.

        Example
        -------
        >>> import numpy as np
        >>> from qmtools import Psi
        >>> R_10 = Psi(1,0,0,0).radial_wave()
        >>> r = np.linspace(0,5,1000)
        >>> R_10(r)
        array([5.19812223e+00, 5.14917339e+00, 5.10068547e+00, 5.05265415e+00,...
        '''
        
        # Normalized object
        self_norm = self.normalize()

        # Internal method to generate the radial wave function
        def _R(n:int,l:int,a0:float) -> Callable[[float],float]:
            return lambda r: np.sqrt((2/(n*a0))**3*factorial(n-l-1)/(2*n*factorial(n+l)))*np.exp(-r/(n*a0))*(2*r/(n*a0))**l*genlaguerre(n-l-1,2*l+1)(r)

        # If <const> is iterable
        if hasattr(self_norm.const,'__iter__'):
            # Output function is the sum of all contribution in Psi
            return lambda r: sum(self_norm.const[i]*_R(self.n[i],self.l[i],a0)(r) for i in range(len(self_norm.const)))
        # If <const> is a single value
        else:
            return _R(self.n,self.l,a0)

    def radial_prob(self,a0:Union[int,float]=0.529):

        '''
        radial_prob
        -----------

        Returns the radial probability density function for the Psi object: r²|R_nl(r)|².
        

        Parameters
        ----------

        a0 : int | float = 0.529
            (Optional) Bohr radius.

        
        @a0: (Optional) Bohr radius.


        Returns
        -------

        rad_prob_nl : Callable[ [ float | Sequence[float] ], float | Sequence[float] ]
            Radial probability density function for the Psi object.
        

        Example
        -------
        >>> import numpy as np
        >>> from qmtools import Psi
        >>> P_rad_10 = Psi(1,0,0,0).P_rad()
        >>> r = np.linspace(0,5,1000)
        >>> P_rad_10(r)
        array([0.00000000e+00, 6.64177354e-04, 2.60691044e-03, 5.75560109e-03,...
        '''

        return lambda r: r**2*np.abs(self.radial_wave(a0)(r))**2

    def wavefunction(self,a0:Union[int,float]=0.529) -> Callable[[Union[float,Sequence[float]],Union[float,Sequence[float]],Union[float,Sequence[float]]], Union[complex,Sequence[complex]]]:

        '''
        wavefunction
        ------------

        Returns the normalized position hydrogen wavefunction of the object Psi in spherical coordinates (r,theta,phi).


        Parameters
        ----------

        a0 : int | float = 0.529
            (Optional) Bohr radius.

        
        @a0: (Optional) Bohr radius.


        Returns
        -------

        psi_nlm : Callable[ [ float | Sequence[float], float | Sequence[float], float | Sequence[float] ], complex | Sequence[complex] ]
        
            Normalized position hydrogen wavefunction of the object Psi in spherical coordinates.


            Parameters
            ----------

            r : float | Sequence[float]
                Radius, 0 < r < inf.

            theta : float | Sequence[float]
                Azimuth coordinate, 0 < theta < 2π.

            phi : float | Sequence[float]
                Polar coordinate, 0 < phi < π.
            

            Returns
            -------

            result : complex | Sequence[complex]
                Result of the operation given the values in spherical coordinates.
        

        Example
        -------
        >>> from qmtools import Psi
        >>> psi_100 = Psi(1,0,0,0).wavefunction()
        >>> r, theta, phi = 1, 3, 2
        >>> psi_100(r,theta,phi)
        (0.22144659149706755+0j)
        '''

        # Normalized object
        self_norm = self.normalize()

        # Internal method to generate respective the wave function
        def _wavefunction(n:int,l:int,ml:int,a0:float) -> Callable[[Union[float,Sequence[float]],Union[float,Sequence[float]],Union[float,Sequence[float]]], Union[complex,Sequence[complex]]]:
            return lambda r,theta,phi: np.sqrt((2/(n*a0))**3*factorial(n-l-1)/(2*n*factorial(n+l)))*np.exp(-r/(n*a0))*(2*r/(n*a0))**l*genlaguerre(n-l-1,2*l+1)(r)*sph_harm(ml,l,theta,phi)

        # If <const> is iterable
        if hasattr(self.const,'__iter__'):
            # Output function is the sum of all contribution in Psi
            return lambda r,theta,phi: sum(self_norm.const[i]*_wavefunction(self.n[i],self.l[i],self.ml[i],a0)(r,theta,phi) for i in range(len(self_norm.const)))
        # If <const> is a single value
        else:
            return _wavefunction(self.n,self.l,self.ml,a0)

    def wavefunction_prob(self,a0:float=0.529) -> Callable[[Union[float,Sequence[float]],Union[float,Sequence[float]],Union[float,Sequence[float]]], Union[float,Sequence[float]]]:

        '''
        wave_function_prob
        ------------------

        Returns the probability density function of the hydrogen wavefunction for the Psi object in spherical coordinates (r, theta, phi).


        Parameters
        ----------

        a0 : int | float = 0.529
            (Optional) Bohr radius.

        
        @a0: (Optional) Bohr radius.


        Returns
        -------

        psi_nlm : Callable[ [ float | Sequence[float], float | Sequence[float], float | Sequence[float] ], complex | Sequence[complex] ]
        
            probability density function of the hydrogen wavefunction for the Psi object in spherical coordinates.


            Parameters
            ----------

            r : float | Sequence[float]
                Radius, 0 < r < inf.

            theta : float | Sequence[float]
                Azimuth coordinate, 0 < theta < 2π.

            phi : float | Sequence[float]
                Polar coordinate, 0 < phi < π.
            

            Returns
            -------

            result : complex | Sequence[complex]
                Result of the operation given the values in spherical coordinates.
        

        Example
        -------
        >>> from qmtools import Psi
        >>> prob_psi_100 = Psi(1,0,0,0).wavefunction_prob()
        >>> r, theta, phi = 1, 3, 2
        >>> prob_psi_100(r,theta,phi)
        0.04903859288566911
        '''

        # |wavefunction|²
        return lambda r,theta,phi: np.abs(self.wavefunction(a0)(r,theta,phi))**2

    def mean_r(self,a0:Union[int,float]=0.529) -> float:

        '''
        mean_r
        ------

        Calculates the mean radius, radius expected value, using the integral from zero to infinity of r*P_rad(r).


        Parameters
        ----------

        a0 : int | float = 0.529
            (Optional) Bohr radius.

        
        @a0: (Optional) Bohr radius.


        Returns
        -------

        mean_radius : float
            radius expected value or mean radius of the Psi object.

        
        Example
        -------
        >>> from qmtools import Psi
        >>> mean_radius = Psi(1,0,0,0).mean_r()
        >>> mean_radius
        0.7934999999999999
        '''

        return quad(lambda r: r*self.radial_prob(a0)(r),0,np.inf)[0]

    def r_bohr(self,a0:Union[int,float]=0.529) -> Union[float,ValueError]:

        '''
        r_bohr
        ------

        Calculates the radius of the Psi object from the Bohr model.


        Parameters
        ----------

        a0 : int | float = 0.529
            (Optional) Bohr radius.

        
        @a0: (Optional) Bohr radius.


        Returns
        -------

        radius_bohr : float | ValueError
            Returns the radius from the Bohr model. If the object has more than one value for each attribute and these are distinct in 'n', principal quantum number, a ValueError will be raised.
        '''

        # If <const> is iterable
        if hasattr(self.const,'__iter__'):
            # If <n> are different from each other
            if not np.equal(*self.n):
                raise ValueError(f"{__class__} object has one or more distinct value in 'n'.")
            return self.n[0]**2*a0
        # If <const> is a single value
        else:
            return self.n**2*a0
