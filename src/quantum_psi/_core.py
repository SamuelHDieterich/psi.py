# BIBLIOTECAS
from __future__ import annotations
from typing import Callable, Sequence, Tuple, Union
import numpy as np
from scipy.integrate import quad
from scipy.special import factorial, genlaguerre, sph_harm

# CLASSE
class Psi:

    def __init__(self,n:Union[int,Sequence[int]],l:Union[int,Sequence[int]],ml:Union[int,Sequence[int]],ms:Union[int,float],const:Union[int,float,Sequence[float]] = 1) -> None:
        self.const = const
        self.n  = n
        self.l  = l
        self.ml = ml
        self.ms = ms
        return None
    
    def __repr__(self) -> str:
        str_ket = "{:.3f}·│ n = {}, l = {}, ml = {}, ms = {} ⧽"
        if hasattr(self.const,'__iter__'):
            output = ''
            for i in range(len(self.const)):
                if self.const[i] < 0:
                    operator = " - "
                elif self.const[i] > 0:
                    operator = ' + ' if i != 0 else '' 
                else:
                    continue
                output = output + operator + str_ket.format(np.abs(self.const[i]),self.n[i],self.l[i],self.ml[i],self.ms[i])
        else:
            output = str_ket.format(self.const,self.n,self.l,self.ml,self.ms)
        return output

    def __mul__(self,other:Union[int,float]) -> Psi:
        if type(other) is not int and type(other) is not float:
            raise Exception("Argumento inválido! Classe Psi só aceita multiplicação com inteiros ou reais.")
        return Psi(self.n,self.l,self.ml,self.ms,self.const*other)
    
    def __rmul__(self,other:Union[int,float]) -> Psi:         
        return self.__mul__(other)
    
    def __truediv__(self,other:Union[int,float]) -> Psi:
        if type(other) is not int and type(other) is not float:
            raise Exception("Argumento inválido! Classe Psi só aceita divisão com inteiros ou reais.")
        return Psi(self.n,self.l,self.ml,self.ms,self.const/other)
    
    def __add__(self,other:Psi) -> Psi:
        if type(other) is not Psi:
            raise Exception("Argumento inválido! A soma com Psi só pode ser feita com outro objeto Psi.")
        return Psi(
            n       = np.array((self.n,other.n)).flatten(),
            l       = np.array((self.l,other.l)).flatten(),
            ml      = np.array((self.ml,other.ml)).flatten(),
            ms      = np.array((self.ms,other.ms)).flatten(),
            const   = np.array((self.const,other.const)).flatten()
        )
    
    def __sub__(self,other:Psi) -> Psi:
        if type(other) is not Psi:
            raise Exception("Argumento inválido! A subtração com Psi só pode ser feita com outro objeto Psi.")
        return Psi(
            n       = np.array((self.n,other.n)).flatten(),
            l       = np.array((self.l,other.l)).flatten(),
            ml      = np.array((self.ml,other.ml)).flatten(),
            ms      = np.array((self.ms,other.ms)).flatten(),
            const   = np.array((self.const,-other.const)).flatten()
        )

    def normalize(self) -> Psi:
        return Psi(self.n,self.l,self.ml,self.ms,self.const/np.sum(self.const**2))

    def R(self,a0:float=0.529) -> Callable[[Union[float,Sequence[float]]], Union[float,Sequence[float]]]:
        
        self = self.normalize() # self = [?]!

        def _R(const:Union[int,float],n:int,l:int,a0:float) -> Callable[[float],float]:
            return lambda r: const*(np.sqrt((2/(n*a0))**3*factorial(n-l-1)/(2*n*factorial(n+l)))*np.exp(-r/(n*a0))*(2*r/(n*a0))**l*genlaguerre(n-l-1,2*l+1)(r))

        if hasattr(self.const,'__iter__'):
            return lambda r: sum(_R(self.const[i],self.n[i],self.l[i],a0)(r) for i in range(len(self.const)))
        else:
            return _R(self.const,self.n,self.l,a0)

    def P_rad(self,a0:float=0.529):
        return lambda r: r**2*np.abs(self.R(a0)(r))**2

    def wave_function(self,a0:float=0.529) -> Callable[[Union[float,Sequence[float]],Union[float,Sequence[float]],Union[float,Sequence[float]]], Union[complex,Sequence[complex]]]:

        self = self.normalize() # self = [?]!

        def _wave_function(const:Union[int,float],n:int,l:int,ml:int,a0:float) -> Callable[[float,float,float], complex]:
            return lambda r,theta,phi: const*(np.sqrt((2/(n*a0))**3*factorial(n-l-1)/(2*n*factorial(n+l)))*np.exp(-r/(n*a0))*(2*r/(n*a0))**l*genlaguerre(n-l-1,2*l+1)(r))*sph_harm(ml,l,theta,phi)

        if hasattr(self.const,'__iter__'):
            return lambda r,theta,phi: sum(_wave_function(self.const[i],self.n[i],self.l[i],self.ml[i],a0)(r,theta,phi) for i in range(len(self.const)))
        else:
            return _wave_function(self.const,self.n,self.l,self.ml,a0)

    def wave_function_prob(self,a0:float=0.529) -> Callable[[Union[float,Sequence[float]],Union[float,Sequence[float]],Union[float,Sequence[float]]], Union[float,Sequence[float]]]:
        return lambda r,theta,phi: np.abs(self.wave_function(a0)(r,theta,phi).real)**2

    def mean_r(self,a0:float=0.529) -> float:
        return quad(lambda r: r*self.P_rad(a0)(r),0,np.inf)[0]

    def r_bohr(self,a0:float=0.529) -> Union[float,Exception]:
        if hasattr(self.const,'__iter__'):
            if not np.equal(*self.n):
                raise Exception("Operação inválida! Objeto Psi apresenta valores diferentes para 'n'.")
            return self.n[0]**2*a0
        else:
            return self.n**2*a0

    from ._graph import plot_data