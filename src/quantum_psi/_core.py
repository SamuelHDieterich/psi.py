'''

Pacote principal de quantum_psi com ferramentas essenciais para a classe Psi.

'''

# BIBLIOTECAS
from __future__ import annotations

from typing import Callable, Sequence, Union

import numpy as np
from scipy.integrate import quad
from scipy.special import factorial, genlaguerre, sph_harm


# CLASSE PRINCIPAL
class Psi:

    '''
    Psi
    ---

    Classe Psi responsável por criar objetos com representação e propriedades similares ao 'ket' da notação de Dirac, possibilitando operações estudadas na mecânica quântica.
    '''

    def __init__(self:Psi,n:Union[int,Sequence[int]],l:Union[int,Sequence[int]],ml:Union[int,Sequence[int]],ms:Union[int,float],const:Union[int,float,Sequence[float]] = 1) -> None:

        '''
        Psi
        ---

        Objetos da classe Psi possuem uma representação e propriedades similares ao 'ket' da notação de Dirac, possibilitando operações estudadas na mecânica quântica.

        Observação: Os parâmetros de inicialização desse objeto permitem passar uma sequência (lista, tupla, etc), contudo, essa prática não é recomendada. O motivo disso está para tratar operações entre funções psi, por exemplo: Ψ = Ψ_1 + Ψ_2. Nesse exemplo, será criado um novo objeto com as propriedades de Ψ_1 e Ψ_2. 


        Parâmetros
        ----------

        n : int | Sequence[int]
            Número quântico principal.
        
        l : int | Sequence[int]
            Número quântico secundário.
        
        ml : int | Sequence[int]
            Número quântico magnético.
        
        ms : int | Sequence[int]
            Número quântico de spin.
        
        const : int | float | Sequence[float] = 1
            (Opcional) Constante que multiplica o 'ket'. Recomenda-se não alterar mannualmente esse parâmetro, pois este funciona para somas e subtração de objetos Psi e normalização de funções de maneira automática.
        

        @n: Número quântico principal.
        @l: Número quântico secundário.
        @ml: Número quântico magnético.
        @ms: Número quântico de spin.
        @const: (Opcional) Constante que multiplica o 'ket'. Recomenda-se não alterar mannualmente esse parâmetro, pois este funciona para somas e subtração de objetos Psi e normalização de funções de maneira automática.


        Exemplo
        -------
        >>> from quantum_psi import Psi
        >>> p = Psi(1,0,0,0)    # 1s
        >>> print(p)
        1.000·│ n = 1, l = 0, ml = 0, ms = 0 ⧽
        '''

        self.const = const  # Constante
        self.n  = n         # Número quântico principal
        self.l  = l         # Número quântico secundário
        self.ml = ml        # Número quântico magnético
        self.ms = ms        # Número quântico de spin
        return None
    
    def __repr__(self:Psi) -> str:

        '''
        __repr__
        --------

        Define o que é mostrado ao usuário pelo objeto Psi, isto é, o `print()`.


        Exemplo
        -------
        >>> p = Psi(3,2,1,1/2)
        >>> print(p)
        1.000·│ n = 3, l = 2, ml = 1, ms = 0.5 ⧽
        '''

        # Estilo de formatação do 'ket'
        str_ket = "{:.3f}·│ n = {}, l = {}, ml = {}, ms = {} ⧽"
        # Se <const> for iterável
        if hasattr(self.const,'__iter__'):
            output = '' # Inicia a várivel de saída
            # Loop em todos os itens do tamanho de <const>
            for i in range(len(self.const)):
                if self.const[i] < 0:   # Se <const> for negativo
                    operator = " - " if i != 0 else '- '
                elif self.const[i] > 0: # Se <const> for positivo
                    operator = ' + ' if i != 0 else '' 
                else:   # Caso for zero ou indeterminado, desconsidera essa parte
                    continue
                # Texto de saída vai ser o que já tinha mais o operador com os valores dentro do 'ket'
                output = output + operator + str_ket.format(np.abs(self.const[i]),self.n[i],self.l[i],self.ml[i],self.ms[i])
        # Caso <const> seja tenha um único valor
        else:
            # Saída "simples"
            output = str_ket.format(self.const,self.n,self.l,self.ml,self.ms)
        # Retorna o texto resultante das operações acima
        return output

    def __mul__(self:Psi,other:Union[int,float]) -> Union[Psi,Exception]:

        '''
        __mul__
        -------

        Define a operação de multiplicação com um objeto Psi. Apenas será válido operações entre um número (inteiro ou real) vezes o objeto Psi.

        A operação `__rmul__` apresenta a mesma propriedade.
        

        Exemplo
        -------
        >>> p = Psi(3,2,1,1/2)
        >>> a = 2   # inteiro
        >>> b = 1/3 # real
        >>> p1 = a*p
        >>> p2 = p*b
        '''

        # Verifica se <other> é inteiro ou real
        if type(other) is not int and type(other) is not float:
            raise Exception("Argumento inválido! Classe Psi só aceita multiplicação com inteiros ou reais.")
        # Faz uma cópia do objeto apenas multiplicando <other> em <const>
        return Psi(self.n,self.l,self.ml,self.ms,self.const*other)
    
    def __rmul__(self:Psi,other:Union[int,float]) -> Union[Psi,Exception]:

        '''
        __rmul__
        --------

        Mesma operação definida em `__mul__`.
        '''

        # Retorna a operação de __mul__
        return self.__mul__(other)
    
    def __truediv__(self,other:Union[int,float]) -> Union[Psi,Exception]:

        '''
        __truediv__
        -----------

        Define a operação de divisão real com um objeto Psi. Apenas será válido operações entre um número (inteiro ou real) dividindo o objeto Psi.


        Exemplo
        -------
        >>> p1 = Psi(2,1,-1,0)
        >>> p = p1/2
        '''

        # Verifica se <other> é inteiro ou real
        if type(other) is not int and type(other) is not float:
            raise Exception("Argumento inválido! Classe Psi só aceita divisão com inteiros ou reais.")
        # Faz uma cópia do objeto apenas dividindo <const> por <other>
        return Psi(self.n,self.l,self.ml,self.ms,self.const/other)
    
    def __add__(self:Psi,other:Psi) -> Union[Psi,Exception]:

        '''
        __add__
        -------

        Define a operação de soma que só é permitida entre dois objetos Psi. Nesse caso, um novo objeto será criado com os atributos dos objetos somados em lista nos atributos correspondentes.


        Exemplo
        -------
        >>> p1 = Psi(2,1,-1,1/2)/2
        >>> p2 = Psi(2,1,0,-1/2)/2
        >>> p = p1 + p2
        >>> print(p)
        0.500·│ n = 2, l = 1, ml = -1, ms = 0.5 ⧽ + 0.500·│ n = 2, l = 1, ml = 0, ms = -0.5 ⧽
        '''

        # Verifica se <other> também é um objeto Psi
        if type(other) is not Psi:
            raise Exception("Argumento inválido! A soma com Psi só pode ser feita com outro objeto Psi.")
        # Cria um novo objeto psi em um array 1D nos atributos com os atributos de <self> e <other> 
        return Psi(
            n       = np.array((self.n,other.n)).flatten(),
            l       = np.array((self.l,other.l)).flatten(),
            ml      = np.array((self.ml,other.ml)).flatten(),
            ms      = np.array((self.ms,other.ms)).flatten(),
            const   = np.array((self.const,other.const)).flatten()
        )
    
    def __sub__(self:Psi,other:Psi) -> Union[Psi,Exception]:

        '''
        __sub__
        -------

        Define a operação de subtração que só é permitida entre dois objetos Psi. Nesse caso, um novo objeto será criado com os atributos dos objetos da subtração em lista nos atributos correspondentes. Diferente de `__add__`, nesse caso, o atributo <const> de <other> será multiplicado por -1 já que está em uma subtração.


        Exemplo
        -------
        >>> p1 = Psi(2,1,-1,1/2)/2
        >>> p2 = Psi(2,1,0,-1/2)/2
        >>> p = p1 - p2
        >>> print(p)
        0.500·│ n = 2, l = 1, ml = -1, ms = 0.5 ⧽ - 0.500·│ n = 2, l = 1, ml = 0, ms = -0.5 ⧽
        '''

        # Verifica se <other> também é um objeto Psi
        if type(other) is not Psi:
            raise Exception("Argumento inválido! A subtração com Psi só pode ser feita com outro objeto Psi.")
        # Cria um novo objeto psi em um array 1D nos atributos com os atributos de <self> e <other> 
        return Psi(
            n       = np.array((self.n,other.n)).flatten(),
            l       = np.array((self.l,other.l)).flatten(),
            ml      = np.array((self.ml,other.ml)).flatten(),
            ms      = np.array((self.ms,other.ms)).flatten(),
            const   = np.array((self.const,-other.const)).flatten() # Multiplica por -1 o <const.other>
        )

    def normalize(self:Psi) -> Psi:

        '''
        normalize
        ---------

        Normaliza o objeto Psi, isto é, garante que a soma das constantes multiplicativas (<const>) ao quadrado resulte em 1. Normalmente, essa operação só será útil caso 2 ou mais objetos Psi tenham sido somados/subtraídos e não é garantido que estes estejam normalizados. Esse é um método usado internamente para outras funções que dependem dessa normalização.


        Retorna
        -------

        normalized_psi : Psi
            Novo objeto Psi com o atributo <const> normalizado.

        
        Exemplo
        -------
        >>> from quantum_psi import Psi
        >>> p1 = Psi(2,1,-1,1/2)
        >>> p2 = Psi(2,1,0,-1/2)
        >>> p = p1 + p2             # Operação não foi normalizada
        >>> print(p)                # Resultado sem normalização
        1.000·│ n = 2, l = 1, ml = -1, ms = 0.5 ⧽ + 1.000·│ n = 2, l = 1, ml = 0, ms = -0.5 ⧽
        >>> print(p.normalize())    # Normalizado
        0.500·│ n = 2, l = 1, ml = -1, ms = 0.5 ⧽ + 0.500·│ n = 2, l = 1, ml = 0, ms = -0.5 ⧽
        '''

        # Retorna uma cópia do objeto fazendo a operação: const = const/sum(const^2)
        return Psi(self.n,self.l,self.ml,self.ms,self.const/np.sum(self.const**2))

    def R(self:Psi,a0:Union[int,float]=0.529) -> Callable[[Union[float,Sequence[float]]], Union[float,Sequence[float]]]:

        '''
        R
        -

        Retorna a função de onda radial, R_nl(r), para o objeto Psi.

        Atenção: nessa operação normaliza o objeto antes de fazer a operação (reescreve).


        Parâmetros
        ----------

        a0 : int | float = 0.529
            (Opcional) Valor utilizado para o raio de Bohr.

        
        @a0: (Opcional)  Valor do raio de Bohr.


        Retorna
        -------

        R_nl : Callable[ [ float | Sequence[float] ], float | Sequence[float] ]
            Função de onda radial dependente de 'r' para o objeto Psi.
        

        Exemplo
        -------
        >>> import numpy as np
        >>> from quantum_psi import Psi
        >>> R_10 = Psi(1,0,0,0).R()
        >>> r = np.linspace(0,5,1000)
        >>> R_10(r)
        array([5.19812223e+00, 5.14917339e+00, 5.10068547e+00, 5.05265415e+00,...
        '''
        
        # Normaliza o objeto
        self = self.normalize() # Atualmente o objeto é reescrito

        # Função interna para gerar a função de onda radial que será chamada conforme a necessidade
        def _R(const:Union[int,float],n:int,l:int,a0:float) -> Callable[[float],float]:
            return lambda r: const*(np.sqrt((2/(n*a0))**3*factorial(n-l-1)/(2*n*factorial(n+l)))*np.exp(-r/(n*a0))*(2*r/(n*a0))**l*genlaguerre(n-l-1,2*l+1)(r))

        # Se <const> for iterável
        if hasattr(self.const,'__iter__'):
            # Função de saída vai ser uma soma das funções de onda radial para cada item com o tamanho da lista de <const>
            return lambda r: sum(_R(self.const[i],self.n[i],self.l[i],a0)(r) for i in range(len(self.const)))
        # Se <const> for um único valor
        else:
            # Retorna a operação de _R
            return _R(self.const,self.n,self.l,a0)

    def P_rad(self:Psi,a0:Union[int,float]=0.529):

        '''
        P_rad
        -----

        Retorna a função de densidade de probabilidade radial para o objeto Psi: r²|R_nl(r)|².
        
        Atenção: nessa operação normaliza o objeto antes de fazer a operação (reescreve).


        Parâmetros
        ----------

        a0 : int | float = 0.529
            (Opcional) Valor utilizado para o raio de Bohr.

        
        @a0: (Opcional)  Valor do raio de Bohr.


        Retorna
        -------

        P_rad_nl : Callable[ [ float | Sequence[float] ], float | Sequence[float] ]
            Função de densidade de probabilidade radial dependente de 'r' para o objeto Psi.
        

        Exemplo
        -------
        >>> import numpy as np
        >>> from quantum_psi import Psi
        >>> P_rad_10 = Psi(1,0,0,0).P_rad()
        >>> r = np.linspace(0,5,1000)
        >>> P_rad_10(r)
        array([0.00000000e+00, 6.64177354e-04, 2.60691044e-03, 5.75560109e-03,...
        '''

        # Retorna uma função anônima dependente de 'r' chamando a função R
        return lambda r: r**2*np.abs(self.R(a0)(r))**2

    def wave_function(self:Psi,a0:Union[int,float]=0.529) -> Callable[[Union[float,Sequence[float]],Union[float,Sequence[float]],Union[float,Sequence[float]]], Union[complex,Sequence[complex]]]:

        '''
        wave_function
        -------------

        Retorna a função de onda de posição normalizada do objeto Psi em coordenadas esféricas (r,theta,phi).

        Atenção: nessa operação normaliza o objeto antes de fazer a operação (reescreve).


        Parâmetros
        ----------

        a0 : int | float = 0.529
            (Opcional) Valor utilizado para o raio de Bohr.

        
        @a0: (Opcional)  Valor do raio de Bohr.


        Retorna
        -------

        psi_nlm : Callable[ [ float | Sequence[float], float | Sequence[float], float | Sequence[float] ], complex | Sequence[complex] ]
        
            Função de onda de posição normalizada dependente de 'r', 'theta' e 'phi' para o objeto Psi.


            Parâmetros
            ----------

            r : float | Sequence[float]
                Raio, 0 < r < inf.

            theta : float | Sequence[float]
                Coordenada azimutal, 0 < theta < 2π.

            phi : float | Sequence[float]
                Coordenada polar, 0 < phi < π.
            

            Retorna
            -------

            result : complex | Sequence[complex]
                Resultado da operação dado os valores nas coordenadas esféricas.
        

        Exemplo
        -------
        >>> from quantum_psi import Psi
        >>> psi_100 = Psi(1,0,0,0).wave_function()
        >>> r, theta, phi = 1, 3, 2
        >>> psi_100(r,theta,phi)
        (0.22144659149706755+0j)
        '''

        # Normaliza o objeto
        self = self.normalize() # Atualmente o objeto é reescrito

        # Função interna para gerar a função de onda respectiva
        def _wave_function(const:Union[int,float],n:int,l:int,ml:int,a0:float) -> Callable[[Union[float,Sequence[float]],Union[float,Sequence[float]],Union[float,Sequence[float]]], Union[complex,Sequence[complex]]]:

            '''
            _wave_function
            --------------

            Função interna para gerar a função de onda respectiva.


            Parâmetros
            ----------

            const : int | float
                Constante que multiplica a função como um todo.

            n : int
                Número quântico principal.
            
            l : int
                Número quântico secundário.
            
            ml : int
                Número quântico magnético.
            
            a0 : float
                Raio de Bohr.
            

            @const: Constante que multiplica a função como um todo.
            @n: Número quântico principal.
            @l: Número quântico secundário.
            @ml: Número quântico magnético.
            @a0: Raio de Bohr.


            Retorna
            -------

            phi_nlm : Callable[[float | Sequence[float], float | Sequence[float], float | Sequence[float]], complex | Sequence[complex]]

                Função de onda de posição normalizada dependente de 'r', 'theta' e 'phi' para o objeto Psi.


                Parâmetros
                ----------

                r : float | Sequence[float]
                    Raio, 0 < r < inf.

                theta : float | Sequence[float]
                    Coordenada azimutal, 0 < theta < 2π.

                phi : float | Sequence[float]
                    Coordenada polar, 0 < phi < π.
                

                Retorna
                -------

                result : complex | Sequence[complex]
                    Resultado da operação dado os valores nas coordenadas esféricas.
            '''

            return lambda r,theta,phi: const*(np.sqrt((2/(n*a0))**3*factorial(n-l-1)/(2*n*factorial(n+l)))*np.exp(-r/(n*a0))*(2*r/(n*a0))**l*genlaguerre(n-l-1,2*l+1)(r))*sph_harm(ml,l,theta,phi)

        # Se <const> for iterável
        if hasattr(self.const,'__iter__'):
            # Aplica a soma das funções de onda para cada item
            return lambda r,theta,phi: sum(_wave_function(self.const[i],self.n[i],self.l[i],self.ml[i],a0)(r,theta,phi) for i in range(len(self.const)))
        # Se <const> for um único valor
        else:
            # Aplica apenas a função internas
            return _wave_function(self.const,self.n,self.l,self.ml,a0)

    def wave_function_prob(self:Psi,a0:float=0.529,real:bool=True) -> Callable[[Union[float,Sequence[float]],Union[float,Sequence[float]],Union[float,Sequence[float]]], Union[float,Sequence[float]]]:

        '''
        wave_function_prob
        ------------------

        Retorna a função de densidade de probabilidade da função de onda do objeto Psi em coordenadas esféricas (r,theta,phi).

        Atenção: nessa operação normaliza o objeto antes de fazer a operação (reescreve).


        Parâmetros
        ----------

        a0 : float = 0.529
            (Opcional) Valor utilizado para o raio de Bohr.

        real : bool = True
            (Opcional) Considerar apenas a parte real para o cálculo da probabilidade. A escolha de `True` ser o valor padrão é para garantir os gráficos adequados para aqueles que forem plotar essa função.

        
        @a0: (Opcional)  Valor do raio de Bohr.
        @real: (Opcional) Considerar apenas a parte real para o cálculo da probabilidade.


        Retorna
        -------

        prob_psi_nlm : Callable[ [ float | Sequence[float], float | Sequence[float], float | Sequence[float] ], float | Sequence[float] ]
        
            Função de densidade de probabilidade da onda de posição normalizada dependente de 'r', 'theta' e 'phi' para o objeto Psi.


            Parâmetros
            ----------

            r : float | Sequence[float]
                Raio, 0 < r < inf.

            theta : float | Sequence[float]
                Coordenada azimutal, 0 < theta < 2π.

            phi : float | Sequence[float]
                Coordenada polar, 0 < phi < π.
            

            Retorna
            -------

            result : float | Sequence[float]
                Resultado da operação dado os valores nas coordenadas esféricas.
        

        Exemplo
        -------
        >>> from quantum_psi import Psi
        >>> prob_psi_100 = Psi(1,0,0,0).wave_function_prob(real=True)
        >>> r, theta, phi = 1, 3, 2
        >>> prob_psi_100(r,theta,phi)
        0.04903859288566911
        '''

        # |wave_fuction.real|²
        if real:
            return lambda r,theta,phi: np.abs(self.wave_function(a0)(r,theta,phi).real)**2
        # |wave_fuction|²
        else:
            return lambda r,theta,phi: np.abs(self.wave_function(a0)(r,theta,phi))**2

    def mean_r(self:Psi,a0:Union[int,float]=0.529) -> float:

        '''
        mean_r
        ------

        Calcula o raio médio, valor esperado do raio, a partir da integral de zero até infinito de r*P_rad(r).

        Atenção: nessa operação normaliza o objeto antes de fazer a operação (reescreve).


        Parâmetros
        ----------

        a0 : int | float = 0.529
            (Opcional) Valor utilizado para o raio de Bohr.
        
        @a0: (Opcional)  Valor do raio de Bohr.


        Retorna
        -------

        mean_radius : float
            Valor esperado do raio ou o valor médio do raio para o objeto Psi.
        '''

        # Integral de zero até infinito de r*P_rad(r) onde P_rad é a função de densidade de probabilidade radial
        return quad(lambda r: r*self.P_rad(a0)(r),0,np.inf)[0]

    def r_bohr(self:Psi,a0:Union[int,float]=0.529) -> Union[float,Exception]:

        '''
        r_bohr
        ------

        Calcula o raio do objeto Psi a partir do modelo de Bohr.


        Parâmetros
        ----------

        a0 : int | float = 0.529
            (Opcional) Valor utilizado para o raio de Bohr.
        
        @a0: (Opcional)  Valor do raio de Bohr.


        Retorna
        -------

        radius_bohr : float | Exception
            Retorna o raio a partir do modelo de Bohr. Caso o objeto tenha mais de um valor para cada atributo e estes são distintos em 'n', número quântico principal, será retornado um erro.
        '''

        # Se <const> for iterável
        if hasattr(self.const,'__iter__'):
            # Se os valores em <n> forem diferentes
            if not np.equal(*self.n):
                raise Exception("Operação inválida! Objeto Psi apresenta valores diferentes para 'n'.")
            return self.n[0]**2*a0
        # Se <const> for um único valor
        else:
            return self.n**2*a0
