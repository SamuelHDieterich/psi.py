'''

Conjunto de funções auxiliares para a criação de gráficos utilizando o pacote quantum-psi.

'''

# BIBLIOTECAS
from __future__ import annotations

from typing import Sequence, Tuple, Union

import numpy as np

from .._core import Psi


# FUNÇÕES
def sphere_points(rmax:Union[int,float],n_points:int) -> Tuple[Union[float,Sequence[float]],Union[float,Sequence[float]],Union[float,Sequence[float]]]:

    '''
    sphere_points
    -------------

    Gera N pontos distribuídos aleatoriamente em coordenadas esféricas dado um raio máximo.

    Observação: Os pontos são gerados a partir do pseudo random generator do NumPy. Caso deseje configurar uma seed, use o comando:
    >>> import numpy as np
    >>> seed = 938792   # Um número inteiro qualquer
    >>> np.random.seed(seed)


    Parâmetros
    ----------

    rmax : int | float
        Raio máximo para os pontos gerados.
    
    n_points : int
        Número de pontos a ser gerados.
    

    @rmax: Raio máximo.
    @n_points: Número de pontos.


    Retorna
    -------

    points : Tuple[ float | Sequence, float | Sequence, float | Sequence ]
        Pontos distribuídos aleatoriamente em coordenadas esféricas (r, theta, phi).


    Exemplo
    -------
    >>> import quantum_psi.graph as pgraph
    >>> print(pgraph.sphere_points(1,100))  # Geração aleatória, resultados podem variar.
    '''

    return rmax*np.cbrt(np.random.random_sample(n_points)), np.random.random_sample(n_points)*(2*np.pi), np.arccos(np.random.random_sample(n_points)*2-1)

def sph2rec(r:Union[float,Sequence[float]],theta:Union[float,Sequence[float]],phi:Union[float,Sequence[float]]) -> Tuple[Union[float,Sequence[float]],Union[float,Sequence[float]],Union[float,Sequence[float]]]:

    '''
    sph2rec
    -------

    Converte pontos em coordendas esféricas para coordenadas retangulares.

    (r, theta, phi) --> (x, y, z)


    Parâmetros
    ----------

    r : float | Sequence[float]
        Raio, 0 < r < inf.

    theta : float | Sequence[float]
        Coordenada azimutal, 0 < theta < 2π.

    phi : float | Sequence[float]
        Coordenada polar, 0 < phi < π.


    @r: Raio, 0 < r < inf.
    @theta: Coordenada azimutal, 0 < theta < 2π.
    @phi: Coordenada polar, 0 < phi < π.


    Retorna
    -------

    points_rec : Tuple[ float | Sequence[float], float | Sequence[float], float | Sequence[float] ]
        Pontos equivalentes em coordenadas retangulares.

    
    Exemplo
    -------
    >>> import quantum_psi.graph as pgraph
    >>> r,theta,phi = pgraph.sphere_points(1,100)
    >>> x,y,z = pgraph.sph2rec(r,theta,phi)
    '''

    return r*np.cos(theta)*np.sin(phi), r*np.sin(theta)*np.sin(phi), r*np.cos(phi)

def gen_data(psi:Psi,rmax:Union[int,float],n_points:int,a0:Union[int,float]=0.529,xyz:bool=True) -> Tuple[Union[float,Sequence[float]],Union[float,Sequence[float]],Union[float,Sequence[float]]]:

    '''
    gen_data
    --------

    Função para a geração de dados da densidade de probabilidade da função de onda de um dado objeto Psi.


    Parâmetros
    ----------

    psi : Psi
        Objeto da classe Psi.
    
    rmax : int | float
        Raio máximo para os pontos gerados.
    
    n_points : int
        Número de pontos a ser gerados.

    a0 : int | float = 0.529
        (Opcional) Valor utilizado para o raio de Bohr.

    xyz : bool = True
        (Opcional) Se True, converte os pontos para coordenadas retangulares, já em False, permanece em coordenadas esféricas.

    
    @psi: Objeto da classe Psi.
    @rmax: Raio máximo.
    @n_points: Número de pontos.
    @a0: (Opcional) Raio de Bohr.
    @xyz: (Opcional) Converte para coordenadas retangulares.


    Retorna
    -------

    (coord1, coord2, coord3, value) : Tuple[ float | Sequence[float], float | Sequence[float], float | Sequence[float], float | Sequence[float] ]
        Tupla com os dados das coordenadas (esféricas ou retangulares) e os dados calculados.
    

    Exemplo
    -------
    >>> from quantum_psi import Psi
    >>> import quantum_psi.quantum as pgraph
    >>> p = Psi(2,1,0,1/2)
    >>> x,y,z,value = pgraph.gen_data(p,8,100000)
    '''

    # Gera os pontos em coordenadas esféricas
    r,theta,phi = sphere_points(rmax,n_points)
    # Calcula o resultado na função de densidade de probabilidade da função de onde
    value = psi.wave_function_prob(a0)(r,theta,phi)
    # Se precisa converter em coordenadas retangulares
    if xyz:
        x,y,z = sph2rec(r,theta,phi)
        # Retorna em coordenadas retangulares
        return x,y,z,value
    else:
        # Retorna em coordenadas esféricas
        return r,theta,phi,value

def clean_data(value:Union[float,Sequence[float]],*coord:list[Union[float,Sequence[float]]],**kwargs:dict[str,Union[float,bool]]) -> list[Union[float,Sequence[float]]]:

    '''
    clean_data
    ----------

    Limpa um conjunto de dados eliminando os valores abaixo de um certo limiar (epsilon). Essa função é especialmente útil para retirar pontos pouco relevantes em um plot, poupando recursos, especialmente se o tamanho do ponto, caso seja um gráfico de dispersão, for proporcional a <value>.


    Parâmetros
    ----------

    value : float | Sequence[float]
        Conjunto de dados que será usado para comparação (resultado de uma função, por exemplo). Se um dado valor aqui for abaixo do limiar (epsilon), este ponto será desconsiderado/deletado.
    
    *coord : list[ float | Sequence[float] ]
        Coordenadas relacionadas com o conjunto de dados <value>, isto é, se um ponto em <value> for deletado, o mesmo acontecerá aqui (na mesma posição).
    
    **kwargs : dict[str,Any]

        epsilon : float = 1E-3
            (Opcional) Valor a ser usado na comparação. Valores abaixo de epsilon serão desconsiderados/deletados.
    
        normalize : bool = True
            (Opcional) Se True, normaliza <value> antes de aplicar a operação.

        norm_value : bool = True
            (Opcional) Se True, retorna o <value> normalizado. Caso contrário, retorna no formato original. Só terá efeito se o parâmetro `normalize` também for verdadeiro.


    @value: Dados para comparação (resultado de uma função).
    @*coord: Coordenadas relacionadas com <value>.
    @**kwargs: Permite configurar: <epsilon>, <normalize> e <norm_value>.


    Retorna
    -------

    [new_value, *new_coord] : list[ float | Sequence[float] ]
        Retorna em uma lista o novo conjunto de dados, <value> e <*coords>, removendo os valores abaixo de um certo limiar.

    
    Exemplo
    -------
    >>> from quantum_psi import Psi
    >>> import quantum_psi.quantum as pgraph
    >>> p = Psi(2,1,0,1/2)
    >>> x,y,z,value = pgraph.gen_data(p,8,100000)
    >>> value,x,y,z = pgraph.clean_data(value)
    '''

    # Pega os valores de <**kwargs> ou seta os padrões
    epsilon     = kwargs.pop('epsilon', 1E-3)
    normalize   = kwargs.pop('normalize',True)
    norm_value  = kwargs.pop('norm_value',True)

    # Copia de <value>
    new_value = value
    if normalize:
        # Normaliza <new_value>
        new_value = value/np.max(value)
    # Encontra os "zeros"
    zeros = np.where(new_value<epsilon)[0]
    if norm_value:
        # Retorna com o valor de <new_value> que, a princípio, é a versão normalizada
        return [np.delete(new_value,zeros)]+[np.delete(i,zeros) for i in coord]
    else:
        # Retorna com <value>, sem normalização
        return [np.delete(value,zeros)]+[np.delete(i,zeros) for i in coord]

def plot_data(psi:Psi,rmax:Union[int,float],n_points:int,a0:float=0.529,xyz:bool=True,**kwargs:dict[str,Union[float,bool]]) -> tuple[Union[float,Sequence[float]],Union[float,Sequence[float]],Union[float,Sequence[float]],Union[float,Sequence[float]]]:

    '''
    plot_data
    ---------

    Script para automatizar o processo de gerar dados para plotagem: gera os dados (gen_data) e "limpa" os mesmos (clean_data).

    Parâmetros
    ----------

    psi : Psi
        Objeto da classe Psi.
    
    rmax : int | float
        Raio máximo para os pontos gerados.
    
    n_points : int
        Número de pontos a ser gerados.

    a0 : int | float = 0.529
        (Opcional) Valor utilizado para o raio de Bohr.

    xyz : bool = True
        (Opcional) Se True, converte os pontos para coordenadas retangulares, já em False, permanece em coordenadas esféricas.

    value : float | Sequence[float]
        Conjunto de dados que será usado para comparação (resultado de uma função, por exemplo). Se um dado valor aqui for abaixo do limiar (epsilon), este ponto será desconsiderado/deletado.
    
    *coord : list[ float | Sequence[float] ]
        Coordenadas relacionadas com o conjunto de dados <value>, isto é, se um ponto em <value> for deletado, o mesmo acontecerá aqui (na mesma posição).
    
    **kwargs : dict[str,Any]

        epsilon : float = 1E-3
            (Opcional) Valor a ser usado na comparação. Valores abaixo de epsilon serão desconsiderados/deletados.
    
        normalize : bool = True
            (Opcional) Se True, normaliza <value> antes de aplicar a operação.

        norm_value : bool = True
            (Opcional) Se True, retorna o <value> normalizado. Caso contrário, retorna no formato original. Só terá efeito se o parâmetro `normalize` também for verdadeiro.


    @psi: Objeto da classe Psi.
    @rmax: Raio máximo.
    @n_points: Número de pontos.
    @a0: (Opcional) Raio de Bohr.
    @xyz: (Opcional) Converte para coordenadas retangulares.
    @value: Dados para comparação (resultado de uma função).
    @*coord: Coordenadas relacionadas com <value>.
    @**kwargs: Permite configurar: <epsilon>, <normalize> e <norm_value>.


    Retorna
    -------

    (coord1, coord2, coord3, value) : tuple[ float | Sequence[float], float | Sequence[float], float | Sequence[float], float | Sequence[float] ]
        Retorna em uma tupla com os dados gerados e já limpos.

    
    Exemplo
    -------
    >>> from quantum_psi import Psi
    >>> import quantum_psi.quantum as pgraph
    >>> p = Psi(2,1,0,1/2)
    >>> x,y,z,value = pgraph.gen_data(p,8,100000)
    >>> value,x,y,z = pgraph.clean_data(value)
    '''

    # Gera os dados
    *coords,value = gen_data(psi,rmax,n_points,a0,xyz)
    # Limpa os dados
    value,*coords = clean_data(value,*coords,**kwargs)
    # Retorna as coordenadas e os valores resultantes
    return *coords,value
