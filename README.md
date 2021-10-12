<div align="center">

# Quantum-psi

Conjunto de ferramentas para estudo de funções de onda na física quântica.

<br>

<img src="https://img.shields.io/static/v1?label=version&message=0.1-alpha0&color=ae35ea&style=flat"/>
&nbsp
<img src="https://img.shields.io/static/v1?label=license&message=GPLv3&color=ae35ea&style=flat"/>

<br>

</div>

## Objetivo

Motivado pelos trabalhos e atividades computacionais da cadeira de fundamentos de física quântica, decidi, até como um desafio pessoal, criar um pacote didático para o estudo desse curso. Para tanto, o material foi pensado em deixar tudo muito bem documentado e direto ao ponto. Dessa forma, a biblioteca conta com funções básicas para aqueles que querem ir no _quick and dirty_, mas também proporcione que os mais curiosos explorem as possibilidades.

## Pré-requisitos

Até o presente momento, não foram realizados testes com versões do Python diferentes da 3.9.7, dessa forma, suponho que versões >=3.9 devem funcionar nativamente. As funções e bibliotecas utilizadas devem, na sua maioria, funcionar com interpretadores mais antigos. Contudo, para a escrita das funções, adicionei _type hints_ - uma vez que o objetivo é ser uma biblioteca bem documentada - que tentem a variar bastante de uma versão para outra.  
Caso queira utilizar a biblioteca e não seja possível utilziar uma versão do python mais recente, sugiro clonar esse repositório e apagar manualmente os _type hints_.

- **Python**: >=3.9
- **Bibliotecas**: NumPy, Scipy, Plotly (opcional) 

## Instalação

O pacote **quantum-psi** está hospedado no Pypi. Assim, é possível instalar essa biblioteca em um ambiente virtual utilizando o `pip` (utilize o comando adequado para o seu sistema):

```
~ pip install quantum-psi
~ pip3 install quantum-psi
~ python -m pip install quantum-psi
~ python3 -m pip install quantum-psi
```

## Licença

A licença para esse projeto é a GPLv3 que, em suma, diz que o código é aberto e aqueles que foram usar esse programa devem deixar os seus projetos abertos também.