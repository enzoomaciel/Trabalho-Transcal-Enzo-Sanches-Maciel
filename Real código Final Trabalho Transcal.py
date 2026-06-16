# Trabalho - Métodos Numéricos na Condução de Calor 2D Transiente
# Universidade Federal de Santa Catarina - UFSC
# Departamento de Engenharia Mecânica - EMC
# Aluno: Enzo Sanches Maciel
# Professor: Joel Boeng

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter
from pathlib import Path
import time

# Pasta onde serão salvos os gráficos e o GIF
pasta_saida = Path(__file__).resolve().parent

# Condições iniciais e tamanho da malha
Ti = 850.0          # Temperatura inicial da barra [°C]
Tinf = 25.0         # Temperatura da água [°C]
Delta = 0.0020   # Espaçamento entre nós [m]

# Dimensões da geometria em L [m]
Lx = 0.10                  # Comprimento total na direção x
Ly = 0.08                  # Altura total na direção y
largura_vertical = 0.03    # Largura do trecho vertical do L
altura_base = 0.05         # Altura da base inferior
espessura_camada = 0.01    # Espessura da camada sinterizada

# Propriedades térmicas do material A: aço carbono
k_a = 45.0          # Condutividade térmica [W/mK]
rho_a = 7800.0      # Densidade [kg/m³]
cp_a = 460.0        # Calor específico [J/kgK]

# Propriedades térmicas do material B: bronze sinterizado
k_b = 8.0           # Condutividade térmica [W/mK]
rho_b = 5200.0      # Densidade [kg/m³]
cp_b = 380.0        # Calor específico [J/kgK]

# Parâmetros numéricos e de contorno
h_inicial = 1500.0        # Coeficiente de convecção inicial [W/m²K]
fator_seguranca = 0.95    # Reduz o dt máximo para garantir estabilidade
T_limite = 120.0          # Temperatura limite do item (a) [°C]
tolerancia_regime = 1e-4  # Critério de parada para regime permanente [°C por passo]
max_passos = 600000       # Número máximo de iterações

def idx(x):
    # Converte uma dimensão física em índice da malha.
    # Exemplo: se x = 0,10 m e Delta = 0,005 m, então o índice é 20.
    valor = x/Delta
    if abs(valor - round(valor)) > 1e-9:
        raise ValueError("Delta precisa dividir exatamente as dimensões da geometria.")
    return int(round(valor))

# Conversão das dimensões físicas para índices da matriz
ix10 = idx(Lx)
iy8 = idx(Ly)
ix3 = idx(largura_vertical)
iy5 = idx(altura_base)
ie = idx(espessura_camada)

if ie < 2:
    raise ValueError("Delta deve gerar pelo menos 2 divisões na espessura da camada sinterizada.")

ix_int = ix3 - ie
iy_int = iy5 - ie
nx = ix10 + 1
ny = iy8 + 1

# Pontos de interesse usados nos gráficos do item (d)
pontos = {
    f"A [{ix3},{iy8}]": (ix3, iy8),
    f"B [{ix10},{iy5}]": (ix10, iy5),
    f"C [{ix_int},{iy_int}]": (ix_int, iy_int),
    f"D [0,0]": (0, 0)
}

def formato():
    # Usei para criar uma matriz indicando quais posições pertencem à peça.
    # True representa nó existente na geometria em L.
    # False representa região fora da peça.
    v = np.zeros((nx, ny), dtype=bool)
    v[0:ix3+1, 0:iy8+1] = True
    v[ix3+1:ix10+1, 0:iy5+1] = True
    return v

def campo_inicial():
    # Cria a matriz de temperaturas.
    # np.nan é usado fora da peça.
    T = np.full((nx, ny), np.nan)
    T[formato()] = Ti
    return T

def nos_da_camada():
    # Identifica os nós pertencentes à camada sinterizada.
    s = set()
    for m in range(ix_int+1, ix3+1):
        for n in range(iy_int+1, iy8+1):
            s.add((m, n))
    for m in range(ix3, ix10+1):
        for n in range(iy_int+1, iy5+1):
            s.add((m, n))
    for n in range(iy_int, iy8+1):
        s.add((ix_int, n))
    for m in range(ix_int, ix10+1):
        s.add((m, iy_int))
    v = formato()
    return sorted([no for no in s if v[no]])

def parametros(h):
    # Calcula o passo de tempo permitido pelo critério de estabilidade.
    # Como o método é explícito, o Delta t precisa ser o mínimo calculado,
    # entre todos os tipos de nós da geometria.
    dt_lim = np.array([
        (rho_a*cp_a*Delta**2)/(4*k_a+4*h*Delta),
        (rho_a*cp_a*Delta**2)/(4*k_a+2*h*Delta),
        ((rho_a*cp_a+rho_b*cp_b)*Delta**2)/(4*(k_a+k_b)+4*h*Delta),
        (rho_b*cp_b*Delta**2)/(4*k_b+2*h*Delta),
        (rho_b*cp_b*Delta**2)/(4*k_b+4*h*Delta),
        (rho_a*cp_a*Delta**2)/(4*k_a+2*h*Delta),
        (rho_a*cp_a*Delta**2)/(4*k_a+2*h*Delta),
        (rho_a*cp_a*Delta**2)/(4*k_a),
        (rho_a*cp_a*Delta**2)/(4*k_a+2*h*Delta),
        (rho_a*cp_a*Delta**2)/(4*k_a+2*h*Delta),
        ((rho_a*cp_a+rho_b*cp_b)*Delta**2)/(4*(k_a+k_b)+4*h*Delta),
        (rho_b*cp_b*Delta**2)/(4*k_b+2*h*Delta),
        (rho_b*cp_b*Delta**2)/(4*k_b+4*h*Delta),
        (rho_b*cp_b*Delta**2)/(4*k_b+2*h*Delta),
        (3*rho_b*cp_b*Delta**2)/(12*k_b+4*h*Delta),
        (rho_b*cp_b*Delta**2)/(4*k_b+2*h*Delta),
        (rho_b*cp_b*Delta**2)/(4*k_b),
        ((rho_a*cp_a+rho_b*cp_b)*Delta**2)/(4*(k_a+k_b)),
        ((3*rho_a*cp_a+rho_b*cp_b)*Delta**2)/(4*(3*k_a+k_b)),
        ((rho_a*cp_a+rho_b*cp_b)*Delta**2)/(4*(k_a+k_b)),
        (rho_a*cp_a*Delta**2)/(4*k_a)
    ])
    nomes = [
        f"Nó [0,{iy8}]",
        f"Nós [1,{iy8}] até [{ix_int-1},{iy8}]",
        f"Nó [{ix_int},{iy8}]",
        f"Nós [{ix_int+1},{iy8}] até [{ix3-1},{iy8}]",
        f"Nó [{ix3},{iy8}]",
        f"Nós [0,1] até [0,{iy8-1}]",
        "Nó [0,0]",
        f"Nós [1,0] até [{ix10-1},0]",
        f"Nó [{ix10},0]",
        f"Nós [{ix10},1] até [{ix10},{iy_int-1}]",
        f"Nó [{ix10},{iy_int}]",
        f"Nós [{ix10},{iy_int+1}] até [{ix10},{iy5-1}]",
        f"Nó [{ix10},{iy5}]",
        f"Nós [{ix3+1},{iy5}] até [{ix10-1},{iy5}]",
        f"Nó [{ix3},{iy5}]",
        f"Nós [{ix3},{iy5+1}] até [{ix3},{iy8-1}]",
        "Nós internos da camada sinterizada",
        "Nós internos da interface vertical",
        f"Nó [{ix_int},{iy_int}]",
        "Nós internos da interface horizontal",
        "Nós internos do material A"
    ]
    ind = int(np.argmin(dt_lim))
    dt_min = float(dt_lim[ind])
    dt = fator_seguranca*dt_min
    Foa = k_a*dt/(rho_a*cp_a*Delta**2)
    Fob = k_b*dt/(rho_b*cp_b*Delta**2)
    Ha = h*dt/(rho_a*cp_a*Delta)
    Hb = h*dt/(rho_b*cp_b*Delta)
    return h, dt, dt_min, nomes[ind], Foa, Fob, Ha, Hb

def passo(T, par):
    # Atualiza a temperatura de todos os nós em um avanço temporal.
    h, dt, dt_min, no_limitante, Foa, Fob, Ha, Hb = par
    N = T.copy()

    # EQUAÇÕES DOS NÓS DO BLOCO SUPERIOR
    # Parte superior do perfil em L, incluindo aço, camada sinterizada
    # e nós de interface entre os materiais A e B.
    N[0,iy8] = 2*Foa*(T[1,iy8]+T[0,iy8-1]) + 4*Ha*Tinf + (1-4*Foa-4*Ha)*T[0,iy8]

    for m in range(1, ix_int):
        N[m,iy8] = Foa*(T[m-1,iy8]+T[m+1,iy8]+2*T[m,iy8-1]) + 2*Ha*Tinf + (1-4*Foa-2*Ha)*T[m,iy8]

    c = 4*dt/((rho_a*cp_a+rho_b*cp_b)*Delta**2)
    N[ix_int,iy8] = c*((k_a/2)*T[ix_int-1,iy8]+(k_b/2)*T[ix_int+1,iy8]+((k_a+k_b)/2)*T[ix_int,iy8-1]+h*Delta*Tinf) + (1-c*(k_a+k_b+h*Delta))*T[ix_int,iy8]

    for m in range(ix_int+1, ix3):
        N[m,iy8] = Fob*(T[m-1,iy8]+T[m+1,iy8]+2*T[m,iy8-1]) + 2*Hb*Tinf + (1-4*Fob-2*Hb)*T[m,iy8]

    N[ix3,iy8] = 2*Fob*(T[ix3-1,iy8]+T[ix3,iy8-1]) + 4*Hb*Tinf + (1-4*Fob-4*Hb)*T[ix3,iy8]

    # EQUAÇÕES DA LATERAL ESQUERDA E DO PONTO D
    # Superfície lateral esquerda com convecção e canto inferior esquerdo.
    for n in range(1, iy8):
        N[0,n] = Foa*(T[0,n+1]+T[0,n-1]+2*T[1,n]) + 2*Ha*Tinf + (1-4*Foa-2*Ha)*T[0,n]

    N[0,0] = 2*Foa*(T[0,1]+T[1,0]) + 2*Ha*Tinf + (1-4*Foa-2*Ha)*T[0,0]


    # EQUAÇÕES DA FACE INFERIOR
    # A face inferior é considerada adiabática, por isso não há
    # troca de calor pela parte de baixo.
    
    for m in range(1, ix10):
        N[m,0] = Foa*(T[m-1,0]+T[m+1,0]+2*T[m,1]) + (1-4*Foa)*T[m,0]

    N[ix10,0] = 2*Foa*(T[ix10-1,0]+T[ix10,1]) + 2*Ha*Tinf + (1-4*Foa-2*Ha)*T[ix10,0]


    # EQUAÇÕES DA LATERAL DIREITA
    # Inclui os nós de material A, os nós de interface e os nós
    # da camada sinterizada na extremidade direita.
    
    for n in range(1, iy_int):
        N[ix10,n] = Foa*(T[ix10,n+1]+T[ix10,n-1]+2*T[ix10-1,n]) + 2*Ha*Tinf + (1-4*Foa-2*Ha)*T[ix10,n]

    N[ix10,iy_int] = c*((k_b/2)*T[ix10,iy_int+1]+(k_a/2)*T[ix10,iy_int-1]+((k_a+k_b)/2)*T[ix10-1,iy_int]+h*Delta*Tinf) + (1-c*(k_a+k_b+h*Delta))*T[ix10,iy_int]

    for n in range(iy_int+1, iy5):
        N[ix10,n] = Fob*(T[ix10,n+1]+T[ix10,n-1]+2*T[ix10-1,n]) + 2*Hb*Tinf + (1-4*Fob-2*Hb)*T[ix10,n]

    N[ix10,iy5] = 2*Fob*(T[ix10-1,iy5]+T[ix10,iy5-1]) + 4*Hb*Tinf + (1-4*Fob-4*Hb)*T[ix10,iy5]

    # EQUAÇÕES DA SUPERFÍCIE HORIZONTAL DA CAMADA SINTERIZADA
    # Parte superior da base, em contato com a água por convecção.

    for m in range(ix3+1, ix10):
        N[m,iy5] = Fob*(T[m-1,iy5]+T[m+1,iy5]+2*T[m,iy5-1]) + 2*Hb*Tinf + (1-4*Fob-2*Hb)*T[m,iy5]

    N[ix3,iy5] = (2/3)*Fob*(T[ix3,iy5+1]+T[ix3+1,iy5]+2*T[ix3-1,iy5]+2*T[ix3,iy5-1]) + (4/3)*Hb*Tinf + (1-4*Fob-(4/3)*Hb)*T[ix3,iy5]


    # EQUAÇÕES DA SUPERFÍCIE VERTICAL DA CAMADA SINTERIZADA
    # Região vertical interna do L, também submetida à convecção.
    for n in range(iy5+1, iy8):
        N[ix3,n] = Fob*(T[ix3,n+1]+T[ix3,n-1]+2*T[ix3-1,n]) + 2*Hb*Tinf + (1-4*Fob-2*Hb)*T[ix3,n]


    # NÓS INTERNOS DA CAMADA SINTERIZADA
    # Nós de material B sem convecção direta, apenas condução
    # para os quatro vizinhos.

    for m in range(ix_int+1, ix3):
        for n in range(iy_int+1, iy8):
            N[m,n] = Fob*(T[m,n+1]+T[m-1,n]+T[m,n-1]+T[m+1,n]) + (1-4*Fob)*T[m,n]

    for m in range(ix3, ix10):
        for n in range(iy_int+1, iy5):
            N[m,n] = Fob*(T[m,n+1]+T[m-1,n]+T[m,n-1]+T[m+1,n]) + (1-4*Fob)*T[m,n]


    # NÓS DE INTERFACE ENTRE MATERIAL A E MATERIAL B
    # Aqui são usadas propriedades combinadas dos dois materiais,
    # pois o volume de controle contém parte de aço e parte de bronze.

    ci = 2*dt/((rho_a*cp_a+rho_b*cp_b)*Delta**2)

    for n in range(iy_int+1, iy8):
        N[ix_int,n] = ci*(k_a*T[ix_int-1,n]+k_b*T[ix_int+1,n]+((k_a+k_b)/2)*(T[ix_int,n+1]+T[ix_int,n-1])) + (1-4*(k_a+k_b)*dt/((rho_a*cp_a+rho_b*cp_b)*Delta**2))*T[ix_int,n]

    cc = 4*dt/((3*rho_a*cp_a+rho_b*cp_b)*Delta**2)
    N[ix_int,iy_int] = cc*(k_a*(T[ix_int-1,iy_int]+T[ix_int,iy_int-1])+((k_a+k_b)/2)*(T[ix_int,iy_int+1]+T[ix_int+1,iy_int])) + (1-4*(3*k_a+k_b)*dt/((3*rho_a*cp_a+rho_b*cp_b)*Delta**2))*T[ix_int,iy_int]

    for m in range(ix_int+1, ix10):
        N[m,iy_int] = ci*(k_b*T[m,iy_int+1]+k_a*T[m,iy_int-1]+((k_a+k_b)/2)*(T[m-1,iy_int]+T[m+1,iy_int])) + (1-4*(k_a+k_b)*dt/((rho_a*cp_a+rho_b*cp_b)*Delta**2))*T[m,iy_int]

    # NÓS INTERNOS DO MATERIAL A
    # Região de aço carbono sem convecção, com condução
    # para os quatro vizinhos.

    for m in range(1, ix_int):
        for n in range(iy_int, iy8):
            N[m,n] = Foa*(T[m,n+1]+T[m-1,n]+T[m,n-1]+T[m+1,n]) + (1-4*Foa)*T[m,n]

    for m in range(1, ix10):
        for n in range(1, iy_int):
            N[m,n] = Foa*(T[m,n+1]+T[m-1,n]+T[m,n-1]+T[m+1,n]) + (1-4*Foa)*T[m,n]

    return N

def calcular_tempo_120(h, guardar=False):
    # Calcula o tempo necessário para que todos os nós da camada
    # sinterizada fiquem abaixo de T_limite = 120 °C.
    par = parametros(h)
    dt = par[1]
    T = campo_inicial()
    nos = nos_da_camada()
    passou = {no: False for no in nos}
    tempos = {}
    historico = {"tempo": [0.0]}
    for nome, no in pontos.items():
        historico[nome] = [T[no]]
    quadros = []
    tempos_quadros = []
    if guardar:
        quadros.append(T.copy())
        tempos_quadros.append(0.0)
    for i in range(1, max_passos+1):
        Told = T
        T = passo(T, par)
        t = i*dt
        for no in nos:
            if not passou[no] and T[no] < T_limite:
                passou[no] = True
                if Told[no] != T[no]:
                    f = (Told[no]-T_limite)/(Told[no]-T[no])
                else:
                    f = 1.0
                tempos[no] = (i-1+f)*dt
        if guardar:
            historico["tempo"].append(t)
            for nome, no in pontos.items():
                historico[nome].append(T[no])
            if i % 40 == 0:
                quadros.append(T.copy())
                tempos_quadros.append(t)
        if all(passou.values()):
            ultimo = max(tempos, key=tempos.get)
            if guardar:
                quadros.append(T.copy())
                tempos_quadros.append(t)
            return tempos[ultimo], ultimo, historico, quadros, tempos_quadros
    raise RuntimeError("A camada sinterizada não atingiu 120 °C dentro de max_passos.")

def calcular_regime_permanente(h, guardar=True):
    # Calcula o tempo até o regime permanente.
    # O critério usado é a maior variação de temperatura entre dois
    # passos consecutivos ser menor que tolerancia_regime.
    par = parametros(h)
    dt = par[1]
    T = campo_inicial()
    v = formato()
    historico = {"tempo": [0.0]}
    for nome, no in pontos.items():
        historico[nome] = [T[no]]
    quadros = []
    tempos_quadros = []
    if guardar:
        quadros.append(T.copy())
        tempos_quadros.append(0.0)
    for i in range(1, max_passos+1):
        Told = T
        T = passo(T, par)
        erro = np.nanmax(np.abs(T[v]-Told[v]))
        t = i*dt
        historico["tempo"].append(t)
        for nome, no in pontos.items():
            historico[nome].append(T[no])
        if guardar and i % 30 == 0:
            quadros.append(T.copy())
            tempos_quadros.append(t)
        if erro < tolerancia_regime:
            if guardar:
                quadros.append(T.copy())
                tempos_quadros.append(t)
            return t, i, erro, historico, quadros, tempos_quadros
    raise RuntimeError("O regime permanente não foi atingido dentro de max_passos.")

def buscar_h(tempo_alvo):
    # Procura um novo coeficiente de convecção para tentar reduzir
    # o tempo do item (a) pela metade. Primeiro são feitos testes
    # em valores pré-definidos e depois é aplicada bisseção.
    h_testes = [3000, 6000, 12000, 25000, 50000, 75000, 100000, 150000, 200000, 300000]
    resultados = []

    for h_teste in h_testes:
        try:
            tempo_teste = calcular_tempo_120(h_teste, guardar=False)[0]
            resultados.append((h_teste, tempo_teste))
            print(f"Teste: h = {h_teste:.0f} W/m²K, tempo = {tempo_teste:.2f} s", flush=True)
        except RuntimeError:
            resultados.append((h_teste, np.inf))
            print(f"Teste: h = {h_teste:.0f} W/m²K, tempo não atingido dentro de max_passos", flush=True)

    h_baixo = h_inicial
    h_alto = None
    t_alto = None

    for h_teste, tempo_teste in resultados:
        if np.isfinite(tempo_teste) and tempo_teste <= tempo_alvo:
            h_alto = h_teste
            t_alto = tempo_teste
            break
        h_baixo = h_teste

    if h_alto is None:
        valores_finitos = [(h, t) for h, t in resultados if np.isfinite(t)]
        if len(valores_finitos) == 0:
            return None, None, None
        h_menor, menor_tempo = min(valores_finitos, key=lambda x: x[1])
        return None, menor_tempo, h_menor

    for i in range(10):
        h_meio = 0.5*(h_baixo + h_alto)
        try:
            tempo_meio = calcular_tempo_120(h_meio, guardar=False)[0]
            print(f"Bisseção {i+1}/10: h = {h_meio:.1f} W/m²K, tempo = {tempo_meio:.2f} s", flush=True)
        except RuntimeError:
            tempo_meio = np.inf
            print(f"Bisseção {i+1}/10: h = {h_meio:.1f} W/m²K, tempo não atingido", flush=True)

        if np.isfinite(tempo_meio) and tempo_meio <= tempo_alvo:
            h_alto = h_meio
            t_alto = tempo_meio
        else:
            h_baixo = h_meio

    return h_alto, t_alto, None

def grafico_temperaturas(historico):
    # Gera o gráfico da evolução temporal dos pontos A, B, C e D.
    nome_delta = f"{Delta:.6f}".replace(".", "_")
    
    arquivo = pasta_saida/f"grafico_temperaturas_ABCD_Delta_{nome_delta}.png"
    plt.figure(figsize=(9,5))

    for nome in pontos:
        plt.plot(historico["tempo"], historico[nome], label=nome)

    tmax = max(historico["tempo"])
    margem_x = 0.04*tmax

    plt.xlabel("Tempo [s]")
    plt.ylabel("Temperatura [°C]")
    plt.title(f"Evolução da temperatura até a estabilização - Delta = {Delta:.6f} m")

    plt.xlim(-margem_x, tmax)
    plt.ylim(0, 900)

    plt.xticks(np.arange(0, tmax + 1, 100))
    plt.yticks(np.arange(0, 901, 100))

    plt.grid(True, which="both", linewidth=0.5, alpha=0.6)
    plt.legend()
    plt.tight_layout()
    plt.savefig(arquivo, dpi=300)
    plt.close()

    arquivo_zoom = pasta_saida/f"grafico_zoom_ABCD_Delta_{nome_delta}.png"
    plt.figure(figsize=(9,5))

    for nome in pontos:
        plt.plot(historico["tempo"], historico[nome], label=nome)

    plt.xlabel("Tempo [s]")
    plt.ylabel("Temperatura [°C]")
    plt.title(f"Zoom da evolução térmica - Delta = {Delta:.6f} m")

    plt.xlim(0, 300)
    plt.ylim(100, 900)

    plt.xticks(np.arange(0, 301, 50))
    plt.yticks(np.arange(100, 901, 100))

    plt.grid(True, which="both", linewidth=0.5, alpha=0.6)
    plt.legend()
    plt.tight_layout()
    plt.savefig(arquivo_zoom, dpi=300)
    plt.close()

    return arquivo

def gif_temperatura(quadros, tempos):
    arquivo = pasta_saida/"distribuicao_temperatura.gif"
    if len(quadros) < 2:
        return None

    tempo_final_gif = (tempo_regime)
    indices = [i for i, t in enumerate(tempos) if t <= tempo_final_gif]
    if len(indices) < 2:
        indices = list(range(len(quadros)))

    quadros = [quadros[i] for i in indices]
    tempos = [tempos[i] for i in indices]

    # Mapa de cores usado na distribuição de temperatura.
    # Os valores np.nan, que representam regiões fora da peça, são mostrados em branco.
    cmap = plt.get_cmap("inferno").copy()
    cmap.set_bad("white")

    fig, ax = plt.subplots(figsize=(9, 5.5), facecolor="white")

    T0 = np.ma.masked_invalid(quadros[0].T)
    img = ax.imshow(
        T0,
        origin="lower",
        cmap=cmap,
        interpolation="bicubic",
        vmin=25,
        vmax=850,
        extent=[-0.5, ix10+0.5, -0.5, iy8+0.5]
    )

    barra = fig.colorbar(img, ax=ax, fraction=0.046, pad=0.04)
    barra.set_label("Temperatura [°C]")

    ax.set_title(f"Distribuição de temperatura - t = {tempos[0]:.1f} s")
    ax.set_xlabel("m")
    ax.set_ylabel("n")
    ax.set_xlim(-0.5, ix10)
    ax.set_ylim(-0.5, iy8)

    passo_x = max(1, ix10//10)
    passo_y = max(1, iy8//8)
    ax.set_xticks(np.arange(0, ix10+1, passo_x))
    ax.set_yticks(np.arange(0, iy8+1, passo_y))
    ax.grid(False)

    for spine in ax.spines.values():
        spine.set_visible(True)

    def atualizar(k):
        T_atual = np.ma.masked_invalid(quadros[k].T)
        img.set_data(T_atual)
        ax.set_title(f"Distribuição de temperatura - t = {tempos[k]:.1f} s")
        return [img]

    animacao = FuncAnimation(fig, atualizar, frames=len(quadros), interval=450)
    animacao.save(arquivo, writer=PillowWriter(fps=2))
    plt.close()
    return arquivo

if __name__ == "__main__":
    # Executa os itens pedidos no trabalho e mede o custo computacional.
    # time.time() registra o horário atual do computador em segundos.
    # Subtraindo fim - início, obtemos quanto tempo real o código levou para rodar.
    inicio_total = time.time()

    par0 = parametros(h_inicial)

    print("")
    print("CRITÉRIO DE ESTABILIDADE")
    print(f"Delta t mínimo permitido = {par0[2]:.6f} s")
    print(f"Delta t usado (coeficiente de segurança de 0,95) = {par0[1]:.6f} s")
    print(f"Nó limitante = {par0[3]}")
    print(f"Fo_a = {par0[4]:.6f}")
    print(f"Fo_b = {par0[5]:.6f}")

    print("")
    print("INFORMAÇÕES DA MALHA")
    print(f"Delta = {Delta:.6f} m")
    print(f"Número de nós em x = {nx}")
    print(f"Número de nós em y = {ny}")
    print(f"Número total de nós da peça = {np.sum(formato())}")

    print("")
    print("ITEM (a)")
    inicio_a = time.time()
    tempo_120, ultimo_ponto, hist_120, quadros_120, tempos_120 = calcular_tempo_120(h_inicial, guardar=True)
    fim_a = time.time()
    custo_a = fim_a - inicio_a
    print(f"Tempo para todos os pontos da camada sinterizada ficarem abaixo de 120 °C = {tempo_120:.3f} s")
    print(f"Último ponto da camada sinterizada = [{ultimo_ponto[0]},{ultimo_ponto[1]}]")
    print(f"Tempo computacional do item A = {custo_a:.3f} s")

    print("")
    print("ITEM (b)", flush=True)
    print("Calculando o novo coeficiente convectivo:", flush=True)
    inicio_b = time.time()
    h_novo, tempo_novo, h_menor = buscar_h(tempo_120/2)
    fim_b = time.time()
    custo_b = fim_b - inicio_b
    if h_novo is None:
        print("Não foi possível encontrar um h que reduza o tempo pela metade até 300000 W/m²K.")
        if tempo_novo is not None:
            print(f"Menor tempo obtido nos testes = {tempo_novo:.3f} s")
            print(f"h correspondente ao menor tempo testado = {h_menor:.3f} W/m²K")
    else:
        print(f"Coeficiente convectivo necessário = {h_novo:.3f} W/m²K")
        print(f"Tempo obtido com esse h = {tempo_novo:.3f} s")
    print(f"Tempo computacional do item B = {custo_b:.3f} s")

    print("")
    print("ITEM (c)")
    inicio_c = time.time()
    tempo_regime, passos_regime, erro_final, hist_regime, quadros_regime, tempos_regime = calcular_regime_permanente(h_inicial, guardar=True)
    fim_c = time.time()
    custo_c = fim_c - inicio_c
    print("")
    print("TEMPERATURAS DOS PONTOS A, B, C E D NO REGIME PERMANENTE")
    print(f"Tempo até regime permanente = {tempo_regime:.3f} s")
    print("")
    for nome in pontos:
        temperatura_final = hist_regime[nome][-1]
        print(f"{nome}: {temperatura_final:.4f} °C")
    print(f"Tempo até regime permanente = {tempo_regime:.3f} s")
    print(f"Número de passos = {passos_regime}")
    print(f"Erro máximo final = {erro_final:.6e} °C")
    print(f"Tempo computacional do item C = {custo_c:.3f} s")
    print("")
    print("ITEM (d)")
    inicio_d = time.time()
    arquivo_grafico = grafico_temperaturas(hist_regime)
    print(f"Gráfico salvo em: {arquivo_grafico}")

    arquivo_gif = gif_temperatura(quadros_regime, tempos_regime)
    fim_d = time.time()
    custo_d = fim_d - inicio_d
    print(f"GIF salvo em: {arquivo_gif}")
    print(f"Tempo computacional do item D = {custo_d:.3f} s")

    fim_total = time.time()
    custo_total = fim_total - inicio_total

    print("")
    print("CUSTO COMPUTACIONAL DA MALHA")
    print(f"Tempo computacional total do código = {custo_total:.3f} s")

    nome_delta_custo = f"{Delta:.6f}".replace(".", "_")
    arquivo_custo = pasta_saida/f"custo_computacional_Delta_{nome_delta_custo}.txt"


