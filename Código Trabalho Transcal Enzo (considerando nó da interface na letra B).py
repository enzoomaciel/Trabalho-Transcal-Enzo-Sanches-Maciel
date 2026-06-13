import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter
from pathlib import Path

pasta_saida = Path(__file__).resolve().parent

Ti = 850.0
Tinf = 25.0
Delta = 0.005

k_a = 45.0
rho_a = 7800.0
cp_a = 460.0

k_b = 8.0
rho_b = 5200.0
cp_b = 380.0

h_inicial = 1500.0
fator_seguranca = 0.95
T_limite = 120.0
tolerancia_regime = 1e-4
max_passos = 600000

pontos = {
    "A [6,16]": (6, 16),
    "B [20,10]": (20, 10),
    "C [4,8]": (4, 8),
    "D [0,0]": (0, 0)
}

def formato():
    v = np.zeros((21, 17), dtype=bool)
    v[0:7, 0:17] = True
    v[7:21, 0:11] = True
    return v

def campo_inicial():
    T = np.full((21, 17), np.nan)
    T[formato()] = Ti
    return T

def nos_da_camada():
    s = set()
    for n in range(9, 17):
        s.add((5, n))
    for n in range(10, 17):
        s.add((6, n))
    for m in range(6, 21):
        s.add((m, 9))
        s.add((m, 10))
    for n in range(8, 17):
        s.add((4, n))
    for m in range(5, 21):
        s.add((m, 8))
    v = formato()
    return sorted([no for no in s if v[no]])

def parametros(h):
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
        "Nó [0,16]",
        "Nós [1,16] até [3,16]",
        "Nó [4,16]",
        "Nó [5,16]",
        "Nó [6,16]",
        "Nós [0,1] até [0,15]",
        "Nó [0,0]",
        "Nós [1,0] até [19,0]",
        "Nó [20,0]",
        "Nós [20,1] até [20,7]",
        "Nó [20,8]",
        "Nó [20,9]",
        "Nó [20,10]",
        "Nós [7,10] até [19,10]",
        "Nó [6,10]",
        "Nós [6,11] até [6,15]",
        "Nós internos da camada sinterizada",
        "Nós internos da interface vertical",
        "Nó [4,8]",
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
    h, dt, dt_min, no_limitante, Foa, Fob, Ha, Hb = par
    N = T.copy()

    N[0,16] = 2*Foa*(T[1,16]+T[0,15]) + 4*Ha*Tinf + (1-4*Foa-4*Ha)*T[0,16]

    for m in range(1,4):
        N[m,16] = Foa*(T[m-1,16]+T[m+1,16]+2*T[m,15]) + 2*Ha*Tinf + (1-4*Foa-2*Ha)*T[m,16]

    c = 4*dt/((rho_a*cp_a+rho_b*cp_b)*Delta**2)
    N[4,16] = c*((k_a/2)*T[3,16]+(k_b/2)*T[5,16]+((k_a+k_b)/2)*T[4,15]+h*Delta*Tinf) + (1-c*(k_a+k_b+h*Delta))*T[4,16]

    N[5,16] = Fob*(T[4,16]+T[6,16]+2*T[5,15]) + 2*Hb*Tinf + (1-4*Fob-2*Hb)*T[5,16]

    N[6,16] = 2*Fob*(T[5,16]+T[6,15]) + 4*Hb*Tinf + (1-4*Fob-4*Hb)*T[6,16]

    for n in range(1,16):
        N[0,n] = Foa*(T[0,n+1]+T[0,n-1]+2*T[1,n]) + 2*Ha*Tinf + (1-4*Foa-2*Ha)*T[0,n]

    N[0,0] = 2*Foa*(T[0,1]+T[1,0]) + 2*Ha*Tinf + (1-4*Foa-2*Ha)*T[0,0]

    for m in range(1,20):
        N[m,0] = Foa*(T[m-1,0]+T[m+1,0]+2*T[m,1]) + (1-4*Foa)*T[m,0]

    N[20,0] = 2*Foa*(T[19,0]+T[20,1]) + 2*Ha*Tinf + (1-4*Foa-2*Ha)*T[20,0]

    for n in range(1,8):
        N[20,n] = Foa*(T[20,n+1]+T[20,n-1]+2*T[19,n]) + 2*Ha*Tinf + (1-4*Foa-2*Ha)*T[20,n]

    N[20,8] = c*((k_b/2)*T[20,9]+(k_a/2)*T[20,7]+((k_a+k_b)/2)*T[19,8]+h*Delta*Tinf) + (1-c*(k_a+k_b+h*Delta))*T[20,8]

    N[20,9] = Fob*(T[20,10]+T[20,8]+2*T[19,9]) + 2*Hb*Tinf + (1-4*Fob-2*Hb)*T[20,9]

    N[20,10] = 2*Fob*(T[19,10]+T[20,9]) + 4*Hb*Tinf + (1-4*Fob-4*Hb)*T[20,10]

    for m in range(7,20):
        N[m,10] = Fob*(T[m-1,10]+T[m+1,10]+2*T[m,9]) + 2*Hb*Tinf + (1-4*Fob-2*Hb)*T[m,10]

    N[6,10] = (2/3)*Fob*(T[6,11]+T[7,10]+2*T[5,10]+2*T[6,9]) + (4/3)*Hb*Tinf + (1-4*Fob-(4/3)*Hb)*T[6,10]

    for n in range(11,16):
        N[6,n] = Fob*(T[6,n+1]+T[6,n-1]+2*T[5,n]) + 2*Hb*Tinf + (1-4*Fob-2*Hb)*T[6,n]

    for n in range(9,16):
        N[5,n] = Fob*(T[5,n+1]+T[4,n]+T[5,n-1]+T[6,n]) + (1-4*Fob)*T[5,n]

    for m in range(6,20):
        N[m,9] = Fob*(T[m,10]+T[m-1,9]+T[m,8]+T[m+1,9]) + (1-4*Fob)*T[m,9]

    ci = 2*dt/((rho_a*cp_a+rho_b*cp_b)*Delta**2)

    for n in range(9,16):
        N[4,n] = ci*(k_a*T[3,n]+k_b*T[5,n]+((k_a+k_b)/2)*(T[4,n+1]+T[4,n-1])) + (1-4*(k_a+k_b)*dt/((rho_a*cp_a+rho_b*cp_b)*Delta**2))*T[4,n]

    cc = 4*dt/((3*rho_a*cp_a+rho_b*cp_b)*Delta**2)
    N[4,8] = cc*(k_a*(T[3,8]+T[4,7])+((k_a+k_b)/2)*(T[4,9]+T[5,8])) + (1-4*(3*k_a+k_b)*dt/((3*rho_a*cp_a+rho_b*cp_b)*Delta**2))*T[4,8]

    for m in range(5,20):
        N[m,8] = ci*(k_b*T[m,9]+k_a*T[m,7]+((k_a+k_b)/2)*(T[m-1,8]+T[m+1,8])) + (1-4*(k_a+k_b)*dt/((rho_a*cp_a+rho_b*cp_b)*Delta**2))*T[m,8]

    for m in range(1,4):
        for n in range(8,16):
            N[m,n] = Foa*(T[m,n+1]+T[m-1,n]+T[m,n-1]+T[m+1,n]) + (1-4*Foa)*T[m,n]

    for m in range(1,20):
        for n in range(1,8):
            N[m,n] = Foa*(T[m,n+1]+T[m-1,n]+T[m,n-1]+T[m+1,n]) + (1-4*Foa)*T[m,n]

    return N

def calcular_tempo_120(h, guardar=False):
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
        quadros.append(T.copy(  ))
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
    arquivo = pasta_saida/"grafico_temperaturas_ABCD_ate_regime.png"
    plt.figure(figsize=(9,5))

    for nome in pontos:
        plt.plot(historico["tempo"], historico[nome], label=nome)

    tmax = max(historico["tempo"])
    margem_x = 0.04*tmax

    plt.xlabel("Tempo [s]")
    plt.ylabel("Temperatura [°C]")
    plt.title("Evolução da temperatura até a estabilização")

    plt.xlim(-margem_x, tmax)
    plt.ylim(0, 900)

    plt.xticks(np.arange(0, tmax + 1, 100))
    plt.yticks(np.arange(0, 901, 100))

    plt.grid(True, which="both", linewidth=0.5, alpha=0.6)
    plt.legend()
    plt.tight_layout()
    plt.savefig(arquivo, dpi=300)
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
        extent=[-0.5, 20.5, -0.5, 16.5]
    )

    barra = fig.colorbar(img, ax=ax, fraction=0.046, pad=0.04)
    barra.set_label("Temperatura [°C]")

    ax.set_title(f"Distribuição de temperatura - t = {tempos[0]:.1f} s")
    ax.set_xlabel("m")
    ax.set_ylabel("n")
    ax.set_xlim(-0.5, 20.0)
    ax.set_ylim(-0.5, 16.0)

    ax.set_xticks(np.arange(0, 21, 2))
    ax.set_yticks(np.arange(0, 17, 2))
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
    par0 = parametros(h_inicial)

    print("")
    print("CRITÉRIO DE ESTABILIDADE")
    print(f"Delta t mínimo permitido = {par0[2]:.6f} s")
    print(f"Delta t usado (coeficiente de segurança de 0,95) = {par0[1]:.6f} s")
    print(f"Nó limitante = {par0[3]}")
    print(f"Fo_a = {par0[4]:.6f}")
    print(f"Fo_b = {par0[5]:.6f}")

    print("")
    print("ITEM (a)")
    tempo_120, ultimo_ponto, hist_120, quadros_120, tempos_120 = calcular_tempo_120(h_inicial, guardar=True)
    print(f"Tempo para todos os pontos da camada sinterizada ficarem abaixo de 120 °C = {tempo_120:.3f} s")
    print(f"Último ponto da camada sinterizada = [{ultimo_ponto[0]},{ultimo_ponto[1]}]")

    print("")
    print("ITEM (b)", flush=True)
    print("Calculando o novo coeficiente convectivo:", flush=True)
    h_novo, tempo_novo, h_menor = buscar_h(tempo_120/2)
    if h_novo is None:
        print("Não foi possível encontrar um h que reduza o tempo pela metade até 300000 W/m²K.")
        if tempo_novo is not None:
            print(f"Menor tempo obtido nos testes = {tempo_novo:.3f} s")
            print(f"h correspondente ao menor tempo testado = {h_menor:.3f} W/m²K")
    else:
        print(f"Coeficiente convectivo necessário = {h_novo:.3f} W/m²K")
        print(f"Tempo obtido com esse h = {tempo_novo:.3f} s")

    print("")
    print("ITEM (c)")
    tempo_regime, passos_regime, erro_final, hist_regime, quadros_regime, tempos_regime = calcular_regime_permanente(h_inicial, guardar=True)
    print(f"Tempo até regime permanente = {tempo_regime:.3f} s")
    print(f"Número de passos = {passos_regime}")
    print(f"Erro máximo final = {erro_final:.6e} °C")
    print("")
    print("ITEM (d)")
    arquivo_grafico = grafico_temperaturas(hist_regime)
    print(f"Gráfico salvo em: {arquivo_grafico}")

    arquivo_gif = gif_temperatura(quadros_regime, tempos_regime)
    print(f"GIF salvo em: {arquivo_gif}")
    print("")