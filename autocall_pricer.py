"""
autocall_pricer.py
==================
Phoenix Autocall Pricer | Monte Carlo (Black-Scholes) | Mono sous-jacent

Modifie les paramètres dans la section PARAMÈTRES, puis:
    pip install numpy pandas yfinance matplotlib
    python autocall_pricer.py
"""

import numpy as np
import matplotlib
matplotlib.use("Agg")          # Backend sans interface graphique (supprimez cette ligne sur votre machine)
import matplotlib.pyplot as plt

try:
    import yfinance as yf
    YF_AVAILABLE = True
except ImportError:
    YF_AVAILABLE = False

# ============================================================
#  PARAMÈTRES (à modifier selon le produit)
# ============================================================

TICKER             = "^STOXX50E"                   # Eurostoxx 50 (sous-jacent standard des autocalls retail)
PERIOD             = "2y"                           # Historique pour la vol réalisée

NOMINAL            = 1_000.0                        # Nominal en EUR
COUPON_RATE        = 0.07                           # Coupon par période d'observation (7% annuel)
RECALL_BARRIER     = 1.00                           # Barrière de rappel (en fraction du spot initial)
COUPON_BARRIER     = 0.70                           # Barrière de versement du coupon
PROTECTION_BARRIER = 0.60                           # Barrière de protection du capital à maturité
OBSERVATION_DATES  = [1.0, 2.0, 3.0, 4.0, 5.0]    # Dates d'observation annuelles (en années)
COUPON_MEMORY      = True                           # True = Phoenix (mémoire des coupons)

RISK_FREE_RATE = 0.030    # Taux OIS EUR (proxy BCE, à actualiser selon conditions de marché)

N_SIMULATIONS = 100_000   # Trajectoires Monte Carlo
SEED          = 42         # Graine aléatoire (assure la reproductibilité)


# ============================================================
#  1. DONNÉES DE MARCHÉ
# ============================================================

def fetch_market_data():
    """
    Spot S0 et volatilité historique annualisée (Eurostoxx 50, Yahoo Finance).

    Volatilité historique: sigma = std( log(S_t / S_{t-1}) ) * sqrt(252)

    Note: c'est la volatilité réalisée sur les 252 dernières séances, utilisée
    comme proxy de la volatilité implicite. En pratique, un desk de structuration
    calibre sur la surface de volatilité implicite extraite des options cotées,
    qui intègre le smile et le skew de volatilité, ignorés ici (Black-Scholes).

    Returns:
        S0    : dernier prix de clôture ajusté (float)
        sigma : volatilité historique annualisée (float)
    """
    if not YF_AVAILABLE:
        print("  yfinance non installé -> paramètres synthétiques (S0=4950, sigma=18.5%)")
        return 4950.0, 0.185

    try:
        data  = yf.download(TICKER, period=PERIOD, auto_adjust=True, progress=False)
        # .squeeze() gère les deux formats yfinance: Series (old) et DataFrame 1-col (new)
        close = data["Close"].squeeze().dropna()
        S0    = float(close.iloc[-1])
        sigma = float(np.log(close / close.shift(1)).dropna().tail(252).std() * np.sqrt(252))
        print(f"  Source: Yahoo Finance ({TICKER}, {len(close)} séances)")
        return S0, sigma
    except Exception as e:
        print(f"  Yahoo Finance indisponible ({e}) -> paramètres synthétiques")
        return 4950.0, 0.185


# ============================================================
#  2. SIMULATION MONTE CARLO (GBM sous mesure risque-neutre)
# ============================================================

def simulate_gbm(S0, r, sigma, obs_dates):
    """
    Trajectoires de prix sous la probabilité risque-neutre Q (modèle Black-Scholes).

    Sous Q, le sous-jacent suit le mouvement brownien géométrique:
        dS = r * S * dt + sigma * S * dW^Q

    Le drift est r (taux sans risque), pas le rendement attendu mu.
    C'est la conséquence de l'absence d'opportunité d'arbitrage: sous Q,
    tout actif a un rendement espéré égal au taux sans risque après neutralisation
    du risque (le prix du risque est absorbé dans le changement de mesure).

    Discrétisation exacte aux dates d'observation (sans erreur numérique):
        S(t + dt) = S(t) * exp( (r - sigma²/2)*dt + sigma*sqrt(dt)*Z ),  Z ~ N(0,1)

    Réduction de variance par variantes antithétiques: pour chaque tirage Z,
    on ajoute -Z. Les deux trajectoires sont négativement corrélées, ce qui
    réduit la variance de l'estimateur Monte Carlo sans coût supplémentaire.

    Args:
        obs_dates : dates d'observation en années (ex: [1.0, 2.0, 3.0, 4.0, 5.0])

    Returns:
        paths : array (N_SIMULATIONS, len(obs_dates) + 1)
                colonne 0 = S0, colonnes suivantes = prix aux dates d'observation
    """
    np.random.seed(SEED)
    T       = obs_dates[-1]
    n_steps = len(obs_dates)
    dt      = T / n_steps

    drift     = (r - 0.5 * sigma**2) * dt
    diffusion = sigma * np.sqrt(dt)

    half      = N_SIMULATIONS // 2
    Z         = np.random.standard_normal((half, n_steps))
    Z         = np.vstack([Z, -Z])                                      # Variantes antithétiques

    log_paths = np.hstack([np.zeros((N_SIMULATIONS, 1)),
                           np.cumsum(drift + diffusion * Z, axis=1)])
    return S0 * np.exp(log_paths)                                        # (N_SIMULATIONS, n_steps+1)


# ============================================================
#  3. PAYOFF DU PHOENIX AUTOCALL
# ============================================================

def compute_payoffs(paths, S0, r, obs_dates, coupon_rate=None):
    """
    Payoffs actualisés et classification par scénario pour chaque trajectoire.

    A chaque date d'observation t_i (i = 1, ..., N):

        (1) Rappel anticipé: S(t_i) >= S0 * RECALL_BARRIER
            Payoff = [NOMINAL + (coupons_en_mémoire + 1) * coupon] * exp(-r * t_i)
            Le produit s'arrête. La mémoire est soldée.

        (2) Coupon versé (sans rappel): S0 * COUPON_BARRIER <= S(t_i) < S0 * RECALL_BARRIER
            Payoff += (coupons_en_mémoire + 1) * coupon * exp(-r * t_i)  [si Phoenix]
            La mémoire est remise à zéro. Le produit continue.

        (3) Pas de coupon: S(t_i) < S0 * COUPON_BARRIER
            Si Phoenix: coupons_en_mémoire += 1. Si Athena: coupon perdu définitivement.

    A maturité (trajectoires non rappelées à aucune date):
        - S(T) >= S0 * PROTECTION_BARRIER: remboursement intégral du NOMINAL * exp(-r*T)
        - S(T) <  S0 * PROTECTION_BARRIER: remboursement = NOMINAL * (S(T)/S0) * exp(-r*T)
          Perte en capital proportionnelle à la performance négative du sous-jacent.

    Note sur le Delta à l'émission: toutes les barrières sont en fraction de S0,
    et la perte en capital est en S(T)/S0. Le produit est donc scale-invariant:
    P(lambda*S0) = lambda*P(S0). Il s'ensuit que dP/dS0 = P/S0 (constant en %
    du nominal), ce qui rend le Delta classique peu informatif à l'émission.
    Pour un produit déjà en vie, les barrières sont fixes et le Delta est significatif.

    Args:
        obs_dates   : dates d'observation effectives (permet le calcul du Theta)
        coupon_rate : taux de coupon à utiliser (défaut: COUPON_RATE global).
                      Paramètre exposé pour la recherche du coupon d'équilibre.

    Returns:
        payoffs   : array (N_SIMULATIONS,) - payoffs actualisés en EUR
        scenarios : array (N_SIMULATIONS,) int
                    0 = rappel anticipé | 1 = maturité, capital protégé | 2 = perte en capital
    """
    if coupon_rate is None:
        coupon_rate = COUPON_RATE

    n_sim = paths.shape[0]

    level_recall      = S0 * RECALL_BARRIER
    level_coupon      = S0 * COUPON_BARRIER
    level_protection  = S0 * PROTECTION_BARRIER

    payoffs           = np.zeros(n_sim)
    recalled          = np.zeros(n_sim, dtype=bool)
    coupons_in_memory = np.zeros(n_sim, dtype=int)

    for i, t in enumerate(obs_dates):
        S_t      = paths[:, i + 1]
        active   = ~recalled
        discount = np.exp(-r * t)

        # (1) Rappel anticipé
        recalled_now = active & (S_t >= level_recall)
        if recalled_now.any():
            n_due                           = coupons_in_memory[recalled_now] + 1
            payoffs[recalled_now]           += (NOMINAL + n_due * coupon_rate * NOMINAL) * discount
            recalled[recalled_now]           = True
            coupons_in_memory[recalled_now]  = 0

        # (2) Coupon versé (sans rappel)
        coupon_paid = active & ~recalled_now & (S_t >= level_coupon)
        if coupon_paid.any():
            n_due = (coupons_in_memory[coupon_paid] + 1) if COUPON_MEMORY else 1
            if COUPON_MEMORY:
                coupons_in_memory[coupon_paid] = 0
            payoffs[coupon_paid] += n_due * coupon_rate * NOMINAL * discount

        # (3) Accumulation en mémoire (Phoenix uniquement)
        if COUPON_MEMORY:
            coupons_in_memory[active & ~recalled_now & ~coupon_paid] += 1

    # Payoff à maturité
    discount_T   = np.exp(-r * obs_dates[-1])
    S_T          = paths[:, -1]
    not_recalled = ~recalled

    above = not_recalled & (S_T >= level_protection)
    below = not_recalled & (S_T <  level_protection)

    payoffs[above] += NOMINAL * discount_T
    if below.any():
        payoffs[below] += NOMINAL * (S_T[below] / S0) * discount_T

    scenarios          = np.zeros(n_sim, dtype=int)  # 0 = rappel (par défaut)
    scenarios[above]   = 1
    scenarios[below]   = 2

    return payoffs, scenarios


# ============================================================
#  4. PRICING (E^Q[payoff actualisé])
# ============================================================

def price_product(paths, S0, r, obs_dates):
    """
    Prix Monte Carlo = E^Q[payoff actualisé], estimé par la moyenne empirique.

    L'estimateur converge vers le vrai prix par la loi des grands nombres.
    L'erreur standard se = std(payoffs) / sqrt(N) décroît en 1/sqrt(N).
    IC 95% par le théorème central-limite: [mean - 1.96*se, mean + 1.96*se].

    Returns:
        price, ci_low, ci_high : prix et bornes IC 95% en EUR
        payoffs, scenarios      : arrays pour les visualisations
    """
    payoffs, scenarios = compute_payoffs(paths, S0, r, obs_dates)
    n    = len(payoffs)
    mean = float(np.mean(payoffs))
    se   = float(np.std(payoffs, ddof=1) / np.sqrt(n))
    return mean, mean - 1.96*se, mean + 1.96*se, payoffs, scenarios


# ============================================================
#  5. GREEKS (bump-and-reval, différences finies centrées)
# ============================================================

def _price_scalar(S0, r, sigma, obs_dates, coupon_rate=None):
    """
    Prix scalaire pour le bump-and-reval et la recherche du coupon d'équilibre.
    Graine SEED fixe pour tous les appels: common random numbers (annulation du bruit).
    """
    paths      = simulate_gbm(S0, r, sigma, obs_dates)
    payoffs, _ = compute_payoffs(paths, S0, r, obs_dates, coupon_rate=coupon_rate)
    return float(np.mean(payoffs))


def compute_greeks(S0, r, sigma):
    """
    Greeks par différences finies centrées.

    La méthode bump-and-reval est la seule approche valide pour un produit
    à payoff discontinu (barrières): les dérivées analytiques n'existent pas
    au sens classique à cause des indicatrices dans le payoff.

    Common random numbers: même graine pour les deux chocs (+h, -h).
    Le bruit stochastique s'annule dans la différence, ce qui réduit
    significativement la variance de l'estimateur du Greek.

    Delta [EUR/pt]:   dP/dS. ~0 à l'émission (scale-invariance, voir compute_payoffs).
                      Significatif pour un produit en vie avec barrières fixes.

    Vega [EUR/unité]: dP/d_sigma. Résultat en EUR par unité de sigma annualisée.
                      Multiplier par 0.01 pour obtenir EUR par +1pp de vol.

    Theta [EUR/jour]: P(T - 1j) - P(T). Variation de prix pour 1 jour ouvré écoulé.
                      Toutes les dates d'observation se décalent de -1/252 an.
    """
    # Delta
    h_S   = S0 * 0.01
    delta = (_price_scalar(S0 + h_S, r, sigma, OBSERVATION_DATES) -
             _price_scalar(S0 - h_S, r, sigma, OBSERVATION_DATES)) / (2 * h_S)

    # Vega
    h_v  = 0.01
    vega = (_price_scalar(S0, r, sigma + h_v, OBSERVATION_DATES) -
            _price_scalar(S0, r, sigma - h_v, OBSERVATION_DATES)) / (2 * h_v)

    # Theta
    dt          = 1 / 252
    obs_shifted = [t - dt for t in OBSERVATION_DATES]
    theta       = (_price_scalar(S0, r, sigma, obs_shifted) -
                   _price_scalar(S0, r, sigma, OBSERVATION_DATES))

    return delta, vega, theta


# ============================================================
#  6. VISUALISATIONS
# ============================================================

def plot_payoff_distribution(payoffs, scenarios, price, ci_low, ci_high):
    """
    Histogramme des payoffs actualisés, coloré par scénario.

    Vert  : rappel anticipé (nominal + coupons, bonne issue pour l'investisseur)
    Orange: maturité, capital intégralement remboursé (au-dessus de la barrière de protection)
    Rouge : maturité, perte en capital (en dessous de la barrière de protection)

    La forme de la distribution illustre le profil de risque asymétrique du produit:
    forte probabilité de rappel concentrée autour de quelques niveaux discrets,
    et queue gauche de pertes potentiellement sévères.
    """
    fig, ax = plt.subplots(figsize=(12, 6))

    palette = {
        0: ("#27ae60", f"Rappel anticipé ({(scenarios==0).mean()*100:.1f}%)"),
        1: ("#e67e22", f"Capital protégé à maturité ({(scenarios==1).mean()*100:.1f}%)"),
        2: ("#c0392b", f"Perte en capital à maturité ({(scenarios==2).mean()*100:.1f}%)"),
    }
    # Bins communs sur toute la plage des payoffs -> barres cohérentes entre scénarios
    bins = np.linspace(payoffs.min(), payoffs.max(), 50)

    for sc, (color, label) in palette.items():
        mask = scenarios == sc
        if mask.any():
            ax.hist(payoffs[mask], bins=bins, color=color, alpha=0.78, label=label)

    ax.axvline(price,   color="#2c3e50", lw=2.2, ls="-",  label=f"Prix MC: {price:,.2f} EUR")
    ax.axvline(NOMINAL, color="#7f8c8d", lw=1.8, ls="--", label=f"Nominal: {NOMINAL:,.0f} EUR")
    ax.axvline(ci_low,  color="#bdc3c7", lw=1.2, ls=":",  label=f"IC 95%: [{ci_low:,.1f}, {ci_high:,.1f}]")
    ax.axvline(ci_high, color="#bdc3c7", lw=1.2, ls=":")

    stats = (f"N = {len(payoffs):,} trajectoires\n"
             f"Moyenne  : {price:,.2f} EUR\n"
             f"Médiane  : {np.median(payoffs):,.2f} EUR\n"
             f"P5 / P95 : {np.percentile(payoffs,5):,.0f} / {np.percentile(payoffs,95):,.0f} EUR")
    ax.text(0.02, 0.97, stats, transform=ax.transAxes, fontsize=9.5,
            va="top", bbox=dict(boxstyle="round", fc="white", alpha=0.88))

    ax.set_xlabel("Payoff actualisé (EUR)", fontsize=11)
    ax.set_ylabel("Nombre de trajectoires", fontsize=11)
    ax.set_title(
        f"Distribution des payoffs | Phoenix Autocall | Eurostoxx 50 | "
        f"{COUPON_RATE*100:.0f}% | {OBSERVATION_DATES[-1]:.0f} ans",
        fontsize=12, fontweight="bold"
    )
    ax.legend(fontsize=9.5)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig("distribution_payoffs.png", dpi=150, bbox_inches="tight")
    print("  -> Sauvegardé: distribution_payoffs.png")


def plot_vega_curve(S0, r, sigma_current):
    """
    Sensibilité du prix à la volatilité sur l'ensemble de la plage plausible (5% - 50%).

    Illustre que le prix n'est pas linéaire en vol: les effets de convexité
    liés aux barrières créent des non-linéarités que Black-Scholes capture
    partiellement (le smile de vol, ignoré ici, les amplifie en réalité).
    """
    print("  Calcul de la courbe prix/vol (20 runs Monte Carlo)...")
    vol_range    = np.linspace(0.05, 0.50, 20)
    prices_curve = np.array([_price_scalar(S0, r, v, OBSERVATION_DATES) for v in vol_range])

    fig, ax1 = plt.subplots(figsize=(12, 6))

    ax1.plot(vol_range * 100, prices_curve, color="#2c3e50", lw=2.5, zorder=3)

    idx = np.argmin(np.abs(vol_range - sigma_current))
    ax1.scatter([sigma_current * 100], [prices_curve[idx]],
                color="#c0392b", s=100, zorder=5,
                label=f"Vol actuelle ({sigma_current*100:.1f}%) -> {prices_curve[idx]:,.2f} EUR")
    ax1.axvline(sigma_current * 100, color="#c0392b", lw=1.8, ls="--")
    ax1.axhline(NOMINAL * np.exp(-RISK_FREE_RATE * OBSERVATION_DATES[-1]),
                color="#7f8c8d", lw=1.2, ls=":",
                label=f"Nominal actualisé ({NOMINAL * np.exp(-RISK_FREE_RATE * OBSERVATION_DATES[-1]):,.0f} EUR)")

    ax1.set_xlabel("Volatilité annualisée (%)", fontsize=11)
    ax1.set_ylabel("Prix du produit (EUR)", fontsize=11)

    # Axe droit: % du nominal
    ax2 = ax1.twinx()
    ymin, ymax = ax1.get_ylim()
    ax2.set_ylim(ymin / NOMINAL * 100, ymax / NOMINAL * 100)
    ax2.set_ylabel("Prix / Nominal (%)", fontsize=11, color="#7f8c8d")
    ax2.tick_params(axis="y", labelcolor="#7f8c8d")

    ax1.set_title(
        f"Sensibilité du prix à la volatilité | Phoenix Autocall | "
        f"{COUPON_RATE*100:.0f}% | {OBSERVATION_DATES[-1]:.0f} ans",
        fontsize=12, fontweight="bold"
    )
    ax1.legend(fontsize=9.5)
    ax1.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig("vega_curve.png", dpi=150, bbox_inches="tight")
    print("  -> Sauvegardé: vega_curve.png")


# ============================================================
#  7. COUPON D'ÉQUILIBRE (dichotomie)
# ============================================================

def find_equilibrium_coupon(S0, r, sigma, target_price=None,
                             coupon_min=0.001, coupon_max=0.30,
                             tol=0.01, max_iter=50):
    """
    Recherche du coupon d'équilibre par dichotomie (bisection).

    Processus réel de structuration: le structureur fixe un prix cible
    (typiquement 100% du nominal = émission au pair), et cherche le coupon
    maximum qu'il peut offrir à l'investisseur en respectant ce prix cible.
    Ce coupon est ensuite réduit de la marge commerciale de la banque.

    La relation prix/coupon est strictement croissante et continue:
    un coupon plus élevé génère des flux plus importants, donc un prix plus élevé.
    La dichotomie exploite cette monotonie: à chaque itération, on divise
    l'intervalle de recherche par deux. Convergence en O(log2((coupon_max -
    coupon_min) / tol)) itérations, soit ~15 itérations pour une précision au
    centime sur un intervalle de 30%.

    Args:
        target_price : prix cible en EUR (défaut: NOMINAL = émission au pair)
        coupon_min   : borne basse de recherche (défaut: 0.1%)
        coupon_max   : borne haute de recherche (défaut: 30%)
        tol          : tolérance en EUR sur le prix (défaut: 1 centime)
        max_iter     : nombre maximum d'itérations (sécurité)

    Returns:
        coupon_eq : taux de coupon d'équilibre (float, ex: 0.0487 pour 4.87%)
    """
    if target_price is None:
        target_price = NOMINAL

    # Les paths sont identiques pour tous les coupons (même graine):
    # on les génère une seule fois pour accélérer la dichotomie.
    paths = simulate_gbm(S0, r, sigma, OBSERVATION_DATES)

    def price_at_coupon(c):
        payoffs, _ = compute_payoffs(paths, S0, r, OBSERVATION_DATES, coupon_rate=c)
        return float(np.mean(payoffs))

    # Vérification des bornes: le prix doit être croissant en coupon
    p_min = price_at_coupon(coupon_min)
    p_max = price_at_coupon(coupon_max)

    if p_min > target_price:
        raise ValueError(f"Prix à coupon_min={coupon_min*100:.1f}% ({p_min:.2f} EUR) "
                         f"déjà supérieur à la cible ({target_price:.2f} EUR). "
                         "Abaissez coupon_min.")
    if p_max < target_price:
        raise ValueError(f"Prix à coupon_max={coupon_max*100:.1f}% ({p_max:.2f} EUR) "
                         f"encore inférieur à la cible ({target_price:.2f} EUR). "
                         "Relevez coupon_max.")

    print(f"    Cible            : {target_price:.2f} EUR ({target_price/NOMINAL*100:.0f}% du nominal)")
    print(f"    Borne basse      : coupon={coupon_min*100:.1f}% -> prix={p_min:.4f} EUR")
    print(f"    Borne haute      : coupon={coupon_max*100:.1f}% -> prix={p_max:.4f} EUR")
    print()

    for n_iter in range(1, max_iter + 1):
        coupon_mid = (coupon_min + coupon_max) / 2
        p_mid      = price_at_coupon(coupon_mid)

        print(f"    Iter {n_iter:2d}: coupon={coupon_mid*100:.4f}%  ->  prix={p_mid:.4f} EUR  "
              f"(écart: {p_mid - target_price:+.4f} EUR)")

        if abs(p_mid - target_price) < tol:
            print(f"\n    Convergence atteinte en {n_iter} itérations.")
            return coupon_mid

        if p_mid < target_price:
            coupon_min = coupon_mid   # Prix trop bas: on monte le coupon
        else:
            coupon_max = coupon_mid   # Prix trop haut: on baisse le coupon

    return (coupon_min + coupon_max) / 2


# ============================================================
#  8. MAIN
# ============================================================

def main():
    print("=" * 62)
    print("  PHOENIX AUTOCALL PRICER")
    print("  Monte Carlo | Black-Scholes | Mono sous-jacent")
    print("=" * 62)

    # 1. Données de marché
    print("\n[1] Données de marché")
    S0, sigma = fetch_market_data()
    r = RISK_FREE_RATE
    print(f"    Spot (S0)         : {S0:>10,.2f}")
    print(f"    Vol historique    : {sigma*100:>9.2f}%  (252 séances, annualisée)")
    print(f"    Taux sans risque  : {r*100:>9.2f}%  (OIS EUR, proxy BCE)")

    # 2. Paramètres produit
    print("\n[2] Paramètres du Phoenix Autocall")
    print(f"    Nominal           : {NOMINAL:>10,.0f} EUR")
    print(f"    Coupon            : {COUPON_RATE*100:>9.1f}% / an")
    print(f"    Barrière rappel   : {RECALL_BARRIER*100:>9.0f}% du spot initial")
    print(f"    Barrière coupon   : {COUPON_BARRIER*100:>9.0f}% du spot initial")
    print(f"    Barrière prot.    : {PROTECTION_BARRIER*100:>9.0f}% du spot initial")
    print(f"    Maturité max      : {OBSERVATION_DATES[-1]:>9.0f} ans")
    print(f"    Mémoire coupon    : {'Oui (Phoenix)' if COUPON_MEMORY else 'Non (Athena)'}")

    # 3. Simulation + pricing
    print(f"\n[3] Simulation & Pricing ({N_SIMULATIONS:,} trajectoires, variantes antithétiques)")
    paths = simulate_gbm(S0, r, sigma, OBSERVATION_DATES)
    price, ci_low, ci_high, payoffs, scenarios = price_product(paths, S0, r, OBSERVATION_DATES)

    print(f"    Prix Monte Carlo  : {price:>10,.4f} EUR")
    print(f"    IC 95%            :  [{ci_low:,.4f}, {ci_high:,.4f}]")
    print(f"    Erreur stat.      : ±{(ci_high-ci_low)/2:,.4f} EUR")
    print(f"    Prix / Nominal    : {price/NOMINAL*100:>9.2f}%")
    print(f"    Répartition:")
    print(f"      Rappels anticipés   : {(scenarios==0).mean()*100:5.1f}%")
    print(f"      Capital protégé     : {(scenarios==1).mean()*100:5.1f}%")
    print(f"      Perte en capital    : {(scenarios==2).mean()*100:5.1f}%")

    # 4. Greeks
    print(f"\n[4] Greeks (bump-and-reval, common random numbers)")
    print("    Calcul en cours...")
    delta, vega, theta = compute_greeks(S0, r, sigma)
    print(f"    Delta  : {delta:>10.4f}  EUR/pt")
    print(f"    Vega   : {vega:>10.4f}  EUR/unité sigma  "
          f"(soit {vega*0.01:+.4f} EUR par +1pp de vol)")
    print(f"    Theta  : {theta:>10.4f}  EUR/jour ouvré")

    # 5. Coupon d'équilibre
    print(f"\n[5] Coupon d'équilibre (dichotomie, émission au pair)")
    print("    Recherche du coupon maximum pour un prix cible = 100% du nominal...")
    print()
    coupon_eq = find_equilibrium_coupon(S0, r, sigma, target_price=NOMINAL)
    print(f"\n    Coupon d'équilibre : {coupon_eq*100:.4f}% par an")
    print(f"    (Le coupon de {COUPON_RATE*100:.1f}% fixé dans PARAMÈTRES était "
          f"{'trop généreux' if COUPON_RATE > coupon_eq else 'en dessous du marché'} "
          f"de {abs(COUPON_RATE - coupon_eq)*100:.2f}pp)")

    # 6. Visualisations
    print(f"\n[6] Visualisations")
    plot_payoff_distribution(payoffs, scenarios, price, ci_low, ci_high)
    plot_vega_curve(S0, r, sigma)

    # 7. Limites du modèle
    print("\n" + "=" * 62)
    print("  Limites du modèle:")
    print("  - Vol constante: pas de smile ni de skew (Black-Scholes).")
    print("    Le smile affecte la probabilité de franchissement des barrières.")
    print("  - Vol historique utilisée comme proxy de la vol implicite.")
    print("  - Dividendes du sous-jacent non modélisés.")
    print("  - Risque de crédit émetteur non capturé.")
    print("=" * 62)


if __name__ == "__main__":
    main()
