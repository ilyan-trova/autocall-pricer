
# Phoenix Autocall Pricer

Moteur de pricing Monte Carlo pour produits structurés autocallables, développé en Python.  
Sous-jacent : **Eurostoxx 50** (données de marché réelles via Yahoo Finance).  
Modèle : **Black-Scholes** sous probabilité risque-neutre.

---

## Fonctionnalités

- Simulation de **100 000 trajectoires GBM** sous mesure risque-neutre Q
- Implémentation complète du mécanisme **Phoenix Autocall** : rappel anticipé, coupon avec mémoire, protection du capital à maturité
- **Réduction de variance** par variantes antithétiques
- **Intervalle de confiance à 95%** sur le prix (théorème central-limite)
- Calcul des **Greeks** par bump-and-reval avec common random numbers (Delta, Vega, Theta)
- **Recherche du coupon d'équilibre** par dichotomie : restitue le processus réel de structuration (émission au pair)
- **2 visualisations** : distribution des payoffs par scénario, courbe de sensibilité prix/volatilité

---

## Installation

```bash
pip install numpy pandas yfinance matplotlib
```

---

## Utilisation

```bash
python autocall_pricer.py
```

Les données de marché (spot, volatilité historique) sont téléchargées automatiquement.  
Les graphiques sont sauvegardés dans le répertoire courant (`distribution_payoffs.png`, `vega_curve.png`).

---

## Paramètres

Tous les paramètres sont modifiables en tête de fichier :

| Paramètre | Défaut | Description |
|---|---|---|
| `TICKER` | `^STOXX50E` | Sous-jacent (Yahoo Finance) |
| `NOMINAL` | 1 000 EUR | Montant nominal |
| `COUPON_RATE` | 7% | Coupon par période d'observation |
| `RECALL_BARRIER` | 100% | Barrière de rappel anticipé |
| `COUPON_BARRIER` | 70% | Barrière de versement du coupon |
| `PROTECTION_BARRIER` | 60% | Barrière de protection du capital |
| `OBSERVATION_DATES` | [1, 2, 3, 4, 5] | Dates d'observation (en années) |
| `COUPON_MEMORY` | True | Mémoire de coupon (Phoenix) |
| `RISK_FREE_RATE` | 3% | Taux OIS EUR (à actualiser) |
| `N_SIMULATIONS` | 100 000 | Trajectoires Monte Carlo |

---

## Résultats (Eurostoxx 50, juillet 2026)

```
Spot (S0)          : 6 204.91
Vol historique     : 15.48%  (252 séances)
Taux sans risque   : 3.00%   (OIS EUR)

Prix Monte Carlo   : 1 059.09 EUR  (105.91% du nominal)
IC 95%             : [1 058.45, 1 059.73]
Erreur statistique : ±0.64 EUR

Rappels anticipés  : 81.4%
Capital protégé    : 15.3%
Perte en capital   :  3.3%

Delta              :  0.0000  (scale-invariant à l'émission)
Vega               : -501.04  EUR/unité sigma  (-5.01 EUR par +1pp vol)
Theta              : +0.1666  EUR/jour ouvré

Coupon d'équilibre :  3.97%  (émission au pair, 15 itérations)
```

---

## Mécanisme du Phoenix Autocall

À chaque date d'observation :

| Condition | Événement |
|---|---|
| S(t) ≥ S₀ × 100% | **Rappel anticipé** : nominal + coupons (+ mémoire) |
| S₀ × 70% ≤ S(t) < S₀ × 100% | **Coupon versé** (+ mémoire accumulée) |
| S(t) < S₀ × 70% | **Pas de coupon** : mise en mémoire (Phoenix) |

À maturité (si jamais rappelé) :

| Condition | Payoff |
|---|---|
| S(T) ≥ S₀ × 60% | Remboursement intégral du nominal |
| S(T) < S₀ × 60% | Nominal × (S(T) / S₀) — perte proportionnelle |

---

## Fondements théoriques

**Mesure risque-neutre** : la simulation s'effectue sous Q (probabilité risque-neutre), non sous P (probabilité historique). Sous Q, tous les actifs ont un rendement espéré égal au taux sans risque r, ce qui garantit l'absence d'opportunité d'arbitrage (AOA). Le prix est l'espérance actualisée du payoff sous Q.

**Discrétisation exacte du GBM** :
```
S(t + dt) = S(t) × exp( (r - σ²/2) × dt + σ × √dt × Z ),   Z ~ N(0,1)
```
Le terme `(r - σ²/2)` est la correction d'Itô (convexité de l'exponentielle).

**Greeks (bump-and-reval)** : les dérivées analytiques n'existent pas pour un payoff à barrières discontinues. On approche chaque Greek par différences finies centrées (erreur en O(h²)), avec common random numbers pour annuler le bruit stochastique.

**Coupon d'équilibre** : la relation prix/coupon est strictement croissante. On inverse cette relation par dichotomie pour trouver le coupon maximum compatible avec une émission au pair. Convergence en O(log₂(n)), soit ~15 itérations.

---

## Limites du modèle

- **Volatilité constante** : Black-Scholes ignore le smile et le skew de volatilité, particulièrement impactants pour les barrières downside d'un autocall
- **Vol historique** utilisée comme proxy de la vol implicite (forward-looking)
- **Dividendes** non modélisés (ajustement recommandé : drift = r - q)
- **Risque de crédit émetteur** non capturé
- **Taux sans risque** fixé manuellement (à actualiser selon les conditions de marché)

---

## Stack technique

`Python` `NumPy` `pandas` `yfinance` `matplotlib`

---

## Structure du code

```
autocall_pricer.py
│
├── fetch_market_data()          # Spot et vol historique (Yahoo Finance)
├── simulate_gbm()               # Trajectoires GBM sous mesure risque-neutre Q
├── compute_payoffs()            # Payoff Phoenix Autocall + classification scénarios
├── price_product()              # Prix Monte Carlo + IC 95%
├── compute_greeks()             # Delta, Vega, Theta (bump-and-reval)
├── find_equilibrium_coupon()    # Coupon d'équilibre par dichotomie
├── plot_payoff_distribution()   # Distribution des payoffs par scénario
├── plot_vega_curve()            # Sensibilité prix/volatilité
└── main()                       # Point d'entrée
```
