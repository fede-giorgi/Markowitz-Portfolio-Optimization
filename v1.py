#%%
import os
from scipy.stats import norm
import pdfplumber
import re
import pandas as pd
import yfinance as yf
import matplotlib.pyplot as plt
from pylatex import Document, Section, Subsection, Tabular, Figure, NoEscape, Command, MiniPage
from scipy.stats import norm, probplot
import numpy as np
import scipy.optimize as sco
from scipy.optimize import minimize

def get_the_tickers():
    tickers = ["EUNL.DE",  #iShares Core MSCI World USD (Acc)
            "IS3N.DE", #iShares MSCI EM IMI USD (Acc) 
            "IUSQ.DE", #iShares Core ACWUI USD (Acc) 
            "SXRT.DE", #iShares Core STOXX 50 EUR (Acc)
            "SXR8.DE", #iShares Core S&P 500 USD (Acc)
            "SXRV.DE", #iShares NASDAQ100 USD (Acc)
            "SXRY.DE", #iShares FTSE MIB EUR EUR (Acc)
            "EXS1.DE", #iShares Core DAX EUR (Acc)
            "AMES.DE", #Amundi IBEX 35 EUR (Acc)
            "GC40.DE", #Amundi CAC 40 EUR (Acc)
            "SXRW.DE", #iShares FTSE 100 GBP (Acc)
            "CEBJ.DE", #iShares MSCI Korea USD (Acc)
            "SXRZ.DE", #iShares Nikkei 225 JPY (Acc)
            "ICGA.DE", #iShares MSCI China USD (Acc)
            "QDV5.DE", #iShares MSCI India USD (Acc)
            "IBC4.DE", #iShares MSCI South Africa USD (Acc)
            "PPFB.DE", #iShares Physical Gold USD (Acc)
            "IS0D.DE", #iShares Oil & Gas USD (Acc)
            "XDW0.DE", #iShares MSCI World Energy USD (Acc)
            ]

    return tickers

def get_asset_data(tickers, start_date="2015-01-01", end_date="2025-01-01"):
    """
    Scarica i gli "Adj Close" dei tickers selezionati nel periodo [start_data - end_date] da Yahoo Finance. Restituisce un DataFrame con colonne = Ticker e indici = Date.
    """
    data = yf.download(tickers, start=start_date, end=end_date, auto_adjust=False)
    price_data = data['Adj Close']
      
    print(f"Fetching data from {start_date} to {end_date}")
    return price_data

def compute_returns(data):
    """
    Calcola i returns giornalieri. Restituisce un DataFrame con i returns percentuali giornalieri, rimuovendo eventuali NaN.
    """
    # Raggruppa per mese prendendo l'ultimo prezzo disponibile per ciascun mese
    # monthly_prices = data.resample('M').last()
    # Calcola il rendimento percentuale  e rimuove i valori NaN
    returns = data.pct_change(fill_method=None).dropna()
    return returns


def analizza_tickers(tickers, returns_df, finestra=30):
    img_dir = 'immagini_tickers'
    os.makedirs(img_dir, exist_ok=True)

    geometry_options = {"top": "3.5cm", "left": "2cm", "right": "2cm", "bottom": "3.5cm"}
    doc = Document('Analisi_Completa_Portafoglio', geometry_options=geometry_options)
    doc.packages.append(NoEscape(r'\usepackage{graphicx}'))


    doc.preamble.append(Command('title', 'Analisi dei tickers'))
    doc.preamble.append(Command('author', 'Federico Giorgi'))
    doc.preamble.append(Command('date', NoEscape('')))
    doc.append(NoEscape('\maketitle'))

    with doc.create(Section('Analisi Completa dei Tickers', numbering=False)):
        for ticker in tickers:
            returns = returns_df[ticker].dropna()

            with doc.create(Subsection(f'Analisi del Ticker {ticker}', numbering=False)):

                # Statistiche riassuntive
                summary = returns.describe().to_frame().round(4)
                with doc.create(Tabular('lr')) as table:
                    for idx, row in summary.iterrows():
                        table.add_row(idx, row[ticker])

                # Immagine 1: Returns Giornalieri
                plt.figure(figsize=(8, 4))
                plt.plot(returns.index, returns, label="Returns giornalieri")
                plt.xlabel("Data")
                plt.ylabel("Rendimento")
                plt.title(f"Returns giornalieri per {ticker}")
                plt.legend()
                plt.grid(True)
                img_returns = os.path.join(img_dir, f'{ticker}_returns_plot.png')
                plt.savefig(img_returns, bbox_inches='tight')
                plt.close()
                img_returns = img_returns.replace('\\', '/')

                # Immagine 2: Equity Curve
                equity_curve = (1 + returns).cumprod()
                plt.figure(figsize=(8, 4))
                plt.plot(equity_curve.index, equity_curve, label="Equity Curve", color="green")
                plt.xlabel("Data")
                plt.ylabel("Valore Normalizzato")
                plt.title(f"Equity Curve per {ticker}")
                plt.legend()
                plt.grid(True)
                img_equity = os.path.join(img_dir, f'{ticker}_equity_curve.png')
                plt.savefig(img_equity, bbox_inches='tight')
                plt.close()
                img_equity = img_equity.replace('\\', '/')

                # Immagine 3: Distribuzione dei returns
                plt.figure(figsize=(8, 4))
                n, bins, _ = plt.hist(returns, bins=50, density=True, alpha=0.6, color='blue', label="Istogramma")
                mu, sigma = norm.fit(returns)
                x = np.linspace(min(bins), max(bins), 100)
                pdf = norm.pdf(x, mu, sigma)
                plt.plot(x, pdf, 'r', linewidth=2, label=f'Fit normale (mu={mu:.4f}, sigma={sigma:.4f})')
                plt.xlabel("Rendimento")
                plt.ylabel("Densità")
                plt.title(f"Distribuzione dei returns per {ticker}")
                plt.legend()
                plt.grid(True)
                img_distribuzione = os.path.join(img_dir, f'{ticker}_distribuzione_returns.png')
                plt.savefig(img_distribuzione, bbox_inches='tight')
                plt.close()
                img_distribuzione = img_distribuzione.replace('\\', '/')

                # Riga con 3 immagini
                with doc.create(Figure(position='htbp')):
                    for img in [img_returns, img_equity, img_distribuzione]:
                        with doc.create(MiniPage(width=NoEscape('0.31\\textwidth'))):
                            doc.append(NoEscape(f"\\includegraphics[width=\\linewidth]{{{img}}}"))

                # Immagine 4: QQ Plot
                plt.figure(figsize=(8, 4))
                probplot(returns, dist="norm", plot=plt)
                plt.title(f"QQ Plot dei returns per {ticker}")
                plt.grid(True)
                img_qqplot = os.path.join(img_dir, f'{ticker}_qq_plot.png')
                plt.savefig(img_qqplot, bbox_inches='tight')
                plt.close()
                img_qqplot = img_qqplot.replace('\\', '/')

                # Immagine 5: Rolling Averages
                media_mobile = returns.rolling(window=finestra).mean()
                varianza_mobile = returns.rolling(window=finestra).var()

                fig, ax1 = plt.subplots(figsize=(8, 4))
                ax1.plot(media_mobile.index, media_mobile, label="Media Mobile", color="blue")
                ax1.set_xlabel("Data")
                ax1.set_ylabel("Media Mobile")
                ax1.legend(loc='upper left')
                ax1.grid(True)

                ax2 = ax1.twinx()
                ax2.plot(varianza_mobile.index, varianza_mobile, label="Varianza Mobile", color="red")
                ax2.set_ylabel("Varianza Mobile")
                ax2.legend(loc='upper right')

                plt.title(f"Rolling Averages per {ticker}")
                img_rolling = os.path.join(img_dir, f'{ticker}_rolling_averages.png')
                plt.savefig(img_rolling, bbox_inches='tight')
                plt.close()
                img_rolling = img_rolling.replace('\\', '/')

                # Riga con 2 immagini
                with doc.create(Figure(position='htbp')):
                    for img in [img_qqplot, img_rolling]:
                        with doc.create(MiniPage(width=NoEscape('0.48\\textwidth'))):
                            doc.append(NoEscape(f"\\includegraphics[width=\\linewidth]{{{img}}}"))

    doc.generate_pdf(clean_tex=False, compiler='pdflatex')


def expected_return(weights, returns):
    return np.sum(returns.mean()*weights)*252

def standard_deviation(weights, cov_matrix):
    variance = weights.T @ cov_matrix @ weights
    return np.sqrt(variance)

def sharpe_ratio(weights, returns, cov_matrix, risk_free_rate):
    return (expected_return(weights, returns) - risk_free_rate) / standard_deviation(weights, cov_matrix)

def min_variance_portfolio(mu, cov):
    """
    Calcola il portafoglio a minima varianza (long-only) vincolato a pesi >= 0, sum(w)=1.
    """
    n = len(mu)
    cons = ({'type': 'eq', 'fun': lambda w: np.sum(w) - 1})
    bounds = tuple((0, 1) for _ in range(n))
    init_guess = np.array(n * [1.0/n])
    
    result = sco.minimize(lambda w: np.dot(w, np.dot(cov, w)),
                          init_guess,
                          method='SLSQP',
                          bounds=bounds,
                          constraints=cons)
    return result.x

def max_sharpe_portfolio(mu, cov, risk_free_rate):
    """
    Calcola i pesi del portafoglio ottimale (portafoglio tangente) 
    in presenza di un risk_free_rate.
    """
    def neg_sharpe_ratio(weights, mu, cov_matrix, risk_free_rate):
        return -sharpe_ratio(weights, mu, cov_matrix, risk_free_rate)

    constraints = {'type': 'eq', 'fun': lambda weights: np.sum(weights) - 1}
    bounds = [(0, 0.4) for _ in range(len(mu))]
    initial_weights = np.array([1/len(mu)]*len(mu))

    optimized_results = minimize(neg_sharpe_ratio, initial_weights, args=(mu, cov, risk_free_rate), method='SLSQP', constraints=constraints, bounds=bounds)
    return optimized_results.x

def allocate_capital(rf_rate, r_tangent, sigma_tangent, capital, target_return=None, target_risk=None):
    if abs(r_tangent - rf_rate) < 1e-9:
        raise ValueError("Il portafoglio tangente ha rendimento identico al risk-free, impossibile calcolare la leva.")
    
    if target_return is None and target_risk is None:
        raise ValueError("Devi specificare almeno target_return o target_risk.")
    
    if target_return is not None and target_risk is not None:
        raise ValueError("Puoi specificare o target_return o target_risk.")
    elif target_return is not None:
        alpha = (target_return - rf_rate) / (r_tangent - rf_rate)
    else:
        alpha = target_risk / sigma_tangent

    # Forza alpha a essere compreso tra 0 e 1
    alpha = max(0, min(1, alpha))

    capital_risky = alpha * capital
    capital_rf = (1 - alpha) * capital

    return alpha, capital_risky, capital_rf

def allocate_to_euros(weights, capital):
    """
    Trasforma i pesi (fra 0 e 1) in importi interi (arrotondati all'euro) 
    la cui somma è esattamente 'capital'.
    """
    capital_int = int(round(capital))
    raw_values = weights * capital_int
    floored_values = np.floor(raw_values).astype(int)
    leftover = capital_int - floored_values.sum()
    fractional_part = raw_values - floored_values
    sorted_idx = np.argsort(-fractional_part)
    i = 0
    while leftover > 0 and i < len(sorted_idx):
        floored_values[sorted_idx[i]] += 1
        leftover -= 1
        i += 1
    return floored_values

def generate_random_portfolios(mu, cov, num_portfolios=10000):
    """
    Genera portafogli casuali (long-only) per illustrare la frontiera.
    Restituisce un array con righe = [rischio, rendimento, Sharpe].
    """
    n = len(mu)
    results = np.zeros((3, num_portfolios))
    for i in range(num_portfolios):
        weights = abs(np.random.random(n))
        weights /= np.sum(weights)
        port_return = np.dot(weights, mu)
        port_risk = np.sqrt(np.dot(weights, np.dot(cov, weights)))
        sharpe = (port_return - 0.025) / port_risk if port_risk != 0 else 0
        results[0, i] = port_risk
        results[1, i] = port_return
        results[2, i] = sharpe
    return results

###############################################################################
#                           FUNZIONI DI PLOTTAGGIO                            #
###############################################################################

def plot_efficient_frontier(mu, cov, w_min, w_opt, random_results, overall_risk, overall_return):
    """
    Plotta i portafogli casuali, il portafoglio min var (X), max Sharpe e il portafoglio complessivo (con risk-free).
    """
    plt.figure(figsize=(20, 12))
    
    # Portafogli casuali
    plt.scatter(random_results[0, :], random_results[1, :],
                c=random_results[2, :], cmap='viridis', alpha=0.5)
    plt.xlim(0, 0.3)
    plt.ylim(0, 0.3)
    plt.colorbar(label='Sharpe Ratio')
    
    # Portafoglio max Sharpe
    risk_opt = np.sqrt(w_opt.T.dot(cov).dot(w_opt))
    ret_opt = np.dot(w_opt, mu)
    plt.scatter(risk_opt, ret_opt, marker='*', color='r', s=500,
                label='Optimal (Max Sharpe) Portfolio')
    
    # Portafoglio min var
    risk_min = np.sqrt(w_min.T.dot(cov).dot(w_min))
    ret_min = np.dot(w_min, mu)
    plt.scatter(risk_min, ret_min, marker='X', color='b', s=200,
                label='Minimum Variance Portfolio')
    
    # Portafoglio complessivo (risky + risk-free)
    plt.scatter(overall_risk, overall_return, marker='D', color='orange', s=300, label='Overall Portfolio (Risk-Free + Risky)')
    
    plt.title("Efficient Frontier e Tradeoff Rischio-Rendimento")
    plt.xlabel("Rischio Annualizzato (Volatilità)")
    plt.ylabel("Rendimento Annualizzato")
    plt.legend()
    plt.show()

def plot_portfolio_distribution(mu, sigma):
    """
    Plotta la distribuzione dei returns del portafoglio assumendo una distribuzione normale.
    """
    # Crea una griglia per l'asse x
    x = np.linspace(mu - 4*sigma, mu + 4*sigma, 1000)
    y = norm.pdf(x, loc=mu, scale=sigma)
    
    plt.figure(figsize=(10, 6))
    plt.plot(x, y)
    plt.xlabel('Rendimento del Portafoglio')
    plt.ylabel('Densità di Probabilità')
    plt.title('Distribuzione dei returns del Portafoglio')
    plt.legend()
    plt.grid(True)
    plt.show()

###############################################################################                            MAIN PROGRAM                                     ###############################################################################

def main(capital, target_return=None, target_risk=None, risk_free_rate=0.025):
    # Scarica i dati e crea un DataFrame
    all_tickers = get_the_tickers()
    data = get_asset_data(all_tickers)

    # Pulizia: rimuove ticker e date completamente NaN
    data.dropna(axis=1, how='all', inplace=True)  # rimuove i ticker senza dati
    data.dropna(axis=0, how='all', inplace=True)  # rimuove le date senza dati

    returns = compute_returns(data)  

    # Analizza i ticker
    analizza_tickers(all_tickers, returns, finestra=30)

    annualized_return = returns.mean() * 252
    annualized_cov = returns.cov() * 252
    
    # Calcola i portafogli MinVar e MaxSharpe
    w_min = min_variance_portfolio(annualized_return, annualized_cov)
    w_opt = max_sharpe_portfolio(annualized_return, annualized_cov, risk_free_rate)
    
    # Calcola expected return e variance per il portafoglio Max Sharpe
    exp_return_opt = expected_return(w_opt, returns)
    std_dev_opt = standard_deviation(w_opt, annualized_cov)
    print(f"Portafoglio Max Sharpe: \nExpected Return = {exp_return_opt:.2%}, \nVolatilità = {std_dev_opt:.2%}")

    # Allocazione del capitale tra risky asset e risk-free
    alpha, capital_risky, capital_rf = allocate_capital(risk_free_rate, exp_return_opt, std_dev_opt, capital, target_return, target_risk)
    capital_risky = allocate_to_euros(w_opt, capital_risky)

    # Calcola il rendimento e la volatilità del portafoglio complessivo
    overall_return = alpha * exp_return_opt + (1 - alpha) * risk_free_rate
    overall_risk = alpha * std_dev_opt
    print(f"Portafoglio Complessivo (Risk-Free + Risky):\nExpected Return = {overall_return:.2%}, \nVolatilità = {overall_risk:.2%}")

    # Creazione del DataFrame con i risultati risky e risk-free
    df_portfolio = pd.DataFrame({
        "ETF": data.columns.to_list(),
        "Allocazione MaxSharpe (EUR)": capital_risky
    })
    df_portfolio = df_portfolio[df_portfolio["Allocazione MaxSharpe (EUR)"] > 0]
    df_risk_free = pd.DataFrame({
        "ETF": ["Risk-Free"],
        "Allocazione MaxSharpe (EUR)": [int(round(capital_rf))]
    })
    df_portfolio = pd.concat([df_portfolio, df_risk_free], ignore_index=True)
    df_portfolio.to_excel("portfolio.xlsx", index=False)
    
    # Genera portafogli casuali e plotta la frontiera
    random_results = generate_random_portfolios(annualized_return, annualized_cov)
    plot_efficient_frontier(annualized_return, annualized_cov, w_min, w_opt, random_results, overall_risk, overall_return)
    
    # Plotta la distribuzione del portafoglio
    plot_portfolio_distribution(overall_return, overall_risk)
    
    # Calcolo dei VaR in termini di perdita assoluta (EUR)
    from scipy.stats import norm
    q1 = norm.ppf(0.01, loc=overall_return, scale=overall_risk)
    q5 = norm.ppf(0.05, loc=overall_return, scale=overall_risk)
    # Se il quantile è negativo, il VaR è la perdita attesa (altrimenti zero)
    VaR_1 = -q1 if q1 < 0 else 0
    VaR_5 = -q5 if q5 < 0 else 0
    VaR_1_eur = capital * VaR_1
    VaR_5_eur = capital * VaR_5
    print(f"VaR al 1%: {VaR_1_eur:.2f} EUR")
    print(f"VaR al 5%: {VaR_5_eur:.2f} EUR")

    return df_portfolio