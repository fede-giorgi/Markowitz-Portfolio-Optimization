#%%
import requests
import pdfplumber
import re
import pandas as pd
import yfinance as yf
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from datetime import datetime, timedelta
import scipy.optimize as sco

###############################################################################
#                       INVESTIMENTO IN DIVERSI ETF                           #
###############################################################################

def get_the_data(verbose = False):
    # Scarica il PDF da iShares
    url = "https://www.ishares.com/us/literature/brochure/ishares-product-list-en-us.pdf"
    pdf_filename = "ishares-product-list-en-us.pdf"

    response = requests.get(url)
    with open(pdf_filename, "wb") as f:
        f.write(response.content)
    print(f"PDF salvato in {pdf_filename}")

    # Definisci un pattern per estrarre Fund Name, Trading Symbol e Expense Ratio
    pattern = re.compile(
        r"^(.*?)\s+([A-Z0-9]+)(?:\s+\d{1,2}/\d{1,2}/\d{2,4})?\s+(\d+\.\d+)$"
    )

    def extract_data_from_text(text):
        records = []
        for line in text.splitlines():
            line = line.strip()
            if not line:
                continue
            match = pattern.match(line)
            if match:
                fund_name = match.group(1).strip()
                trading_symbol = match.group(2).strip()
                expense_ratio = match.group(3).strip()
                records.append({
                    "Fund Name": fund_name,
                    "Trading Symbol": trading_symbol,
                    "Expense Ratio": expense_ratio
                })
        return records

    # Estrai i dati da tutte le pagine del PDF
    all_records = []
    with pdfplumber.open(pdf_filename) as pdf:
        total_pages = len(pdf.pages)
        if verbose:
            print(f"Numero totale di pagine nel PDF: {total_pages}")
        for i, page in enumerate(pdf.pages, start=1):
            text = page.extract_text()
            if text:
                page_records = extract_data_from_text(text)
                if verbose:    
                    print(f"Pagina {i}: trovati {len(page_records)} record")
                all_records.extend(page_records)
            else:
                if verbose:
                    print(f"Pagina {i}: nessun testo estratto")

    if all_records:
        df = pd.DataFrame(all_records)
        df.to_csv("ishares_extracted_full.csv", index=False)
        if verbose:
            print("I dati estratti sono stati salvati in 'ishares_extracted_full.csv'")
            print(df.head())
    else:
        if verbose:
            print("Nessun dato estratto. Verifica il pattern o la struttura del PDF.")
    return df

###############################################################################
#                FUNZIONI PER IL DOWNLOAD E LA PULIZIA DEI DATI               #
###############################################################################

def get_asset_data(tickers, use_adj_close=True):
    """
    Scarica i dati (Adj Close o Close) da Yahoo Finance.
    Restituisce un DataFrame con le colonne = Ticker e gli indici = Date.
    """
    start_date = "2015-01-01"
    end_date = "2025-01-01"
    
    if use_adj_close:
        data = yf.download(tickers, start=start_date, end=end_date, auto_adjust=False)
        price_data = data['Adj Close']
    else:
        data = yf.download(tickers, start=start_date, end=end_date)
        price_data = data['Close']
        
    print(f"Fetching data from {start_date} to {end_date}")
    return price_data

def compute_returns(data):
    """
    Calcola i rendimenti mensili utilizzando il prezzo di chiusura dell'ultimo giorno di ogni mese.
    Restituisce un DataFrame con i rendimenti percentuali mensili, rimuovendo eventuali NaN.
    """
    # Raggruppa per mese prendendo l'ultimo prezzo disponibile di ogni mese
    monthly_data = data.resample('M').last()
    # Calcola i rendimenti percentuali mensili e rimuovi eventuali NaN
    monthly_returns = monthly_data.pct_change().dropna()
    return monthly_returns

###############################################################################
#               FUNZIONI DI PORTAFOGLIO (MIN VAR, MAX SHARPE, ECC.)           #
###############################################################################

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

def max_sharpe_portfolio(mu, cov, risk_free_rate=0.025):
    """
    Calcola i pesi del portafoglio ottimale (portafoglio tangente) 
    in presenza di un tasso privo di rischio.

    La formula utilizzata è:
        w = inv(cov) * (mu - risk_free_rate) / (ones.T * inv(cov) * (mu - risk_free_rate))
    
    Argomenti:
      mu: array-like, vettore dei rendimenti attesi
      cov: array-like, matrice di covarianza degli asset
      risk_free_rate: float, tasso privo di rischio

    Ritorna:
      weights: numpy array, pesi ottimali per ciascun asset
    """
    mu = np.array(mu)
    cov = np.array(cov)
    ones = np.ones(len(mu))
    
    # Calcola i rendimenti in eccesso rispetto al tasso privo di rischio
    excess_returns = mu - risk_free_rate
    
    # Calcola l'inversa della matrice di covarianza
    inv_cov = np.linalg.inv(cov)
    
    # Calcola il numeratore: inv(cov) * (mu - risk_free_rate)
    numerator = np.dot(inv_cov, excess_returns)
    
    # Calcola il denominatore: somma dei pesi (1^T * inv(cov) * (mu - risk_free_rate))
    denominator = np.dot(ones, numerator)
    
    # Calcola i pesi normalizzati
    weights = numerator / denominator
    return weights

def generate_random_portfolios(mu, cov, num_portfolios=100000):
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

def plot_efficient_frontier(mu, cov, w_min, w_opt, random_results):
    """
    Plotta i portafogli casuali, il portafoglio min var (X) e max Sharpe (*).
    """
    plt.figure(figsize=(10, 6))
    
    # Portafogli casuali
    plt.scatter(random_results[0, :], random_results[1, :],
                c=random_results[2, :], cmap='viridis', alpha=0.5)
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
    
    plt.title("Efficient Frontier e Tradeoff Rischio-Rendimento")
    plt.xlabel("Rischio Annualizzato (Volatilità)")
    plt.ylabel("Rendimento Annualizzato")
    plt.legend()
    plt.show()

###############################################################################
#                   FUNZIONE PER ALLOCARE IL CAPITALE IN EURO                #
###############################################################################

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

def allocate_to_euros_commission(weights, capital, commission=1):
    """
    Alloca il capitale in euro tenendo conto di una commissione fissa per ogni operazione,
    includendo solo gli ETF con pesi positivi (gli ETF con peso zero riceveranno zero).
    
    Il capitale effettivo investibile è ridotto del costo commissione per ogni ETF con peso positivo.
    """
    weights = np.array(weights)
    # Identifica gli indici con peso positivo
    pos_indices = np.where(weights > 0)[0]
    n_assets = len(pos_indices)
    effective_capital = capital - commission * n_assets
    if effective_capital < 0:
        raise ValueError("Il capitale non è sufficiente per coprire le commissioni.")
    
    # Estrae i pesi positivi e li normalizza
    pos_weights = weights[pos_indices]
    pos_weights = pos_weights / pos_weights.sum()
    
    # Alloca il capitale effettivo solo per gli ETF con peso positivo
    allocated_pos = allocate_to_euros(pos_weights, effective_capital)
    
    # Crea un array di allocazione completo: per gli ETF non attivi (peso zero) assegna 0
    allocated = np.zeros_like(weights, dtype=int)
    allocated[pos_indices] = allocated_pos
    
    return allocated

###############################################################################
#                            MAIN PROGRAM                                     #
###############################################################################

def main(capital):
    """
    Esegue l'intero flusso: 
    1) utilizza una lista predefinita di ETF (inclusi bond),
    2) download dati con yfinance,
    3) calcolo di portafoglio Min Var e Max Sharpe (con risk free rate del 2,5%),
    4) allocazione in euro tenendo conto di commissioni da 1€,
    5) esportazione in Excel,
    6) plot della frontiera e stampa di expected return e variance.
    """
    assert capital > 0, "Il capitale deve essere un valore positivo."

    all_tickers = [
    "EZU",   # iShares MSCI Eurozone ETF
    "IEUR",  # iShares Core MSCI Europe ETF
    "EWU",   # iShares MSCI United Kingdom ETF
    "EUFN",  # iShares MSCI Europe Financials ETF
    "IEV",   # iShares Europe ETF
    "EWG",   # iShares MSCI Germany ETF
    "EWL",   # iShares MSCI Switzerland ETF
    "EWP",   # iShares MSCI Spain ETF
    "HEZU",  # iShares Currency Hedged MSCI Eurozone ETF
    "EWQ",   # iShares MSCI France ETF
    "EWI",   # iShares MSCI Italy ETF
    "EWD",   # iShares MSCI Sweden ETF
    "EPOL",  # iShares MSCI Poland ETF
    "EIS",   # iShares MSCI Israel ETF
    "EWN",   # iShares MSCI Netherlands ETF
    "EDEN",  # iShares MSCI Denmark ETF
    "TUR",   # iShares MSCI Turkey ETF
    "IEUS",  # iShares MSCI Europe Small-Cap ETF
    "EWO",   # iShares MSCI Austria ETF
    "EIRL",  # iShares MSCI Ireland ETF
    "EWUS",  # iShares MSCI United Kingdom Small-Cap ETF
    "ENOR",  # iShares MSCI Norway ETF
    "EFNL",  # iShares MSCI Finland ETF
    "EWK"    # iShares MSCI Belgium ETF
]

    # Scarica i dati e crea un DataFrame
    #df = get_the_data()
    #all_tickers = df["Trading Symbol"].to_list()
    data = get_asset_data(all_tickers, use_adj_close=True)
    print("Forma iniziale di 'data':", data.shape)


    # Pulizia: rimuove ticker e date completamente NaN
    data.dropna(axis=1, how='all', inplace=True)  # rimuove i ticker senza dati
    data.dropna(axis=0, how='all', inplace=True)  # rimuove le date senza dati
    print("Forma di 'data' dopo drop NaN:", data.shape)
    
    available_tickers = data.columns.to_list()
    if len(available_tickers) < 2:
        print("Non ci sono abbastanza ticker con dati validi per costruire un portafoglio.")
        return None
    
    # 3. Calcola i rendimenti e le statistiche annualizzate
    returns = compute_returns(data)
    if returns.empty or returns.shape[1] < 2:
        print("Non ci sono abbastanza dati di rendimento per calcolare un portafoglio.")
        return None
    
    mean_monthly_return = returns.mean()
    annualized_return = mean_monthly_return * 12
    annualized_cov = returns.cov() * 12
    
    # 4. Calcola i portafogli MinVar e MaxSharpe (long-only) 
    #    con risk free rate del 2,5% per il portafoglio Max Sharpe
    w_min = min_variance_portfolio(annualized_return, annualized_cov)
    w_opt = max_sharpe_portfolio(annualized_return, annualized_cov, risk_free_rate=0.025)
    
    # Calcola expected return e variance per il portafoglio Max Sharpe
    exp_return_opt = np.dot(w_opt, annualized_return)
    variance_opt = np.dot(w_opt, np.dot(annualized_cov, w_opt))
    std_dev_opt = np.sqrt(variance_opt)
    print(f"Portafoglio Max Sharpe: Expected Return = {exp_return_opt:.2%}, "
        f"Volatilità = {std_dev_opt:.2%}")

    # 5. Calcola l'allocazione in euro considerando le commissioni da 1€ per ogni ETF
    alloc_min = allocate_to_euros_commission(w_min, capital, commission=1)
    alloc_opt = allocate_to_euros_commission(w_opt, capital, commission=1)
    
    # Crea un DataFrame con i risultati
    df_portfolio = pd.DataFrame({
        "ETF": available_tickers,
        "Allocazione MinVar (EUR)": alloc_min,
        "Allocazione MaxSharpe (EUR)": alloc_opt
    })
    
    # Esporta in Excel
    df_portfolio.to_excel("portfolio.xlsx", index=False)
    print("\nIl file 'portfolio.xlsx' è stato creato con successo.")
    
    # 6. Genera portafogli casuali e plotta la frontiera
    random_results = generate_random_portfolios(annualized_return, annualized_cov)
    plot_efficient_frontier(annualized_return, annualized_cov, w_min, w_opt, random_results)
    
    # Restituisce il DataFrame per eventuali utilizzi successivi
    return df_portfolio
# %%
