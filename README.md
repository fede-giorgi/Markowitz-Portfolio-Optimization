# 📈 Markowitz Portfolio Optimization

This project applies **Modern Portfolio Theory (MPT)** to real financial data.  
It starts with the **analysis of individual ETFs**, studying their return distributions, and then moves to the construction of optimal portfolios. 
---

## ETF Analysis

The first step focuses on the statistical properties of each asset: mean, volatility, and normality tests.  
For instance, the distribution of returns for the Spanish market (AMES.DE, IBEX 35 ETF) shows how individual series are modeled before entering the portfolio optimization stage:

![Return Distribution](AMES.DE_return_distribution.png)  

---

## Portfolio Optimization

Once the assets are analyzed, the tool generates thousands of random portfolios to simulate the **efficient frontier**.  
The result highlights the trade-off between risk and return, as well as the position of key portfolios:

- **Maximum Sharpe Ratio Portfolio**  
- **Minimum Variance Portfolio**  
- **Overall portfolio** combining risk-free and risky assets  

![Efficient Frontier](efficient_frontier.png)  

For the period **2015–2025**, the optimal portfolio achieved an **expected return of ~16.3%** with an **annualized volatility of ~9%**.
