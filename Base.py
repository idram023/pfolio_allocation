#P1 
# Apple(AAPL), LVMH(MC.PA), Coca-Cola(KO), Pioneer(PXD), Vertex(VTNR)
import numpy as np
import pandas as pd
from pandas_datareader import data as wb
import yfinance as yf
import matplotlib.pyplot as plt
%matplotlib inline
import scipy.optimize as sco 

#P2 # 
Creating a list of Stock Tickers
tickers = ['AAPL', 'MSFT', 'GOOG', 'JNJ', 'NVDA', '^GSPC', 'KO', 'LVMUY', 'PXD', 'ADBE', 'AVGO', 'ORCL', 'DELL', 'BABA', 'WMT', 'CSCO', 'RIOT', 'DIS', 'WBD', 'COIN', 'PARA', 'QCOM', 'PFE', 'MMM', 'LLY', 'PYPL', 'AMD', 'ABBV', 'CVX', 'INTC', 'AMZN', 'IBM', 'F', 'T', 'BA', 'META', 'TSLA', 'AXP', 'BX', 'PG', 'NFLX', 'MS', 'UAL', 'BAC', 'GS', 'BLK', 'WFC', 'JPM', 'C', 'DAL', 'DELL']
t = tickers
end = "2024-04-16"
start = "2014-04-16"

# Creating an empty DataFrame with the same number of rows as the number of tickers
pf_data = pd.DataFrame(index=pd.date_range(start, end))

# Downloading the adjusted close price for each ticker and add it to the DataFrame
pf_data = pf_data.join(yf.download(tickers=t, start=start, end=end)['Adj Close'])

#P3
# Viewing data
pf_data = pf_data.dropna()

#P4
# Plotting Normalized Returns
(pf_data / pf_data.iloc[0] * 100).plot(figsize=(10, 5))
plt.show()

#P5
# Plotting Daily Returns Volatility
returns = pf_data.pct_change()



plt.figure(figsize=(14, 7))

for i in tickers:

    plt.plot(returns.index, returns[i], lw=2, alpha=0.8,label=i)

plt.legend(loc='lower center', fontsize=14)

plt.ylabel('daily returns')

#P6
# Defining functions for our analysis
def portfolio_annualised_performance(weights, mean_returns, cov_matrix):
    returns = np.sum(mean_returns * weights) * 251
    std = np.sqrt(np.dot(weights.T, np.dot(cov_matrix, weights))) * np.sqrt(251)
    return std, returns

def random_portfolios(num_portfolios, mean_returns, cov_matrix, risk_free_rate):
    results = np.zeros((3, num_portfolios))
    weights_record = []
    for i in range(num_portfolios):
        weights = np.random.random(len(mean_returns))
        weights /= np.sum(weights)
        weights_record.append(weights)
        portfolio_std_dev, portfolio_return = portfolio_annualised_performance(weights, mean_returns, cov_matrix)
        results[0,i] = portfolio_std_dev
        results[1,i] = portfolio_return
        results[2,i] = (portfolio_return - risk_free_rate) / portfolio_std_dev
    return results, weights_record

def neg_sharpe_ratio(weights, mean_returns, cov_matrix, risk_free_rate):
    p_var, p_ret = portfolio_annualised_performance(weights, mean_returns, cov_matrix)
    return -(p_ret - risk_free_rate) / p_var

def max_sharpe_ratio(mean_returns, cov_matrix, risk_free_rate):
    num_assets = len(mean_returns)
    args = (mean_returns, cov_matrix, risk_free_rate)
    constraints = ({'type': 'eq', 'fun' : lambda x: np.sum(x)-1})
    bound = (0.0, 1.0)
    bounds = tuple(bound for asset in range(num_assets))
    result = sco.minimize(neg_sharpe_ratio, num_assets * [1./num_assets,], args=args, method = 'SLSQP', bounds=bounds, constraints=constraints)
    return result

def portfolio_volatility(weights, mean_returns, cov_matrix):
    return portfolio_annualised_performance(weights, mean_returns, cov_matrix)[0]

def min_variance(mean_returns, cov_matrix):
    num_assets = len(mean_returns)
    args = (mean_returns, cov_matrix)
    constraints = ({'type': 'eq', 'fun' : lambda x: np.sum(x)-1})
    bound = (0.0, 1.0)
    bounds = tuple(bound for asset in range(num_assets))
    result = sco.minimize(portfolio_volatility, num_assets * [1./num_assets,], args=args, method = 'SLSQP', bounds=bounds, constraints=constraints)
    return result

def efficient_return(mean_returns, cov_matrix, target):
    num_assets = len(mean_returns)
    args = (mean_returns, cov_matrix)

def portfolio_return(weights):
    return portfolio_annualised_performance(weights, mean_returns, cov_matrix)[1]
    constraints = ({'type': 'eq', 'fun' : lambda x: portfolio_return(x) - target},
                   {'type': 'eq', 'fun': lambda x: np.sum(x) - 1})
    bounds = tuple((0,1) for asset in range(num_assets))
    result = sco.minimize(portfolio_volatility, num_assets * [1./rixnum_assets,], args=args, method = 'SLSQP', bounds=bounds, constraints=constraints)
    return result

def efficient_frontier(mean_returns, cov_matrix, returns_range):
        efficients = []
        for ret in returns_range:
            efficients.append(efficient_return(mean_returns, cov_matrix, ret))
            return efficients

def display_simulated_ef_with_random(mean_returns, cov_matrix, num_portfolio, risk_free_rate):

    results, weights = random_portfolios(num_portfolios, mean_returns, cov_matrix, risk_free_rate)

    max_sharpe_idx = np.argmax(results[2])
    sdp, rp = results[0, max_sharpe_idx], results[1, max_sharpe_idx]
    max_sharpe_allocation = pd.DataFrame(weights[max_sharpe_idx], index=pf_data.columns, columns = ['allocation'])
    max_sharpe_allocation.allocation = [round(i * 100, 2) for i in max_sharpe_allocation.allocation]
    max_sharpe_allocation = max_sharpe_allocation.T

    min_vol_idx = np.argmin(results[0])
    sdp_min, rp_min = results[0,min_vol_idx], results[1, min_vol_idx]
    min_vol_allocation = pd.DataFrame(weights[min_vol_idx], index=pf_data.columns, columns = ['allocation'])
    min_vol_allocation.allocation = [round(i * 100, 2) for i in min_vol_allocation.allocation]
    min_vol_allocation = min_vol_allocation.T

    print('-' * 80)
    print('Maximum Sharpe Ratio Portfolio Allocation\n')
    print('Annualised Return:', round(rp, 2))
    print('Annualised Volatility:', round(sdp, 2))
    print('\n')
    print(max_sharpe_allocation)
    print('-' * 80)
    print('Minimum Volatility Portfolio Allocation\n')
    print('Annualised Return:', round(rp_min, 2))
    print('Annualised Volatility:', round(sdp_min, 2))
    print('\n')
    print(min_vol_allocation)

    plt.figure(figsize = (10,7))
    plt.scatter(results[0,:], results[1,:], c = results[2,:], cmap = 'YlGnBu', marker = 'o', s=10, alpha=0.3)
    plt.colorbar()
    plt.scatter(sdp, rp, marker = '*', color = 'r', s = 500, label = 'Maximum Sharpe Ratio')
    plt.scatter(sdp_min, rp_min, marker = '*', color = 'g', s=500, label = 'Minimum Volatility')
    plt.title('Simulated Portfolio Optimization based on Efficient Frontier')
    plt.xlabel('Annualised Volatility')
    plt.ylabel('Annualised Returns')
    plt.legend(labelspacing = 0.8)

def display_calculated_ef_with_random(mean_returns, cov_matrix, num_portfolios, risk_free_rate):
    results, _ = random_portfolios(num_portfolios, mean_returns, cov_matrix, risk_free_rate)

    max_sharpe = max_sharpe_ratio(mean_returns, cov_matrix, risk_free_rate)
    sdp, rp = portfolio_annualised_performance(max_sharpe['x'], mean_returns, cov_matrix)
    max_sharpe_allocation = pd.DataFrame(max_sharpe.x, index=pf_data.columns, columns = ['allocation'])
    max_sharpe_allocation.allocation = [round(i*100, 2) for i in max_sharpe_allocation.allocation]
    max_sharpe_allocation = max_sharpe_allocation.T

    min_vol = min_variance(mean_returns, cov_matrix)
    sdp_min, rp_min = portfolio_annualised_performance(min_vol['x'], mean_returns, cov_matrix)
    min_vol_allocation = pd.DataFrame(min_vol.x, index=pf_data.columns, columns = ['allocation'])
    min_vol_allocation.allocation = [round(i*100, 2) for i in min_vol_allocation.allocation]
    min_vol_allocation = min_vol_allocation.T

    print('-' * 80)
    print('Maximum Sharpe Ratio Portfolio Allocation\n')
    print('Annualised Return:', round(rp, 2))
    print('Annualised Volatility:', round(sdp, 2))
    print('\n')
    print(max_sharpe_allocation)
    print('-' * 80)
    print('Minimum Volatility Portfolio Allocation\n')
    print('Annualised Return:', round(rp_min, 2))
    print('Annualised Volatility:', round(sdp_min, 2))
    print('\n')
    print(min_vol_allocation)

    plt.figure(figsize = (10,7))
    plt.scatter(results[0,:], results[1,:], c = results[2,:], cmap = 'YlGnBu', marker = 'o', s=10, alpha=0.3)
    plt.colorbar()
    plt.scatter(sdp, rp, marker = '*', color = 'r', s = 500, label = 'Maximum Sharpe Ratio')
    plt.scatter(sdp_min, rp_min, marker = '*', color = 'g', s=500, label = 'Minimum Volatility')

    target = np.linspace(rp_min, 0.40, 50)
    efficient_portfolios = efficient_frontier(mean_returns, cov_matrix, target)

    plt.title('Calculated Portfolio Optimization based on Efficient Frontier')
    plt.xlabel('Annualised Volatility')
    plt.ylabel('Annualised Returns')
    plt.legend(labelspacing = 0.8)

#P7
# Variables needed for the functions
returns = pf_data.pct_change()
mean_returns = returns.mean()
cov_matrix = returns.cov()
num_portfolios = 1000000
risk_free_rate = 0.046280

#P8
# Simulated Portfolios based on Efficient Frontier
display_simulated_ef_with_random(mean_returns, cov_matrix, num_portfolios, risk_free_rate)

#P9
# Calculating Portfolios based on Efficient Frontier
display_calculated_ef_with_random(mean_returns, cov_matrix, num_portfolios, risk_free_rate)

#P10
def display_simulated_ef_with_random_sfr(mean_returns,cov_matrix,num_portfolios, risk_free_rate):
    results, weights = random_portfolios(num_portfolios, mean_returns, cov_matrix, risk_free_rate)
    max_sharpe_idx = np.argmax(results[2])
    sdp, rp = results[0,max_sharpe_idx], results[1,max_sharpe_idx]
    max_sharpe_allocation = pd.DataFrame(weights[max_sharpe_idx], index = pf_data.columns, columns = ['allocation'])
    max_sharpe_allocation.allocation = [round(i*100,2) for i in max_sharpe_allocation.allocation]
    max_sharpe_allocation = max_sharpe_allocation.T

    min_vol_idx = np.argmin(results[0])
    sdp_min, rp_min = results[0,min_vol_idx], results[1, min_vol_idx]
    min_vol_allocation = pd.DataFrame(weights[min_vol_idx], index=pf_data.columns, columns = ['allocation'])
    min_vol_allocation.allocation = [round(i * 100, 2) for i in min_vol_allocation.allocation]
    min_vol_allocation = min_vol_allocation.T

    print('-'*80)
    print('Maximum Sharpe Ratio Portfolio Allocation\n')
    print('Annualised Return:', round(rp,2))
    print('Annualised Volatility:', round(sdp, 2))
    print('\n')
    print(max_sharpe_allocation)
    print('-'*80)
    print('Minimum Volatility Portfolio Allocation\n')
    print('Annualised Return:', round(rp_min, 2))
    print('Annualised Volatility:', round(sdp, 2))
    print('\n')
    print(min_vol_allocation)

    plt.figure(figsize=(10,7))

    plt.scatter(results[0,:], results[1,:], c=results[2,:], cmap='YlGnBu', marker='o', s=10, alpha=0.3)
    plt.colorbar()
    plt.scatter(sdp, rp, marker='*', color='r', s=500, label='Maximum Sharpe ratio')
    plt.title('Simulated Portfolio Optimization based on Efficient Frontier')
    plt.xlabel('Annualised Volatility')
    plt.ylabel('Annualised Returns')
    plt.legend(labelspacing = 0.8)

def display_calculated_ef_with_random_sfr(mean_returns, cov_matrix, num_portfolio, risk_free_rate):
    results, _ = random_portfolios(num_portfolios, mean_returns, cov_matrix, risk_free_rate)

    max_sharpe = max_sharpe_ratio(mean_returns, cov_matrix, risk_free_rate)
    sdp, rp = portfolio_annualised_performance(max_sharpe['x'], mean_returns, cov_matrix)
    max_sharpe_allocation = pd.DataFrame(max_sharpe.x, index = pf_data.columns, columns = ['allocation'])
    max_sharpe_allocation.allocation = [round(i*100,2) for i in max_sharpe_allocation.allocation]
    max_sharpe_allocation = max_sharpe_allocation.T

    min_vol = min_variance(mean_returns, cov_matrix)
    sdp_min, rp_min = portfolio_annualised_performance(min_vol['x'], mean_returns, cov_matrix)
    min_vol_allocation= pd.DataFrame(min_vol.x, index=pf_data.columns, columns=['allocation'])
    min_vol_allocation.allocation = [round(i*100, 2) for i in min_vol_allocation.allocation]
    min_vol_allocation = min_vol_allocation.T

    print('-'*80)
    print('Maximum Sharpe Ratio Portfolio Allocation\n')
    print('Annualised Return:', round(rp,2))
    print('Annualised Volatility:', round(sdp, 2))
    print('\n')
    print(max_sharpe_allocation)
    print('-'*80)
    print('Minimum Volatility Portfolio Allocation\n')
    print('Annualised Return:', round(rp_min, 2))
    print('Annualised Volatility:', round(sdp, 2))
    print('\n')
    print(min_vol_allocation)

    plt.figure(figsize=(10,7))

    plt.scatter(results[0,:], results[1,:], c=results[2,:], cmap='YlGnBu', marker='o', s=10, alpha=0.3)
    plt.colorbar()
    plt.scatter(sdp, rp, marker='*', color='r', s=500, label='Maximum Sharpe ratio')
    target = np.linspace(rp_min, 0.60, 50)
    efficient_portfolios = efficient_frontier(mean_returns, cov_matrix, target)
    plt.title('Calculated Portfolio Optimization based on Efficient Frontier')
    plt.xlabel('Annualised Volatility')
    plt.ylabel('Annualised Returns')
    plt.legend(labelspacing = 0.8)

#P11
returns = pf_data.pct_change()
mean_returns = returns.mean()
cov_matrix = returns.cov()
num_portfolios = 1000000
risk_free_rate = 0.046280

#P12
# Simulating Portfolios based on Efficient Frontier (Minimum Acceptable Return 20%)
display_simulated_ef_with_random_sfr(mean_returns, cov_matrix, num_portfolios, risk_free_rate)

#P13
# Calculating Portfolios based on Efficient Frontier
display_calculated_ef_with_random_sfr(mean_returns, cov_matrix, num_portfolios, risk_free_rate)

#P14
def utility_optimal_portfolio(data, risk_aversion_coeff):
    # Importing libraries
    from pypfopt import EfficientFrontier
    from pypfopt import risk_models
    from pypfopt import expected_returns
    from pypfopt import objective_functions

    # Expected Returns
    mu = expected_returns.mean_historical_return(pf_data)

    # Expected Volatility
    Sigma = risk_models.sample_cov(pf_data)
    ef = EfficientFrontier(mu, Sigma) #Setup
    ef.add_objective(objective_functions.L2_reg) # add a secondary objective
    weights = ef.max_quadratic_utility(risk_aversion = risk_aversion_coeff, market_neutral = False) # find the portfolio that maximises utility
    ret, vol, sharpe_r = ef.portfolio_performance(risk_free_rate=0.04532)

    # loop to iterate for values
    res = dict()
    for key in weights :
    # Rounding to K using round()
        res[key] = round(weights[key], 3)

    return 'Allocation ' + str(res), 'Annualised Return ' + str(round(ret, 3)), 'Annualised Volatility ' + str(round(vol, 3)), 'Sharpe Ratio '+ str(round(sharpe_r, 3))

#P15
# Aggresive Investor
utility_optimal_portfolio(pf_data, 1)

#P16
# Moderate Investor
utility_optimal_portfolio(pf_data, 4)

#P17
# Risk_Averse Invesotr
utility_optimal_portfolio(pf_data, 10)
