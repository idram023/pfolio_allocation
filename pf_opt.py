import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import pypfopt as pfopt
from scipy.optimize import minimize
from typing import List, Tuple, Dict, Any
from matplotlib.pyplot import Axes
import yfinance as yf

# Constants
START_DATE = "2014-04-16"
END_DATE = "2024-04-16"
RISK_FREE_RATE = 0.046280
NUM_PORTFOLIOS = 1000000
stock_tickers = ['AAPL', 'MSFT', 'GOOG', 'JNJ', 'NVDA', '^GSPC', 'KO', 'LVMUY', 'PXD', 'ADBE', 'AVGO', 'ORCL', 'DELL', 'BABA', 'WMT', 'CSCO', 'RIOT', 'DIS', 'WBD', 'COIN', 'PARA', 'QCOM', 'PFE', 'MMM', 'LLY', 'PYPL', 'AMD', 'ABBV', 'CVX', 'INTC', 'AMZN', 'IBM', 'F', 'T', 'BA', 'META', 'TSLA', 'AXP', 'BX', 'PG', 'NFLX', 'MS', 'UAL', 'BAC', 'GS', 'BLK', 'WFC', 'JPM', 'C', 'DAL', 'DELL']

# Function to download stock prices
def download_stock_prices(stock_tickers: List[str], start_date: str, end_date: str) -> pd.DataFrame:
    """
    Downloads stock prices for a given list of tickers and date range.
    """
    price_data = yf.download(tickers=stock_tickers, start=start_date, end=end_date)["Adj Close"]
    return price_data

# Function to calculate daily returns
def calculate_daily_returns(price_data: pd.DataFrame) -> pd.DataFrame:
    """
    Calculates daily returns for a given price data.
    """
    daily_returns = price_data.pct_change()
    return daily_returns

# Function to plot normalized returns
def plot_normalized_returns(returns: pd.DataFrame) -> None:
    """
    Plots normalized returns for a given returns data.
    """
    (returns / returns.iloc[0] * 100).plot(figsize=(10, 5))
    plt.show()

# Function to plot daily returns volatility
def plot_daily_returns_volatility(returns: pd.DataFrame) -> None:
    """
    Plots daily returns volatility for a given returns data.
    """
    plt.figure(figsize=(14, 7))
    for ticker in stock_tickers:
        sns.lineplot(x=returns.index, y=returns[ticker], label=ticker)
    plt.legend(loc="lower center", fontsize=14)
    plt.ylabel("daily returns")

# Function to generate random portfolios
def random_portfolios(num_portfolios: int, mean_returns: np.ndarray, cov_matrix: np.ndarray) -> Tuple[np.ndarray, List[np.ndarray]]:
    """
    Generates random portfolios and calculates their performance.
    """
    np.random.seed(0)
    results = np.zeros((num_portfolios, 3))
    weights_record = []
    for i in range(num_portfolios):
        weights = np.random.random(len(mean_returns))
        weights /= np.sum(weights)
        weights = np.clip(weights, 0, 1)
        results[i,:] = portfolio_annualised_performance(weights, mean_returns, cov_matrix)
        weights_record.append(weights)
    return results, weights_record

# Function to calculate portfolio annualised performance
def portfolio_annualised_performance(weights: np.ndarray, mean_returns: np.ndarray, cov_matrix: np.ndarray) -> Tuple[float, float]:
    """
    Calculates annualised performance of a portfolio.
    """
    returns = np.sum(mean_returns * weights) * 251
    std = np.sqrt(np.dot(weights.T, np.dot(cov_matrix, weights))) * np.sqrt(251)
    return std, returns

# Function to calculate portfolio volatility
def portfolio_volatility(weights: np.ndarray, mean_returns: np.ndarray, cov_matrix: np.ndarray) -> float:
    """
    Calculates portfolio volatility.
    """
    return portfolio_annualised_performance(weights, mean_returns, cov_matrix)[0]

# Function to calculate portfolio Sharpe ratio
def portfolio_sharpe_ratio(weights: np.ndarray, mean_returns: np.ndarray, cov_matrix: np.ndarray, risk_free_rate: float) -> float:
    """
    Calculates Sharpe ratio of a portfolio.
    """
    portfolio_return = portfolio_annualised_performance(weights, mean_returns, cov_matrix)[1]
    sharpe_ratio = (portfolio_return - risk_free_rate) / portfolio_volatility(weights, mean_returns, cov_matrix)
    return sharpe_ratio

# Function to calculate portfolio allocation
def portfolio_allocation(weights: np.ndarray, mean_returns: np.ndarray) -> pd.DataFrame:
    """
    Calculates portfolio allocation.
    """
    portfolio_allocation = pd.DataFrame(weights, index=stock_tickers, columns=["allocation"])
    portfolio_allocation.allocation = [round(i * 100, 2) for i in portfolio_allocation.allocation]
    portfolio_allocation = portfolio_allocation.T
    return portfolio_allocation

# Function to calculate maximum Sharpe ratio portfolio
def max_sharpe_ratio(mean_returns: np.ndarray, cov_matrix: np.ndarray, risk_free_rate: float) -> Tuple[np.ndarray, float]:
    """
    Finds portfolio with maximum Sharpe ratio.
    """
    num_assets = len(mean_returns)
    args = (mean_returns, cov_matrix, risk_free_rate)
    constraints = ({'type': 'eq', 'fun' : lambda x: np.sum(x)-1})
    bound = (0.0, 1.0)
    bounds = tuple(bound for asset in range(num_assets))
    result = minimize(portfolio_volatility, num_assets * [1./num_assets,], args=args, method = 'SLSQP', bounds=bounds, constraints=constraints)
    max_sharpe_portfolio = result.x
    max_sharpe_ratio = portfolio_sharpe_ratio(max_sharpe_portfolio, mean_returns, cov_matrix, risk_free_rate)
    return max_sharpe_portfolio, max_sharpe_ratio

# Function to calculate efficient frontier
def efficient_frontier(mean_returns: np.ndarray, cov_matrix: np.ndarray, returns_range: np.ndarray) -> List[Dict[str, Any]]:
    """
    Calculates efficient frontier for a given range of returns.
    """
    efficient_portfolios = []
    for ret in returns_range:
        args = (mean_returns, cov_matrix, ret)
        constraints = ({'type': 'eq', 'fun' : lambda x: np.sum(x)-1})
        bound = (0.0, 1.0)
        bounds = tuple(bound for asset in range(len(mean_returns)))
        result = minimize(portfolio_volatility, len(mean_returns) * [1./len(mean_returns),], args=args, method = 'SLSQP', bounds=bounds, constraints=constraints)
        efficient_portfolios.append({
            "return": ret,
            "volatility": portfolio_volatility(result.x, mean_returns, cov_matrix),
            "allocation": portfolio_allocation(result.x, mean_returns)
        })
    return efficient_portfolios

# Function to display simulated efficient frontier with random portfolios
def display_simulated_ef_with_random(mean_returns: np.ndarray, cov_matrix: np.ndarray, num_portfolios: int, risk_free_rate: float) -> None:
    """
    Displays simulated efficient frontier with random portfolios.
    """
    results, weights = random_portfolios(num_portfolios, mean_returns, cov_matrix)
    max_sharpe_idx = np.argmax(results[:, 2])
    sdp, rp = results[max_sharpe_idx, 0], results[max_sharpe_idx, 1]
    max_sharpe_allocation = portfolio_allocation(weights[max_sharpe_idx], mean_returns)
    max_sharpe_allocation.allocation = [round(i * 100, 2) for i in max_sharpe_allocation.allocation]
    max_sharpe_allocation = max_sharpe_allocation.T

    min_vol_idx = np.argmin(results[:, 0])
    sdp_min, rp_min = results[min_vol_idx, 0], results[min_vol_idx, 1]
    min_vol_allocation = portfolio_allocation(weights[min_vol_idx], mean_returns)
    min_vol_allocation.allocation = [round(i * 100, 2) for i in min_vol_allocation.allocation]
    min_vol_allocation = min_vol_allocation.T

    max_sharpe_r = rp_max_sharpe_portfolio = pd.DataFrame(rp, index=["max sharpe portfolio"])
    min_vol_r = rp_min_vol_portfolio = pd.DataFrame(rp_min, index=["min vol portfolio"])

    ax = plot_efficient_frontier(mean_returns, cov_matrix, risk_free_rate)
    ax.set_title('Simulated Efficient Frontier with Random Portfolios')
    plt.xlabel('Expected Return')
    plt.ylabel('Volatility')
    plt.show()

    # Plot efficient frontier
    ax1 = plt.subplot(121)
    ax1.set_title('Efficient Frontier')
    ax1.plot(results[:, 1], results[:, 0], color='blue', label='Efficient Frontier')
    ax1.scatter(rp_max_sharpe_portfolio, sdp_max_sharpe_portfolio, color='red', label='Max Sharpe Ratio')
    ax1.scatter(rp_min_vol_portfolio, sdp_min_vol_portfolio, color='green', label='Min Volatility')
    ax1.set_xlabel('Volatility')
    ax1.set_ylabel('Expected Return')
    ax1.legend()

    # Plot allocation pie charts
    ax2 = plt.subplot(122)
    ax2.pie(max_sharpe_allocation.allocation, labels=max_sharpe_allocation.index, autopct='%1.1f%%')
    ax2.set_title('Max Sharpe Ratio Portfolio Allocation')

    ax3 = plt.subplot(122)
    ax3.pie(min_vol_allocation.allocation, labels=min_vol_allocation.index, autopct='%1.1f%%')
    ax3.set_title('Min Volatility Portfolio Allocation')

    plt.show()

# Function to display calculated efficient frontier with random portfolios
def display_calculated_ef_with_random(mean_returns: np.ndarray, cov_matrix: np.ndarray, num_portfolios: int, risk_free_rate: float) -> None:
    """
    Displays calculated efficient frontier with random portfolios.
    """
    results, _ = random_portfolios(num_portfolios, mean_returns, cov_matrix)

    max_sharpe = max_sharpe_ratio(mean_returns, cov_matrix, risk_free_rate)
    sdp, rp = portfolio_annualised_performance(max_sharpe[0], mean_returns, cov_matrix)
    max_sharpe_allocation = portfolio_allocation(max_sharpe[0], mean_returns)
    max_sharpe_allocation.allocation = [round(i * 100, 2) for i in max_sharpe_allocation.allocation]
    max_sharpe_allocation = max_sharpe_allocation.T

    min_vol = min_variance(mean_returns, cov_matrix)
    sdp_min, rp_min = portfolio_annualised_performance(min_vol[0], mean_returns, cov_matrix)
    min_vol_allocation = portfolio_allocation(min_vol[0], mean_returns)
    min_vol_allocation.allocation = [round(i * 100, 2) for i in min_vol_allocation.allocation]
    min_vol_allocation = min_vol_allocation.T

    max_sharpe_r = rp_max_sharpe_portfolio = pd.DataFrame(rp, index=["max sharpe portfolio"])
    min_vol_r = rp_min_vol_portfolio = pd.DataFrame(rp_min, index=["min vol portfolio"])

    ax = plot_efficient_frontier(mean_returns, cov_matrix, risk_free_rate)
    ax.set_title('Calculated Efficient Frontier with Random Portfolios')
    plt.xlabel('Expected Return')
    plt.ylabel('Volatility')
    plt.show()

    # Plot efficient frontier
    ax1 = plt.subplot(121)
    ax1.set_title('Efficient Frontier')
    ax1.plot(results[:, 1], results[:, 0], color='blue', label='Efficient Frontier')
    ax1.scatter(rp_max_sharpe_portfolio, sdp_max_sharpe_portfolio, color='red', label='Max Sharpe Ratio')
    ax1.scatter(rp_min_vol_portfolio, sdp_min_vol_portfolio, color='green', label='Min Volatility')
    ax1.set_xlabel('Volatility')
    ax1.set_ylabel('Expected Return')
    ax1.legend()

    # Plot allocation pie charts
    ax2 = plt.subplot(122)
    ax2.pie(max_sharpe_allocation.allocation, labels=max_sharpe_allocation.index, autopct='%1.1f%%')
    ax2.set_title('Max Sharpe Ratio Portfolio Allocation')

    ax3 = plt.subplot(122)
    ax3.pie(min_vol_allocation.allocation, labels=min_vol_allocation.index, autopct='%1.1f%%')
    ax3.set_title('Min Volatility Portfolio Allocation')

    plt.show()

# Function to display simulated efficient frontier with random portfolios and Sharpe ratio constraint
def display_simulated_ef_with_random_sfr(mean_returns: np.ndarray, cov_matrix: np.ndarray, num_portfolios: int, risk_free_rate: float) -> None:
    """
    Displays simulated efficient frontier with random portfolios and Sharpe ratio constraint.
    """
    results, weights = random_portfolios(num_portfolios, mean_returns, cov_matrix)
    max_sharpe_idx = np.argmax(results[:, 2])
    sdp, rp = results[max_sharpe_idx, 0], results[max_sharpe_idx, 1]
    max_sharpe_allocation = portfolio_allocation(weights[max_sharpe_idx], mean_returns)
    max_sharpe_allocation.allocation = [round(i * 100, 2) for i in max_sharpe_allocation.allocation]
    max_sharpe_allocation = max_sharpe_allocation.T

    min_vol_idx = np.argmin(results[:, 0])
    sdp_min, rp_min = results[min_vol_idx, 0], results[min_vol_idx, 1]
    min_vol_allocation = portfolio_allocation(weights[min_vol_idx], mean_returns)
    min_vol_allocation.allocation = [round(i * 100, 2) for i in min_vol_allocation.allocation]
    min_vol_allocation = min_vol_allocation.T



    max_sharpe_r = rp_max_sharpe_portfolio = pd.DataFrame(rp, index=["max sharpe portfolio"])
    min_vol_r = rp_min_vol_portfolio = pd.DataFrame(rp_min, index=["min vol portfolio"])

    ax = plot_efficient_frontier(mean_returns, cov_matrix, risk_free_rate)
    ax.set_title('Simulated Efficient Frontier with Random Portfolios and Sharpe Ratio Constraint')
    plt.xlabel('Expected Return')
    plt.ylabel('Volatility')
    plt.show()

    # Plot efficient frontier
    ax1 = plt.subplot(121)
    ax1.set_title('Efficient Frontier')
    ax1.plot(results[:, 1], results[:, 0], color='blue', label='Efficient Frontier')
    ax1.scatter(rp_max_sharpe_portfolio, sdp_max_sharpe_portfolio, color='red', label='Max Sharpe Ratio')
    ax1.scatter(rp_min_vol_portfolio, sdp_min_vol_portfolio, color='green', label='Min Volatility')
    ax1.set_xlabel('Volatility')
    ax1.set_ylabel('Expected Return')
    ax1.legend()

    # Plot allocation pie charts
    ax2 = plt.subplot(122)
    ax2.pie(max_sharpe_allocation.allocation, labels=max_sharpe_allocation.index, autopct='%1.1f%%')
    ax2.set_title('Max Sharpe Ratio Portfolio Allocation')

    ax3 = plt.subplot(122)
    ax3.pie(min_vol_allocation.allocation, labels=min_vol_allocation.index, autopct='%1.1f%%')
    ax3.set_title('Min Volatility Portfolio Allocation')

    plt.show()

# Function to display calculated efficient frontier with random portfolios and Sharpe ratio constraint
def display_calculated_ef_with_random_sfr(mean_returns: np.ndarray, cov_matrix: np.ndarray, num_portfolios: int, risk_free_rate: float) -> None:
    """
    Displays calculated efficient frontier with random portfolios and Sharpe ratio constraint.
    """
    results, _ = random_portfolios(num_portfolios, mean_returns, cov_matrix)

    max_sharpe = max_sharpe_ratio_sfr(mean_returns, cov_matrix, risk_free_rate)
    sdp, rp = portfolio_annualised_performance(max_sharpe[0], mean_returns, cov_matrix)
    max_sharpe_allocation = portfolio_allocation(max_sharpe[0], mean_returns)
    max_sharpe_allocation.allocation = [round(i * 100, 2) for i in max_sharpe_allocation.allocation]
    max_sharpe_allocation = max_sharpe_allocation.T

    min_vol = min_variance(mean_returns, cov_matrix)
    sdp_min, rp_min = portfolio_annualised_performance(min_vol[0], mean_returns, cov_matrix)
    min_vol_allocation = portfolio_allocation(min_vol[0], mean_returns)
    min_vol_allocation.allocation = [round(i * 100, 2) for i in min_vol_allocation.allocation]
    min_vol_allocation = min_vol_allocation.T

    max_sharpe_r = rp_max_sharpe_portfolio = pd.DataFrame(rp, index=["max sharpe portfolio"])
    min_vol_r = rp_min_vol_portfolio = pd.DataFrame(rp_min, index=["min vol portfolio"])

    ax = plot_efficient_frontier(mean_returns, cov_matrix, risk_free_rate)
    ax.set_title('Calculated Efficient Frontier with Random Portfolios and Sharpe Ratio Constraint')
    plt.xlabel('Expected Return')
    plt.ylabel('Volatility')
    plt.show()

    # Plot efficient frontier
    ax1 = plt.subplot(121)
    ax1.set_title('Efficient Frontier')
    ax1.plot(results[:, 1], results[:, 0], color='blue', label='Efficient Frontier')
    ax1.scatter(rp_max_sharpe_portfolio, sdp_max_sharpe_portfolio, color='red', label='Max Sharpe Ratio')
    ax1.scatter(rp_min_vol_portfolio, sdp_min_vol_portfolio, color='green', label='Min Volatility')
    ax1.set_xlabel('Volatility')
    ax1.set_ylabel('Expected Return')
    ax1.legend()

    # Plot allocation pie charts
    ax2 = plt.subplot(122)
    ax2.pie(max_sharpe_allocation.allocation, labels=max_sharpe_allocation.index, autopct='%1.1f%%')
    ax2.set_title('Max Sharpe Ratio Portfolio Allocation')

    ax3 = plt.subplot(122)
    ax3.pie(min_vol_allocation.allocation, labels=min_vol_allocation.index, autopct='%1.1f%%')
    ax3.set_title('Min Volatility Portfolio Allocation')

    plt.show()

# Function to calculate portfolio volatility using pypfopt
def portfolio_volatility_pypfopt(weights: np.ndarray, mean_returns: np.ndarray, cov_matrix: np.ndarray) -> float:
    """
    Calculates portfolio volatility using pypfopt.
    """
    portfolio = pfopt.Portfolio(mean_returns, weights)
    return portfolio.volatility()

# Function to calculate maximum Sharpe ratio portfolio using pypfopt
def max_sharpe_ratio_pypfopt(mean_returns: np.ndarray, cov_matrix: np.ndarray, risk_free_rate: float) -> Tuple[np.ndarray, float]:
    """
    Finds portfolio with maximum Sharpe ratio using pypfopt.
    """
    portfolio = pfopt.EfficientFrontier(mean_returns, cov_matrix)
    max_sharpe_portfolio = portfolio.max_sharpe(risk_free_rate=risk_free_rate)
    return max_sharpe_portfolio.weights, max_sharpe_portfolio.expected_return

# Function to calculate efficient frontier using pypfopt
def efficient_frontier_pypfopt(mean_returns: np.ndarray, cov_matrix: np.ndarray, returns_range: np.ndarray) -> List[Dict[str, Any]]:
    """
    Calculates efficient frontier for a given range of returns using pypfopt.
    """
    portfolio = pfopt.EfficientFrontier(mean_returns, cov_matrix)
    efficient_portfolios = []
    for ret in returns_range:
        portfolio.add_objective(lambda w: -w @ mean_returns, weight_constraint='eq', weight_constraint_value=1)
        portfolio.add_constraint(lambda w: w @ mean_returns >= ret)
        portfolio.add_constraint(lambda w: w @ cov_matrix @ w <= 1)
        portfolio.add_constraint(lambda w: w >= 0)
        portfolio.minimize()
        efficient_portfolios.append({
            "return": portfolio.portfolio.expected_return,
            "volatility": portfolio.portfolio.volatility,
            "allocation": portfolio.portfolio.weights
        })
    return efficient_portfolios

# Function to plot efficient frontier
def plot_efficient_frontier(mean_returns: np.ndarray, cov_matrix: np.ndarray, risk_free_rate: float)-> Axes:
    """
    Plots efficient frontier.
    """
    returns_range = np.linspace(np.min(mean_returns), np.max(mean_returns), 100)
    efficient_portfolios = efficient_frontier_pypfopt(mean_returns, cov_matrix, returns_range)
    x = [portfolio["volatility"] for portfolio in efficient_portfolios]
    y = [portfolio["return"] for portfolio in efficient_portfolios]
    plt.plot(x, y, color='blue', label='Efficient Frontier')
    plt.xlabel('Volatility')
    plt.ylabel('Expected Return')
    plt.scatter(portfolio_volatility_pypfopt(max_sharpe_portfolio_pypfopt[0], mean_returns, cov_matrix), max_sharpe_portfolio_pypfopt[1], color='red', label='Max Sharpe Ratio')
    plt.scatter(portfolio_volatility_pypfopt(min_vol_portfolio_pypfopt[0], mean_returns, cov_matrix), min_vol_portfolio_pypfopt[1], color='green', label='Min Volatility')
    plt.legend()
    return plt.gca()

# Main function
def main() -> None:
    # Download stock prices
    stock_tickers = ['AAPL', 'MSFT', 'GOOG', 'JNJ', 'NVDA', '^GSPC', 'KO', 'LVMUY', 'PXD', 'ADBE', 'AVGO', 'ORCL', 'DELL', 'BABA', 'WMT', 'CSCO', 'RIOT', 'DIS', 'WBD', 'COIN', 'PARA', 'QCOM', 'PFE', 'MMM', 'LLY', 'PYPL', 'AMD', 'ABBV', 'CVX', 'INTC', 'AMZN', 'IBM', 'F', 'T', 'BA', 'META', 'TSLA', 'AXP', 'BX', 'PG', 'NFLX', 'MS', 'UAL', 'BAC', 'GS', 'BLK', 'WFC', 'JPM', 'C', 'DAL', 'DELL']
    price_data = download_stock_prices(stock_tickers, START_DATE, END_DATE)

    # Calculate daily returns
    daily_returns = calculate_daily_returns(price_data)

    # Plot normalized returns
    plot_normalized_returns(daily_returns)

    # Plot daily returns volatility
    plot_daily_returns_volatility(daily_returns)

    # Calculate mean returns and covariance matrix
    mean_returns = daily_returns.mean()
    cov_matrix = daily_returns.cov()

    # Generate random portfolios
    num_portfolios = NUM_PORTFOLIOS
    results, weights = random_portfolios(num_portfolios, mean_returns, cov_matrix)

    # Calculate maximum Sharpe ratio portfolio
    max_sharpe_portfolio, max_sharpe_ratio = max_sharpe_ratio(mean_returns, cov_matrix, risk_free_rate)

    # Calculate minimum variance portfolio
    min_vol_portfolio = min_variance(mean_returns, cov_matrix)

    # Display simulated efficient frontier with random portfolios
    display_simulated_ef_with_random(mean_returns, cov_matrix, num_portfolios, risk_free_rate)

    # Display calculated efficient frontier with random portfolios
    display_calculated_ef_with_random(mean_returns, cov_matrix, num_portfolios, risk_free_rate)

    # Display simulated efficient frontier with random portfolios and Sharpe ratio constraint
    display_simulated_ef_with_random_sfr(mean_returns, cov_matrix, num_portfolios, risk_free_rate)

    # Display calculated efficient frontier with random portfolios and Sharpe ratio constraint
    display_calculated_ef_with_random_sfr(mean_returns, cov_matrix, num_portfolios, risk_free_rate)

    # Calculate portfolio volatility using pypfopt
    max_sharpe_portfolio_pypfopt = max_sharpe_ratio_pypfopt(mean_returns, cov_matrix, risk_free_rate)
    min_vol_portfolio_pypfopt = min_variance_pypfopt(mean_returns, cov_matrix)

    # Plot efficient frontier using pypfopt
    plot_efficient_frontier(mean_returns, cov_matrix, risk_free_rate)

if __name__ == '__main__':
    main()
