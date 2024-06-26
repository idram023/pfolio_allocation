import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import pypfopt as pfopt
from scipy.optimize import minimize
from typing import List, Tuple, Dict, Any
from matplotlib.pyplot import Axes
import yfinance as yf
import numexpr as ne
from functools import partial

# Constants
START_DATE = "2014-04-16"
END_DATE = "2024-04-16"
RISK_FREE_RATE = 0.046280
NUM_PORTFOLIOS = 100000
stock_tickers = ['AAPL', 'MSFT', 'GOOG', 'JNJ', 'NVDA', '^GSPC', 'KO', 'LVMUY', 'PXD', 'ADBE', 'AVGO', 'ORCL', 'DELL', 'BABA', 'WMT', 'CSCO', 'RIOT', 'DIS', 'WBD', 'COIN', 'PARA', 'QCOM', 'PFE', 'MMM', 'LLY', 'PYPL', 'AMD', 'ABBV', 'CVX', 'INTC', 'AMZN', 'IBM', 'F', 'T', 'BA', 'META', 'TSLA', 'AXP', 'BX', 'PG', 'NFLX', 'MS', 'UAL', 'BAC', 'GS', 'BLK', 'WFC', 'JPM', 'C', 'DAL']

def download_stock_prices(stock_tickers: List[str], start_date: str, end_date: str) -> pd.DataFrame:
    """
    Downloads stock prices for a given list of tickers and date range.
    """
    try:
        price_data = yf.download(tickers=stock_tickers, start=start_date, end=end_date)["Adj Close"]
        return price_data
    except Exception as e:
        print(f"Error downloading stock prices: {e}")
        return None

def calculate_daily_returns(price_data: pd.DataFrame) -> pd.DataFrame:
    """
    Calculates daily returns for a given price data.
    """
    try:
        daily_returns = price_data.pct_change()
        return daily_returns
    except Exception as e:
        print(f"Error calculating daily returns: {e}")
        return None

def plot_normalized_returns(returns: pd.DataFrame) -> None:
    """
    Plots normalized returns for a given returns data.
    """
    try:
        (returns / returns.iloc[0] * 100).plot(figsize=(10, 5))
        plt.show()
    except Exception as e:
        print(f"Error plotting normalized returns: {e}")

def plot_daily_returns_volatility(returns: pd.DataFrame) -> None:
    """
    Plots daily returns volatility for a given returns data.
    """
    try:
        plt.figure(figsize=(14, 7))
        for ticker in stock_tickers:
            sns.lineplot(x=returns.index, y=returns[ticker], label=ticker)
        plt.legend(loc="lower center", fontsize=14)
        plt.ylabel("daily returns")
    except Exception as e:
        print(f"Error plotting daily returns volatility: {e}")
def portfolio_volatility(weights: np.ndarray, mean_returns: np.ndarray, cov_matrix: np.ndarray) -> float:
    """
    Calculates portfolio volatility.
    """
    try:
        return np.sqrt(np.dot(weights.T, np.dot(cov_matrix, weights))) * np.sqrt(251)
    except ValueError:
        return float('inf')
    except RuntimeWarning:
        return float('inf')

def portfolio_sharpe_ratio(weights: np.ndarray, mean_returns: np.ndarray, cov_matrix: np.ndarray, risk_free_rate: float) -> float:
    """
    Calculates Sharpe ratio of a portfolio.
    """
    portfolio_return = np.sum(mean_returns * weights) * 251
    sharpe_ratio = (portfolio_return - risk_free_rate) / portfolio_volatility(weights, mean_returns, cov_matrix)
    return sharpe_ratio

def max_sharpe_ratio(mean_returns: np.ndarray, cov_matrix: np.ndarray, risk_free_rate: float) -> Tuple[np.ndarray, float]:
    """
    Calculates the maximum Sharpe ratio portfolio.
    """
    num_assets = len(mean_returns)
    initial_weights = np.ones(num_assets) / num_assets
    constraints = ({'type': 'eq', 'fun': lambda x: np.sum(x) - 1})
    bounds = tuple((0, 1) for asset in range(num_assets))
    result = minimize(portfolio_volatility, initial_weights, args=(mean_returns, cov_matrix), method='SLSQP', bounds=bounds, constraints=constraints)
    weights = result.x
    sharpe_ratio = (mean_returns @ weights - risk_free_rate) / portfolio_volatility(weights, mean_returns, cov_matrix)
    return weights, sharpe_ratio

def min_variance(mean_returns: np.ndarray, cov_matrix: np.ndarray) -> Tuple[np.ndarray, float]:
    """
    Finds portfolio with minimum variance.
    """
    num_assets = len(mean_returns)
    args = (mean_returns, cov_matrix)
    constraints = ({'type': 'eq', 'fun' : lambda x: np.sum(x)-1})
    bound = (0.0, 1.0)
    bounds = tuple(bound for asset in range(num_assets))
    result = minimize(portfolio_volatility, num_assets * [1./num_assets,], args=args, method = 'SLSQP', bounds=bounds, constraints=constraints)
    min_vol_portfolio = result.x
    min_vol_variance = np.sqrt(np.dot(min_vol_portfolio.T, np.dot(cov_matrix, min_vol_portfolio))) * np.sqrt(251)
    return min_vol_portfolio, min_vol_variance


def random_portfolios(num_portfolios: int, mean_returns: np.ndarray, cov_matrix: np.ndarray) -> Tuple[np.ndarray, List[np.ndarray]]:
    """
    Generates random portfolios and calculates their performance.
    """
    np.random.seed(0)
    num_assets = len(mean_returns)
    results = np.zeros((num_portfolios, 3))
    weights_record = []
    for i in range(num_portfolios):
        weights = np.random.random(num_assets)
        weights /= np.sum(weights)
        weights = np.clip(weights, 0, 1)
        weights = weights.reshape(num_assets,)
        results[i,:] = portfolio_annualised_performance_vectorized(weights, mean_returns, cov_matrix)
        weights_record.append(weights)
    return results, weights_record

def portfolio_annualised_performance_vectorized(weights: np.ndarray, mean_returns: np.ndarray, cov_matrix: np.ndarray) -> Tuple[float, float]:
    """
    Calculates the annualised performance of a portfolio using NumPy vectorized operations.
    """
    num_assets = len(mean_returns)
    returns = np.sum(mean_returns * weights)
    std_dev = np.sqrt(np.dot(weights.T, np.dot(cov_matrix, weights)))
    annualised_return = (1 + returns) ** 252 - 1

def plot_efficient_frontier(mean_returns: np.ndarray, cov_matrix: np.ndarray, risk_free_rate: float, ax: Axes = None) -> Axes:
    """
    Plots efficient frontier.
    """
    returns_range = np.linspace(np.min(mean_returns), np.max(mean_returns), 100)
    efficient_portfolios = efficient_frontier(mean_returns, cov_matrix, returns_range)
    x = [portfolio["volatility"] for portfolio in efficient_portfolios]
    y = [portfolio["return"] for portfolio in efficient_portfolios]
    if ax is None:
        plt.plot(x, y, color='blue', label='Efficient Frontier')
    else:
        ax.plot(x, y, color='blue', label='Efficient Frontier')
    ax.set_xlabel('Volatility')
    ax.set_ylabel('Expected Return')
    ax.set_title('Efficient Frontier')
    ax.scatter(portfolio_volatility_pypfopt(max_sharpe_portfolio_pypfopt[0], mean_returns, cov_matrix), max_sharpe_portfolio_pypfopt[1], color='red', label='Max Sharpe Ratio')
    ax.scatter(portfolio_volatility_pypfopt(min_vol_portfolio_pypfopt[0], mean_returns, cov_matrix), min_vol_portfolio_pypfopt[1], color='green', label='Min Volatility')
    ax.legend()
    return ax

def efficient_frontier(mean_returns: np.ndarray, cov_matrix: np.ndarray, returns_range: np.ndarray) -> List[Dict[str, Any]]:
    """
    Calculates efficient frontier for a given range of returns.
    """
    efficient_portfolios = []
    args = (mean_returns, cov_matrix)
    for ret in returns_range:
        constraints = ({'type': 'eq', 'fun' : lambda x: np.sum(x)-1})
        bound = (0.0, 1.0)
        bounds = tuple(bound for asset in range(len(mean_returns)))
        portfolio_volatility_partial = partial(portfolio_volatility, *args)

        # Add print statements to debug the function
        print("Arguments passed to portfolio_volatility_partial:", args)
        print("Returns range:", ret)

        result = minimize(portfolio_volatility_partial, len(mean_returns) * [1./len(mean_returns),], method='SLSQP', bounds=bounds, constraints=constraints)

        portfolio_volatility_value = portfolio_volatility(*result.x, *args)
        print("Volatility calculated by portfolio_volatility:", portfolio_volatility_value)

        efficient_portfolios.append({
            "return": ret,
            "volatility": portfolio_volatility_value,
            "allocation": portfolio_allocation(result.x, mean_returns)
        })
    return efficient_portfolios

def portfolio_allocation(weights: np.ndarray, mean_returns: np.ndarray) -> pd.DataFrame:
    """
    Calculates portfolio allocation.
    """
    weights = np.array(weights).squeeze()
    if weights.size == len(mean_returns):
        portfolio_allocation = pd.DataFrame(weights, index=stock_tickers, columns=["allocation"])
        portfolio_allocation.allocation = [round(i * 100, 2) for i in portfolio_allocation.allocation]
        portfolio_allocation = portfolio_allocation.T
    else:
        raise ValueError("The length of weights does not match the number of assets.")
    return portfolio_allocation

def display_simulated_ef_with_random(mean_returns: np.ndarray, cov_matrix: np.ndarray, num_portfolios: int, risk_free_rate: float) -> None:
    """
    Displays simulated efficient frontier with random portfolios.
    """
    results, weights_record = random_portfolios(num_portfolios, mean_returns, cov_matrix)
    max_sharpe_idx = np.argmax(results[:, 2])
    sdp, rp = results[max_sharpe_idx, 0], results[max_sharpe_idx, 1]
    max_sharpe_allocation = portfolio_allocation(weights_record[max_sharpe_idx], mean_returns)
    max_sharpe_allocation = max_sharpe_allocation.round(2) * 100
    max_sharpe_allocation = max_sharpe_allocation.T

    min_vol_idx = np.argmin(results[:, 0])
    sdp_min, rp_min = results[min_vol_idx, 0], results[min_vol_idx, 1]
    min_vol_allocation = portfolio_allocation(weights_record[min_vol_idx], mean_returns)
    min_vol_allocation = min_vol_allocation.round(2) * 100
    min_vol_allocation = min_vol_allocation.T

    max_sharpe_r = rp_max_sharpe_portfolio = pd.Series(rp, index=["max sharpe portfolio"])
    min_vol_r = rp_min_vol_portfolio = pd.Series(rp_min, index=["min vol portfolio"])

    fig, ax = plt.subplots(figsize=(10, 5))
    plot_efficient_frontier(mean_returns, cov_matrix, risk_free_rate, ax=ax)
    ax.set_title('Simulated Efficient Frontier with Random Portfolios')
    plt.xlabel('Expected Return')
    plt.ylabel('Volatility')
    plt.show()

    # Plot efficient frontier
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.set_title('Efficient Frontier')
    ax.plot(results[:, 1], results[:, 0], color='blue', label='Efficient Frontier')
    ax.scatter(rp_max_sharpe_portfolio, sdp_max_sharpe_portfolio, color='red', label='Max Sharpe Ratio')
    ax.scatter(rp_min_vol_portfolio, sdp_min_vol_portfolio, color='green', label='Min Volatility')
    ax.set_xlabel('Volatility')
    ax.set_ylabel('Expected Return')
    ax.legend()
    plt.show()

    # Plot allocation pie charts
    fig, axs = plt.subplots(1, 2, figsize=(10, 5))
    axs[0].pie(max_sharpe_allocation.allocation, labels=max_sharpe_allocation.index, autopct='%1.1f%%')
    axs[0].set_title('Max Sharpe Ratio Portfolio Allocation')
    axs[1].pie(min_vol_allocation.allocation, labels=min_vol_allocation.index, autopct='%1.1f%%')
    axs[1].set_title('Min Volatility Portfolio Allocation')
    plt.show()

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

    fig, ax = plt.subplots(figsize=(10, 5))
    plot_efficient_frontier(mean_returns, cov_matrix, risk_free_rate, ax=ax)
    ax.set_title('Calculated Efficient Frontier with Random Portfolios')
    plt.xlabel('Expected Return')
    plt.ylabel('Volatility')
    plt.show()

    # Plot efficient frontier
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.set_title('Efficient Frontier')
    ax.plot(results[:, 1], results[:, 0], color='blue', label='Efficient Frontier')
    ax.scatter(rp_max_sharpe_portfolio, sdp_max_sharpe_portfolio, color='red', label='Max Sharpe Ratio')
    ax.scatter(rp_min_vol_portfolio, sdp_min_vol_portfolio, color='green', label='Min Volatility')
    ax.set_xlabel('Volatility')
    ax.set_ylabel('Expected Return')
    ax.legend()
    plt.show()

    # Plot allocation pie charts
    fig, axs = plt.subplots(1, 2, figsize=(10, 5))
    axs[0].pie(max_sharpe_allocation.allocation, labels=max_sharpe_allocation.index, autopct='%1.1f%%')
    axs[0].set_title('Max Sharpe Ratio Portfolio Allocation')
    axs[1].pie(min_vol_allocation.allocation, labels=min_vol_allocation.index, autopct='%1.1f%%')
    axs[1].set_title('Min Volatility Portfolio Allocation')
    plt.show()

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

    fig, ax = plt.subplots(figsize=(10, 5))
    plot_efficient_frontier(mean_returns, cov_matrix, risk_free_rate, ax=ax)
    ax.set_title('Simulated Efficient Frontier with Random Portfolios and Sharpe Ratio Constraint')
    plt.xlabel('Expected Return')
    plt.ylabel('Volatility')
    plt.show()

    # Plot efficient frontier
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.set_title('Efficient Frontier')
    ax.plot(results[:, 1], results[:, 0], color='blue', label='Efficient Frontier')
    ax.scatter(rp_max_sharpe_portfolio, sdp_max_sharpe_portfolio, color='red', label='Max Sharpe Ratio')
    ax.scatter(rp_min_vol_portfolio, sdp_min_vol_portfolio, color='green', label='Min Volatility')
    ax.set_xlabel('Volatility')
    ax.set_ylabel('Expected Return')
    ax.legend()
    plt.show()

    # Plot allocation pie charts
    fig, axs = plt.subplots(1, 2, figsize=(10, 5))
    axs[0].pie(max_sharpe_allocation.allocation, labels=max_sharpe_allocation.index, autopct='%1.1f%%')
    axs[0].set_title('Max Sharpe Ratio Portfolio Allocation')
    axs[1].pie(min_vol_allocation.allocation, labels=min_vol_allocation.index, autopct='%1.1f%%')
    axs[1].set_title('Min Volatility Portfolio Allocation')
    plt.show()

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

    fig, ax = plt.subplots(figsize=(10, 5))
    plot_efficient_frontier(mean_returns, cov_matrix, risk_free_rate, ax=ax)
    ax.set_title('Calculated Efficient Frontier with Random Portfolios and Sharpe Ratio Constraint')
    plt.xlabel('Expected Return')
    plt.ylabel('Volatility')
    plt.show()

    # Plot efficient frontier
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.set_title('Efficient Frontier')
    ax.plot(results[:, 1], results[:, 0], color='blue', label='Efficient Frontier')
    ax.scatter(rp_max_sharpe_portfolio, sdp_max_sharpe_portfolio, color='red', label='Max Sharpe Ratio')
    ax.scatter(rp_min_vol_portfolio, sdp_min_vol_portfolio, color='green', label='Min Volatility')
    ax.set_xlabel('Volatility')
    ax.set_ylabel('Expected Return')
    ax.legend()
    plt.show()

    # Plot allocation pie charts
    fig, axs = plt.subplots(1, 2, figsize=(10, 5))
    axs[0].pie(max_sharpe_allocation.allocation, labels=max_sharpe_allocation.index, autopct='%1.1f%%')
    axs[0].set_title('Max Sharpe Ratio Portfolio Allocation')
    axs[1].pie(min_vol_allocation.allocation, labels=min_vol_allocation.index, autopct='%1.1f%%')
    axs[1].set_title('Min Volatility Portfolio Allocation')
    plt.show()

def max_sharpe_ratio_sfr(mean_returns: np.ndarray, cov_matrix: np.ndarray, risk_free_rate: float) -> Tuple[np.ndarray, float]:
    num_assets = len(mean_returns)
    args = (mean_returns, cov_matrix, -1.0, np.eye(num_assets), np.ones(num_assets) / num_assets)
    constraints = ({'type': 'eq', 'fun': lambda x: np.dot(x, x)},
                    {'type': 'eq', 'fun': lambda x: np.dot(mean_returns, x)},
                    {'type': 'eq', 'fun': lambda x: np.dot(cov_matrix, x[0] * x[0] * np.dot(np.diag(x[1:]), x[1:])) + risk_free_rate, 'args': (mean_returns, cov_matrix, risk_free_rate)},
                    {'type': 'eq', 'fun': lambda x: x[0]},
                    {'type': 'eq', 'fun': lambda x: x[1:]},
                    {'type': 'ineq', 'fun': lambda x: x[0]},
                    {'type': 'ineq', 'fun': lambda x: np.sum(x[1:]) - 1})
    result = minimize(sharpe_ratio_sfr, args=args, method='SLSQP', bounds=[(0, 1), (0, 1)] * num_assets, constraints=constraints)
    return result.x, -result.fun

def min_variance_pypfopt(mean_returns: np.ndarray, cov_matrix: np.ndarray) -> Tuple[np.ndarray, float]:
    """
    Finds portfolio with minimum variance using pypfopt.
    """
    portfolio = pfopt.Portfolio(mean_returns, cov_matrix)
    min_vol_portfolio = portfolio.min_volatility()
    min_vol_variance = portfolio.portfolio.volatility
    return min_vol_portfolio.weights, min_vol_variance

def max_sharpe_ratio_pypfopt(mean_returns: np.ndarray, cov_matrix: np.ndarray, risk_free_rate: float) -> Tuple[np.ndarray, float]:
    """
    Finds portfolio with maximum Sharpe ratio using pypfopt.
    """
    portfolio = pfopt.Portfolio(mean_returns, cov_matrix)
    max_sharpe_portfolio = portfolio.max_sharpe(risk_free_rate=risk_free_rate)
    return max_sharpe_portfolio.weights, max_sharpe_portfolio.expected_return

def efficient_frontier_pypfopt(mean_returns: np.ndarray, cov_matrix: np.ndarray, returns_range: np.ndarray) -> List[Dict[str, Any]]:
    """
    Calculates efficient frontier for a given range of returns using pypfopt.
    """
    portfolio = pfopt.EfficientFrontier(mean_returns, cov_matrix)
    efficient_portfolios = []
    for ret in returns_range:
        constraints = ({'type': 'eq', 'fun' : lambda x: np.sum(x)-1})
        bound = (0.0, 1.0)
        bounds = tuple(bound for asset in range(len(mean_returns)))
        portfolio_volatility_partial = partial(portfolio_volatility, *args)
        result = minimize(portfolio_volatility_partial, len(mean_returns) * [1./len(mean_returns),], method='SLSQP', bounds=bounds, constraints=constraints)

        print("Arguments passed to portfolio_volatility_partial:", result.x)

        portfolio_volatility_value = portfolio_volatility(*result.x, *args)
        print("Volatility calculated by portfolio_volatility:", portfolio_volatility_value)

        efficient_portfolios.append({
            "return": ret,
            "volatility": portfolio_volatility_value,
            "allocation": portfolio_allocation(result.x, mean_returns)
        })
    return efficient_portfolios

def main() -> None:
    # Download stock prices
    price_data = download_stock_prices(stock_tickers, START_DATE, END_DATE)

    if price_data is None:
        print("Error downloading stock prices")
        return

    # Calculate daily returns
    daily_returns = calculate_daily_returns(price_data)

    if daily_returns is None:
        print("Error calculating daily returns")
        return

    # Plot normalized returns
    plot_normalized_returns(daily_returns)

    # Plot daily returns volatility
    plot_daily_returns_volatility(daily_returns)

    # Calculate mean returns and covariance matrix
    mean_returns = daily_returns.mean()
    cov_matrix = daily_returns.cov()

    # Add a small positive constant to the diagonal elements
    cov_matrix_plus_epsilon = cov_matrix + np.eye(cov_matrix.shape[0]) * 1e-8

    # Compute the Cholesky decomposition of the covariance matrix
    cholesky_decomposition = np.linalg.cholesky(cov_matrix_plus_epsilon)

    # Compute the inverse of the covariance matrix using the Cholesky decomposition
    cov_matrix_inv = np.linalg.inv(cholesky_decomposition)
    cov_matrix_inv = cov_matrix_inv.T @ cov_matrix_inv

    # Define risk-free rate
    risk_free_rate = RISK_FREE_RATE

    # Generate random portfolios
    num_portfolios = NUM_PORTFOLIOS
    results, weights = random_portfolios(num_portfolios, mean_returns, cov_matrix)

    # Calculate maximum Sharpe ratio portfolio
    max_sharpe_portfolio, max_sharpe_ratio_value = max_sharpe_ratio(mean_returns, cov_matrix, risk_free_rate)

    # Calculate minimum variance portfolio
    min_vol_portfolio, min_vol_variance = min_variance(mean_returns, cov_matrix)

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
    fig, ax = plt.subplots(figsize=(10, 5))
    plot_efficient_frontier(mean_returns, cov_matrix, risk_free_rate, ax=ax)
    ax.set_title('Efficient Frontier using pypfopt')
    plt.xlabel('Volatility')
    plt.ylabel('Expected Return')
    plt.show()

if __name__ == '__main__':
    main()
