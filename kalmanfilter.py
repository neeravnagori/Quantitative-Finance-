import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime, timedelta

class KalmanFilter:
    def __init__(self, q, r, x=0, p=1, a=1, h=1):
        self.q = q
        self.r = r
        self.x = x
        self.p = p
        self.a = a
        self.h = h
        self.k = 0

    def filter(self, z):
        self.p = self.a * self.p * self.a + self.q
        self.k = self.p * self.h / (self.h * self.p * self.h + self.r)
        self.x = self.a * self.x + self.k * (z - self.h * self.a * self.x)
        self.p = (1 - self.k * self.h) * self.p
        return self.x


def generate_synthetic_stock_data(symbol, listing_price, n_days=253):
    dates = pd.date_range(end=datetime.now(), periods=n_days, freq='D')
    np.random.seed(42)
    
    base = np.linspace(listing_price, listing_price + 56, n_days)
    trend = np.cumsum(np.random.normal(0.15, 0.5, n_days)) * 0.08
    momentum = np.sin(np.linspace(0, 6*np.pi, n_days)) * 8
    noise = np.random.normal(0, 6.5, n_days)
    
    prices = base + trend + momentum + noise
    prices = np.maximum(prices, listing_price - 15)
    
    return pd.DataFrame({
        'Date': dates,
        'Close': prices,
        'Open': prices + np.random.normal(0, 2.5, n_days)
    })


def apply_kalman_filter(prices, q=1e-5, r=0.15):
    kf = KalmanFilter(q=q, r=r, x=prices[0], p=1)
    return np.array([kf.filter(price) for price in prices])


def calculate_metrics(original, filtered):
    residuals = original - filtered
    original_std = original.std()
    filtered_std = filtered.std()
    noise_std = residuals.std()
    
    returns_original = np.diff(original) / original[:-1] * 100
    returns_filtered = np.diff(filtered) / filtered[:-1] * 100
    
    metrics = {
        'original_std': original_std,
        'filtered_std': filtered_std,
        'noise_std': noise_std,
        'noise_reduction_pct': ((original_std - filtered_std) / original_std) * 100,
        'original_return_std': returns_original.std(),
        'filtered_return_std': returns_filtered.std(),
        'return_smoothing_pct': ((returns_original.std() - returns_filtered.std()) / returns_original.std()) * 100,
        'residuals': residuals,
        'returns_original': returns_original,
        'returns_filtered': returns_filtered
    }
    return metrics


def plot_analysis(dates, original, filtered, metrics, title="Stock Price Analysis"):
    fig = plt.figure(figsize=(16, 11))
    gs = fig.add_gridspec(4, 2, hspace=0.35, wspace=0.3)

    ax1 = fig.add_subplot(gs[0, :])
    ax1.plot(dates, original, 'o-', label='Observed Prices', color='#e74c3c', 
             linewidth=1.5, markersize=2, alpha=0.7)
    ax1.plot(dates, filtered, '-', label='Kalman Filtered', color='#3498db', linewidth=2.5)
    ax1.fill_between(dates, original, filtered, alpha=0.15, color='gray')
    ax1.set_ylabel('Price (₹)', fontsize=11, fontweight='bold')
    ax1.set_title(title, fontsize=13, fontweight='bold', pad=15)
    ax1.legend(loc='upper left', fontsize=10)
    ax1.grid(True, alpha=0.3, linestyle='--')

    ax2 = fig.add_subplot(gs[1, :])
    residuals = metrics['residuals']
    noise_std = metrics['noise_std']
    ax2.bar(dates, residuals, width=0.6, label='Noise', color='#f39c12', alpha=0.7)
    ax2.axhline(y=0, color='black', linestyle='-', linewidth=0.8)
    ax2.axhline(y=noise_std, color='red', linestyle='--', linewidth=1.5, 
                label=f'±1σ (±₹{noise_std:.2f})')
    ax2.axhline(y=-noise_std, color='red', linestyle='--', linewidth=1.5)
    ax2.set_ylabel('Noise (₹)', fontsize=11, fontweight='bold')
    ax2.set_title(f'Extracted Noise (Reduction: {metrics["noise_reduction_pct"]:.2f}%)', 
                  fontsize=12, fontweight='bold', pad=10)
    ax2.legend(loc='upper right', fontsize=10)
    ax2.grid(True, alpha=0.3, axis='y')

    ax3 = fig.add_subplot(gs[2, 0])
    ax3.plot(dates[1:], metrics['returns_original'], 'o-', label='Original Returns', 
             color='#e74c3c', linewidth=1, markersize=3, alpha=0.6)
    ax3.plot(dates[1:], metrics['returns_filtered'], '-', label='Filtered Returns', 
             color='#3498db', linewidth=2)
    ax3.axhline(y=0, color='black', linestyle='-', linewidth=0.8, alpha=0.5)
    ax3.set_ylabel('Return (%)', fontsize=10, fontweight='bold')
    ax3.set_title('Daily Returns Comparison', fontsize=11, fontweight='bold')
    ax3.legend(fontsize=9)
    ax3.grid(True, alpha=0.3)

    ax4 = fig.add_subplot(gs[2, 1])
    ax4.hist(residuals, bins=25, color='#f39c12', alpha=0.7, edgecolor='#e67e22')
    ax4.axvline(x=0, color='red', linestyle='--', linewidth=2)
    ax4.set_xlabel('Noise Value (₹)', fontsize=10, fontweight='bold')
    ax4.set_ylabel('Frequency', fontsize=10, fontweight='bold')
    ax4.set_title('Noise Distribution', fontsize=11, fontweight='bold')
    ax4.grid(True, alpha=0.3, axis='y')

    ax5 = fig.add_subplot(gs[3, 0])
    ax5.plot(dates, original, 'o', label='Observed', color='#e74c3c', 
             markersize=3, alpha=0.5, linestyle='none')
    ax5.plot(dates, filtered, '-', label='Filtered', color='#3498db', linewidth=2)
    ax5.set_ylabel('Price (₹)', fontsize=10, fontweight='bold')
    ax5.set_xlabel('Date', fontsize=10, fontweight='bold')
    ax5.set_title('Signal Smoothing', fontsize=11, fontweight='bold')
    ax5.legend(fontsize=9)
    ax5.grid(True, alpha=0.3)

    ax6 = fig.add_subplot(gs[3, 1])
    cumulative_filtered = np.cumprod(1 + metrics['returns_filtered']/100) * 100 - 100
    cumulative_original = np.cumprod(1 + metrics['returns_original']/100) * 100 - 100
    ax6.plot(dates[1:], cumulative_original, label='Original', color='#e74c3c', linewidth=2, alpha=0.7)
    ax6.plot(dates[1:], cumulative_filtered, label='Filtered', color='#3498db', linewidth=2.5)
    ax6.axhline(y=0, color='black', linestyle='-', linewidth=0.8, alpha=0.5)
    ax6.set_ylabel('Cumulative Return (%)', fontsize=10, fontweight='bold')
    ax6.set_xlabel('Date', fontsize=10, fontweight='bold')
    ax6.set_title('Cumulative Returns', fontsize=11, fontweight='bold')
    ax6.legend(fontsize=9)
    ax6.grid(True, alpha=0.3)

    plt.tight_layout()
    return fig


def print_report(symbol, df, filtered_prices, metrics):
    print("\n" + "="*75)
    print(f"KALMAN FILTER ANALYSIS: {symbol}")
    print("="*75 + "\n")
    
    print(f"Period: {df['Date'].iloc[0].date()} to {df['Date'].iloc[-1].date()}")
    print(f"Trading Days: {len(df)}")
    print(f"Price Range: ₹{df['Close'].min():.2f} - ₹{df['Close'].max():.2f}")
    print(f"Average Price: ₹{df['Close'].mean():.2f}\n")
    
    print("KALMAN FILTER RESULTS:")
    print("─"*75)
    print(f"Original Volatility (σ):          ₹{metrics['original_std']:.4f}")
    print(f"Filtered Volatility (σ):          ₹{metrics['filtered_std']:.4f}")
    print(f"Extracted Noise (σ):              ₹{metrics['noise_std']:.4f}")
    print(f"Noise Reduction:                  {metrics['noise_reduction_pct']:.2f}%\n")
    
    print(f"Original Return Volatility:       {metrics['original_return_std']:.4f}%")
    print(f"Filtered Return Volatility:       {metrics['filtered_return_std']:.4f}%")
    print(f"Return Smoothing:                 {metrics['return_smoothing_pct']:.2f}%")
    
    print("\n" + "─"*75)
    print("LAST 15 TRADING DAYS:")
    print("─"*75)
    
    tail_data = pd.DataFrame({
        'Date': df['Date'].iloc[-15:].values,
        'Observed': df['Close'].iloc[-15:].values.round(2),
        'Filtered': filtered_prices[-15:].round(2),
        'Noise': metrics['residuals'][-15:].round(2)
    })
    tail_data.reset_index(drop=True, inplace=True)
    print(tail_data.to_string(index=False))
    
    print("\n" + "="*75)
    print(f"Underlying Trend: ₹{filtered_prices[0]:.2f} → ₹{filtered_prices[-1]:.2f}")
    print(f"Net Movement: {filtered_prices[-1] - filtered_prices[0]:.2f} rupees")
    print("="*75 + "\n")


if __name__ == "__main__":
    symbol = "ARSSBL"
    listing_price = 432.0
    
    df = generate_synthetic_stock_data(symbol, listing_price, n_days=253)
    
    filtered_prices = apply_kalman_filter(df['Close'].values, q=1e-5, r=0.15)
    
    metrics = calculate_metrics(df['Close'].values, filtered_prices)
    
    print_report(symbol, df, filtered_prices, metrics)
    
    fig = plot_analysis(df['Date'], df['Close'].values, filtered_prices, metrics,
                       title=f"ANAND RATHI (ARSSBL) - Kalman Filter Applied")
    
    fig.savefig('kalman_analysis.png', dpi=150, bbox_inches='tight')
    print("✓ Chart saved: kalman_analysis.png\n")
