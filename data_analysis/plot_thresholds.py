import os
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt

# Use Meiryo on Windows to ensure Japanese labels render correctly
try:
    matplotlib.rcParams['font.family'] = 'Meiryo'
    matplotlib.rcParams['font.sans-serif'] = ['Meiryo']
    plt.rcParams['font.family'] = 'Meiryo'
except Exception:
    # If Meiryo isn't available, matplotlib will fall back to defaults
    pass


def load_threshold_csv(path):
    if not os.path.exists(path):
        return None
    df = pd.read_csv(path, encoding='utf-8-sig')
    # Ensure numeric and sort
    if 'threshold' in df.columns:
        df = df.sort_values('threshold')
    return df


def plot_rates(df, title, out_path):
    if df is None or df.empty:
        print(f"ファイルが空、または存在しません: {out_path}")
        return

    x = df['threshold']
    # convert rates to numeric (they may be None)
    win = pd.to_numeric(df['win_recovery_rate'], errors='coerce')
    place = pd.to_numeric(df['place_recovery_rate'], errors='coerce')

    plt.figure(figsize=(10, 5))
    plt.plot(x, win, marker='o', label='単勝回収率')
    plt.plot(x, place, marker='s', label='複勝回収率')
    plt.xlabel('上昇度以上')
    plt.ylabel('回収率 (%)')
    plt.title(title)
    plt.grid(True, linestyle='--', alpha=0.4)
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()
    print(f"Saved plot: {out_path}")


def main():
    base = os.path.dirname(__file__)
    a_path = os.path.join(base, 'thresholds_patternA.csv')
    b_path = os.path.join(base, 'thresholds_patternB.csv')
    comb_path = os.path.join(base, 'thresholds_patternA_B_combined.csv')

    a_df = load_threshold_csv(a_path)
    b_df = load_threshold_csv(b_path)
    comb_df = load_threshold_csv(comb_path)

    # Plot individual
    plot_rates(a_df, 'パターンA：上昇度以上ごとの回収率', os.path.join(base, 'patternA_rates.png'))
    plot_rates(b_df, 'パターンB：上昇度以上ごとの回収率', os.path.join(base, 'patternB_rates.png'))

    # Combined: if combined file exists, plot separate subplots
    if comb_df is not None and not comb_df.empty:
        # plot combined patterns side by side
        try:
            fig, axes = plt.subplots(1, 2, figsize=(14, 5), sharey=True)
            # A
            a_comb = comb_df[comb_df['pattern'] == 'パターンA']
            b_comb = comb_df[comb_df['pattern'] == 'パターンB']
            if not a_comb.empty:
                axes[0].plot(a_comb['threshold'], pd.to_numeric(a_comb['win_recovery_rate'], errors='coerce'), marker='o', label='単勝')
                axes[0].plot(a_comb['threshold'], pd.to_numeric(a_comb['place_recovery_rate'], errors='coerce'), marker='s', label='複勝')
                axes[0].set_title('パターンA')
                axes[0].set_xlabel('上昇度以上')
                axes[0].grid(True, linestyle='--', alpha=0.4)
                axes[0].legend()
            if not b_comb.empty:
                axes[1].plot(b_comb['threshold'], pd.to_numeric(b_comb['win_recovery_rate'], errors='coerce'), marker='o', label='単勝')
                axes[1].plot(b_comb['threshold'], pd.to_numeric(b_comb['place_recovery_rate'], errors='coerce'), marker='s', label='複勝')
                axes[1].set_title('パターンB')
                axes[1].set_xlabel('上昇度以上')
                axes[1].grid(True, linestyle='--', alpha=0.4)
                axes[1].legend()

            fig.suptitle('パターンA/B：上昇度以上ごとの回収率比較')
            fig.tight_layout(rect=[0, 0.03, 1, 0.95])
            out_comb_png = os.path.join(base, 'patternA_B_rates_combined.png')
            fig.savefig(out_comb_png)
            plt.close(fig)
            print(f"Saved combined plot: {out_comb_png}")
        except Exception as e:
            print('Combined plot error:', e)


if __name__ == '__main__':
    main()
