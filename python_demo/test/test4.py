import sys
import os
import scipy
import time

# 現在のファイルのディレクトリ（test）の親ディレクトリ（プロジェクトルート）を取得し、パスに追加する
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from src import utils
from src import constants
from src import dtcwpt_stateful_copy
from src import dtcwpt_stateful
import numpy as np
import matplotlib.pyplot as plt

def generate_random_destinations(max_level):
        """
        制約「XLが存在すればXHも存在する」を満たすランダムなターゲット集合を生成する。
        """
        # 初期状態: レベル1分解
        current_leaves = ['']
        
        # 深さごとに確率的に分解
        import random
        for _ in range(max_level):
            next_leaves = []
            for node in current_leaves:
                # 70%の確率で分解、ただしルート('')は必ず分解
                if node == '' or random.random() < 0.6:
                    child_L = node + 'L'
                    child_H = node + 'H'
                    # dests.add(child_L)
                    # dests.add(child_H)
                    next_leaves.append(child_L)
                    next_leaves.append(child_H)
                else:
                    next_leaves.append(node)
            current_leaves = next_leaves
            
        return current_leaves

def generate_full_destinations(max_level):
    """
    指定レベルまで完全に分解するためのターゲットリストを生成
    再帰ロジックに基づき、子ノードが存在するかをチェックするため、
    分解したい最終レベルのノード名までを含める必要があります。
    """
    dests = []
    
    def add_paths(current_path, level):
        if level > max_level:
            return
        
        # 現在のパスを追加 (例: 'L', 'LH' ...)
        if (current_path != '' and level == max_level):
            dests.append(current_path)
            
        # 次のレベルへ
        add_paths(current_path + 'L', level + 1)
        add_paths(current_path + 'H', level + 1)

    # ルートから開始
    add_paths('', 0)
    return dests

def generate_wt_destinations(max_level):
    """
    指定レベルまで完全に分解するためのターゲットリストを生成
    再帰ロジックに基づき、子ノードが存在するかをチェックするため、
    分解したい最終レベルのノード名までを含める必要があります。
    """
    dests = []
    
    def add_paths(current_path, level):
        if (level > max_level) or (len(current_path) >= 2 and current_path[-2] == 'H'):
            return
        
        # 現在のパスを追加 (例: 'L', 'LH' ...)
        if ((current_path != '' and level == max_level) or current_path.endswith('H')):
            dests.append(current_path)
            
        # 次のレベルへ
        add_paths(current_path + 'L', level + 1)
        add_paths(current_path + 'H', level + 1)

    # ルートから開始
    add_paths('', 0)
    return dests

def main():
    # --- 入力信号生成 ---
    # fs = 44100
    length = 1 << 15
    # テスト用に単純な矩形波やランダム波形を使用
    # inputR = np.random.random(size=length) * 2 - 1
    # print(len(inputR))

    t = np.linspace(0, 1, length)
    iLs = np.random.random(size=length) * 2 - 1
    iRs = np.random.random(size=length) * 2 - 1
    iL = np.random.random(size=length) * 2 - 1
    iR = np.random.random(size=length) * 2 - 1
    # iLs = scipy.signal.chirp(t, f0=1, f1=length/2, t1=1, method='linear')
    # iRs = scipy.signal.chirp(t, f0=1, f1=length/4, t1=1, method='linear')
    # iL = scipy.signal.chirp(t, f0=1, f1=length/6, t1=1, method='linear')
    # iR = scipy.signal.chirp(t, f0=1, f1=length/8, t1=1, method='linear')

    # t = np.linspace(0, 1, length)
    # inputL = np.sin(2 * np.pi * 50 * t) + 0.5 * np.sin(2 * np.pi * 120 * t)
    
    # input = np.ones(length)
    
    # dests = generate_wt_destinations(6)
    # dests = generate_full_destinations(8)
    dests = generate_random_destinations(7)

    print(dests)

    print(f"Input Signal Length: {length}")

    # --- 解析---
    start = time.perf_counter()
    (max_delay2, outputL2, outputR2) = dtcwpt_stateful_copy.dtcwptMorph(iLs, iRs, iL, iR, dests)
    end = time.perf_counter() #計測終了
    print('{:.5f}'.format((end-start)))

    start = time.perf_counter()
    (max_delay1, outputL1, outputR1) = dtcwpt_stateful.dtcwptMorph(iLs, iRs, iL, iR, dests)
    end = time.perf_counter() #計測終了
    print('{:.5f}'.format((end-start)))

    print("end")

    print(len(outputL1) - max_delay1)
    print(len(outputR1) - max_delay1)
    outputL1 = np.array(outputL1[max_delay1:max_delay1+length])
    outputR1 = np.array(outputR1[max_delay1:max_delay1+length])

    print(len(outputL2) - max_delay2)
    print(len(outputR2) - max_delay2)
    outputL2 = np.array(outputL2[max_delay2:max_delay2+length])
    outputR2 = np.array(outputR2[max_delay2:max_delay2+length])
    
    # 誤差計算 (MSE)
    mse = np.mean((outputL2 - outputL1) ** 2)
    print(f"\nReconstruction MSE: {mse:.10f}")
    
    # プロット
    plt.figure(figsize=(10, 6))
    
    plt.subplot(2, 1, 1)
    plt.plot(outputL1, label='Original', alpha=0.7)
    plt.plot(outputL2, label='Reconstructed', linestyle='--', alpha=0.7)
    plt.title(f'Original vs Reconstructed (MSE: {mse:.2e})')
    plt.legend()
    plt.grid(True)

    plt.subplot(2, 1, 2)
    plt.plot(outputL2 - outputL1, color='red', label='Error')
    plt.title('Reconstruction Error')
    plt.legend()
    plt.grid(True)
    
    plt.tight_layout()
    plt.show()

    # テスト判定 (誤差が許容範囲内か)
    assert mse < 1e-5, "Reconstruction error is too high!"
    print("\nTest Passed: Signal reconstructed successfully.")

    # 誤差計算 (MSE)
    mse = np.mean((outputR2 - outputR1) ** 2)
    print(f"\nReconstruction MSE R: {mse:.10f}")
    
    # プロット
    plt.figure(figsize=(10, 6))
    
    plt.subplot(2, 1, 1)
    plt.plot(outputR2, label='Original', alpha=0.7)
    plt.plot(outputR1, label='Reconstructed', linestyle='--', alpha=0.7)
    plt.title(f'Original vs Reconstructed (MSE: {mse:.2e})')
    plt.legend()
    plt.grid(True)

    plt.subplot(2, 1, 2)
    plt.plot(outputR2 - outputR1, color='red', label='Error')
    plt.title('Reconstruction Error')
    plt.legend()
    plt.grid(True)
    
    plt.tight_layout()
    plt.show()

    # テスト判定 (誤差が許容範囲内か)
    assert mse < 1e-5, "Reconstruction error is too high!"
    print("\nTest Passed: Signal reconstructed successfully.")

if __name__ == "__main__":
    main()