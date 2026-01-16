import numpy as np
import numpy.typing as npt
from src import constants
from typing import Generator, Tuple, Dict, Optional, List, Set, Union, Final, cast
import copy
import math
import dataclasses

# ==============================================================================
# Constants & Data Structures
# ==============================================================================

@dataclasses.dataclass
class MorphParam:
  """
  モーフィング制御パラメータ。
  VSTプラグインのFloatParameterとして管理されることを想定。
  """
  mag: Final[float] = 0.01   # Magnitude Interpolation Ratio (0.0: Main, 1.0: SideChain)
  phase: Final[float] = 1.0  # Phase Mixing Ratio (0.0: Main, 1.0: SideChain)
  thr: Final[float] = 0.01   # Magnitude Threshold for Phase processing


# ==============================================================================
# Signal Processing Functions / Classes
# ==============================================================================

def morphBuffer(a_re: npt.NDArray[np.float64], # Mutable: In-place modification
                a_im: npt.NDArray[np.float64], # Mutable: In-place modification
                b_re: npt.NDArray[np.float64], # Immutable: Read-only context
                b_im: npt.NDArray[np.float64], # Immutable: Read-only context
                param: MorphParam) -> None:
  """
  2つの複素数信号バッファをモーフィングし、結果をAのバッファに上書きする (In-Place)。
    
  Algorithm:
    1. 振幅 (Magnitude): 単純な線形補間
    2. 位相 (Phase): 単位ベクトルの重み付き平均 (N-Lerp) により合成
    
  Args:
      a_re, a_im: Main信号 (Mutable / 出力先)
      b_re, b_im: SideChain信号 (Immutable)
      param: モーフィングパラメータ
    
  Note:
      Rustでは `zipped iterator` を使用してSIMD化が期待されるセクション。
  """
  current_phase_coef: float = 0.0
  num_samples: Final[int] = a_re.shape[0]


  # ゼロ除算防止
  eps: Final[float] = 1e-37

  for i in range(num_samples):
    # 1. 振幅の計算
    mag_a: float = math.sqrt(a_re[i] * a_re[i] + a_im[i] * a_im[i])
    mag_b: float = math.sqrt(b_re[i] * b_re[i] + b_im[i] * b_im[i])

    # 位相合成係数の調整 (信号レベルが低い場合は位相操作を抑制)
    if mag_b < param.thr:
      current_phase_coef = param.phase * mag_b*0.9/param.thr
    else:
      current_phase_coef = param.phase

    # 2. 振幅の線形補間 (Target Magnitude)
    target_mag: float = mag_a * (1.0 - param.mag) + mag_b * param.mag

    # 3. 単位ベクトル化 (Unit Vector)
    # Aの単位ベクトル
    scale_a: float = 1.0 / (mag_a + eps)
    ua_re: float = a_re[i] * scale_a
    ua_im: float = a_im[i] * scale_a

    # Bの単位ベクトル
    scale_b: float = 1.0 / (mag_b + eps)
    ub_re: float = b_re[i] * scale_b
    ub_im: float = b_im[i] * scale_b

    # 4. 単位ベクトルの合成 (Phase Mixing)
    # ベクトル加算後に正規化を行う
    mix_re: float = ua_re * (1.0 - current_phase_coef) + ub_re * current_phase_coef
    mix_im: float = ua_im * (1.0 - current_phase_coef) + ub_im * current_phase_coef

    # 合成ベクトルの正規化
    mix_mag: float = math.sqrt(mix_re**2 + mix_im**2)
    scale_mix: float = 1.0 / (mix_mag + eps)

    unit_mix_re: float = mix_re * scale_mix
    unit_mix_im: float = mix_im * scale_mix

    # 5. 最終合成 (Target Mag * Unit Phase)
    a_re[i] = target_mag * unit_mix_re
    a_im[i] = target_mag * unit_mix_im


class DelayBuffer:
  """
  遅延を実現するリングバッファクラス。
  """
  def __init__(self, delay_samples: int, block_size: int) -> None:
    # バッファサイズは ブロックサイズ + 遅延量 以上の余裕を持つ2のべき乗 (ビットマスク計算用)
    size_calc: int = 2
    while (size_calc <= block_size * 2 + delay_samples):
      size_calc = size_calc << 1
    self.size: Final[int] = size_calc

    # ダブルバッファ（ミラーリング）領域の確保
    self.buffer: npt.NDArray[np.float64] = np.zeros((self.size << 1), dtype=np.float64) # ここで確保した[0]が遅延成分になる

    self.write_cursor: int = 0
    self.delay: Final[int] = delay_samples

  def process(self, input_block: npt.NDArray[np.float64], output_buffer: npt.NDArray[np.float64]) -> None:
    """
      入力ブロックを内部バッファに書き込み、delay_samples前のデータをoutput_bufferに書き出す。
        
      Args:
        input_block: 入力データ (Read Only)
        output_buffer: 書き込み先 (Mutable, In-Place)
    """
    # 遅延なしの場合はコピーのみ（入力と出力が別オブジェクトの場合）
    if self.delay == 0:
      if input_block is not output_buffer:
        output_buffer[:] = input_block
      return

    n: Final[int] = len(input_block)
 
    # 1. 書き込み
    # 常に現在の時刻の位置に書き込む
    # ダブルバッファの両方に書き込む
    end_pos: Final[int] = self.write_cursor + n
 
    if end_pos <= self.size:
      # 折り返しなし
      self.buffer[self.write_cursor:end_pos] = input_block
      self.buffer[self.write_cursor+self.size:end_pos+self.size] = input_block
    else:
      # 折り返し書き込み
      remain: Final[int] = self.size - self.write_cursor
      self.buffer[self.write_cursor:self.size] = input_block[:remain]
      self.buffer[self.write_cursor+self.size:self.size+self.size] = input_block[:remain]
      self.buffer[:n-remain] = input_block[remain:]
      self.buffer[self.size:self.size+n-remain] = input_block[remain:]

    # 2. 読み出し
    # 現在の書き込み終了位置(Head)から、N+Delay分戻った位置が読み出し開始位置
    current_head: Final[int] = (self.write_cursor + n) & (self.size - 1)
    read_cursor: Final[int] = (current_head - n - self.delay) & (self.size - 1)
    read_end: Final[int] = read_cursor + n

    # 次回の書き込み位置更新
    self.write_cursor = current_head

    output_buffer[:] = self.buffer[read_cursor:read_end]


class StatefulFilter:
  """
  リングバッファを用いたステートフル（状態保持）なFIRフィルタクラス。
  1ch信号をインターリーブして保持し、メモリコピーなしで畳み込みを行う。
  """

  def __init__(self, kernel: npt.NDArray[np.float64], delay: int) -> None:
    self.kernel: Final[npt.NDArray[np.float64]] = kernel
    self.size: Final[int] = len(kernel) + 1

    # リングバッファの境界チェックを排除するためのミラーバッファ
    self.mirrorbuffer: npt.NDArray[np.float64] = np.zeros(self.size*2, dtype=np.float64)
    self.delay: Final[int] = delay
    self.cursor: int = 0

  def pushSample(self, x: float) -> None:
    """サンプルを1つバッファに追加する"""
    self.mirrorbuffer[self.cursor] = x
    self.mirrorbuffer[self.cursor + self.size] = x
    self.cursor += 1
    if(self.cursor >= self.size):
      self.cursor = 0

  def filtering(self) -> float:
    """
    現在のバッファ内容とカーネルの畳み込み計算を行う。
    Rust移植時はSIMD命令 (fused multiply-add) による最適化対象。
    """
    # 最新のサンプル位置から遡って畳み込みを行う
    idx: int = self.cursor - 1
    if idx < self.size:
      idx += self.size

    ret: float = 0.0

    # カーネルとの積和演算
    # ミラーバッファのおかげで idx -= 1 しても範囲外参照の心配がない
    for k in self.kernel:
      ret += k * self.mirrorbuffer[idx]
      idx -= 1
    return ret


class AnalysisNode:
  """
  分解（Analysis）ノード。
  入力信号に対してLow/Highフィルタを適用し、ダウンサンプリングを行う。
  """
  def __init__(self, filter_low: StatefulFilter, filter_high: StatefulFilter) -> None:
    self.filter_low: Final[StatefulFilter] = filter_low
    self.filter_high: Final[StatefulFilter] = filter_high
    # パリティフラグ: ダウンサンプリング(1/2間引き)の制御用
    # False(偶数番目) -> 計算実行, True(奇数番目) -> 計算スキップ
    self.parity: bool = False

  def updateBuffer(self, x: float,
                   work_buffer: npt.NDArray,
                   active_flags: npt.NDArray,
                   low_idx: int,
                   high_idx: int) -> bool:
    """
    1サンプルを入力し、ダウンサンプリング条件に応じて計算結果を返す。

    Args:
        x: 入力サンプル
        work_buffer: 書き込み先となる全ノード共有バッファ (Mutable)
        active_flags: 有効データフラグ配列 (Mutable)
        low_idx, high_idx: 書き込み先のインデックス

    Returns:
        bool:
            - 計算スキップ時: False
            - 計算実行時: True
    """
    self.filter_low.pushSample(x)
    self.filter_high.pushSample(x)

    # ダウンサンプリング: 2回に1回だけフィルタ計算を行う
    if(self.parity):
      self.parity = False
      return False
    else:
      work_buffer[low_idx] = self.filter_low.filtering()
      work_buffer[high_idx] = self.filter_high.filtering()
      active_flags[low_idx] = True
      active_flags[high_idx] = True
      self.parity = True
      return True


class SynthesisNode:
  """
  合成（再構成）用のノードクラス。
  Low/High入力を受け取り、アップサンプリング（ゼロ挿入）を行って加算する。
  """
  def __init__(self, filter_low: StatefulFilter, filter_high: StatefulFilter) -> None:
    self.filter_low: Final[StatefulFilter] = filter_low
    self.filter_high: Final[StatefulFilter] = filter_high

  def synthesize_block(self, low_buf: npt.NDArray[np.float64],
                       high_buf: npt.NDArray[np.float64],
                       out_buf: npt.NDArray[np.float64],
                       out_cursor: int,
                       num_samples: int) -> None:
    """
      Low成分とHigh成分を入力し、アップサンプリングを行って2つの出力サンプル(Even, Odd)を生成しout_bufに書き込む。

      Args:
        low_buf: Low成分の入力バッファ (Read Only)
        high_buf: High成分の入力バッファ (Read Only)
        out_buf: 出力先 (Mutable, In-Place更新)
        out_cursor: 書き込み開始位置
        num_samples: 処理する有効サンプル数
    """
    for i in range(num_samples):
      # Even Sample (Input)
      self.filter_low.pushSample(low_buf[i])
      self.filter_high.pushSample(high_buf[i])
      out_buf[out_cursor + 2*i] = self.filter_low.filtering() + self.filter_high.filtering()

      # Odd Sample (Zero Padding / Up-sampling)
      self.filter_low.pushSample(0)
      self.filter_high.pushSample(0)
      out_buf[out_cursor + 2*i + 1] = self.filter_low.filtering() + self.filter_high.filtering()


# ==============================================================================
# Topology & Processing Management
# ==============================================================================

class TopologyPlanner:
  """
  フィルタバンクの構造（トポロジー）を管理するクラス。
  ユーザーが指定したパス（例: "LL", "LHR"）に基づき、必要なノードと処理順序を決定する。
  完全二分木を配列（Heap構造）で表現する。
  Root = 1, Left(i) = 2*i, Right(i) = 2*i+1
  """
  def __init__(self, destinations_list: List[str]):
    # ルートはインデックス1。深さ8まで対応可能なサイズを確保
    # 1-based indexing: 
    # index i -> left: 2*i, right: 2*i + 1
    self.MAX_SIZE = 1024  # 2^10 (深さ9程度まで余裕を持たせる)
    self.is_active = np.zeros(self.MAX_SIZE, dtype=bool)
        
    # 実行計画リスト
    self.analysis_order: List[int] = []  # 分解順序 (親 -> 子)
    self.synthesis_order: List[int] = [] # 合成順序 (子 -> 親)
       
    # 構築実行
    self._build_topology(destinations_list)

    self.destinations: List[int] = [self._path_to_index(dest) for dest in destinations_list]

    # モーフィング対象（葉ノード）のリスト作成
    self.morphing_list: List[int] = []
    for dest in self.destinations:
      if (dest & (dest - 1)) == 0: # 100000..
        continue
      elif (dest & (dest + 1)) == 0: # 111111..
        continue
      self.morphing_list.append(dest)



  def _path_to_index(self, path: str) -> int:
    """文字列パス('LH'など)を整数インデックス(5など)に変換"""
    idx = 1 # Root
    for char in path:
      idx = idx << 1      # 左シフト (*2)
      if char == 'H':
        idx |= 1        # ビットを立てる (+1)
    return idx

  def _build_topology(self, destinations: List[str]):
    """
    destinationsから、必要な全ノードを特定し、処理順序リストを作成する
    """
    # 1. 必要なノードを全てマークする
    for path in destinations:
      idx = self._path_to_index(path)
      self.is_active[idx] = True
            
      # そのノードからルートまで遡って全てActiveにする
      curr = idx
      while curr >= 1:
        self.is_active[curr] = True
        curr = curr >> 1  # 親へ戻る

    # 2. Analysis実行計画の作成 (親 -> 子)
    # 単純にインデックス順に走査すれば、必ず親は子より先に来る
    for i in range(1, self.MAX_SIZE):
      if self.is_active[i]:
        # 子が存在し、かつActiveなら分解処理が必要（Inner Node）
        left_child = i << 1
        if left_child < self.MAX_SIZE and self.is_active[left_child]:
          self.analysis_order.append(i)

    # 3. Synthesis実行計画の作成 (子 -> 親)
    # 「子が両方揃っている親」のみが合成を行う
    # インデックスの大きい方から小さい方へ走査
    for i in range(self.MAX_SIZE - 1, 0, -1):
      if self.is_active[i]:
        left_child = i << 1
        # 子が存在し、かつActiveであれば合成処理が必要
        if left_child < self.MAX_SIZE and self.is_active[left_child]:
          self.synthesis_order.append(i)

  def get_plan(self):
    return self.analysis_order, self.synthesis_order



def calculateDelay(target_idx: int, nodes: List[SynthesisNode], node_ids: List[int]) -> int:
  """
    ターゲットノードに至るまでの合成フィルタ群による遅延総量を計算する。

  Args:
    target_idx (int): ターゲットとなる葉ノードのインデックス
    nodes (List[SynthesisNode]): ノード配列
    node_ids: nodesに対応するID配列

  Returns:
    int: 計算された総遅延サンプル数
  """
  # ID -> DenseIndex の逆引き辞書を作成
  # Rustでは HashMap<usize, usize> に相当
  id_map = {tree_id: i for i, tree_id in enumerate(node_ids)}

  total_delay: int = 0
  coef: int = 1 # 初期倍率
    
  # ターゲットのビット長を取得
  # 最上位ビットは常にRoot(1)を表すため、その次のビットから走査を開始する
  num_bits: Final[int] = target_idx.bit_length()
    
  # 現在追跡中の親ノード (Rootから開始)
  current_node_idx: int = 1
    
  # 上位ビットから下位ビットへ向かって走査
  # bit_pos: (num_bits - 2) から 0 まで
  for bit_pos in range(num_bits - 2, -1, -1):
        
    # 現在のビット位置の値を取り出す (0 or 1)
    # 0 -> Left Childへ進む, 1 -> Right Childへ進む
    direction = (target_idx >> bit_pos) & 1
      
    # 辞書を使って node List のどこにあるか特定する
    if current_node_idx in id_map:
      dense_idx = id_map[current_node_idx]
      node = nodes[dense_idx]

      # 使用するフィルタに応じた遅延を取得
      if direction == 0: #Left
        delay = node.filter_low.delay
        current_node_idx = current_node_idx << 1
      else: #Right
        delay = node.filter_high.delay
        current_node_idx = (current_node_idx << 1) | 1

    else:
      delay = 0
      if direction == 0:
        current_node_idx = current_node_idx << 1
      else:
        current_node_idx = (current_node_idx << 1) | 1

    total_delay += delay * coef
    coef = coef << 1

  return total_delay


class DTCWPTProcessor:
  """
  DT-CWPT処理のメインエンジン。
  バッファ管理、処理フローの制御を行う。
  Rust移植時は `struct Processor` に相当。
  """

  def __init__(self, sample_rate: float, max_block_size: int, destinations: List[str], channel_num: int): 
    '''
    prepare to play
    Args: 
      sample_rate: サンプルレート
      max_block_size: ホストから渡される最大バッファサイズ
      destination: DT-CWPTの分割先
    '''

    # --- 基本設定 ---
    self.channel_num = channel_num
    self.sample_rate: float = sample_rate
    self.topology_planner: Final[TopologyPlanner] = TopologyPlanner(destinations)

    # --- 解析次の作業用バッファ (Allocation Avoidance) ---
    # Analysis時に1サンプルごとの結果を一時保持する配列
    self._analysis_work_buffer: npt.NDArray[np.float64] = np.zeros(self.topology_planner.MAX_SIZE)
    self._analysis_active_flags: npt.NDArray[np.bool_] = np.zeros(self.topology_planner.MAX_SIZE, dtype=bool)

    # ノード定義
    self.main_analysis_nodes_re: List[List[AnalysisNode]] = [[] for _ in range(self.channel_num)]
    self.main_analysis_nodes_im: List[List[AnalysisNode]] = [[] for _ in range(self.channel_num)]
    self.main_analysis_nodes_ids: List[int] = []

    self.synthesis_nodes_re: List[List[SynthesisNode]] = [[] for _ in range(self.channel_num)]
    self.synthesis_nodes_im: List[List[SynthesisNode]] = [[] for _ in range(self.channel_num)]
    self.synthesis_nodes_ids: List[int] = []


    # init nodes for targets
    analysis_order, synthesis_order = self.topology_planner.get_plan()

    for i in range(self.channel_num):
      node_re = AnalysisNode(
        StatefulFilter(constants.CDF_H0R, constants.CDF_D0R),
        StatefulFilter(constants.CDF_H1R, constants.CDF_D1R),
        )
      node_im = AnalysisNode(
        StatefulFilter(constants.CDF_H0I, constants.CDF_D0I),
        StatefulFilter(constants.CDF_H1I, constants.CDF_D1I),
        )
      self.main_analysis_nodes_re[i].append(node_re)
      self.main_analysis_nodes_im[i].append(node_im)
    self.main_analysis_nodes_ids.append(1)


    for idx in analysis_order:
      if idx == 1: continue
      self.main_analysis_nodes_ids.append(idx)
      if (idx % 2) == 1:
        for i in range(self.channel_num):
          node_re = AnalysisNode(
            StatefulFilter(constants.PACKET_H0, constants.PACKET_D0),
            StatefulFilter(constants.PACKET_H1, constants.PACKET_D1),
            )
          node_im = AnalysisNode(
            StatefulFilter(constants.PACKET_H0, constants.PACKET_D0),
            StatefulFilter(constants.PACKET_H1, constants.PACKET_D1),
            )
          self.main_analysis_nodes_re[i].append(node_re)
          self.main_analysis_nodes_im[i].append(node_im)
      else:
        for i in range(self.channel_num):
          node_re = AnalysisNode(
            StatefulFilter(constants.QSHIFT14_H0R, constants.QSHIFT14_D0R),
            StatefulFilter(constants.QSHIFT14_H1R, constants.QSHIFT14_D1R),
            )
          node_im = AnalysisNode(
            StatefulFilter(constants.QSHIFT14_H0I, constants.QSHIFT14_D0I),
            StatefulFilter(constants.QSHIFT14_H1I, constants.QSHIFT14_D1I),
            )
          self.main_analysis_nodes_re[i].append(node_re)
          self.main_analysis_nodes_im[i].append(node_im)

    # 2. Synthesis Nodes Initialization
    for idx in synthesis_order:
      self.synthesis_nodes_ids.append(idx)

      if idx == 1: # Root
        for i in range(self.channel_num):
          node_re = SynthesisNode(
            StatefulFilter(constants.CDF_G0R, constants.CDF_D0R),
            StatefulFilter(constants.CDF_G1R, constants.CDF_D1R),
          )
          node_im = SynthesisNode(
            StatefulFilter(constants.CDF_G0I, constants.CDF_D0I),
            StatefulFilter(constants.CDF_G1I, constants.CDF_D1I),
            )
          self.synthesis_nodes_re[i].append(node_re)
          self.synthesis_nodes_im[i].append(node_im)
      else:
        if (idx % 2) == 1:
          for i in range(self.channel_num):
            node_re = SynthesisNode(
              StatefulFilter(constants.PACKET_G0, constants.PACKET_D0),
              StatefulFilter(constants.PACKET_G1, constants.PACKET_D1),
              )
            node_im = SynthesisNode(
              StatefulFilter(constants.PACKET_G0, constants.PACKET_D0),
              StatefulFilter(constants.PACKET_G1, constants.PACKET_D1),
              )
            self.synthesis_nodes_re[i].append(node_re)
            self.synthesis_nodes_im[i].append(node_im)
        else:
          for i in range(self.channel_num):
            node_re = SynthesisNode(
              StatefulFilter(constants.QSHIFT14_G0R, constants.QSHIFT14_D0R),
              StatefulFilter(constants.QSHIFT14_G1R, constants.QSHIFT14_D1R),
              )
            node_im = SynthesisNode(
              StatefulFilter(constants.QSHIFT14_G0I, constants.QSHIFT14_D0I),
              StatefulFilter(constants.QSHIFT14_G1I, constants.QSHIFT14_D1I),
              )
            self.synthesis_nodes_re[i].append(node_re)
            self.synthesis_nodes_im[i].append(node_im)

    self.sc_analysis_nodes_re: List[List[AnalysisNode]] = copy.deepcopy(self.main_analysis_nodes_re)
    self.sc_analysis_nodes_im: List[List[AnalysisNode]] = copy.deepcopy(self.main_analysis_nodes_im)
    self.sc_analysis_nodes_ids: List[int] = copy.deepcopy(self.main_analysis_nodes_ids)

    # initialize delay buffers
    self.main_delay_buffers_re: List[List[DelayBuffer]] = [[] for _ in range(self.channel_num)]
    self.main_delay_buffers_im: List[List[DelayBuffer]] = [[] for _ in range(self.channel_num)]
    self.sc_delay_buffers_re: List[List[DelayBuffer]] = [[] for _ in range(self.channel_num)]
    self.sc_delay_buffers_im: List[List[DelayBuffer]] = [[] for _ in range(self.channel_num)]

    # calculate max delay
    self.max_delay: int = 0
    path_delays_re: List[Optional[int]] = [None]*self.topology_planner.MAX_SIZE
    path_delays_im: List[Optional[int]] = [None]*self.topology_planner.MAX_SIZE

    for dest in self.topology_planner.destinations:
      d_re: int = calculateDelay(dest, self.synthesis_nodes_re[0], self.synthesis_nodes_ids)
      d_im: int = calculateDelay(dest, self.synthesis_nodes_im[0], self.synthesis_nodes_ids)
      path_delays_re[dest] = d_re
      path_delays_im[dest] = d_im
      self.max_delay = max(self.max_delay, d_re, d_im)

    # calculate delay for destinations
    for dest in self.topology_planner.destinations:
      depth: int = dest.bit_length() - 1
      rate_factor: int = 1 << depth
      # サブバンドドメインでの遅延差分を計算
      # Re
      diff_re_input_domain: int = self.max_delay - path_delays_re[dest] # type: ignore
      diff_re_subband: int = diff_re_input_domain // rate_factor 
      for i in range(self.channel_num):
        self.main_delay_buffers_re[i].append(DelayBuffer(diff_re_subband, max_block_size))
        self.sc_delay_buffers_re[i].append(DelayBuffer(diff_re_subband, max_block_size))

      # Im
      diff_im_input_domain: int = self.max_delay - path_delays_im[dest] # type: ignore
      diff_im_subband: int = diff_im_input_domain // rate_factor
      for i in range(self.channel_num):
        self.main_delay_buffers_im[i].append(DelayBuffer(diff_im_subband, max_block_size))
        self.sc_delay_buffers_im[i].append(DelayBuffer(diff_im_subband, max_block_size))


    # init results container
    self.main_results_re: List[Optional[npt.NDArray[np.float64]]] = [None]*self.topology_planner.MAX_SIZE
    self.main_results_im: List[Optional[npt.NDArray[np.float64]]] = [None]*self.topology_planner.MAX_SIZE
    self.main_cursors_re: List[Optional[int]] = [None]*self.topology_planner.MAX_SIZE
    self.main_cursors_im: List[Optional[int]] = [None]*self.topology_planner.MAX_SIZE
    self.sc_results_re: List[Optional[npt.NDArray[np.float64]]] = [None]*self.topology_planner.MAX_SIZE
    self.sc_results_im: List[Optional[npt.NDArray[np.float64]]] = [None]*self.topology_planner.MAX_SIZE
    self.sc_cursors_re: List[Optional[int]] = [None]*self.topology_planner.MAX_SIZE
    self.sc_cursors_im: List[Optional[int]] = [None]*self.topology_planner.MAX_SIZE

    # Re
    for target in self.topology_planner.analysis_order:
      depth = target.bit_length() - 1
      output_size = max_block_size >> depth

      self.main_results_re[target] = np.zeros(output_size, dtype=np.float64)
      self.sc_results_re[target] = np.zeros(output_size, dtype=np.float64)
      self.main_cursors_re[target] = 0
      self.sc_cursors_re[target] = 0

    for target in self.topology_planner.destinations:
      depth = target.bit_length() - 1
      output_size = max_block_size >> depth

      self.main_results_re[target] = np.zeros(output_size, dtype=np.float64)
      self.sc_results_re[target] = np.zeros(output_size, dtype=np.float64)
      
      # カーソル初期化
      self.main_cursors_re[target] = 0
      self.sc_cursors_re[target] = 0
    # Im
    for target in self.topology_planner.analysis_order:
      depth = target.bit_length() - 1
      output_size = max_block_size >> depth

      self.main_results_im[target] = np.zeros(output_size, dtype=np.float64)
      self.sc_results_im[target] = np.zeros(output_size, dtype=np.float64)
      self.main_cursors_im[target] = 0
      self.sc_cursors_im[target] = 0

    for target in self.topology_planner.destinations:
      depth = target.bit_length() - 1
      output_size = max_block_size >> depth

      self.main_results_im[target] = np.zeros(output_size, dtype=np.float64)
      self.sc_results_im[target] = np.zeros(output_size, dtype=np.float64)
      self.main_cursors_im[target] = 0
      self.sc_cursors_im[target] = 0

  def get_latency(self) -> int:
    """システム全体のレイテンシ（サンプル数）を返す"""
    return self.max_delay
  
  def analysisProcess(self, input_block: npt.NDArray[np.float64],
                      active_nodes: List[AnalysisNode],
                      node_ids: List[int],
                      results: List[Optional[npt.NDArray[np.float64]]],
                      cursors: List[Optional[int]]) -> None: # type: ignore
    '''
    入力ブロックをサンプルごとに処理し、ツリー構造に従って分解する。
    再帰呼び出しを使わず、フラグ配列を用いた反復処理で実装。
  
    Args:
      input_block: 入力信号配列
      active_nodes: 全ノードを含む配列
      node_ids: ノードのID
      results: 結果書き込み先。
      cursors: 書き込み位置管理。
      max_nodex: 最大ノード数
    '''

    # --- 1. ルートノードの初期化 ---
    self._analysis_active_flags.fill(False)

    for x in input_block:
      self._analysis_work_buffer[1] = x
      self._analysis_active_flags[1] = True

      # --- 2. 分解プロセス (Analysis) ---
      # 実行計画に従って親ノードを走査
      for node, my_id in zip(active_nodes, node_ids):
        # 親データが無効(前の階層で間引かれた等)ならスキップ
        if not self._analysis_active_flags[my_id]:
          continue

        # 分解実行

        # 子インデックスを算出
        left_idx = my_id << 1
        right_idx = (my_id << 1) | 1

        # nodes配列のバッファに追加
        node.updateBuffer(self._analysis_work_buffer[my_id],
                                self._analysis_work_buffer,
                                self._analysis_active_flags,
                                left_idx,
                                right_idx) # type: ignore


      # --- 3. 結果保存 (Save Results) ---
      # 保存対象のノードだけチェックして書き込み
      for target_idx in self.topology_planner.destinations:
        if self._analysis_active_flags[target_idx]:
          # シンプルな書き込みとカーソルインクリメント (元のロジック通り)
          cursor = cursors[target_idx]
          results[target_idx][cursor] = self._analysis_work_buffer[target_idx] # type: ignore
          cursors[target_idx] += 1 # type: ignore

      # --- 4. クリーンアップ ---
      # 次のサンプルのためにフラグを折る
      # (C++なら memset(is_active, 0, ...) が高速)
      # Python実装として、activeになった可能性のある箇所だけ戻すのが安全だが
      # ここでは簡易的に全リセット相当の挙動を想定
        
      # 最適化: analysis_orderにある子ノードとRootだけFalseに戻せば良い
      self._analysis_active_flags[1] = False
      for pid in node_ids:
        self._analysis_active_flags[pid << 1] = False
        self._analysis_active_flags[(pid << 1) | 1] = False


  def synthesisProcess(self, analysed_data: List[Optional[npt.NDArray[np.float64]]],
                      nodes: List[SynthesisNode],
                      node_ids: List[int],
                      cursors: List[Optional[int]]) -> None: 
    """
    分解されたサブバンド信号を合成して親ノードのバッファに書き込む。

    Args:
      analysed_data: 各ノードのバッファデータを持つコンテナ。
                   (Synthesis処理では、子ノードのデータを読み、親ノードのデータに書き込む)
      nodes: SynthesisNodeオブジェクトの配列。
      node_ids: ノードのID(Synthesis_order)
      cursors: 親ノードへの書き込み位置を管理するカーソル配列。
    """
    #実行計画に基づき、深い階層の親から順に処理
    for node, parent_idx in zip(nodes, node_ids):
      # 1. 親子関係の解決 (ビット演算)
      left_idx = parent_idx << 1
      right_idx = (parent_idx << 1) | 1

      # 2. バッファとノードの取得
      # 事前のTopologyPlannerにより、synthesis_orderにある親は必ず子を持つことが保証されている前提
      low_buf = analysed_data[left_idx]
      high_buf = analysed_data[right_idx]
      out_buf = analysed_data[parent_idx]

      # バッファが存在しない、またはノード未定義の場合はスキップ(安全性担保)
      if (low_buf is None) or (high_buf is None) or (node is None) or (out_buf is None):
        continue

      # 3. 合成処理 (Loop over samples)
      # 子ノード(Low)が実際に何サンプル出力したかを取得
      num_samples = cursors[left_idx]
      write_cursor = cursors[parent_idx]

      node.synthesize_block(low_buf, high_buf, out_buf, write_cursor, num_samples) #type: ignore
            
      # カーソル位置の更新
      cursors[parent_idx] = write_cursor + (num_samples * 2) #type: ignore


  def process_block(self, main_in: npt.NDArray, sc_in: npt.NDArray, param: MorphParam) -> None:
    """
    1ブロック分の信号処理パイプライン。
    Analysis -> Morphing -> Delay -> Synthesis -> Reconstruction
        
    Args:
      main_in: メイン入力 (In-Placeで出力に書き換わる)
      sc_in: サイドチェーン入力 (Read Only)
      param: モーフィングパラメータ
    """
    # Morphing parameter
    # こいつはプラグイン化したときにsmoothパラメータとしていじれるようにする
    

    channels_num = main_in.shape[0]
    buffer_size = main_in.shape[1]

    for channel_idx in range(channels_num):
      main_buf = main_in[channel_idx]
      sc_buf = sc_in[channel_idx]

      # init cursor for all targets in every channel loop
      for target in self.topology_planner.analysis_order:
        self.main_cursors_re[target] = 0
        self.sc_cursors_re[target] = 0
      for target in self.topology_planner.destinations:
        self.main_cursors_re[target] = 0
        self.sc_cursors_re[target] = 0

      for target in self.topology_planner.analysis_order:
        self.main_cursors_im[target] = 0
        self.sc_cursors_im[target] = 0
      for target in self.topology_planner.destinations:
        self.main_cursors_im[target] = 0
        self.sc_cursors_im[target] = 0

      # analysis
      self.analysisProcess(main_buf,
                      self.main_analysis_nodes_re[channel_idx],
                      self.main_analysis_nodes_ids,
                      self.main_results_re,
                      self.main_cursors_re)
      self.analysisProcess(main_buf,
                      self.main_analysis_nodes_im[channel_idx],
                      self.main_analysis_nodes_ids,
                      self.main_results_im,
                      self.main_cursors_im)
      
      self.analysisProcess(sc_buf,
                      self.sc_analysis_nodes_re[channel_idx],
                      self.main_analysis_nodes_ids,
                      self.sc_results_re,
                      self.sc_cursors_re)
      self.analysisProcess(sc_buf,
                      self.sc_analysis_nodes_im[channel_idx],
                      self.main_analysis_nodes_ids,
                      self.sc_results_im,
                      self.sc_cursors_im)


      # morphing
      for dest in self.topology_planner.morphing_list:

        # カーソルからサブバンド長を取得
        re_subband_len = self.main_cursors_re[dest]
        im_subband_len = self.main_cursors_im[dest]

        morphBuffer(self.main_results_re[dest][:re_subband_len], #type: ignore
                    self.main_results_im[dest][:im_subband_len], #type: ignore
                    self.sc_results_re[dest][:re_subband_len],   #type: ignore
                    self.sc_results_im[dest][:im_subband_len],   #type: ignore
                    param)


      # 遅延処理
      # Re
      for dest, buf in zip(self.topology_planner.destinations, self.main_delay_buffers_re[channel_idx]):
        # サブバンド長を取得
        current_subband_len = self.main_cursors_re[dest]

        # 1. 今のバッファの解析結果および遅延処理後の結果の格納先を取り出す(スライス)
        current_main_samples = self.main_results_re[dest][:current_subband_len] #type: ignore

        # 2. StereoDelayBufferに通す
        buf.process(current_main_samples, current_main_samples) #type: ignore

      # Im
      for dest, buf in zip(self.topology_planner.destinations, self.main_delay_buffers_im[channel_idx]):

        # サブバンド長を取得
        current_subband_len = self.main_cursors_im[dest]

        # 1. 今のバッファの解析結果および遅延処理後の結果の格納先を取り出す(スライス)
        current_main_samples = self.main_results_im[dest][:current_subband_len] #type: ignore

        # 2. StereoDelayBufferに通す
        buf.process(current_main_samples, current_main_samples) #type: ignore


      # 再合成
      self.synthesisProcess(self.main_results_re,
                       self.synthesis_nodes_re[channel_idx],
                       self.synthesis_nodes_ids,
                       self.main_cursors_re)
      
      self.synthesisProcess(self.main_results_im,
                       self.synthesis_nodes_im[channel_idx],
                       self.synthesis_nodes_ids,
                       self.main_cursors_im)

      # Synthesis結果(Root node 1)を取り出して平均化
      root_re = self.main_results_re[1][:buffer_size] #type: ignore
      root_im = self.main_results_im[1][:buffer_size] #type: ignore
      
      main_in[channel_idx][:buffer_size] = (root_re + root_im) / 2.0

    return None
  

def dtcwptMorph(
    main_signal_L: npt.NDArray, 
    main_signal_R: npt.NDArray, 
    sc_signal_L: npt.NDArray, 
    sc_signal_R: npt.NDArray, 
    destinations: List[str]
) -> Tuple[int, List[float], List[float]]:
  """
    Stateful DT-CWPT処理のメイン関数 (Test Harness)。
    VSTホスト(DAW)の挙動を模倣して、ブロックごとにProcessorを呼び出す。
  """

  BUFFER_SIZE = 512 # ホストのブロックサイズ設定
  MAX_BLOCK_SIZE = 4096 # プラグインが許容する最大ブロックサイズ
  CHANNEL_NUM = 2
  SAMPLE_RATE = 44100

  # プロセッサの初期化 (ここでメモリ確保とLatency計算が行われる)
  processor = DTCWPTProcessor(SAMPLE_RATE, MAX_BLOCK_SIZE, destinations, CHANNEL_NUM)
  latency = processor.get_latency()

  # 入力信号の結合 (Stereo)
  # shape: (2, N)
  main_signal_stereo = np.array([main_signal_L, main_signal_R])
  sc_signal_stereo = np.array([sc_signal_L, sc_signal_R])
  
  num_input_samples = main_signal_stereo.shape[1]
  
  # 出力バッファ (Listで蓄積して最後に結合)
  output_L: List[float] = []
  output_R: List[float] = []

  # --- 1. メイン信号の処理ループ ---
  # バッファサイズごとにスライスしてProcessorに投げる
  cursor = 0

  params = MorphParam(mag=0.01, phase=1.0, thr=0.01)
  while cursor < num_input_samples:
    # 今回のブロックサイズ (最後は短くなる可能性がある)
    current_block_size = min(BUFFER_SIZE, num_input_samples - cursor)
    
    # スライス取得
    block_main = main_signal_stereo[:, cursor : cursor + current_block_size]
    block_sc = sc_signal_stereo[:, cursor : cursor + current_block_size]
    
    # Process !!
    processor.process_block(block_main, block_sc, params)
    
    # 結果蓄積
    output_L.extend(block_main[0].copy())
    output_R.extend(block_main[1].copy())
    
    cursor += current_block_size

  # --- 2. テール(残響)処理ループ ---
  # 入力が終わった後、内部の遅延バッファに残っている音を吐き出させるために
  # 無音(Silence)を入力し続ける。DAWのLatency Compensation挙動の模倣。
  
  samples_processed_tail = 0
  
  # Latency分だけ無音を流す
  while samples_processed_tail < latency:
    current_block_size = min(BUFFER_SIZE, latency - samples_processed_tail)
    
    # 無音バッファ
    silence_in = np.zeros((CHANNEL_NUM, current_block_size), dtype=np.float64)
    
    # Process Silence
    processor.process_block(silence_in, silence_in, params)
    
    output_L.extend(silence_in[0].copy())
    output_R.extend(silence_in[1].copy())
    
    samples_processed_tail += current_block_size

  # 結果返却
  # 戻り値: (Latencyサンプル数, L出力, R出力)
  # 実際の音は Latency 分だけ遅れて出てきているため、
  # 厳密な波形比較をする場合は output_L[latency:] と input_L を比較することになる
  return (latency, output_L, output_R)