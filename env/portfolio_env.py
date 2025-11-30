import gym
import pandas as pd
import torch
from gym import spaces
import numpy as np
from stable_baselines3.common.vec_env import DummyVecEnv


class StockPortfolioEnv(gym.Env):
    def __init__(self, args, corr=None, ts_features=None, features=None,
             ind=None, pos=None, neg=None, returns=None, pyg_data=None,
             benchmark_return=None, mode="train", reward_net=None, device='cuda:0',
             ind_yn=False, pos_yn=False, neg_yn=False, risk_profile=None):
        super(StockPortfolioEnv, self).__init__()
        
        print("\n" + "="*80)
        print("[ENV INIT] Starting StockPortfolioEnv initialization")
        print("="*80)
        
        self.current_step = 0
        self.max_step = returns.shape[0] - 1
        self.done = False
        self.reward = 0.0
        self.net_value = 1.0
        self.peak_value = 1.0
        self.net_value_s = [1.0]
        self.daily_return_s = [0.0]
        self.num_stocks = returns.shape[-1]
        self.benchmark_return = benchmark_return

        # Track portfolio weights history
        self.weights_history = []
        self.dates = []
        self.prev_weights = np.zeros(self.num_stocks, dtype=np.float32)

        self.corr_tensor = corr
        self.ts_features_tensor = ts_features
        self.features_tensor = features
        self.ind_tensor = ind
        self.pos_tensor = pos
        self.neg_tensor = neg
        self.pyg_data_batch = pyg_data
        self.ror_batch = returns
        self.ind_yn = ind_yn
        self.pos_yn = pos_yn
        self.neg_yn = neg_yn
        self.mode = mode
        self.reward_net = reward_net
        self.device = device

        self.risk_profile = risk_profile or {}
        self.risk_score = float(self.risk_profile.get('risk_score', getattr(args, 'risk_score', 0.5)))
        self.max_weight_cap = self.risk_profile.get('max_weight', None)
        self.min_weight_floor = self.risk_profile.get('min_weight', 0.0)
        self.sector_cap = self.risk_profile.get('sector_cap', 0.60)
        self.rebalance_window = 21
        self.action_temperature = 0.5 + 2.0 * (1.0 - self.risk_score)

        # Debug: Print input shapes
        print(f"[ENV INIT] returns shape: {returns.shape}")
        print(f"[ENV INIT] num_stocks: {self.num_stocks}")
        print(f"[ENV INIT] max_step: {self.max_step}")
        
        if ts_features is not None:
            print(f"[ENV INIT] ts_features shape: {ts_features.shape}")
            print(f"[ENV INIT] ts_features ndim: {ts_features.ndim}")
        else:
            print("[ENV INIT] WARNING: ts_features is None!")
        
        if ind is not None:
            print(f"[ENV INIT] ind shape: {ind.shape}")
        if pos is not None:
            print(f"[ENV INIT] pos shape: {pos.shape}")
        if neg is not None:
            print(f"[ENV INIT] neg shape: {neg.shape}")

        # Action space
        self.action_space = spaces.Box(
            low=-10.0,
            high=10.0,
            shape=(self.num_stocks,),
            dtype=np.float32,
        )

        self.top_k = max(1, int(0.1 * self.num_stocks))

        # CRITICAL: Extract lookback and feat_dim correctly
        print("\n[ENV INIT] Extracting lookback and feat_dim...")
        
        if ts_features is not None:
            ts_shape = ts_features.shape
            print(f"[ENV INIT] ts_features shape: {ts_shape}")
            
            # Handle different tensor formats
            if len(ts_shape) == 4:
                # (batch, num_stocks, lookback, feat_dim)
                self.lookback = ts_shape[2]
                self.feat_dim = ts_shape[3]
                print(f"[ENV INIT] 4D tensor detected: lookback={self.lookback}, feat_dim={self.feat_dim}")
            elif len(ts_shape) == 3:
                # (num_stocks, lookback, feat_dim)
                self.lookback = ts_shape[1]
                self.feat_dim = ts_shape[2]
                print(f"[ENV INIT] 3D tensor detected: lookback={self.lookback}, feat_dim={self.feat_dim}")
            else:
                raise ValueError(f"[ENV INIT] ERROR: ts_features has unexpected shape {ts_shape}. Expected 3D or 4D.")
        else:
            # Fallback to args if ts_features not provided
            self.lookback = getattr(args, "lookback", 30)
            self.feat_dim = getattr(args, "input_dim", 6)
            print(f"[ENV INIT] Using fallback from args: lookback={self.lookback}, feat_dim={self.feat_dim}")

        print(f"[ENV INIT] Final values: lookback={self.lookback}, feat_dim={self.feat_dim}")

        # Calculate observation space size
        adj_size = self.num_stocks * self.num_stocks
        print(f"\n[ENV INIT] Calculating observation space:")
        print(f"[ENV INIT]   Adjacency matrix size: {adj_size} (for each of 3 matrices)")
        print(f"[ENV INIT]   Total adjacency: {adj_size * 3}")
        
        ts_flat_size = self.num_stocks * self.lookback * self.feat_dim
        print(f"[ENV INIT]   TS features flat size: {self.num_stocks} * {self.lookback} * {self.feat_dim} = {ts_flat_size}")
        
        prev_weights_size = self.num_stocks
        print(f"[ENV INIT]   Previous weights size: {prev_weights_size}")
        
        obs_len = (adj_size * 3) + ts_flat_size + prev_weights_size
        print(f"[ENV INIT]   TOTAL OBSERVATION SIZE: {obs_len}")
        print(f"[ENV INIT]     = {adj_size * 3} + {ts_flat_size} + {prev_weights_size}")

        # Observation space
        self.observation_space = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(obs_len,),
            dtype=np.float32,
        )
        
        print(f"[ENV INIT] Observation space shape: {self.observation_space.shape}")
        print("="*80 + "\n")

    def load_observation(self, ts_yn=False, ind_yn=False, pos_yn=False, neg_yn=False):
        """Load observation at current step with detailed debugging."""
        
        print(f"\n[LOAD_OBS] Step {self.current_step}")
        
        # Adjacency matrices
        if torch.is_tensor(self.ind_tensor):
            ind_matrix = self.ind_tensor[self.current_step].cpu().numpy()
            pos_matrix = self.pos_tensor[self.current_step].cpu().numpy()
            neg_matrix = self.neg_tensor[self.current_step].cpu().numpy()
        else:
            ind_matrix = self.ind_tensor[self.current_step]
            pos_matrix = self.pos_tensor[self.current_step]
            neg_matrix = self.neg_tensor[self.current_step]

        print(f"[LOAD_OBS]   ind_matrix shape: {ind_matrix.shape}")
        print(f"[LOAD_OBS]   pos_matrix shape: {pos_matrix.shape}")
        print(f"[LOAD_OBS]   neg_matrix shape: {neg_matrix.shape}")

        # Time-series features
        if torch.is_tensor(self.ts_features_tensor):
            ts_data = self.ts_features_tensor[self.current_step].cpu().numpy()
        else:
            ts_data = self.ts_features_tensor[self.current_step]

        print(f"[LOAD_OBS]   ts_data shape: {ts_data.shape}")
        print(f"[LOAD_OBS]   Expected shape: ({self.num_stocks}, {self.lookback}, {self.feat_dim})")

        if ts_data.shape != (self.num_stocks, self.lookback, self.feat_dim):
            print(f"[LOAD_OBS] ERROR: Shape mismatch!")
            print(f"[LOAD_OBS]   Got: {ts_data.shape}")
            print(f"[LOAD_OBS]   Expected: ({self.num_stocks}, {self.lookback}, {self.feat_dim})")
            raise ValueError(
                f"ts_features shape {ts_data.shape} does not match expected "
                f"({self.num_stocks}, {self.lookback}, {self.feat_dim})"
            )

        obs_parts = []
        zeros_mat = np.zeros(self.num_stocks * self.num_stocks, dtype=np.float32)
        
        part_ind = ind_matrix.flatten() if ind_yn else zeros_mat
        part_pos = pos_matrix.flatten() if pos_yn else zeros_mat
        part_neg = neg_matrix.flatten() if neg_yn else zeros_mat
        part_ts = ts_data.flatten()
        part_prev = self.prev_weights.astype(np.float32)
        
        print(f"[LOAD_OBS]   part_ind size: {part_ind.shape[0]}")
        print(f"[LOAD_OBS]   part_pos size: {part_pos.shape[0]}")
        print(f"[LOAD_OBS]   part_neg size: {part_neg.shape[0]}")
        print(f"[LOAD_OBS]   part_ts size: {part_ts.shape[0]}")
        print(f"[LOAD_OBS]   part_prev size: {part_prev.shape[0]}")
        
        obs_parts = [part_ind, part_pos, part_neg, part_ts, part_prev]
        self.observation = np.concatenate(obs_parts).astype(np.float32)
        
        print(f"[LOAD_OBS]   Final observation size: {self.observation.shape[0]}")
        print(f"[LOAD_OBS]   Expected size: {self.observation_space.shape[0]}")
        
        if self.observation.shape[0] != self.observation_space.shape[0]:
            print(f"[LOAD_OBS] ERROR: Observation size mismatch!")
            raise ValueError(
                f"Observation size {self.observation.shape[0]} doesn't match "
                f"observation space size {self.observation_space.shape[0]}"
            )
        
        self.ror = self.ror_batch[self.current_step].cpu()
    def reset(self):
        self.current_step = 0
        self.done = False
        self.reward = 0.0
        self.net_value = 1.0
        self.peak_value = 1.0
        self.net_value_s = [1.0]
        self.daily_return_s = [0.0]
        self.weights_history = []
        self.dates = []
        self.prev_weights = np.zeros(self.num_stocks, dtype=np.float32)
        self.load_observation(ind_yn=self.ind_yn, pos_yn=self.pos_yn, neg_yn=self.neg_yn)
        return self.observation

    def seed(self, seed):
        return np.random.seed(seed)

    def step(self, actions):
        self.done = self.current_step == self.max_step
        if self.done:
            if self.mode == "test":
                print("=================================")
                print(f"Final Net Value: {self.net_value:.4f}")
                metrics, benchmark_metrics = self.evaluate()
                print(f"ARR: {metrics.get('arr'):.4f} | AVOL: {metrics.get('avol'):.4f} | Sharpe: {metrics.get('sharpe'):.4f}")
                print(f"MDD: {metrics.get('mdd'):.4f} | CR: {metrics.get('cr'):.4f} | IR: {metrics.get('ir'):.4f}")
                if benchmark_metrics:
                    print(
                        f"Benchmark ARR: {benchmark_metrics.get('arr'):.4f} | "
                        f"AVOL: {benchmark_metrics.get('avol'):.4f} | "
                        f"Sharpe: {benchmark_metrics.get('sharpe'):.4f} | "
                        f"MDD: {benchmark_metrics.get('mdd'):.4f} | "
                        f"CR: {benchmark_metrics.get('cr'):.4f}"
                    )
                print("=================================")
        else:
            self.current_step += 1
            self.load_observation(ind_yn=self.ind_yn, pos_yn=self.pos_yn, neg_yn=self.neg_yn)

            prev_weights = self.weights_history[-1] if self.weights_history else self.prev_weights
            is_rebalancing_day = (self.current_step % self.rebalance_window == 0) or (self.current_step == 1)

            if is_rebalancing_day:
                action_scores = np.array(actions).flatten()
                temp = max(self.action_temperature, 1e-4)
                exp_actions = np.exp((action_scores - np.max(action_scores)) / temp)
                weights = exp_actions / exp_actions.sum()
                weights = self._apply_risk_constraints(weights)

                turnover = np.sum(np.abs(weights - prev_weights))
                transaction_cost_factor = 0.001
                turnover_penalty = transaction_cost_factor * turnover
            else:
                stock_returns = np.array(self.ror)
                new_values = prev_weights * (1 + stock_returns)
                portfolio_value_change = np.sum(new_values)
                if portfolio_value_change > 0:
                    weights = new_values / portfolio_value_change
                else:
                    weights = prev_weights
                turnover_penalty = 0.0

            self.weights_history.append(weights.copy())
            self.dates.append(self.current_step)
            self.prev_weights = weights.copy()

            if self.mode == "test" and self.current_step % self.rebalance_window == 0:
                print(f"Step {self.current_step} [Rebalance={is_rebalancing_day}]")
                if is_rebalancing_day:
                    print(f"  Turnover: {turnover:.4f} | Cost: {turnover_penalty:.6f}")

            # Reward
            if self.reward_net is not None:
                state_tensor = torch.FloatTensor(self.observation).to(self.device)
                action_tensor = torch.FloatTensor(weights).to(self.device)
                wealth_info = torch.FloatTensor([self.net_value, self.peak_value]).to(self.device)
                with torch.no_grad():
                    raw_reward = self.reward_net(
                        state_tensor,
                        action_tensor,
                        wealth_info,
                        risk_score=self.risk_score
                    ).mean().cpu().item()
            else:
                raw_reward = np.dot(weights, np.array(self.ror))

            self.reward = raw_reward - turnover_penalty

            # Wealth tracking
            step_return = np.dot(weights, np.array(self.ror))
            self.net_value *= (1 + step_return)
            self.peak_value = max(self.peak_value, self.net_value)
            self.daily_return_s.append(step_return)
            self.net_value_s.append(self.net_value)

        return self.observation, self.reward, self.done, {}

    def _apply_risk_constraints(self, weights):
        """Apply individual weight caps AND sector exposure caps."""
        
        # 1. Apply Individual Stock Cap (Existing Logic)
        if self.max_weight_cap is not None and self.max_weight_cap > 0:
            weights = np.minimum(weights, self.max_weight_cap)
            if self.min_weight_floor > 0:
                weights = np.maximum(weights, self.min_weight_floor)
        
        # 2. Apply Sector Cap (New Logic)
        if self.sector_cap is not None and self.ind_yn:
            # We need the industry matrix to know which stocks belong to which sector.
            # We grab it from the current step's tensor.
            if torch.is_tensor(self.ind_tensor):
                ind_matrix = self.ind_tensor[self.current_step].cpu().numpy()
            else:
                ind_matrix = self.ind_tensor[self.current_step]
            
            # Identify unique sectors.
            # ind_matrix rows are "masks" for each sector. 
            # We iterate through unique rows to find unique sectors.
            # (Note: This assumes the matrix is a clean block-diagonal or similar structure where 
            # rows for stocks in the same sector are identical)
            unique_sectors = np.unique(ind_matrix, axis=0)
            
            for sector_mask in unique_sectors:
                # sector_mask is 1.0 for stocks in this sector, 0.0 otherwise
                if sector_mask.sum() == 0: continue # Skip empty/padding rows
                
                # Calculate total weight currently in this sector
                current_sector_weight = np.dot(weights, sector_mask)
                
                # If violation, scale down ONLY this sector's stocks
                if current_sector_weight > self.sector_cap:
                    scale_factor = self.sector_cap / current_sector_weight
                    # Apply scale factor where mask is 1
                    weights = np.where(sector_mask == 1, weights * scale_factor, weights)

        # 3. Final Normalization
        # Caps might have reduced the total sum below 1.0, so we re-normalize.
        total = weights.sum()
        if total > 1e-8:
            return weights / total
        
        # Fallback if everything became zero (shouldn't happen)
        return np.ones_like(weights) / len(weights)

    def get_sb_env(self):
        e = DummyVecEnv([lambda: self])
        obs = e.reset()
        return e, obs

    def get_df_net_value(self):
        df_net_value = pd.DataFrame(self.net_value_s)
        df_net_value.columns = ["net_value"]
        return df_net_value

    def get_df_daily_return(self):
        df_daily_return = pd.DataFrame(self.daily_return_s)
        df_daily_return.columns = ["daily_return"]
        return df_daily_return

    def get_weights_history(self):
        if not self.weights_history:
            return np.array([])
        return np.array(self.weights_history)

    def get_weights_dataframe(self, tickers=None):
        weights_array = self.get_weights_history()
        if weights_array.size == 0:
            return pd.DataFrame()
        if tickers is not None and len(tickers) == weights_array.shape[1]:
            columns = tickers
        else:
            columns = [f"stock_{i}" for i in range(weights_array.shape[1])]
        df = pd.DataFrame(weights_array, columns=columns)
        df.insert(0, "step", self.dates)
        return df

    def evaluate(self):
        df_daily_return = self.get_df_daily_return()
        metrics = dict(arr=0, avol=0, sharpe=0, mdd=0, cr=0, ir=0)
        benchmark_metrics = {}

        if df_daily_return["daily_return"].std() != 0:
            arr = (1 + df_daily_return["daily_return"].mean()) ** 252 - 1
            avol = df_daily_return["daily_return"].std() * (252 ** 0.5)
            sharpe = (252 ** 0.5) * df_daily_return["daily_return"].mean() / df_daily_return["daily_return"].std()
            df_daily_return["cumulative_return"] = (1 + df_daily_return["daily_return"]).cumprod()
            running_max = df_daily_return["cumulative_return"].cummax()
            drawdown = df_daily_return["cumulative_return"] / running_max - 1
            mdd = drawdown.min()
            cr = arr / abs(mdd) if mdd != 0 else 0

            metrics.update(arr=arr, avol=avol, sharpe=sharpe, mdd=mdd, cr=cr)

            if self.benchmark_return is not None and len(self.benchmark_return) == len(df_daily_return):
                ex_return = df_daily_return["daily_return"] - self.benchmark_return.reset_index(drop=True)
                if ex_return.std() != 0:
                    ir = ex_return.mean() / ex_return.std() * (252 ** 0.5)
                    benchmark_arr = (1 + self.benchmark_return.mean()) ** 252 - 1
                    benchmark_avol = self.benchmark_return.std() * (252 ** 0.5)
                    benchmark_sharpe = (
                        (252 ** 0.5) * self.benchmark_return.mean() / self.benchmark_return.std()
                        if self.benchmark_return.std() != 0 else 0
                    )
                    benchmark_mdd = 0
                    benchmark_cr = 0
                    benchmark_metrics = dict(
                        arr=benchmark_arr,
                        avol=benchmark_avol,
                        sharpe=benchmark_sharpe,
                        mdd=benchmark_mdd,
                        cr=benchmark_cr,
                    )
                    metrics["ir"] = ir

        return metrics, benchmark_metrics
    def get_net_value_history(self) -> np.ndarray:
        """Return portfolio value history as numpy array."""
        return np.array(self.net_value_s)

    def get_daily_returns_history(self) -> np.ndarray:
        """Return daily returns history as numpy array."""
        return np.array(self.daily_return_s)