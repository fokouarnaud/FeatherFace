"""
Training Monitoring Utilities for FeatherFace
Provides real-time monitoring, metrics tracking, and progress visualization
"""

import time
import json
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional, Union, Any
from collections import defaultdict, deque
import logging
from datetime import datetime


class MetricsTracker:
    """Track and visualize training metrics in real-time"""
    
    def __init__(self, save_dir: Union[str, Path] = "experiments/logs", 
                 window_size: int = 100):
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(parents=True, exist_ok=True)
        
        self.metrics = defaultdict(list)
        self.timestamps = []
        self.window_size = window_size
        self.smoothed_metrics = defaultdict(lambda: deque(maxlen=window_size))
        
        # Training state
        self.start_time = None
        self.current_epoch = 0
        self.best_metrics = {}
        
        # Files for persistent storage
        self.metrics_file = self.save_dir / 'training_metrics.json'
        self.csv_file = self.save_dir / 'training_log.csv'
        
    def start_training(self):
        """Mark the start of training"""
        self.start_time = time.time()
        logging.info("📊 Training metrics tracking started")
        
    def log_epoch(self, epoch: int, metrics: Dict[str, float], 
                  learning_rate: Optional[float] = None,
                  additional_info: Optional[Dict[str, Any]] = None):
        """Log metrics for an epoch"""
        self.current_epoch = epoch
        timestamp = time.time()
        self.timestamps.append(timestamp)
        
        # Store metrics
        for key, value in metrics.items():
            self.metrics[key].append(float(value))
            self.smoothed_metrics[key].append(float(value))
            
        # Store learning rate if provided
        if learning_rate is not None:
            self.metrics['learning_rate'].append(float(learning_rate))
            
        # Track best metrics
        for key, value in metrics.items():
            if 'loss' in key.lower():
                # For losses, lower is better
                if key not in self.best_metrics or value < self.best_metrics[key]['value']:
                    self.best_metrics[key] = {'value': value, 'epoch': epoch}
            else:
                # For other metrics (accuracy, mAP), higher is better
                if key not in self.best_metrics or value > self.best_metrics[key]['value']:
                    self.best_metrics[key] = {'value': value, 'epoch': epoch}
        
        # Calculate elapsed time and ETA
        elapsed = timestamp - self.start_time if self.start_time else 0
        
        # Create log entry
        log_entry = {
            'epoch': epoch,
            'timestamp': timestamp,
            'elapsed_time': elapsed,
            **metrics
        }
        
        if learning_rate is not None:
            log_entry['learning_rate'] = learning_rate
            
        if additional_info:
            log_entry.update(additional_info)
            
        # Save to files
        self._save_metrics(log_entry)
        
        # Log to console
        self._log_to_console(epoch, metrics, learning_rate, elapsed)
        
    def _save_metrics(self, log_entry: Dict):
        """Save metrics to persistent storage"""
        # Save to JSON (complete metrics history)
        all_metrics = {
            'training_start': self.start_time,
            'current_epoch': self.current_epoch,
            'best_metrics': self.best_metrics,
            'metrics_history': dict(self.metrics),
            'timestamps': self.timestamps
        }
        
        with open(self.metrics_file, 'w') as f:
            json.dump(all_metrics, f, indent=2)
            
        # Save to CSV (for easy analysis)
        df = pd.DataFrame([log_entry])
        if self.csv_file.exists():
            df.to_csv(self.csv_file, mode='a', header=False, index=False)
        else:
            df.to_csv(self.csv_file, index=False)
            
    def _log_to_console(self, epoch: int, metrics: Dict[str, float], 
                       learning_rate: Optional[float], elapsed: float):
        """Log metrics to console with nice formatting"""
        # Format metrics for display
        metric_strs = []
        for key, value in metrics.items():
            if 'loss' in key.lower():
                metric_strs.append(f"{key}: {value:.4f}")
            elif 'map' in key.lower() or 'acc' in key.lower():
                metric_strs.append(f"{key}: {value:.3f}")
            else:
                metric_strs.append(f"{key}: {value:.3f}")
                
        metric_str = " | ".join(metric_strs)
        
        # Calculate time estimates
        time_per_epoch = elapsed / epoch if epoch > 0 else 0
        eta_str = ""
        
        if time_per_epoch > 0:
            eta_mins = (time_per_epoch * (100 - epoch)) / 60  # Assuming 100 epochs
            eta_str = f" | ETA: {eta_mins:.1f}min"
            
        # Log rate info
        lr_str = f" | LR: {learning_rate:.2e}" if learning_rate else ""
        
        logging.info(f"Epoch {epoch:3d} | {metric_str}{lr_str} | "
                    f"Time: {elapsed/60:.1f}min{eta_str}")
        
    def get_smoothed_metrics(self, metric_name: str, window: Optional[int] = None) -> List[float]:
        """Get smoothed version of a metric"""
        if metric_name not in self.metrics:
            return []
            
        values = self.metrics[metric_name]
        window = window or min(len(values), 10)
        
        if len(values) < window:
            return values
            
        smoothed = []
        for i in range(len(values)):
            start_idx = max(0, i - window + 1)
            smoothed.append(np.mean(values[start_idx:i+1]))
            
        return smoothed
        
    def plot_metrics(self, metrics: Optional[List[str]] = None, 
                    save_path: Optional[Path] = None,
                    show_smoothed: bool = True) -> plt.Figure:
        """Plot training metrics"""
        if not self.metrics:
            logging.warning("No metrics to plot")
            return None
            
        # Default to all metrics if none specified
        if metrics is None:
            metrics = list(self.metrics.keys())
            if 'learning_rate' in metrics:
                metrics.remove('learning_rate')  # Plot LR separately
                
        # Determine subplot layout
        n_metrics = len(metrics)
        has_lr = 'learning_rate' in self.metrics
        n_plots = n_metrics + (1 if has_lr else 0)
        
        if n_plots == 1:
            fig, axes = plt.subplots(1, 1, figsize=(10, 6))
            axes = [axes]
        elif n_plots <= 2:
            fig, axes = plt.subplots(1, n_plots, figsize=(15, 6))
        else:
            rows = (n_plots + 1) // 2
            fig, axes = plt.subplots(rows, 2, figsize=(15, 6 * rows))
            axes = axes.flatten()
            
        plot_idx = 0
        epochs = list(range(1, len(self.metrics[metrics[0]]) + 1))
        
        # Plot main metrics
        for metric in metrics:
            if metric not in self.metrics:
                continue
                
            ax = axes[plot_idx]
            values = self.metrics[metric]
            
            ax.plot(epochs, values, alpha=0.7, label='Raw', linewidth=1)
            
            if show_smoothed and len(values) > 5:
                smoothed = self.get_smoothed_metrics(metric)
                ax.plot(epochs, smoothed, label='Smoothed', linewidth=2)
                
            ax.set_title(f'{metric.replace("_", " ").title()}')
            ax.set_xlabel('Epoch')
            ax.set_ylabel(metric)
            ax.grid(True, alpha=0.3)
            ax.legend()
            
            # Mark best value
            if metric in self.best_metrics:
                best = self.best_metrics[metric]
                ax.axhline(y=best['value'], color='red', linestyle='--', alpha=0.5,
                          label=f'Best: {best["value"]:.4f} (epoch {best["epoch"]})')
                ax.legend()
                
            plot_idx += 1
            
        # Plot learning rate if available
        if has_lr and plot_idx < len(axes):
            ax = axes[plot_idx]
            lr_values = self.metrics['learning_rate']
            ax.plot(epochs, lr_values, 'orange', linewidth=2)
            ax.set_title('Learning Rate')
            ax.set_xlabel('Epoch')
            ax.set_ylabel('Learning Rate')
            ax.set_yscale('log')
            ax.grid(True, alpha=0.3)
            plot_idx += 1
            
        # Hide unused subplots
        for i in range(plot_idx, len(axes)):
            axes[i].set_visible(False)
            
        plt.tight_layout()
        
        # Save if path provided
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            logging.info(f"Metrics plot saved to {save_path}")
            
        return fig
        
    def get_summary(self) -> Dict[str, Any]:
        """Get training summary statistics"""
        if not self.metrics:
            return {}
            
        summary = {
            'total_epochs': self.current_epoch,
            'training_time_hours': (time.time() - self.start_time) / 3600 if self.start_time else 0,
            'best_metrics': self.best_metrics,
            'current_metrics': {key: values[-1] for key, values in self.metrics.items() if values}
        }
        
        return summary
        
    def load_previous_session(self, metrics_file: Optional[Path] = None) -> bool:
        """Load metrics from a previous training session"""
        file_path = metrics_file or self.metrics_file
        
        if not file_path.exists():
            logging.info("No previous session found")
            return False
            
        try:
            with open(file_path, 'r') as f:
                data = json.load(f)
                
            self.start_time = data.get('training_start')
            self.current_epoch = data.get('current_epoch', 0)
            self.best_metrics = data.get('best_metrics', {})
            
            # Restore metrics history
            for key, values in data.get('metrics_history', {}).items():
                self.metrics[key] = values
                
            self.timestamps = data.get('timestamps', [])
            
            logging.info(f"Loaded previous session: {self.current_epoch} epochs")
            return True
            
        except Exception as e:
            logging.error(f"Failed to load previous session: {e}")
            return False


class PerformanceMonitor:
    """Monitor and track model performance metrics"""
    
    def __init__(self):
        self.training_times = []
        self.inference_times = []
        self.memory_usage = []
        
    def time_operation(self, operation_name: str = "operation"):
        """Context manager for timing operations"""
        from contextlib import contextmanager
        
        @contextmanager
        def timer():
            start = time.time()
            try:
                yield
            finally:
                duration = time.time() - start
                self.training_times.append(duration)
                logging.info(f"{operation_name} completed in {duration:.2f}s")
                
        return timer()
        
    def benchmark_inference(self, model, input_tensor, num_runs: int = 100):
        """Benchmark model inference speed"""
        model.eval()
        times = []
        
        # Warmup
        with torch.no_grad():
            for _ in range(10):
                _ = model(input_tensor)
                
        # Benchmark
        with torch.no_grad():
            for _ in range(num_runs):
                start = time.time()
                _ = model(input_tensor)
                if torch.cuda.is_available():
                    torch.cuda.synchronize()
                times.append(time.time() - start)
                
        stats = {
            'mean_time_ms': np.mean(times) * 1000,
            'std_time_ms': np.std(times) * 1000,
            'min_time_ms': np.min(times) * 1000,
            'max_time_ms': np.max(times) * 1000,
            'fps': 1.0 / np.mean(times)
        }
        
        self.inference_times.extend(times)
        
        logging.info(f"Inference benchmark: {stats['mean_time_ms']:.2f}±{stats['std_time_ms']:.2f}ms "
                    f"({stats['fps']:.1f} FPS)")
        
        return stats


def create_training_dashboard(metrics_tracker: MetricsTracker, 
                            update_interval: int = 10) -> None:
    """Create a live training dashboard (for Jupyter notebooks)"""
    try:
        from IPython.display import display, clear_output
        import matplotlib.pyplot as plt
        
        def update_dashboard():
            clear_output(wait=True)
            
            # Plot metrics
            fig = metrics_tracker.plot_metrics()
            plt.show()
            
            # Show summary table
            summary = metrics_tracker.get_summary()
            if summary:
                print("\n=== Training Summary ===")
                print(f"Epoch: {summary.get('total_epochs', 0)}")
                print(f"Training time: {summary.get('training_time_hours', 0):.2f} hours")
                
                print("\n=== Best Metrics ===")
                for metric, info in summary.get('best_metrics', {}).items():
                    print(f"{metric}: {info['value']:.4f} (epoch {info['epoch']})")
                    
        return update_dashboard
        
    except ImportError:
        logging.warning("IPython not available, dashboard disabled")
        return lambda: None


# Convenience function for notebook setup
def setup_training_monitoring(experiment_name: str = None) -> MetricsTracker:
    """Quick setup for training monitoring in notebooks"""
    if experiment_name:
        save_dir = Path("experiments/logs") / experiment_name
    else:
        save_dir = Path("experiments/logs") / datetime.now().strftime("%Y%m%d_%H%M%S")
        
    tracker = MetricsTracker(save_dir=save_dir)
    
    logging.info(f"📊 Training monitoring setup complete")
    logging.info(f"📁 Logs will be saved to: {save_dir}")
    
    return tracker


if __name__ == "__main__":
    # Demo usage
    import torch
    
    # Setup logging
    logging.basicConfig(level=logging.INFO)
    
    # Create tracker
    tracker = MetricsTracker()
    tracker.start_training()
    
    # Simulate training
    for epoch in range(1, 11):
        # Simulate metrics
        train_loss = 1.0 - (epoch * 0.05) + np.random.normal(0, 0.05)
        val_loss = train_loss + 0.1 + np.random.normal(0, 0.03)
        val_map = 0.5 + (epoch * 0.03) + np.random.normal(0, 0.02)
        lr = 0.001 * (0.9 ** (epoch // 3))
        
        tracker.log_epoch(
            epoch=epoch,
            metrics={
                'train_loss': train_loss,
                'val_loss': val_loss,
                'val_map': val_map
            },
            learning_rate=lr
        )
        
        time.sleep(0.1)  # Simulate training time
        
    # Plot results
    fig = tracker.plot_metrics()
    plt.show()
    
    # Print summary
    summary = tracker.get_summary()
    print("\nTraining Summary:")
    for key, value in summary.items():
        print(f"{key}: {value}")