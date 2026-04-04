import trl
import inspect
from trl import DPOTrainer
print(inspect.getdoc(DPOTrainer.compute_metrics) if hasattr(DPOTrainer, 'compute_metrics') else 'No compute_metrics doc')
