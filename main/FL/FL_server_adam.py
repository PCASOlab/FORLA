import os
import time
import shutil
import torch
os.environ['WORKING_DIR_IMPORT_MODE'] = 'train_miccai'  # Change this to your target mode
# os.environ['WORKING_DIR_IMPORT_MODE'] = 'eval_miccai'  # Change this to your target mode


print("Current working directory:", os.getcwd())
from working_dir_root import Output_root,Linux_computer
Output_root = Output_root + "FL/Dino/"

# SHARED_ROOT = "./shared_models"
SHARED_ROOT = Output_root + "shared_models/"
FED_MIN_CLIENTS = 2
FED_CLIENT_TIMEOUT = 300  # 5 minutes
COMPONENTS = ["initializer", "adapter", "processor", "decoder"]
Average_student = False

class ServerFedHelper:
    def __init__(self):
        # self.global_version = self._read_global_version()
        self.global_version = 0
        global_dir = os.path.join(SHARED_ROOT, "global")
        self.global_dir = global_dir
        os.makedirs(global_dir, exist_ok=True)
        with open(os.path.join(global_dir, "version.txt"), "w") as f:
                f.write(str(self.global_version))
        # Initialize global model from client if needed
        if not self._global_model_exists():
            self._initialize_global_from_client()

        # FedAdam parameters
        self.beta1 = 0.9      # Momentum decay
        self.beta2 = 0.999    # Variance decay
        self.epsilon = 1e-8   # Numerical stability
        self.server_lr = 0.01 # Server learning rate
        
        # FedAdam state
        self.m = None         # First moment estimate
        self.v = None         # Second moment estimate
        self.t = 0            # Time step counter
        self._clean_previous_state()  # Add this line

        self._load_fedadam_state()  # Load previous state if exists
    def _global_model_exists(self):
        """Check if any component of global model exists"""
        return any(
                    os.path.exists(os.path.join(self.global_dir, f"{comp}.pth"))  # Added closing )
                    for comp in COMPONENTS
                )

    def _initialize_global_from_client(self):
        """Initialize global model from latest client model"""
        print("Initializing global model from client...")
        
        # Find latest client model (using client_1 as default)
        client_dirs = [d for d in os.listdir(SHARED_ROOT) 
                     if d.startswith("client_")]
        if not client_dirs:
            raise RuntimeError("No client models available for initialization")

        # Use client_1 if exists
        source_client = "client_1" if "client_1" in client_dirs else client_dirs[0]
        source_dir = os.path.join(SHARED_ROOT, source_client)

        # Copy client model to global
        tmp_dir = os.path.join(self.global_dir, "tmp")
        os.makedirs(tmp_dir, exist_ok=True)
        
        try:
            for comp in COMPONENTS:
                src = os.path.join(source_dir, f"{comp}.pth")
                dst = os.path.join(tmp_dir, f"{comp}.pth")
                if os.path.exists(src):
                    shutil.copy(src, dst)
                    print(f"Copied {comp} from {source_client}")

            # Atomic replace
            for fname in os.listdir(tmp_dir):
                os.replace(
                    os.path.join(tmp_dir, fname),
                    os.path.join(self.global_dir, fname)
                )

            print(f"Global model initialized from {source_client}")
        finally:
            shutil.rmtree(tmp_dir, ignore_errors=True)
    def _load_fedadam_state(self):
        """Load FedAdam optimizer state from disk"""
        state_path = os.path.join(SHARED_ROOT, "global", "fedadam_state.pth")
        if os.path.exists(state_path):
            state = torch.load(state_path)
            self.m = state['m']
            self.v = state['v']
            self.t = state['t']
            print(f"Loaded FedAdam state at t={self.t}")

    def _save_fedadam_state(self):
        """Persist FedAdam optimizer state"""
        state_path = os.path.join(SHARED_ROOT, "global", "fedadam_state.pth")
        torch.save({
            'm': self.m,
            'v': self.v,
            't': self.t
        }, state_path)
    def _read_global_version(self):
        try:
            with open(os.path.join(SHARED_ROOT, "global", "version.txt"), "r") as f:
                return int(f.read())
        except:
            return 0

    def _get_active_clients(self):
        clients = []
        for entry in os.listdir(SHARED_ROOT):
            if entry.startswith("client_") and os.path.isdir(os.path.join(SHARED_ROOT, entry)):
                status_file = os.path.join(SHARED_ROOT, entry, "status.txt")
                if not os.path.exists(status_file):
                    continue
                
                try:
                    with open(status_file, "r") as f:
                        content = f.read().strip()
                        if not content:  # Empty file
                            print(f"Empty status file in {entry}")
                            continue
                        
                        timestamp = float(content)
                        if time.time() - timestamp < FED_CLIENT_TIMEOUT:
                            clients.append(entry)
                except ValueError as e:
                    print(f"Invalid timestamp in {entry}: {e}")
                except Exception as e:
                    print(f"Error processing {entry}: {e}")
                    
        return clients

    def _acquire_lock(self):
        lock_file = os.path.join(SHARED_ROOT, "fed.lock")
        for _ in range(10):  # 10 attempts
            try:
                fd = os.open(lock_file, os.O_CREAT | os.O_EXCL | os.O_RDWR)
                os.close(fd)
                return True
            except FileExistsError:
                time.sleep(1)
        return False

    def _release_lock(self):
        lock_file = os.path.join(SHARED_ROOT, "fed.lock")
        if os.path.exists(lock_file):
            os.remove(lock_file)

    def aggregate_models(self):
        if not self._acquire_lock():
            return False
            
        try:
            clients = self._get_active_clients()
            if len(clients) < FED_MIN_CLIENTS:
                print(f"Not enough clients ({len(clients)}) for aggregation")
                return False

            # Load previous global model
            global_dir = os.path.join(SHARED_ROOT, "global")
            prev_global = self._load_full_global_model()

            # Collect client updates
            client_updates = []
            for client in clients:
                client_dir = os.path.join(SHARED_ROOT, client)
                client_model = self._load_client_model(client_dir)
                client_update = self._compute_update(prev_global, client_model)
                client_updates.append(client_update)

            # FedAdam aggregation logic
            avg_update = self._average_updates(client_updates)
            new_global = self._apply_fedadam(prev_global, avg_update)

            # Save new global model
            self._save_full_global_model(new_global)
            self.global_version += 1
            self._save_fedadam_state()
            with open(os.path.join(global_dir, "version.txt"), "w") as f:
                f.write(str(self.global_version))
            
            return True
            
        finally:
            self._release_lock()

    def _load_full_global_model(self):
        """Load all components into a single state dict with prefixed keys"""
        state = {}
        for comp in COMPONENTS:
            path = os.path.join(SHARED_ROOT, "global", f"{comp}.pth")
            if os.path.exists(path):
                comp_state = torch.load(path, map_location='cpu')
                # Prefix each key with component name
                for key, value in comp_state.items():
                    prefixed_key = f"{comp}.{key}"
                    state[prefixed_key] = value
        return state

    def _load_client_model(self, client_dir):
        """Load client model components with prefixed keys"""
        state = {}
        for comp in COMPONENTS:
            path = os.path.join(client_dir, f"{comp}.pth")
            if os.path.exists(path):
                comp_state = torch.load(path, map_location='cpu')
                for key, value in comp_state.items():
                    prefixed_key = f"{comp}.{key}"
                    state[prefixed_key] = value
        return state

    def _compute_update(self, prev_global, client_model):
        """Calculate parameter-wise update (client - global)"""
        return {k: client_model[k] - prev_global[k] for k in prev_global}

    def _average_updates(self, updates):
        """Average updates across clients"""
        avg = {}
        for key in updates[0]:
            tensors = [u[key].float() for u in updates]
            avg[key] = torch.stack(tensors).mean(dim=0)
        return avg

    def _apply_fedadam(self, prev_global, avg_update):
        """Apply FedAdam update rule"""
        # Initialize moments if first run
        if self.m is None:
            self.m = {k: torch.zeros_like(v) for k,v in avg_update.items()}
            self.v = {k: torch.zeros_like(v) for k,v in avg_update.items()}

        self.t += 1
        
        # Update moments
        self.m = {k: self.beta1*self.m[k] + (1-self.beta1)*avg_update[k] 
                 for k in self.m}
        self.v = {k: self.beta2*self.v[k] + (1-self.beta2)*(avg_update[k]**2) 
                 for k in self.v}

        # Bias correction
        m_hat = {k: m / (1 - self.beta1**self.t) for k,m in self.m.items()}
        v_hat = {k: v / (1 - self.beta2**self.t) for k,v in self.v.items()}

        # Apply update
        new_global = {}
        for k in prev_global:
            update = self.server_lr * m_hat[k] / (torch.sqrt(v_hat[k]) + self.epsilon)
            new_global[k] = prev_global[k] + update
            
        return new_global

    def _save_full_global_model(self, state):
        """Split and save components from prefixed state dict"""
        tmp_dir = os.path.join(SHARED_ROOT, "global", "tmp")
        os.makedirs(tmp_dir, exist_ok=True)
        
        components = {comp: {} for comp in COMPONENTS}
        for key in state:
            parts = key.split('.', 1)  # Split into [component, subkey]
            if len(parts) == 2:
                comp, subkey = parts
                if comp in COMPONENTS:
                    components[comp][subkey] = state[key]
        
        # Save components without prefixes
        for comp in COMPONENTS:
            comp_path = os.path.join(tmp_dir, f"{comp}.pth")
            torch.save(components[comp], comp_path)
        
        # Atomic replace
        for fname in os.listdir(tmp_dir):
            src = os.path.join(tmp_dir, fname)
            dst = os.path.join(SHARED_ROOT, "global", fname)
            os.replace(src, dst)
    # Add to ServerFedHelper class
    def _clean_previous_state(self):
        """Clear all client status files and lock file"""
        # Clear client status files
        for entry in os.listdir(SHARED_ROOT):
            if entry.startswith("client_"):
                client_dir = os.path.join(SHARED_ROOT, entry)
                status_file = os.path.join(client_dir, "status.txt")
                if os.path.exists(status_file):
                    os.remove(status_file)
                    
        # Clear any existing lock
        lock_file = os.path.join(SHARED_ROOT, "fed.lock")
        if os.path.exists(lock_file):
            os.remove(lock_file)
            
        print("Cleaned previous server state")

if __name__ == "__main__":
    server = ServerFedHelper()
    while True:
        if server.aggregate_models():
            print(f"Aggregated new global model v{server.global_version}")
        time.sleep(1)  # Check every minute