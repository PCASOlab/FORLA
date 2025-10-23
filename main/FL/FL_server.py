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
FED_MIN_CLIENTS = 4
FED_CLIENT_TIMEOUT = 300  # 5 minutes
COMPONENTS = ["initializer", "adapter", "processor", "decoder"]
Average_student = False
class ServerFedHelper:
    def __init__(self):
        self._clean_previous_state()  # Add this line
        # self.global_version = self._read_global_version()
        self.global_version = 0
        global_dir = os.path.join(SHARED_ROOT, "global")
        os.makedirs(global_dir, exist_ok=True)

        with open(os.path.join(global_dir, "version.txt"), "w") as f:
                f.write(str(self.global_version))
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

            # Load all client models
            client_weights = []
            for client in clients:
                client_dir = os.path.join(SHARED_ROOT, client)
                if Average_student:
                    client_dir = os.path.join(SHARED_ROOT, client,"student")

                try:
                    weights = {}
                    for comp in COMPONENTS:
                        path = os.path.join(client_dir, f"{comp}.pth")
                        weights[comp] = torch.load(path, map_location='cpu')
                    client_weights.append(weights)
                except Exception as e:
                    print(f"Error loading {client}: {e}")
                    continue

            # Average weights
            avg_weights = {}
            for comp in COMPONENTS:
                avg_weights[comp] = {}
                for key in client_weights[0][comp]:
                    tensors = [cw[comp][key].float() for cw in client_weights]
                    avg = torch.stack(tensors).mean(dim=0)
                    avg_weights[comp][key] = avg

            # Save new global model
            global_dir = os.path.join(SHARED_ROOT, "global")
            tmp_dir = os.path.join(global_dir, "tmp")
            
            os.makedirs(tmp_dir, exist_ok=True)
            for comp in COMPONENTS:
                torch.save(avg_weights[comp], os.path.join(tmp_dir, f"{comp}.pth"))
            
            # Atomic replace
            for fname in os.listdir(tmp_dir):
                src = os.path.join(tmp_dir, fname)
                dst = os.path.join(global_dir, fname)
                os.replace(src, dst)
            
            # Update version
            self.global_version += 1
            with open(os.path.join(global_dir, "version.txt"), "w") as f:
                f.write(str(self.global_version))
            
            return True
            
        finally:
            self._release_lock()
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