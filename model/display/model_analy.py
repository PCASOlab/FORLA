import torch
import torch.nn as nn
from thop import profile, clever_format
from ptflops import get_model_complexity_info
import numpy as np

def analyze_model(model, input_shape=(1, 3, 16, 224, 224), device='cuda'):
    """
    Analyze model size, parameters, and computational requirements
    
    Args:
        model: Your model instance
        input_shape: Input tensor shape (batch, channels, depth, height, width)
        device: Device to run analysis on
    """
    print("=" * 80)
    print("MODEL ANALYSIS")
    print("=" * 80)
    
    # Move model to device
    model = model.to(device)
    model.eval()
    
    # Create dummy input
    dummy_input = torch.randn(input_shape).to(device)
    dummy_flows = torch.randn(input_shape).to(device)  # For input_flows
    
    # 1. Total Parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    print(f"\n1. PARAMETER COUNT:")
    print(f"   Total parameters: {total_params:,}")
    print(f"   Trainable parameters: {trainable_params:,}")
    print(f"   Non-trainable parameters: {total_params - trainable_params:,}")
    print(f"   Percentage trainable: {100 * trainable_params / total_params:.2f}%")
    
    # 2. Model Size
    param_size = 0
    for param in model.parameters():
        param_size += param.nelement() * param.element_size()
    buffer_size = 0
    for buffer in model.buffers():
        buffer_size += buffer.nelement() * buffer.element_size()
    
    size_all_mb = (param_size + buffer_size) / 1024**2
    print(f"\n2. MODEL SIZE:")
    print(f"   Total size: {size_all_mb:.2f} MB")
    print(f"   Parameters: {param_size / 1024**2:.2f} MB")
    print(f"   Buffers: {buffer_size / 1024**2:.2f} MB")
    
    # 3. Forward/Backward Pass Memory (Approximate)
    print(f"\n3. MEMORY ESTIMATION:")
    try:
        # Forward pass memory
        with torch.no_grad():
            torch.cuda.empty_cache()
            initial_memory = torch.cuda.memory_allocated() if device == 'cuda' else 0
            
            # Run forward pass
            with torch.cuda.device(device):
                output = model(dummy_input, dummy_flows, None, Enable_student=False)
            
            final_memory = torch.cuda.memory_allocated() if device == 'cuda' else 0
            forward_memory = (final_memory - initial_memory) / 1024**2
            
            print(f"   Forward pass memory: {forward_memory:.2f} MB")
            
            # Estimate backward pass memory (roughly 2-3x forward)
            backward_memory_estimate = forward_memory * 2.5
            print(f"   Backward pass estimate: {backward_memory_estimate:.2f} MB")
            print(f"   Total training memory: {forward_memory + backward_memory_estimate:.2f} MB")
            
    except Exception as e:
        print(f"   Memory estimation failed: {e}")
    
    # 4. FLOPs and Computational Complexity
    print(f"\n4. COMPUTATIONAL COMPLEXITY:")
    try:
        # Use thop for FLOPs calculation
        macs, params = profile(model, inputs=(dummy_input, dummy_flows, None, False), verbose=False)
        gflops = macs / 1e9
        
        print(f"   FLOPs: {macs:,}")
        print(f"   GFLOPs: {gflops:.2f}")
        print(f"   Parameters: {params:,}")
        
        # Calculate FLOPs per parameter
        if params > 0:
            flops_per_param = macs / params
            print(f"   FLOPs/parameter: {flops_per_param:.2f}")
            
    except Exception as e:
        print(f"   FLOPs calculation failed: {e}")
        # Alternative method
        try:
            macs, params = get_model_complexity_info(
                model, 
                input_shape[1:],  # Remove batch dimension
                as_strings=False,
                print_per_layer_stat=False,
                verbose=False
            )
            print(f"   MACs: {macs:,}")
            print(f"   Parameters: {params:,}")
        except:
            print("   Both FLOPs calculation methods failed")
    
    # 5. Layer-wise Analysis
    print(f"\n5. LAYER-WISE ANALYSIS:")
    print("-" * 50)
    
    # Analyze backbone
    if hasattr(model, 'backbone'):
        backbone_params = sum(p.numel() for p in model.backbone.parameters())
        print(f"   Backbone: {backbone_params:,} params ({100 * backbone_params / total_params:.1f}%)")
    
    # Analyze VideoNets
    if hasattr(model, 'VideoNets'):
        videonets_params = sum(p.numel() for p in model.VideoNets.parameters())
        print(f"   VideoNets: {videonets_params:,} params ({100 * videonets_params / total_params:.1f}%)")
    
    # Analyze VideoNets_S
    if hasattr(model, 'VideoNets_S'):
        videonets_s_params = sum(p.numel() for p in model.VideoNets_S.parameters())
        print(f"   VideoNets_S: {videonets_s_params:,} params ({100 * videonets_s_params / total_params:.1f}%)")
    
    # Analyze other components
    if hasattr(model, 'resnet'):
        resnet_params = sum(p.numel() for p in model.resnet.parameters())
        print(f"   ResNet: {resnet_params:,} params ({100 * resnet_params / total_params:.1f}%)")
    
    # 6. Memory Requirements for Different Batch Sizes
    print(f"\n6. MEMORY FOR DIFFERENT BATCH SIZES:")
    batch_sizes = [1, 2, 4, 8, 16, 32]
    
    for bs in batch_sizes:
        try:
            # Estimate memory for different batch sizes
            single_batch_memory = forward_memory if 'forward_memory' in locals() else size_all_mb
            estimated_memory = single_batch_memory * bs * 3.5  # Factor for gradients and optimizer states
            
            if estimated_memory < 1024:
                print(f"   Batch size {bs:2d}: ~{estimated_memory:.0f} MB")
            else:
                print(f"   Batch size {bs:2d}: ~{estimated_memory/1024:.1f} GB")
                
        except:
            break
    
    # 7. Training/Inference Speed (if CUDA available)
    if device == 'cuda':
        print(f"\n7. PERFORMANCE BENCHMARK:")
        try:
            # Warm up
            for _ in range(10):
                _ = model(dummy_input, dummy_flows, None, Enable_student=False)
            
            # Time forward pass
            torch.cuda.synchronize()
            start_time = torch.cuda.Event(enable_timing=True)
            end_time = torch.cuda.Event(enable_timing=True)
            
            start_time.record()
            for _ in range(50):
                _ = model(dummy_input, dummy_flows, None, Enable_student=False)
            end_time.record()
            torch.cuda.synchronize()
            
            elapsed_time = start_time.elapsed_time(end_time) / 50  # ms per forward
            fps = 1000 / elapsed_time
            
            print(f"   Forward pass: {elapsed_time:.2f} ms")
            print(f"   Throughput: {fps:.1f} FPS")
            print(f"   Batch throughput: {fps * input_shape[0]:.1f} samples/sec")
            
        except Exception as e:
            print(f"   Performance benchmark failed: {e}")
    
    print("=" * 80)
    
    return {
        'total_params': total_params,
        'trainable_params': trainable_params,
        'model_size_mb': size_all_mb,
        'gflops': gflops if 'gflops' in locals() else 0
    }