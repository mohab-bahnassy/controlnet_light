"""
Test script to verify that all lightweight ControlNet variants can be instantiated properly.
"""

import torch
from cldm.model import create_model

def count_parameters(model):
    """Count the number of trainable parameters in a model."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def test_model_variant(config_path, variant_name):
    """Test instantiation and basic functionality of a model variant."""
    print(f"\n{'='*60}")
    print(f"Testing {variant_name}")
    print(f"{'='*60}")
    
    try:
        # Create model
        print(f"Loading config: {config_path}")
        model = create_model(config_path).cpu()
        
        # Count parameters in control model
        control_params = count_parameters(model.control_model)
        total_params = count_parameters(model)
        
        print(f"✓ Model instantiated successfully")
        print(f"  Control model parameters: {control_params:,}")
        print(f"  Total model parameters: {total_params:,}")
        
        # Test forward pass with dummy data
        batch_size = 1
        h, w = 64, 64  # Latent space size
        
        x_noisy = torch.randn(batch_size, 4, h, w)
        t = torch.randint(0, 1000, (batch_size,))
        hint = torch.randn(batch_size, 3, h*8, w*8)  # Full resolution hint
        context = torch.randn(batch_size, 77, 768)  # Text conditioning
        
        # Test control model forward pass
        print("Testing control model forward pass...")
        with torch.no_grad():
            control_output = model.control_model(x_noisy, hint, t, context)
        
        print(f"✓ Forward pass successful")
        print(f"  Number of control outputs: {len(control_output)}")
        print(f"  Control output shapes: {[tuple(o.shape) for o in control_output[:3]]}...")
        
        return {
            'name': variant_name,
            'config': config_path,
            'control_params': control_params,
            'total_params': total_params,
            'success': True
        }
        
    except Exception as e:
        print(f"✗ Error testing {variant_name}: {str(e)}")
        import traceback
        traceback.print_exc()
        return {
            'name': variant_name,
            'config': config_path,
            'success': False,
            'error': str(e)
        }

def main():
    """Test all model variants."""
    print("="*60)
    print("Lightweight ControlNet Variants Test Suite")
    print("="*60)
    
    # Test configurations
    variants = [
        ('Original ControlNet', './models/cldm_v15.yaml'),
        ('LightControlNet (50% channels)', './models/cldm_v15_light.yaml'),
        ('TinyControlNet (25% channels)', './models/cldm_v15_tiny.yaml'),
        ('EfficientControlNet (Depthwise Sep.)', './models/cldm_v15_efficient.yaml'),
        ('SimpleCNNControlNet (Simple CNN)', './models/cldm_v15_simple_cnn.yaml'),
    ]
    
    results = []
    for variant_name, config_path in variants:
        result = test_model_variant(config_path, variant_name)
        results.append(result)
    
    # Summary
    print(f"\n{'='*60}")
    print("SUMMARY")
    print(f"{'='*60}\n")
    
    print(f"{'Variant':<40} {'Status':<10} {'Control Params':<20}")
    print("-" * 70)
    
    baseline_params = None
    for result in results:
        if result['success']:
            status = "✓ PASS"
            params_str = f"{result['control_params']:,}"
            
            # Calculate reduction percentage
            if baseline_params is None:
                baseline_params = result['control_params']
                reduction = "-"
            else:
                reduction_pct = (1 - result['control_params'] / baseline_params) * 100
                reduction = f"({reduction_pct:.1f}% smaller)"
            
            print(f"{result['name']:<40} {status:<10} {params_str:<15} {reduction}")
        else:
            status = "✗ FAIL"
            print(f"{result['name']:<40} {status:<10} {'N/A':<15}")
    
    print("\n" + "="*60)
    successful = sum(1 for r in results if r['success'])
    print(f"Results: {successful}/{len(results)} variants passed")
    print("="*60)
    
    return all(r['success'] for r in results)

if __name__ == "__main__":
    import sys
    success = main()
    sys.exit(0 if success else 1)

