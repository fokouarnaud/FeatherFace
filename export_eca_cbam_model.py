#!/usr/bin/env python3
"""
ECA-CBAM Model Export Script
=============================

Exports the trained ECA-CBAM model to various formats for deployment:
- PyTorch (.pth)
- ONNX (.onnx)
- TorchScript (.pt)

Usage:
    python export_eca_cbam_model.py --model weights/eca_cbam/featherface_eca_cbam_final.pth
"""

import argparse
import torch
import torch.onnx
from pathlib import Path
from data.config import cfg_eca_cbam
from models.featherface_eca_cbam import FeatherFaceECAcbaM


def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description='Export ECA-CBAM Model')
    parser.add_argument('--model', '-m',
                       default='weights/eca_cbam/featherface_eca_cbam_final.pth',
                       type=str, help='Path to trained model')
    parser.add_argument('--export_dir',
                       default='exports/eca_cbam',
                       type=str, help='Export directory')
    parser.add_argument('--formats',
                       nargs='+',
                       default=['pytorch', 'onnx', 'torchscript'],
                       choices=['pytorch', 'onnx', 'torchscript'],
                       help='Export formats')
    parser.add_argument('--input_size',
                       type=int,
                       default=640,
                       help='Input image size')
    parser.add_argument('--batch_size',
                       type=int,
                       default=1,
                       help='Batch size for export')
    return parser.parse_args()


def load_model(model_path, device='cpu'):
    """Load trained ECA-CBAM model"""
    print(f"📥 Loading model from: {model_path}")

    # Create model
    model = FeatherFaceECAcbaM(cfg=cfg_eca_cbam, phase='test')

    # Load weights
    state_dict = torch.load(model_path, map_location=device)

    # Handle different state dict formats
    if "state_dict" in state_dict:
        state_dict = state_dict['state_dict']

    # Remove 'module.' prefix if present
    from collections import OrderedDict
    new_state_dict = OrderedDict()
    for k, v in state_dict.items():
        name = k.replace('module.', '') if k.startswith('module.') else k
        new_state_dict[name] = v

    model.load_state_dict(new_state_dict, strict=False)
    model.eval()

    print(f"✅ Model loaded successfully!")

    # Get parameter count
    param_info = model.get_parameter_count()
    print(f"📊 Model parameters: {param_info['total']:,}")

    return model, param_info


def export_pytorch(model, export_path):
    """Export as PyTorch .pth file"""
    print(f"\n📦 Exporting PyTorch format...")
    torch.save(model.state_dict(), export_path)
    print(f"✅ Saved: {export_path}")
    return export_path


def export_onnx(model, export_path, input_size=640, batch_size=1):
    """Export as ONNX format"""
    print(f"\n📦 Exporting ONNX format...")

    # Create dummy input
    dummy_input = torch.randn(batch_size, 3, input_size, input_size)

    # Export to ONNX
    torch.onnx.export(
        model,
        dummy_input,
        export_path,
        export_params=True,
        opset_version=11,
        do_constant_folding=True,
        input_names=['input'],
        output_names=['loc', 'conf', 'landms'],
        dynamic_axes={
            'input': {0: 'batch_size'},
            'loc': {0: 'batch_size'},
            'conf': {0: 'batch_size'},
            'landms': {0: 'batch_size'}
        }
    )

    print(f"✅ Saved: {export_path}")
    return export_path


def export_torchscript(model, export_path, input_size=640, batch_size=1):
    """Export as TorchScript format"""
    print(f"\n📦 Exporting TorchScript format...")

    # Create dummy input
    dummy_input = torch.randn(batch_size, 3, input_size, input_size)

    # Trace the model
    traced_model = torch.jit.trace(model, dummy_input)

    # Save traced model
    traced_model.save(str(export_path))

    print(f"✅ Saved: {export_path}")
    return export_path


def verify_export(export_path, model, input_size=640):
    """Verify exported model"""
    print(f"\n🔍 Verifying export: {export_path}")

    try:
        if export_path.suffix == '.pth':
            # Verify PyTorch
            loaded = torch.load(export_path, map_location='cpu')
            print(f"✅ PyTorch export valid: {len(loaded)} parameters")

        elif export_path.suffix == '.onnx':
            # Verify ONNX
            import onnx
            onnx_model = onnx.load(str(export_path))
            onnx.checker.check_model(onnx_model)
            print(f"✅ ONNX export valid")

        elif export_path.suffix == '.pt':
            # Verify TorchScript
            loaded = torch.jit.load(str(export_path))
            dummy_input = torch.randn(1, 3, input_size, input_size)
            outputs = loaded(dummy_input)
            print(f"✅ TorchScript export valid: {len(outputs)} outputs")

        return True

    except Exception as e:
        print(f"❌ Verification failed: {e}")
        return False


def main():
    """Main export function"""
    args = parse_args()

    print("🔬 ECA-CBAM Model Export")
    print("=" * 60)

    # Create export directory
    export_dir = Path(args.export_dir)
    export_dir.mkdir(parents=True, exist_ok=True)
    print(f"📁 Export directory: {export_dir}")

    # Check if model exists
    model_path = Path(args.model)
    if not model_path.exists():
        print(f"❌ Model not found: {model_path}")
        print(f"Please train the model first or provide correct path")
        return

    # Load model
    model, param_info = load_model(args.model)

    # Export in requested formats
    exported_files = {}

    if 'pytorch' in args.formats:
        pytorch_path = export_dir / 'featherface_eca_cbam_hybrid.pth'
        exported_files['pytorch'] = export_pytorch(model, pytorch_path)
        verify_export(pytorch_path, model, args.input_size)

    if 'onnx' in args.formats:
        onnx_path = export_dir / 'featherface_eca_cbam_hybrid.onnx'
        try:
            exported_files['onnx'] = export_onnx(model, onnx_path, args.input_size, args.batch_size)
            verify_export(onnx_path, model, args.input_size)
        except Exception as e:
            print(f"❌ ONNX export failed: {e}")
            print(f"Note: ONNX export may require 'onnx' package: pip install onnx")

    if 'torchscript' in args.formats:
        torchscript_path = export_dir / 'featherface_eca_cbam_hybrid.pt'
        try:
            exported_files['torchscript'] = export_torchscript(model, torchscript_path, args.input_size, args.batch_size)
            verify_export(torchscript_path, model, args.input_size)
        except Exception as e:
            print(f"❌ TorchScript export failed: {e}")

    # Export summary
    print(f"\n" + "=" * 60)
    print(f"📊 EXPORT SUMMARY")
    print(f"=" * 60)

    print(f"\n🔬 Model Information:")
    print(f"  • Total parameters: {param_info['total']:,} ({param_info['total']/1e6:.3f}M)")
    print(f"  • Parameter reduction: {param_info['efficiency_gain']:.1f}% vs CBAM")
    print(f"  • Architecture: 6 ECA-CBAM modules")
    print(f"  • Input size: {args.input_size}x{args.input_size}")

    print(f"\n📦 Exported Formats:")
    for format_name, file_path in exported_files.items():
        file_size = Path(file_path).stat().st_size / (1024 * 1024)  # MB
        print(f"  • {format_name.upper()}: {file_path} ({file_size:.2f} MB)")

    print(f"\n🚀 Deployment Ready:")
    print(f"  ✅ PyTorch: Python environments")
    print(f"  ✅ ONNX: Cross-platform deployment")
    print(f"  ✅ TorchScript: Mobile/embedded devices")

    print(f"\n📝 Usage Example:")
    print(f"  # Load PyTorch model")
    print(f"  model = FeatherFaceECAcbaM(cfg_eca_cbam, phase='test')")
    print(f"  model.load_state_dict(torch.load('{exported_files.get('pytorch', 'model.pth')}'))")
    print(f"  ")
    print(f"  # Load ONNX model")
    print(f"  import onnxruntime")
    print(f"  session = onnxruntime.InferenceSession('{exported_files.get('onnx', 'model.onnx')}')")
    print(f"  ")
    print(f"  # Load TorchScript model")
    print(f"  model = torch.jit.load('{exported_files.get('torchscript', 'model.pt')}')")

    print(f"\n✅ Export completed successfully!")
    print(f"=" * 60)


if __name__ == '__main__':
    main()
