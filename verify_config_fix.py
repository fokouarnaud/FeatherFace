#!/usr/bin/env python3
"""
Verify that cfg_mnet_v2 configuration is properly centralized
"""

def check_config_centralization():
    """Check that cfg_mnet_v2 is only defined in one place"""
    print("=== Configuration Centralization Check ===")
    
    try:
        # This should work - centralized config
        import sys
        sys.path.append('.')
        from data.config import cfg_mnet_v2
        
        print("✅ Successfully imported cfg_mnet_v2 from data.config")
        print(f"   name: '{cfg_mnet_v2['name']}'")
        print(f"   out_channel_v2: {cfg_mnet_v2['out_channel_v2']}")
        print(f"   epochs: {cfg_mnet_v2['epoch']}")
        
        return True
        
    except ImportError as e:
        print(f"❌ Failed to import cfg_mnet_v2: {e}")
        return False

def check_no_duplicate_definition():
    """Check that retinaface_v2.py no longer defines cfg_mnet_v2"""
    print("\n=== Duplicate Definition Check ===")
    
    try:
        with open('models/retinaface_v2.py', 'r') as f:
            content = f.read()
        
        # Check for cfg_mnet_v2 definition
        if 'cfg_mnet_v2 = {' in content:
            print("❌ Found cfg_mnet_v2 definition in retinaface_v2.py")
            return False
        else:
            print("✅ No duplicate cfg_mnet_v2 definition in retinaface_v2.py")
            return True
            
    except FileNotFoundError:
        print("❌ Could not find models/retinaface_v2.py")
        return False

def check_consistent_usage():
    """Check that all files use the correct import"""
    print("\n=== Import Consistency Check ===")
    
    # Files that should import from data.config
    files_to_check = [
        'train_v2.py',
        'test_v1_v2_comparison.py'
    ]
    
    consistent = True
    
    for filename in files_to_check:
        try:
            with open(filename, 'r') as f:
                content = f.read()
            
            if 'from data import' in content and 'cfg_mnet_v2' in content:
                print(f"✅ {filename}: Uses correct import from data")
            elif 'from data.config import' in content and 'cfg_mnet_v2' in content:
                print(f"✅ {filename}: Uses correct import from data.config")
            elif 'cfg_mnet_v2' in content:
                print(f"⚠️  {filename}: Contains cfg_mnet_v2 but import unclear")
                consistent = False
            else:
                print(f"ℹ️  {filename}: Does not use cfg_mnet_v2")
                
        except FileNotFoundError:
            print(f"⚠️  {filename}: File not found")
    
    return consistent

def main():
    """Run all checks"""
    print("FeatherFace V2 Configuration Consistency Verification")
    print("=" * 60)
    
    checks = [
        check_config_centralization(),
        check_no_duplicate_definition(),
        check_consistent_usage()
    ]
    
    passed = sum(checks)
    total = len(checks)
    
    print("\n" + "=" * 60)
    print("VERIFICATION SUMMARY")
    print("=" * 60)
    print(f"Checks passed: {passed}/{total}")
    
    if passed == total:
        print("\n🎉 Configuration is properly centralized!")
        print("✅ No more inconsistency issues with cfg_mnet_v2")
        return 0
    else:
        print("\n❌ Some issues remain. Check the details above.")
        return 1

if __name__ == "__main__":
    import sys
    sys.exit(main())