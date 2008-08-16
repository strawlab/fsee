def get_module_for_optics(optics=None):
    if optics == 'synthetic':
        import fsee.eye_geometry.precomputed_synthetic as precomputed
    elif optics == 'buchner71':
        # fused namespace for backwards compatibility
        import fsee.eye_geometry.precomputed_buchner71_fused as precomputed
    else:
        raise ValueError("unknown optics")
    return precomputed
