def get_module_for_optics(optics=None):
    if optics == 'synthetic':
        import fsee.eye_geometry.precomputed_synthetic as precomputed
    elif optics == 'buchner71':
        # fused namespace for backwards compatibility
        import fsee.eye_geometry.precomputed_buchner71_fused as precomputed
    else:
        module_name = "fsee.eye_geometry.precomputed_%s" % optics 
        try:
            # if we give a non-empty fromlist to __import__,
            # it returns the module fsee.eye_geometry.precomputed_XYZ,
            # otherwise it returns fsee. Strange, but that's how
            # the documentation says.
            # See: http://docs.python.org/library/functions.html#__import__ 
            fromlist = ['edge_slicer']
            precomputed = __import__(module_name, globals(), locals(),fromlist)
        except ImportError:
            raise ValueError("unknown optics %s (looked for module)" % 
                (optics, module_name) )
    return precomputed
