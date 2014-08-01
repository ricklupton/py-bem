from bem import BEMModel, AerofoilDatabase, Blade

def get_test_model(radii=None):
    root_length = 1.25
    blade = Blade.from_yaml('tests/data/Bladed_demo_a_modified/blade.yaml')
    if radii is not None:
        x = radii - root_length
        blade = blade.resample(x)
    db = AerofoilDatabase('tests/data/aerofoils.npz')
    model = BEMModel(blade, root_length, num_blades=3, aerofoil_database=db)
    return model
