from bem import BEMModel, AerofoilDatabase, Blade

def get_test_model(radii=None):
    blade = Blade.from_yaml('tests/data/Bladed_demo_a_modified/blade.yaml')
    db = AerofoilDatabase('tests/data/aerofoils.npz')
    root_length = 1.25
    model = BEMModel(blade, root_length=root_length,
                     num_blades=3, aerofoil_database=db, radii=radii)
    return model
