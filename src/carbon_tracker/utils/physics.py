def biomass_to_carbon(biomass_value):
    """
    Convert Above-Ground Biomass (AGB) to Carbon Stock.
    Standard IPCC factor is 0.47.
    """
    return biomass_value * 0.47

def carbon_to_co2_equivalent(carbon_value):
    """
    Convert Carbon Stock to CO2 equivalent.
    Ratio of molecular weights (44/12) is approx 3.67.
    """
    return carbon_value * 3.67