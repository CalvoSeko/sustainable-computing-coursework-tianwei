def calculate_co2_emission(energy_J, intensity_gco2_per_kwh):
    """
    Calculate CO2 emissions based on energy consumption and carbon intensity.

    Parameters:
    energy_kwh (float): Energy consumption in kilowatt-hours (kWh).
    intensity_gco2_per_kwh (float): Carbon intensity in grams of CO2 per kWh.

    Returns:
    float: CO2 emissions in grams.
    """
    energy_kwh = energy_J / 3.6e6
    return energy_kwh * intensity_gco2_per_kwh
for i in range(12):
    J = input("input energy in J:")
    print(calculate_co2_emission(float(J), 274))