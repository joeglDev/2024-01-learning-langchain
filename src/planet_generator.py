"""Module providing a class which models a planet"""

from math import pi
from random import choice
from basic_llm import BasicLlm


llm = BasicLlm()
NAME_PROMPT = "Give me a short name for a planet. The planet may be real or fictional. Please only give a brief one to two word answer."  # pylint: disable=C0301
CLIMATE_TYPE = [
    "Arid",
    "Desert",
    "Savanna",
    "Alpine",
    "Arctic",
    "Tundra",
    "Tropical",
    "Continental",
    "Ocean",
]
GAS_GIANT_CLIMATE = [
    "Ammonia clouds",
    "Water clouds",
    "Cloudless",
    "Alkali-metal clouds",
    "Silicate clouds",
]


class Planet:
    """
    Class which represents an abstract planet.
    - radius: int Kilometers/ Km
    - distance_from_star: int Astronomical Units/ AU
    - density: int Kg/m^3
    """

    def __create_planet_name():  # pylint: disable=E0211
        name = llm.get_completion(NAME_PROMPT).replace('"', "")
        if len(name.split()) > 1:
            truncated_name = "".join(name.split()[:2])
            return truncated_name
        return name

    def __init__(
        self,
        radius=6371,
        distance_from_star=1,
        density=5515,
        name=__create_planet_name(),
    ):
        self.radius: int = radius
        self.distance_from_star: int = distance_from_star
        self.density: int = density
        self.name: str = name
        # mass, type, climate,

    def get_circumference(self) -> int:
        """Calculates circumference of planet"""

        circumference: int = round(2 * pi * self.radius, 3)
        return circumference

    def get_mass(self) -> float:
        """Calculates mass of planet."""

        volume = (4.0 / 3.0) * pi * self.radius**3
        mass = round(self.density * volume, 3)
        return mass

    def print_planet_properties(self):
        """Prints own properties."""

        own_properties = [
            attr
            for attr in dir(self)
            if not callable(getattr(self, attr)) and not attr.startswith("__")
        ]
        for own_property in own_properties:
            print(f"{own_property.capitalize()}: {getattr(self, own_property)}")


class RockyPlanet(Planet):
    """
    Class which represents a rocky planet.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.type = "Rocky"
        self.climate_type = choice(CLIMATE_TYPE)


class GasGiantPlanet(Planet):
    """
    Class which represents a gas giant planet.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.type = "Gas Giant"
        self.climate_type = choice(GAS_GIANT_CLIMATE)

    def scream(self):
        """Gives a electromagnetic scream noise simulating a planet's EM spectrum."""
        return "AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA"


""" 
generic_planet = RockyPlanet(name="Holy Terra")
generic_planet.print_planet_properties()
print(f"The circumference is: {generic_planet.get_circumference()} Km.")
print(f"The mass is: {generic_planet.get_mass() / (10**12)} Tg.")

jupiter = GasGiantPlanet(
    density=1326, radius=69911, distance_from_star=5.2, name="Jupiter"
)
jupiter.print_planet_properties()
print(f"{jupiter.name} says {jupiter.scream()}")
"""
