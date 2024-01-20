"""Module providing a class which models a planet"""

from math import pi
from basic_llm import BasicLlm


llm = BasicLlm()
NAME_PROMPT = "Give me a short name for a planet. The planet may be real or fictional. Please only give a brief one to two word answer."  # pylint: disable=C0301


class Planet:
    """
    Class which represents an abstract planet.
    - radius: int Kilometers/ Km
    - distance_from_star: int Astronomical Units/ AU
    - density: int Kg/m^3
    """

    def __init__(
        self,
        radius=6371,
        distance_from_star=1,
        density=5515,
        name=llm.get_completion(NAME_PROMPT).replace('"', ''),
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


generic_planet = Planet()
generic_planet.print_planet_properties()
print(f"The circumference is: {generic_planet.get_circumference()} Km.")
print(f"The mass is: {generic_planet.get_mass() / (10**12)} Tg.")
