"""
Defines contact materials and their corresponding physics behavior groupings.
"""

from enum import IntEnum


class ContactMaterial(IntEnum):
    Concrete = 0
    Pavement = 1
    Grass = 2
    Ice = 3
    Metal = 4
    Sand = 5
    Dirt = 6
    Turbo = 7
    Asphalt = 16
    WetDirtRoad = 17
    WetAsphalt = 18
    WetPavement = 19
    WetGrass = 20
    Turbo2 = 26


# Maps material to a broad physics category
physics_group_map = {
    ContactMaterial.Concrete: 0,
    ContactMaterial.Pavement: 0,
    ContactMaterial.Asphalt: 0,
    ContactMaterial.WetAsphalt: 0,
    ContactMaterial.WetPavement: 0,

    ContactMaterial.Grass: 1,
    ContactMaterial.WetGrass: 1,

    ContactMaterial.Sand: 2,
    ContactMaterial.Dirt: 2,
    ContactMaterial.WetDirtRoad: 2,

    ContactMaterial.Turbo: 3,
    ContactMaterial.Turbo2: 3,

    ContactMaterial.Ice: 4,
    ContactMaterial.Metal: 4,
}
