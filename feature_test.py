from svganim import *


class Mover(Behavior):
    def update(self, deltaTime: float):
        self.owner.relativeTransform = (
            self.owner.relativeTransform.translate(Vector(1, 1) * 0 * deltaTime)
            .rotate(100 * deltaTime * 1)
            .scaleAdd(Vector(1, 1) * 1 * deltaTime)
        )


rectPrefab = PrefabFactory(
    (
        RectMesh,
        {"width": 100, "height": 200},
    ),
)

trianglePrefab = PrefabFactory(
    (
        PolyMesh,
        {
            "centerOfMass": Vector(100, 100),
            "points": [Vector(0, 0), Vector(200, 0), Vector(200, 200)],
        },
    ),
    (Mover, {}),
    children=[rectPrefab],
)


world = World(deltaTime=1 / 24)
world.placeActorFromPrefab(trianglePrefab)

world.simulateTo(10)
world.render("test.svg")
