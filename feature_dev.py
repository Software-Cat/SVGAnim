from svganim import *
from svganim import Actor


class Mover(Behavior):
    def __init__(self, ownerActor: Actor, moveDir: Vector, speed: float = 100) -> None:
        super().__init__(ownerActor)
        self.moveDir = moveDir
        self.speed = speed

    def update(self, deltaTime: float):
        self.owner.localTransform = self.owner.localTransform.translate(
            self.moveDir * self.speed * deltaTime
        )


class Rotor(Behavior):
    def __init__(self, ownerActor: Actor, speed: float = 100) -> None:
        super().__init__(ownerActor)
        self.speed = speed

    def update(self, deltaTime: float):
        self.owner.localTransform = self.owner.localTransform.rotate(
            self.speed * deltaTime
        )


rectPrefab = PrefabFactory(
    (
        RectMesh,
        {"width": 100, "height": 200},
    ),
    (Mover, {"moveDir": Vector(0, 100)}),
    defaultTransform=Transform(Vector(-100, -100)),
)

trianglePrefab = PrefabFactory(
    (
        PolyMesh,
        {
            "centerOfMass": Vector(100, 100),
            "points": [Vector(0, 0), Vector(200, 0), Vector(200, 200)],
        },
    ),
    # (Mover, {"moveDir": Vector(1, 1)}),
    (Rotor, {"speed": 180}),
    defaultTransform=Transform(Vector(100, 100)),
    children=[rectPrefab],
)


world = World(deltaTime=1 / 24)
world.placeActorFromPrefab(trianglePrefab)

world.simulateTo(10)
world.render("test.svg")
