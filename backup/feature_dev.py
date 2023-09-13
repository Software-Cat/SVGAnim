from svganim import *


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


circlePrefab = PrefabFactory((EllipseMesh, {"rx": 10, "ry": 10, "fill": "red"}))


class Tracker(Behavior):
    def start(self):
        col = self.owner.getComponentOfType(Collider)
        col.callbacks.append(self.placeTracker)

    def placeTracker(self, oth, res):
        print(res.overlap_v)

    # def update(self, deltaTime: float):
    #     points = self.mesh.points
    #     points = [self.owner.getAbsoluteTransform().applyToVec(p) for p in points]
    #     for p in points:
    #         self.owner.world.placeActorFromPrefab(circlePrefab, p)


rectPrefab = PrefabFactory(
    (
        RectMesh,
        {"width": 100, "height": 200},
    ),
    (ConvexPolyCollider, {}),
    defaultTransform=Transform(Vector(0, 0)),
)

trianglePrefab = PrefabFactory(
    (
        PolyMesh,
        {
            "centerOfMass": Vector(100, 100),
            "points": [Vector(0, 0), Vector(200, 0), Vector(200, 200)],
        },
    ),
    (ConvexPolyCollider, {}),
    (Rotor, {"speed": 180}),
    (Tracker, {}),
    defaultTransform=Transform(Vector(150, 150)),
)

world = World(deltaTime=1 / 24)
world.placeActorFromPrefab(trianglePrefab)
world.placeActorFromPrefab(rectPrefab)

world.simulateTo(10)
world.render("test.svg")
