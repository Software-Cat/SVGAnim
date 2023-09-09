import random
from svganim import *


class Mover(Behavior):
    def __init__(self, ownerActor: Actor, moveDir: Vector, speed: float) -> None:
        super().__init__(ownerActor)
        self.moveDir = moveDir.norm()
        self.speed = speed

    def update(self, deltaTime: float):
        self.owner.transform = self.owner.transform.translate(
            self.moveDir * self.speed * deltaTime
        )


ellipsePrefab = PrefabFactory(
    (EllipseMesh, {"centerOfMass": Vector(0, 0)}),
    (Mover, {"moveDir": Vector(1, 1), "speed": 100}),
)


class Spawner(Behavior):
    def __init__(self, ownerActor: Actor, spawnPeriod: float) -> None:
        super().__init__(ownerActor)
        self.spawnPeriod = spawnPeriod

    def start(self):
        def spawn(dt: float) -> float:
            placed = self.owner.world.placeActorFromPrefab(
                ellipsePrefab, Transform(Vector(0, 0), scale=Vector(25, 25))
            )
            mover = placed.getComponentOfType(Mover)
            mover.moveDir = Vector(random.uniform(-1, 1), random.uniform(-1, 1)).norm()
            return self.spawnPeriod

        spawnCoroutine = Coroutine(spawn)
        self.owner.startCoroutine(spawnCoroutine)


spawnerPrefab = PrefabFactory((Spawner, {"spawnPeriod": 1}))
world = World(deltaTime=0.05)
world.placeActorFromPrefab(spawnerPrefab, Transform(Vector(0, 0)))


world.simulateTo(10)
world.render("test.svg")
