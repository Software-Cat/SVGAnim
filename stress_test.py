import random
from svganim import *


class Mover(Behavior):
    def __init__(self, ownerActor: Actor, moveDir: Vector, speed: float) -> None:
        super().__init__(ownerActor)
        self.moveDir = moveDir.norm()
        self.speed = speed

    def start(self):
        def destroySelf(dt) -> float:
            self.destroyActor(self.owner)
            return 0

        destroyCoroutine = Coroutine(destroySelf)
        self.owner.startCoroutine(destroyCoroutine, 2)

    def update(self, deltaTime: float):
        self.owner.transform = self.owner.transform.translate(
            self.moveDir * self.speed * deltaTime
        )


rectPrefab = PrefabFactory(
    (RectangleMesh, {"centerOfMass": Vector(0, 0)}),
    (Mover, {"moveDir": Vector(1, 1), "speed": 100}),
    defaultTransform=Transform(scale=Vector(50, 25)),
)


class Spawner(Behavior):
    def __init__(self, ownerActor: Actor, spawnPeriod: float) -> None:
        super().__init__(ownerActor)
        self.spawnPeriod = spawnPeriod

    def start(self):
        def spawn(dt: float) -> float:
            placed = self.owner.world.placeActorFromPrefab(rectPrefab)
            mover = placed.getComponentOfType(Mover)
            mover.moveDir = Vector(random.uniform(-1, 1), random.uniform(-1, 1)).norm()
            return self.spawnPeriod

        spawnCoroutine = Coroutine(spawn)
        self.owner.startCoroutine(spawnCoroutine)


spawnerPrefab = PrefabFactory(
    (Spawner, {"spawnPeriod": 0.1}), defaultTransform=Transform()
)
world = World(deltaTime=0.05)
world.placeActorFromPrefab(spawnerPrefab)


world.simulateTo(10)
world.render("test.svg")
