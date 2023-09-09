import random
from svganim import *


class Mover(Behavior):
    def __init__(self, ownerActor: Actor, moveDir: Vector, speed: float) -> None:
        super().__init__(ownerActor)
        self.moveDir = moveDir.normalize()
        self.speed = speed

    def start(self):
        def destroySelf(dt) -> float:
            self.destroyActor(self.owner)
            return 0

        destroyCoroutine = Coroutine(destroySelf)
        self.owner.startCoroutine(destroyCoroutine, 2)

    def update(self, deltaTime: float):
        self.owner.relativeTransform = self.owner.relativeTransform.translate(
            self.moveDir * self.speed * deltaTime
        )


rectPrefab = PrefabFactory(
    (RectMesh, {"width": 25, "height": 25}),
    (Mover, {"moveDir": Vector(1, 1), "speed": 100}),
)


class Spawner(Behavior):
    def __init__(self, ownerActor: Actor, spawnPeriod: float) -> None:
        super().__init__(ownerActor)
        self.spawnPeriod = spawnPeriod

    def start(self):
        def spawn(dt: float) -> float:
            placed = self.owner.world.placeActorFromPrefab(rectPrefab)
            mover = placed.getComponentOfType(Mover)
            mover.moveDir = Vector(
                random.uniform(-1, 1), random.uniform(-1, 1)
            ).normalize()
            return self.spawnPeriod

        spawnCoroutine = Coroutine(spawn)
        self.owner.startCoroutine(spawnCoroutine)


spawnerPrefab = PrefabFactory((Spawner, {"spawnPeriod": 0.1}))
world = World(deltaTime=0.05)
world.placeActorFromPrefab(spawnerPrefab)


world.simulateTo(10)
world.render("test.svg")
