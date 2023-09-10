import random
from svganim import *


class Mover(Behavior):
    def __init__(self, ownerActor: Actor, moveDir: Vector, speed: float) -> None:
        super().__init__(ownerActor)
        self.moveDir = moveDir.normalize()
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


class Destroyer(Behavior):
    def __init__(self, ownerActor: Actor, delay: float) -> None:
        super().__init__(ownerActor)
        self.delay = delay

    def start(self):
        def destroySelf(dt) -> float:
            self.destroyActor(self.owner)
            return 0

        destroyCoroutine = Coroutine(destroySelf)
        self.owner.startCoroutine(destroyCoroutine, self.delay)


rectPrefab = PrefabFactory(
    (RectMesh, {"width": 25, "height": 25}),
    (Mover, {"moveDir": Vector(1, 1), "speed": 200}),
    (Destroyer, {"delay": 2}),
)


class Spawner(Behavior):
    def __init__(self, ownerActor: Actor, spawnPeriod: float) -> None:
        super().__init__(ownerActor)
        self.spawnPeriod = spawnPeriod

    def start(self):
        def spawn(dt: float) -> float:
            placed = self.owner.world.placeActorFromPrefab(
                rectPrefab, parent=self.owner
            )
            mover = placed.getComponentOfType(Mover)
            mover.moveDir = Vector(
                random.uniform(-1, 1), random.uniform(-1, 1)
            ).normalize()
            return self.spawnPeriod

        spawnCoroutine = Coroutine(spawn)
        self.owner.startCoroutine(spawnCoroutine)


spawnerPrefab = PrefabFactory(
    (Spawner, {"spawnPeriod": 0.1}),
    (Mover, {"moveDir": Vector(1, 1), "speed": 0}),
    (Rotor, {"speed": 360}),
)
world = World(deltaTime=0.05)
world.placeActorFromPrefab(spawnerPrefab)


world.simulateTo(10)
world.render("test.svg")
