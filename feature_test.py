from svganim import *


class Mover(Behavior):
    def update(self, deltaTime: float):
        self.owner.transform = self.owner.transform.translate(
            Vector(1, 1) * 0 * deltaTime
        ).rotate(100 * deltaTime)


rectPrefab = PrefabFactory(
    (RectangleMesh, {"centerOfMass": Vector(25, 12.5)}),
    (Mover, {}),
    defaultTransform=Transform(Vector(-100, -100), scale=Vector(50, 25)),
)


world = World(deltaTime=0.05)
world.placeActorFromPrefab(rectPrefab, None)


world.simulateTo(10)
world.render("test.svg")
