from svganim import *


class MyCircle(Circle):
    def __init__(self, transform: Transform, world: World, **kwargs) -> None:
        super().__init__(transform, world, r=10, **kwargs)

    def update(self, deltaTime: float):
        self.transform = self.transform.translate(delta=Vector(100, 100) * deltaTime)


class MyManager(Manager):
    def update(self, deltaTime: float):
        pass


world = World()
world.spawnActor(MyCircle, Transform(Vector(0, 0)))
world.spawnActor(MyManager, Transform(Vector(0, 0)))

world.simulateTo(10)
world.render("test.svg")
