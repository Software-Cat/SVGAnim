from abc import ABC, abstractmethod
from copy import deepcopy
import random
from typing import Any, Callable, Optional, Self, TypeVar
from typing import Type
import drawsvg as draw

from dataclasses import dataclass


@dataclass(frozen=True)
class Vector:
    x: float
    y: float

    def __add__(self, other: Self):
        return Vector(self.x + other.x, self.y + other.y)

    def __sub__(self, other: Self):
        return Vector(self.x - other.x, self.y - other.y)

    def __mul__(self, other: float):
        return Vector(self.x * other, self.y * other)

    def __truediv__(self, other: float):
        return Vector(self.x / other, self.y / other)

    def mag(self):
        """Magnitude"""
        return (self.x**2 + self.y**2) ** 0.5

    def norm(self):
        """Normalized"""
        return self / self.mag()


@dataclass(frozen=True)
class Transform:
    pos: Vector
    rot: float = 0
    scale: Vector = Vector(1, 1)

    def translate(self, delta: Vector):
        return Transform(self.pos + delta, self.rot, self.scale)

    def rotate(self, delta: float):
        return Transform(self.pos, self.rot + delta, self.scale)


class Behavior(ABC):
    def __init__(self, ownerActor: "Actor") -> None:
        self.owner = ownerActor

    def start(self):
        pass

    def update(self, deltaTime: float):
        pass


class Mesh(Behavior):
    def __init__(self, ownerActor: "Actor", centerOfMass: Vector) -> None:
        """Center of mass expressed in local coordinates"""
        super().__init__(ownerActor)
        self.centerOfMass = centerOfMass

    def start(self):
        return super().start()

    def update(self, deltaTime: float):
        return super().update(deltaTime)

    @abstractmethod
    def render(self) -> Optional[draw.DrawingElement]:
        pass


class EllipseMesh(Mesh):
    def render(self) -> Optional[draw.DrawingElement]:
        timeOfSpawn = min(self.owner.transformHistory.keys())

        circle = draw.Ellipse(
            self.owner.transformHistory[timeOfSpawn].pos.x,
            self.owner.transformHistory[timeOfSpawn].pos.y,
            self.owner.transformHistory[timeOfSpawn].scale.x,
            self.owner.transformHistory[timeOfSpawn].scale.y,
        )
        circle.add_key_frame(time=0, visibility="hidden")
        circle.add_key_frame(time=timeOfSpawn, visibility="visible")

        for time, trans in self.owner.transformHistory.items():
            circle.add_key_frame(  # Todo: Refactor to use draw.AnimateTransform
                time=time,
                cx=trans.pos.x,
                cy=trans.pos.y,
                rx=trans.scale.x,
                ry=trans.scale.y,
            )

        return circle


T = TypeVar("T")


class Actor:
    def __init__(self, transform: Transform, world: "World") -> None:
        self.transform: Transform = transform
        self.world: World = world
        self.transformHistory: dict[float, Transform] = {}
        self.coroutines: list[Coroutine] = []
        self.components: list[Behavior] = []

    def start(self):
        for behavior in self.components:
            behavior.start()

    def update(self, deltaTime: float):
        for behavior in self.components:
            behavior.update(deltaTime)
        for c in self.coroutines:
            c.systemUpdate(self.world.deltaTime)
        self.recordTransform()

    def recordTransform(self):
        self.transformHistory[self.world.time] = deepcopy(self.transform)

    def render(self) -> Optional[draw.DrawingElement]:
        for comp in self.components:
            if isinstance(comp, Mesh):
                return comp.render()

    def startCoroutine(self, cor: "Coroutine", initialDelay: float = 0) -> "Coroutine":
        cor.waitTime = initialDelay
        cor.alreadyWaited = 0
        cor.ownerActor = self
        self.coroutines.append(cor)
        # We want to start the coroutine without making it think time passed, hence dt=0
        cor.systemUpdate(deltaTime=0)
        return cor

    def stopCoroutine(self, cor: "Coroutine"):
        self.coroutines.remove(cor)

    def getComponentOfType(self, clazz: Type[T]) -> T:
        for comp in self.components:
            if isinstance(comp, clazz):
                return comp
        raise ValueError()

    def getComponentsOfType(self, clazz: Type[T]) -> list[T]:
        return [comp for comp in self.components if isinstance(comp, clazz)]


class PrefabFactory:
    def __init__(self, *behaviorParams: tuple[Type[Behavior], dict[str, Any]]) -> None:
        self.behaviorParams = behaviorParams

    def build(self, transform: Transform, world: "World") -> Actor:
        newActor = Actor(transform, world)
        for comp, params in self.behaviorParams:
            compInst = comp(newActor, **params)
            newActor.components.append(compInst)
        return newActor


class World:
    def __init__(self, width=800, height=800, deltaTime=0.1, gravity=Vector(0, 1)):
        self.time: float = 0
        self.deltaTime = deltaTime
        self.simulatingActors: list[Actor] = []
        self.destroyedActors: list[Actor] = []
        self.drawing = None
        self.simulationUpTo = -1
        self.width = width
        self.height = height

    def destroyActor(self, actor: Actor):
        a = self.simulatingActors[self.simulatingActors.index(actor)]
        self.simulatingActors.remove(a)
        self.destroyedActors.append(a)

    def placeBuiltActor(self, actor: Actor):
        """Places actor already instantiated from factory into the world"""
        actor.start()
        actor.recordTransform()
        self.simulatingActors.append(actor)
        return actor

    def placeActorFromPrefab(self, factory: PrefabFactory, transform: Transform):
        """Places actor already instantiated from factory into the world"""
        return self.placeBuiltActor(factory.build(transform, self))

    def update(self):
        self.time += self.deltaTime
        for actor in self.simulatingActors:
            actor.update(self.deltaTime)

    def simulateTo(self, endTime: float):
        self.time = 0
        while self.time < endTime:
            self.update()
        self.simulationUpTo = self.time

    def render(self, filename: str):
        if self.simulationUpTo <= 0:
            raise ValueError()

        self.drawing = draw.Drawing(
            self.width,
            self.height,
            origin="center",
            animation_config=draw.types.SyncedAnimationConfig(
                # Animation configuration
                duration=self.time,  # Seconds
                show_playback_progress=True,
                show_playback_controls=True,
                repeat_count="indefinite",
            ),
        )

        shapes = (actor.render() for actor in self.simulatingActors)
        for shape in shapes:
            self.drawing.append(shape)
        self.drawing.save_svg(filename)


class Collider(ABC):
    def __init__(self, owner: Actor, callback: Callable[[Self], None]) -> None:
        self.owner = owner
        self.callback = callback

    @abstractmethod
    def checkCollision(self, other: Self) -> bool:
        pass


class CircleCollider(Collider):
    def __init__(self, radius, *args) -> None:
        super().__init__(*args)
        self.radius = radius

    def checkCollision(self, other: Collider) -> bool:
        if isinstance(other, CircleCollider):
            centerDist = (
                (self.owner.transform.pos.x - other.owner.transform.pos.x) ** 2
                + (self.owner.transform.pos.y - other.owner.transform.pos.y) ** 2
            ) ** 0.5
            radiusSep = self.radius + other.radius
            return centerDist <= radiusSep
        raise NotImplementedError()


class BoxCollider(Collider):
    pass


class Empty(Actor):
    def createAnimation(self):
        return None


class Coroutine:
    def __init__(self, asyncUpdateFunc: Callable[[float], float]) -> None:
        self.waitTime: float = 0
        self.alreadyWaited: float = 0
        self.userAsyncUpdateFunction = asyncUpdateFunc
        self.id = random.randint(0, 1000)
        self.ownerActor: Actor

    def systemAsyncUpdate(self, asyncDeltaTime: float):
        """
        Generator function, yields new time to wait.
        asyncDeltaTime: how long was waited since last execution of coroutine.
        """
        return self.userAsyncUpdateFunction(asyncDeltaTime)

    def systemUpdate(self, deltaTime: float):
        if self.waitTime < 0:
            self.ownerActor.stopCoroutine(self)
            return

        self.alreadyWaited += deltaTime
        if self.alreadyWaited >= self.waitTime:
            self.waitTime = self.systemAsyncUpdate(self.alreadyWaited)
            self.alreadyWaited = 0
