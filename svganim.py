from abc import ABC, abstractmethod
from copy import deepcopy
from typing import Self
from typing import Type
import drawsvg as draw

from dataclasses import dataclass


@dataclass(frozen=True)
class Vector:
    x: float
    y: float

    def __add__(self, other: Self):
        return Vector(self.x + other.x, self.y + other.y)

    def __mul__(self, other: float):
        return Vector(self.x * other, self.y * other)


@dataclass(frozen=True)
class Transform:
    pos: Vector
    rot: float = 0
    scale: Vector = Vector(1, 1)

    def translate(self, delta: Vector):
        return Transform(self.pos + delta, self.rot, self.scale)


class Actor(ABC):
    def __init__(self, transform: Transform, world: "World", **kwargs) -> None:
        self.transform: Transform = transform
        self.world: World = world
        self.transformHistory: dict[float, Transform] = dict()
        self.constructorKwargs = kwargs

    def start(self):
        pass

    def update(self, deltaTime: float):
        pass

    def systemUpdate(self, deltaTime: float):
        self.update(deltaTime)
        self.recordTransform()

    def recordTransform(self):
        self.transformHistory[self.world.time] = deepcopy(self.transform)

    @abstractmethod
    def createAnimation(self) -> draw.DrawingBasicElement:
        pass


class World:
    def __init__(self, width=800, height=800, deltaTime=0.1, gravity=1):
        self.time: float = 0
        self.deltaTime = deltaTime
        self.actors: list[Actor] = []
        self.coroutines: list[Coroutine] = []
        self.drawing = None
        self.simulationUpTo = -1
        self.width = width
        self.height = height

    def spawnActor(self, clazz: Type[Actor], transform: Transform, **kwargs):
        newActor = clazz(transform, self, **kwargs)
        newActor.start()
        newActor.recordTransform()
        self.actors.append(newActor)

    def spawnCoroutine(self, clazz: Type["Coroutine"]):
        newCor = clazz()
        newCor.systemUpdate(self.deltaTime)
        self.coroutines.append(newCor)

    def tick(self):
        self.time += self.deltaTime
        for actor in self.actors:
            actor.systemUpdate(self.deltaTime)
        for c in self.coroutines:
            # We want to start the coroutine without making it think time passed, hence dt=0
            c.systemUpdate(deltaTime=0)

    def simulateTo(self, endTime: float):
        self.time = 0
        while self.time < endTime:
            self.tick()
        self.simulationUpTo = self.time

    def render(self, filename: str):
        if self.simulationUpTo < 0:
            raise ValueError()

        self.drawing = draw.Drawing(
            self.width,
            self.height,
            origin="center",
            animation_config=draw.types.SyncedAnimationConfig(
                # Animation configuration
                duration=self.time,  # Seconds
                show_playback_progress=True,
                repeat_count="indefinite",
            ),
        )

        shapes = (actor.createAnimation() for actor in self.actors)
        for shape in shapes:
            self.drawing.append(shape)
        self.drawing.save_svg(filename)


class Circle(Actor):
    def __init__(self, transform: Transform, world: World, **kwargs) -> None:
        super().__init__(transform, world, **kwargs)

    def createAnimation(self):
        timeOfSpawn = min(self.transformHistory.keys())

        circle = draw.Circle(
            self.transformHistory[timeOfSpawn].pos.x,
            self.transformHistory[timeOfSpawn].pos.y,
            **self.constructorKwargs,
        )
        circle.add_key_frame(time=0, visibility="hidden")
        circle.add_key_frame(time=timeOfSpawn, visibility="visible")

        for time, trans in self.transformHistory.items():
            circle.add_key_frame(time=time, cx=trans.pos.x, cy=trans.pos.y)

        return circle


class Manager(Actor):
    def createAnimation(self):
        return None


class Coroutine(ABC):
    def __init__(self) -> None:
        self.waitTime: float = 0
        self.alreadyWaited: float = 0

    @abstractmethod
    def asyncUpdate(self, asyncDeltaTime: float):
        """
        Generator function, yields new time to wait.
        asyncDeltaTime: how long was waited since last execution of coroutine.
        """
        return 10

    def systemUpdate(self, deltaTime: float):
        self.alreadyWaited += deltaTime
        if self.alreadyWaited >= self.waitTime:
            self.waitTime = self.asyncUpdate(self.alreadyWaited)
            self.alreadyWaited = 0
