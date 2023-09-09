from abc import ABC, abstractmethod
from collections import defaultdict
import random
from typing import Any, Callable, Optional, Self, TypeVar, Generic
import drawsvg as draw

from dataclasses import dataclass

T = TypeVar("T")


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

    def __neg__(self):
        return self * -1

    def mag(self):
        """Magnitude"""
        return (self.x**2 + self.y**2) ** 0.5

    def norm(self):
        """Normalized"""
        return self / self.mag()


@dataclass(frozen=True)
class Transform:
    pos: Vector = Vector(0, 0)
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

    def destroyActor(self, actor: "Actor"):
        self.owner.world.destroyActor(actor)

    def lateUpdate(self, deltaTime: float):
        pass


class Timeline(ABC, Generic[T]):
    """
    Implementation: every frame we call render, which stores the necessary attributes into the timeline.
    It's no longer the actor's job to store intransient attributes, it's now the behavior's job.
    Any attribute which doesn't exist in a timeline is trainsient and will not be considered at all during render.
    At the end we apply the timeline's animation output onto the shape to create the svg.
    """

    def __init__(
        self,
        attributeName: str,
        animationClass: type[draw.Animate] = draw.Animate,
        initialDefault: Optional[T] = None,
        **kwargs: str,
    ) -> None:
        self.attributeName = attributeName
        self.animationClass = animationClass
        self.keyframes: dict[float, T] = defaultdict()
        self.initialDefault = initialDefault
        self.kwargs = kwargs

    def addKeyframe(self, time: float, value: T):
        assert time >= 0
        self.keyframes[time] = value

    def removeKeyframe(self, time: float):
        del self.keyframes[time]

    @abstractmethod
    def timestampRepr(self, value: T) -> str:
        pass

    def toAnimations(self, duration: float = -1) -> list[draw.Animate]:
        # Filling first frame with default value
        if not 0 in self.keyframes.keys():
            if self.initialDefault == None:
                self.keyframes[0] = self.keyframes[min(self.keyframes.keys())]
            else:
                self.keyframes[0] = self.initialDefault

        # Extending animation to desired length
        if duration >= 0:
            self.keyframes = {k: v for k, v in self.keyframes.items() if k <= duration}
            self.keyframes[duration] = self.keyframes[max(self.keyframes.keys())]

        durFloat = max(self.keyframes.keys())
        dur = f"{str(durFloat)}s"
        items = sorted(self.keyframes.items())
        values = ";".join(self.timestampRepr(v) for k, v in items)
        if durFloat == 0:
            keyTimes = "0"
        else:
            keyTimes = ";".join(str(k / durFloat) for k, v in items)
        repeatCount = "indefinite"
        fill = "freeze"

        return [
            self.animationClass(
                attributeName=self.attributeName,
                dur=dur,
                from_or_values=values,
                keyTimes=keyTimes,
                repeatCount=repeatCount,
                fill=fill,
                **self.kwargs,
            )
        ]

    def extendAnimations(self, shape: draw.DrawingBasicElement, duration: float = -1):
        shape.extend_anim(self.toAnimations(duration))


class VectorTimeline(Timeline[Vector]):
    def __init__(self, type: str, additive: str) -> None:
        super().__init__(
            "transform",
            animationClass=draw.AnimateTransform,
            type=type,
            additive=additive,
        )

    def timestampRepr(self, value: Vector) -> str:
        return f"{value.x},{value.y}"


class FloatTimeline(Timeline[float]):
    def timestampRepr(self, value: Vector) -> str:
        return str(value)


class BoolTimeline(Timeline[bool]):
    def __init__(
        self,
        attributeName: str,
        mapping: dict[bool, str],
        animationClass: type[draw.Animate] = draw.Animate,
        initialDefault: bool | None = None,
        **kwargs: str,
    ) -> None:
        super().__init__(attributeName, animationClass, initialDefault, **kwargs)
        self.mapping = mapping

    def timestampRepr(self, value: bool) -> str:
        return self.mapping[value]


# class RotationTimeline(FloatTimeline):
#     def __init__(
#         self,
#         initialDefault: Optional[float] = None,
#         **kwargs: str,
#     ) -> None:
#         super().__init__(
#             "transform",
#             animationClass=draw.AnimateTransform,
#             initialDefault=initialDefault,
#             **kwargs,
#         )


class TransformTimeline(Timeline[Transform]):
    def __init__(self) -> None:
        self.posTimeline = VectorTimeline("translate", additive="sum")
        self.rotTimeline = FloatTimeline(
            "transform",
            animationClass=draw.AnimateTransform,
            type="rotate",
            additive="sum",
        )
        self.scaleTimeline = VectorTimeline("scale", additive="sum")

    def addKeyframe(self, time: float, value: Transform):
        self.posTimeline.addKeyframe(time, value.pos)
        self.rotTimeline.addKeyframe(time, value.rot)
        self.scaleTimeline.addKeyframe(time, value.scale)

    def removeKeyframe(self, time: float):
        self.posTimeline.removeKeyframe(time)
        self.rotTimeline.removeKeyframe(time)
        self.scaleTimeline.removeKeyframe(time)

    def toAnimations(self, duration: float = -1) -> list[draw.Animate]:
        return (
            self.posTimeline.toAnimations(duration)
            + self.rotTimeline.toAnimations(duration)
            + self.scaleTimeline.toAnimations(duration)
        )

    def timestampRepr(self, value: Transform) -> str:
        return ""


class Mesh(Behavior):  # Todo: implement color timeline track
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
        ellipse = draw.Ellipse(
            0,
            0,
            1,
            1,
        )

        self.owner.lifecycleTimeline.extendAnimations(
            ellipse, self.owner.world.simulationUpTo
        )
        self.owner.transformTimeline.extendAnimations(
            ellipse, self.owner.world.simulationUpTo
        )

        return ellipse


class RectangleMesh(Mesh):
    def render(self) -> Optional[draw.DrawingElement]:
        rect = draw.Rectangle(0, 0, 1, 1)

        self.owner.lifecycleTimeline.extendAnimations(
            rect, self.owner.world.simulationUpTo
        )
        self.owner.transformTimeline.extendAnimations(
            rect, self.owner.world.simulationUpTo
        )

        return rect


class Actor:
    def __init__(self, transform: Transform, world: "World") -> None:
        self.transform: Transform = transform
        self.world: World = world
        self.transformTimeline = TransformTimeline()
        self.lifecycleTimeline = BoolTimeline(
            "visibility",
            mapping={False: "hidden", True: "visible"},
            initialDefault=False,
        )
        self.coroutines: list[Coroutine] = []
        self.components: list[Behavior] = []

    def start(self):
        for behavior in self.components:
            behavior.start()
        self.lifecycleTimeline.addKeyframe(self.world.time, True)

    def onDestroy(self):
        # Subtraction is a magic correction, not sure why it works but it just does.
        self.lifecycleTimeline.addKeyframe(
            self.world.time - self.world.deltaTime * 2, False
        )

    def update(self, deltaTime: float):
        for behavior in self.components:
            behavior.update(deltaTime)
        for c in self.coroutines:
            c.systemUpdate(self.world.deltaTime)

    def lateUpdate(self, deltaTime: float):
        for behavior in self.components:
            behavior.lateUpdate(deltaTime)
        self.transformTimeline.addKeyframe(self.world.time, self.transform)

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

    def getComponentOfType(self, clazz: type[T]) -> T:
        for comp in self.components:
            if isinstance(comp, clazz):
                return comp
        raise ValueError()

    def getComponentsOfType(self, clazz: type[T]) -> list[T]:
        return [comp for comp in self.components if isinstance(comp, clazz)]


class PrefabFactory:
    def __init__(
        self,
        *behaviorParams: tuple[type[Behavior], dict[str, Any]],
        defaultTransform: Transform,
    ) -> None:
        self.behaviorParams = behaviorParams
        self.defaultTransform = defaultTransform

    def build(
        self,
        world: "World",
        posOverride: Optional[Vector] = None,
        rotOverride: Optional[float] = None,
        scaleOverride: Optional[Vector] = None,
    ) -> Actor:
        transform = self.defaultTransform

        if posOverride:
            transform = Transform(posOverride, transform.rot, transform.scale)
        if rotOverride:
            transform = Transform(transform.pos, rotOverride, transform.scale)
        if scaleOverride:
            transform = Transform(transform.pos, transform.rot, scaleOverride)

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
        a.onDestroy()
        self.simulatingActors.remove(a)
        self.destroyedActors.append(a)

    def placeBuiltActor(self, actor: Actor):
        """Places actor already instantiated from factory into the world"""
        actor.start()
        self.simulatingActors.append(actor)
        return actor

    def placeActorFromPrefab(
        self,
        factory: PrefabFactory,
        posOverride: Optional[Vector] = None,
        rotOverride: Optional[float] = None,
        scaleOverride: Optional[Vector] = None,
    ):
        """Places actor already instantiated from factory into the world"""
        return self.placeBuiltActor(
            factory.build(self, posOverride, rotOverride, scaleOverride)
        )

    def start(self):
        for actor in self.simulatingActors:
            actor.lateUpdate(self.deltaTime)

    def update(self):
        self.time += self.deltaTime
        for actor in self.simulatingActors:
            actor.update(self.deltaTime)
        for actor in self.simulatingActors:
            actor.lateUpdate(self.deltaTime)

    def simulateTo(self, endTime: float):
        self.time = 0
        self.start()
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

        shapes = (
            actor.render() for actor in self.simulatingActors + self.destroyedActors
        )
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
