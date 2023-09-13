from abc import ABC, abstractmethod
from typing import Any, Callable, Optional, Self, TypeVar, Generic
from dataclasses import dataclass

import collections
import math
import drawsvg as draw
import itertools
import collision

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
        return Vector(-self.x, -self.y)

    def __eq__(self, other: Any):
        if not isinstance(other, Vector):
            return False
        return self.x == other.x and self.y == other.y

    def __ne__(self, other: Any):
        if not isinstance(other, Vector):
            return True
        return self.x != other.x or self.y != other.y

    def __getitem__(self, index: int):
        return [self.x, self.y][index]

    def __contains__(self, value):
        return value == self.x or value == self.y

    def __len__(self):
        return 2

    def __repr__(self):
        return self.__str__()

    def __str__(self):
        return f"Vector{{{self.x}, {self.y}}}]"

    def copy(self):
        return Vector(self.x, self.y)

    def perp(self):
        return Vector(self.y, -self.x)

    def rotate(self, angle: float):
        """Angle in radians"""
        return Vector(
            self.x * math.cos(angle) - self.y * math.sin(angle),
            self.x * math.sin(angle) + self.y * math.cos(angle),
        )

    def reverse(self):
        return Vector(-self.x, -self.y)

    def normalize(self):
        dot = self.magnitude()
        return self / dot

    def project(self, other: Self):
        amt = self.dot(other) / other.magnitudeSquared()

        return Vector(amt * other.x, amt * other.y)

    def projectN(self, other: Self):
        amt = self.dot(other)

        return Vector(amt * other.x, amt * other.y)

    def reflect(self, axis: Self):
        v = Vector(self.x, self.y)
        v = v.project(axis) * 2
        v = -v

        return v

    def reflect_n(self, axis: Self):
        v = Vector(self.x, self.y)
        v = v.projectN(axis) * 2
        v = -v

        return v

    def dot(self, other: Self):
        return self.x * other.x + self.y * other.y

    def magnitudeSquared(self):
        return self.dot(self)

    def magnitude(self):
        return math.sqrt(self.magnitudeSquared())

    def matrixprod(self, other: Self):
        return Vector(self.x * other.x, self.y * other.y)

    def toCollisionRepr(self):
        return collision.Vector(self.x, self.y)


@dataclass(frozen=True)
class Transform:
    pos: Vector = Vector(0, 0)
    rot: float = 0
    scale: Vector = Vector(1, 1)

    def translate(self, delta: Vector):
        return Transform(self.pos + delta, self.rot, self.scale)

    def rotate(self, delta: float):
        return Transform(self.pos, self.rot + delta, self.scale)

    def scaleAdd(self, delta: Vector):
        return Transform(self.pos, self.rot, self.scale + delta)

    def scaleMult(self, multiplier: Vector):
        return Transform(self.pos, self.rot, self.scale.matrixprod(multiplier))

    def applyTo(self, other: Self) -> Self:
        return Transform(
            other.pos + self.pos.rotate(math.radians(other.rot)),
            other.rot + self.rot,
            other.scale.matrixprod(self.scale),
        )

    def applyToVec(self, other: Vector):
        return (other.rotate(math.radians(self.rot)) + self.pos).matrixprod(self.scale)


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
        self.keyframes: dict[float, T] = collections.defaultdict()
        self.initialDefault = initialDefault
        self.kwargs = kwargs

    def addKeyframe(self, time: float, value: T):
        assert time >= 0
        self.keyframes[time] = value

    def removeKeyframe(self, time: float):
        del self.keyframes[time]

    @abstractmethod
    def timestampRepr(self, key: float, value: T) -> str:
        return str(value)

    def toAnimations(
        self,
        mesh: "Mesh",
        duration: float = -1,
    ) -> list[draw.Animate]:
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
        values = ";".join(self.timestampRepr(k, v) for k, v in items)
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

    def extendAnimations(
        self,
        shape: draw.DrawingBasicElement,
        mesh: "Mesh",
        duration: float = -1,
    ):
        shape.extend_anim(self.toAnimations(mesh, duration=duration))


class VectorTimeline(Timeline[Vector]):
    def __init__(self, type: str, additive: str) -> None:
        super().__init__(
            "transform",
            animationClass=draw.AnimateTransform,
            type=type,
            additive=additive,
        )

    def timestampRepr(self, key: float, value: Vector) -> str:
        return f"{value.x},{value.y}"


class FloatTimeline(Timeline[float]):
    def timestampRepr(self, key: float, value: float) -> str:
        return str(value)


class MappedTimeline(Timeline[T]):
    def __init__(
        self,
        attributeName: str,
        mapping: dict[T, str],
        animationClass: type[draw.Animate] = draw.Animate,
        initialDefault: Optional[T] = None,
        **kwargs: str,
    ) -> None:
        super().__init__(attributeName, animationClass, initialDefault, **kwargs)
        self.mapping = mapping

    def timestampRepr(self, key: float, value: T) -> str:
        return self.mapping[value]


class TransformTimeline(Timeline[Transform]):
    def __init__(self) -> None:
        self.posTimeline = VectorTimeline("translate", additive="sum")
        self.rotTimeline = FloatTimeline(
            "transform",
            animationClass=draw.AnimateTransform,
            initialDefault=0,
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

    def toAnimations(self, mesh: "Mesh", duration: float = -1) -> list[draw.Animate]:
        return (
            self.posTimeline.toAnimations(mesh, duration)
            + self.rotTimeline.toAnimations(mesh, duration)
            + self.scaleTimeline.toAnimations(mesh, duration)
        )

    def timestampRepr(self, key: float, value: Transform) -> str:
        return ""


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


class Mesh(Behavior):
    def __init__(
        self,
        ownerActor: "Actor",
    ) -> None:
        """Center of mass expressed in local coordinates"""
        super().__init__(ownerActor)

    def start(self):
        return super().start()

    def update(self, deltaTime: float):
        return super().update(deltaTime)

    @abstractmethod
    def render(self) -> Optional[draw.DrawingElement]:
        pass


class EllipseMesh(Mesh):
    def __init__(
        self, ownerActor: "Actor", rx: float = 100, ry: float = 100, **kwargs
    ) -> None:
        super().__init__(ownerActor)
        self.rx = rx
        self.ry = ry
        self.kwargs = kwargs

    def render(self) -> draw.DrawingElement:
        ellipse = draw.Ellipse(0, 0, self.rx, self.ry, **self.kwargs)

        self.owner.lifecycleTimeline.extendAnimations(
            ellipse, self, self.owner.world.simulationUpTo
        )
        self.owner.absoluteTransformTimeline.extendAnimations(
            ellipse, self, self.owner.world.simulationUpTo
        )

        return ellipse


class PolyMesh(Mesh):
    def __init__(
        self,
        ownerActor: "Actor",
        points: list[Vector],
        centerOfMass: Vector = Vector(0, 0),
    ) -> None:
        super().__init__(ownerActor)
        self.points = [p - centerOfMass for p in points]

    def render(self) -> draw.DrawingElement:
        poly = draw.Lines(
            *itertools.chain.from_iterable((p.x, p.y) for p in self.points),
            close=True,
        )

        self.owner.lifecycleTimeline.extendAnimations(
            poly, self, self.owner.world.simulationUpTo
        )
        self.owner.absoluteTransformTimeline.extendAnimations(
            poly, self, self.owner.world.simulationUpTo
        )

        return poly


class RectMesh(PolyMesh):
    def __init__(
        self,
        ownerActor: "Actor",
        width: float,
        height: float,
    ) -> None:
        super().__init__(
            ownerActor,
            [Vector(0, 0), Vector(width, 0), Vector(width, height), Vector(0, height)],
            Vector(width / 2, height / 2),
        )


class Collider(Behavior):
    def __init__(
        self,
        owner: "Actor",
        callbacks: list[Callable[["Collider", collision.Response], None]] = [],
        isTrigger=False,
    ) -> None:
        self.owner = owner
        self.isTrigger = isTrigger
        self.callbacks: list[Callable[["Collider", collision.Response], None]] = []
        self.callbacks.extend(callbacks)

    @abstractmethod
    def checkCollision(self, other: "Collider") -> bool:
        pass

    @abstractmethod
    def toCollisionRepr(self) -> collision.Poly | collision.Circle:
        pass

    def onCollide(self, other: "Collider", response: collision.Response):
        for callback in self.callbacks:
            callback(other, response)
        for callback in other.callbacks:
            callback(self, response)


class ConvexPolyCollider(Collider):
    def __init__(
        self,
        owner: "Actor",
        callbacks: list[Callable[["Collider", collision.Response], None]] = [],
        isTrigger=False,
        fromMesh=True,  # Todo: implement non from-mesh colliders
    ) -> None:
        super().__init__(owner, callbacks, isTrigger)
        self.polyMesh: PolyMesh

    def start(self):
        self.polyMesh = self.owner.getComponentOfType(PolyMesh)

    def toCollisionRepr(self):
        absTrans = self.owner.getAbsoluteTransform()  # Todo: let scale affect collider
        return collision.Poly(collision.Vector(0, 0), [absTrans.applyToVec(p).toCollisionRepr() for p in self.polyMesh.points], 0)  # type: ignore

    def checkCollision(self, other: Collider) -> bool:
        response = collision.Response()
        hasCollided = collision.collide(
            self.toCollisionRepr(), other.toCollisionRepr(), response
        )
        if hasCollided:
            self.onCollide(other, response)
        return hasCollided


class Actor:
    def __init__(self, localTransform: Transform, world: "World") -> None:
        self.localTransform: Transform = localTransform
        self.world: World = world
        self.absoluteTransformTimeline = TransformTimeline()
        self.lifecycleTimeline = MappedTimeline(
            "visibility",
            mapping={False: "hidden", True: "visible"},
            initialDefault=False,
        )
        self.coroutines: list[Coroutine] = []
        self.components: list[Behavior] = []
        self.hasStarted = False

        self.children: list[Actor] = []
        self.parent: Optional[Actor] = None

    def getAbsoluteTransform(self) -> Transform:
        transChain = [self.localTransform]
        next = self.parent
        while next:
            transChain.insert(0, next.localTransform)
            next = next.parent

        absoluteTrans = self.world.sceneGraph.localTransform
        for item in transChain:
            absoluteTrans = item.applyTo(absoluteTrans)
        return absoluteTrans

    def start(self):
        if not self.hasStarted:
            for behavior in self.components:
                behavior.start()
            for child in self.children:
                child.start()
            self.lifecycleTimeline.addKeyframe(self.world.time, True)
            self.hasStarted = True

    def onDestroy(self):
        for child in self.children:
            child.onDestroy()
        # Subtraction is a magic correction, not sure why it works but it just does.
        self.lifecycleTimeline.addKeyframe(
            self.world.time - self.world.deltaTime * 2, False
        )

    def update(self, deltaTime: float):
        for behavior in self.components:
            behavior.update(deltaTime)
        for c in self.coroutines:
            c.update(self.world.deltaTime)
        for child in self.children:
            child.update(deltaTime)

    def lateUpdate(self, deltaTime: float):
        for behavior in self.components:
            behavior.lateUpdate(deltaTime)
        for child in self.children:
            child.lateUpdate(deltaTime)
        self.absoluteTransformTimeline.addKeyframe(
            self.world.time, self.getAbsoluteTransform()
        )

    def render(self) -> list[Optional[draw.DrawingElement]]:
        shapes = []
        for comp in self.components:
            if isinstance(comp, Mesh):
                shapes.append(comp.render())
        for child in self.children:
            shapes.extend(child.render())
        return shapes

    def startCoroutine(self, cor: "Coroutine", initialDelay: float = 0) -> "Coroutine":
        cor.waitTime = initialDelay
        cor.alreadyWaited = 0
        cor.owner = self
        self.coroutines.append(cor)
        # We want to start the coroutine without making it think time passed, hence dt=0
        cor.update(deltaTime=0)
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

    def getComponentsOfTypeInclChildren(self, clazz: type[T]) -> list[T]:
        totalComps = [comp for comp in self.components if isinstance(comp, clazz)]
        for child in self.children:
            totalComps += child.getComponentsOfTypeInclChildren(clazz)
        return totalComps


class PrefabFactory:
    def __init__(
        self,
        *behaviorParams: tuple[type[Behavior], dict[str, Any]],
        defaultTransform: Transform = Transform(),
        children: list[Self] = [],
    ) -> None:
        self.behaviorParams = behaviorParams
        self.defaultTransform = defaultTransform
        self.childFactories = children

    def build(
        self,
        world: "World",
        posOverride: Optional[Vector] = None,
        rotOverride: Optional[float] = None,
        scaleOverride: Optional[Vector] = None,
        parent: Optional[Actor] = None,
    ) -> Actor:
        transform = self.defaultTransform

        if posOverride:
            transform = Transform(posOverride, transform.rot, transform.scale)
        if rotOverride:
            transform = Transform(transform.pos, rotOverride, transform.scale)
        if scaleOverride:
            transform = Transform(transform.pos, transform.rot, scaleOverride)

        newActor = Actor(transform, world)
        newActor.parent = parent
        for comp, params in self.behaviorParams:
            compInst = comp(newActor, **params)
            newActor.components.append(compInst)

        for cf in self.childFactories:
            child = cf.build(world, parent=newActor)
            newActor.children.append(child)

        return newActor


class World:
    def __init__(self, width=800, height=800, deltaTime=0.1, gravity=Vector(0, 1)):
        self.time: float = 0
        self.deltaTime = deltaTime
        self.drawing = None
        self.simulationUpTo = -1
        self.width = width
        self.height = height

        self.sceneGraph = Actor(Transform(), self)
        self.destroyedGraph = Actor(Transform(), self)

    def destroyActor(self, actor: Actor):
        if actor.parent:
            actor.parent.children.remove(actor)
        actor.onDestroy()
        self.destroyedGraph.children.append(actor)

    def placeBuiltActor(self, actor: Actor, parent: Optional[Actor] = None):
        """Places actor already instantiated from factory into the world"""
        if parent:
            parent.children.append(actor)
            actor.parent = parent
        else:
            self.sceneGraph.children.append(actor)
            actor.parent = self.sceneGraph

        actor.start()
        return actor

    def placeActorFromPrefab(
        self,
        factory: PrefabFactory,
        posOverride: Optional[Vector] = None,
        rotOverride: Optional[float] = None,
        scaleOverride: Optional[Vector] = None,
        parent: Optional[Actor] = None,
    ):
        """Instantiates instance from factory and place into world"""
        newActor = factory.build(
            self, posOverride, rotOverride, scaleOverride, parent=parent
        )
        return self.placeBuiltActor(newActor, parent=parent)

    def start(self):
        self.sceneGraph.lateUpdate(0)

    def tick(self):
        self.time += self.deltaTime

        # Update
        self.sceneGraph.update(self.deltaTime)

        # Physics tick
        colliders = self.sceneGraph.getComponentsOfTypeInclChildren(Collider)
        for a, b in itertools.combinations(colliders, 2):
            a.checkCollision(b)

        # Late update
        self.sceneGraph.lateUpdate(self.deltaTime)

    def simulateTo(self, endTime: float):
        self.time = 0
        self.start()
        while self.time < endTime:
            self.tick()
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

        shapes = self.sceneGraph.render() + self.destroyedGraph.render()
        for shape in shapes:
            self.drawing.append(shape)
        self.drawing.save_svg(filename)


class Coroutine:
    def __init__(self, asyncUpdateFunc: Callable[[float], float]) -> None:
        self.waitTime: float = 0
        self.alreadyWaited: float = 0
        self.userAsyncUpdateFunction = asyncUpdateFunc
        self.owner: Actor

    def asyncUpdate(self, asyncDeltaTime: float):
        """
        Generator function, yields new time to wait.
        asyncDeltaTime: how long was waited since last execution of coroutine.
        """
        return self.userAsyncUpdateFunction(asyncDeltaTime)

    def update(self, deltaTime: float):
        if self.waitTime < 0:
            self.owner.stopCoroutine(self)
            return

        self.alreadyWaited += deltaTime
        if self.alreadyWaited >= self.waitTime:
            self.waitTime = self.asyncUpdate(self.alreadyWaited)
            self.alreadyWaited = 0
