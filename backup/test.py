from svganim import *

t1 = Transform(rot=1)

v1 = Vector(0, 10)

v2 = t1.applyToVec(v1)

print(v2)
