
n = 1000
import random
indices = [random.randrange(0,n) for i in range(1000)]

class OutcomeState:

    def __init__(self):
        self.x = False
        self.y = 0
    
    def manip(self):
        self.x = not self.x
        self.y = 2
    
    def copy_from(self,other):
        self.x = other.x
        self.y = other.y

class OOPClass:

    def __init__(self):
        self.os = [OutcomeState() for x in range(n)]
    
    def copy_from(self,other):
        for i in range(len(self.os)):
            self.os[i].copy_from(other.os[i])
        # for selfo, othero in zip(self.os,other.os):
        #     selfo.copy_from(othero)

    def manip(self):
        for i in indices:
            self.os[i].manip()
    
import numpy as np

class NPClass:

    def __init__(self):
        self.xs = np.full((n), False, dtype=bool)
        self.ys = np.full((n), 0, dtype=np.uint8)

    def manip(self):
        for i in indices:
            self.xs[i] = not self.xs[i]
            self.ys[i] = 2
        
    def copy_from(self, other):
        np.copyto(self.xs,other.xs)
        np.copyto(self.ys,other.ys)

m = 20
OOPobjs = [OOPClass() for i in range(m)]
NPobjs = [NPClass() for i in range(m)]

def OOPManip():
    for obj in OOPobjs:
        obj.manip()
def NPManip():
    for obj in NPobjs:
        obj.manip()

def OOPCopy():
    for i in range(m-1):
        OOPobjs[i].copy_from(OOPobjs[i+1])
def NPCopy():
    for i in range(m-1):
        NPobjs[i].copy_from(NPobjs[i+1])
    

import timeit
print("OOPManip:",
    timeit.timeit("OOPManip()",globals=globals(),number=1000))
print("NPManip:",
    timeit.timeit("NPManip()",globals=globals(),number=1000))
print("")
print("OOPCopy():",
    timeit.timeit("OOPCopy()",globals=globals(),number=1000))
print("NPCopy():",
    timeit.timeit("NPCopy()",globals=globals(),number=1000))

print("Conclusions: OOPManip is 25percent faster then NPManip")
print("But NPCopy is like at least 100-1000 times faster than OOPCopy!")
print("-> Let's use a NPClass design for the grid state because of the recurring",
    "copying involved.")
    