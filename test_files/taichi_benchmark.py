import time
import taichi as ti
ti.init(arch=ti.gpu)

@ti.kernel
def benchmark():
    x = 1.0
    for i in range(1000):
        for j in range(10000):
            x += ti.sin(float(i + j))

start = time.time()
benchmark()
ti.sync()
end = time.time()
print(end - start)

start = time.time()
benchmark()
end = time.time()
print(end - start) # less than previous one.