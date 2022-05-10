import sys
import os
import taichi as ti
import time
import math
import numpy as np
import Mesh2Particle as M2P
import Simulation as Sim
import sys, getopt



def main(argv):
    ti.init(arch=ti.gpu)

    particl_radius = 0.5
    try:
       opts, args = getopt.getopt(argv,"hi:r:",["input=","radius="])
    except getopt.GetoptError:
       print("ShowVoxel.py -i <input>")
       sys.exit(2)
    for opt, arg in opts:
       if opt == '-h':
          print ("ShowVoxel.py -i <input>")
          sys.exit()
       elif opt in ("-i", "--input"):
          obj_name = arg
       elif opt in ("-r", "--radius"):
          particl_radius = float(arg)

    window = ti.ui.Window("PBD", (512, 512), vsync=True)
    canvas = window.get_canvas()
    scene = ti.ui.Scene()
    camera = ti.ui.make_camera()
    
    sim = Sim.Simulation()
    sim.load_particle_file("model/" + obj_name+ ".rigid", 0.1, [0.0,0.0,0.0], [1.0,0.0,0.0,1.0], True, [0.0,0.0,0.0])
    sim.build()
    
    while window.running:
        camera.position(sim.centrenp[0,0], sim.centrenp[0,1], sim.centrenp[0,2]+np.linalg.norm(sim.sizenp)*2.0)
        camera.lookat(sim.centrenp[0,0], sim.centrenp[0,1], sim.centrenp[0,2])
        scene.set_camera(camera)
        scene.particles(sim.position, per_vertex_color=sim.color, radius=particl_radius)
        scene.point_light(pos=(sim.centrenp[0,0], sim.centrenp[0,1], sim.centrenp[0,2]+np.linalg.norm(sim.sizenp)*1.5), color=(0.5, 0.5, 0.5))
        canvas.scene(scene)
        window.show()

if __name__ == "__main__":
   main(sys.argv[1:])










