import sys
import math
import taichi as ti
import numpy as np
import Simulation as Sim
import UtilsFunc as UF
import SceneData as SCD

WRITE_PIC = 1

def add_plane(sim):
   shape = SCD.Shape()
   shape.setType(UF.SHAPE_QUAD)
   shape.setV1([0.0,0.0,10.0])
   shape.setV2([10.0, 0.0,0.0])
   sim.add_static_mesh(shape, [0.0,0.0,0.0], [0.0,0.0,0.0,1.0],[1.0, 1.0, 1.0])


def main(argv):
   ti.init(arch=ti.gpu, dynamic_index=True)
    
   particl_radius = 0.5
    
   window = ti.ui.Window("PBD", (512, 512), vsync=True)
   canvas = window.get_canvas()
   scene = ti.ui.Scene()
   camera = ti.ui.make_camera()
   
   sim = Sim.Simulation()
   sim.load_particle_file("model/T.rigid", 0.1, [0.0,0.0,0.0], [1.0,0.0,0.0,1.0], True, [-25.0, 3.0, 0.0], 0.1)
   sim.load_particle_file("model/A.rigid", 0.1, [0.0,0.0,0.0], [0.0,0.0,0.0,1.0], True, [-15.0, 6.0, 0.0], 0.1)
   sim.load_particle_file("model/I.rigid", 0.1, [0.0,0.0,0.0], [0.0,0.0,0.0,1.0], True, [-5.0,  9.0, 0.0], 0.1)
   sim.load_particle_file("model/C.rigid", 0.1, [0.0,0.0,0.0], [0.0,0.0,0.0,1.0], True, [8.0,  12.0, 0.0], 0.1)
   sim.load_particle_file("model/H.rigid", 0.1, [0.0,0.0,0.0], [0.0,0.0,1.0,1.0], True, [20.0, 6.0, 0.0], 0.1)
   sim.load_particle_file("model/I.rigid", 0.1, [0.0,0.0,0.0], [0.0,0.0,0.0,1.0], True, [30.0, 3.0, 0.0], 0.1)

   #add_plane(sim)
   sim.add_static_mesh("model/cube.obj",[0.0,0.0,0.0],[0.0,0.0,0.0,1.0],[30.0, 0.1, 30.0])

   sim.build()
   #sim.export()

   yaw = 0.0
   pitch = 0.6
   scale = 100.0

   
   sim.init_sim()
   while window.running:
      sim.sim_one_frame()
      camera.position(scale*math.cos(pitch)*math.sin(yaw), scale*math.sin(pitch)+5.0, scale*math.cos(pitch)*math.cos(yaw))
      camera.lookat(0.0,   5.0, 0.0)
      scene.set_camera(camera)
      scene.particles(sim.position, per_vertex_color=sim.color, radius=particl_radius)
      scene.point_light(pos=(sim.centrenp[0,0], sim.centrenp[0,1], sim.centrenp[0,2]+np.linalg.norm(sim.sizenp)*1.5), color=(0.5, 0.5, 0.5))
      canvas.scene(scene)

      if WRITE_PIC== 1:
         filename = 'result/' + str(sim.frame_cpu[0]).zfill(6) + '.png'
         window.write_image(filename)
      
      window.show()

      if sim.frame_cpu[0]>500:
         break

      
      #yaw += 0.01
   
if __name__ == "__main__":
   main(sys.argv[1:])










