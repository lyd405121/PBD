import sys
import taichi as ti
import numpy as np
import Mesh2Particle as M2P
import Simulation as Sim
import sys, getopt


def main(argv):
    ti.init(arch=ti.gpu)

    obj_name = "H"
    space   = 0.5
    #print(argv)

    try:
        opts, args = getopt.getopt(argv,"-h-i:-s:", ["help","input=","space="])
    except getopt.GetoptError:
        print("GenVoxel.py -i <input>")
        sys.exit(2)

    for opt, arg in opts:
        if opt == '-h':
            print ("GenVoxel.py -i <input> -s <space>")
            sys.exit()
        elif opt in ("-i", "--input"):
            obj_name = arg
        elif opt in ("-s", "--space"):
            space = float(arg)

    ti.init(arch=ti.gpu)
    particle = M2P.Mesh2Particle()
    particle.load_obj("model/" + obj_name+ ".obj", space)
    particle.build()
    particle.export()

if __name__ == "__main__":
   main(sys.argv[1:])





