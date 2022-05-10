import taichi as ti
import math
import numpy as np
import taichi as ti

MAX_NEIGHBOUR = 31
DEBUG_MODE = 0
DEBUG_CELL_NUM = 21
DEBUG_PARTICLE_NUM = 5

@ti.data_oriented
class UniformGrid:
    def __init__(self,  spacing, maxParticle, domainMinNp, domainMaxNp):

        self.spacing            = spacing
        self.invSpacing         = 1.0 / spacing
        self.domainMaxNp        = domainMaxNp
        self.domainMinNp        = domainMinNp
        self.max_particle       = maxParticle


        self.grid               = ti.field(dtype=ti.i32)
        self.particle           = ti.Vector.field(2, dtype=ti.i32)
        self.neighbour          = ti.field(dtype=ti.i32)

        self.blockSize          = ti.Vector.field(3, dtype=ti.i32, shape=(1))
        self.min_boundary       = ti.Vector.field(3, dtype=ti.f32, shape=(1))
        self.max_boundary       = ti.Vector.field(3, dtype=ti.f32, shape=(1))


        #cal gird col row depth 
        self.blocknp   = np.ones(shape=(1,3), dtype=np.int32)
        for i in range(3):
            self.blocknp [0, i]    = int(    (self.domainMaxNp[0, i] - self.domainMinNp[0, i]) / self.spacing + 1  )
        self.cell_num = self.blocknp [0, 0]*self.blocknp [0, 1]*self.blocknp [0, 2]+1

        if DEBUG_MODE:
            self.cell_num = DEBUG_CELL_NUM

        print("block_size:",self.blocknp , "cell_num:",self.cell_num)
        ti.root.dense(ti.i,  int(self.cell_num)  ).place(self.grid)
        ti.root.dense(ti.i,  self.max_particle).place(self.particle)
        ti.root.dense(ti.ij, [self.max_particle, MAX_NEIGHBOUR+1] ).place(self.neighbour)

        self.has_build = 0

    def debug_init(self):
        if DEBUG_MODE :
        #https://github.com/matthias-research/pages/blob/master/tenMinutePhysics/11-hashing.pdf
            par_np      = self.particle.to_numpy()
            par_np[0,0] = 5
            par_np[4,0] = 5
            par_np[1,0] = 11
            par_np[3,0] = 11
            par_np[2,0] = 13
            self.particle.from_numpy(par_np)

            grid_np = np.zeros(shape=(self.cell_num), dtype=np.int32)
            grid_np[5] =2
            grid_np[11]=2
            grid_np[13] =1
            self.grid.from_numpy(grid_np)

    def debug_print_grid(self, prefix_str):
        if DEBUG_MODE :
            gird_np = self.grid.to_numpy()
            print(prefix_str,end=' ')
            for i in range(self.cell_num):
                print(gird_np[i],end=' ')
            print("")    

    def debug_print_particle(self, particle_num):
        if DEBUG_MODE :
            par_np = self.particle.to_numpy()
            print("particle",end=' ')
            for i in range(particle_num):
                print(par_np[i,0],par_np[i,1],end=',')
            print("") 


    def build_grid(self, pos, particle_num):
        if self.has_build == 0:
            self.max_boundary.from_numpy(self.domainMaxNp )
            self.min_boundary.from_numpy(self.domainMinNp )
            self.blockSize.from_numpy(self.blocknp)
            self.has_build = 1

        self.particle_to_grid(pos, particle_num)

        self.debug_init()
        self.debug_print_grid("init")

        self.inclusive_scan()
        self.debug_print_grid("scan")  

        self.sort_particle( particle_num)
        self.debug_print_grid("sort")
        self.debug_print_particle(particle_num)

        self.query_neighbour(particle_num)

    @ti.kernel
    def particle_to_grid(self, pos:ti.template(), particle_num:ti.i32 ):
        for i in range(self.cell_num):
            self.grid[i] = 0



        for i in range(particle_num):
            self.particle[i][0] = -1
            self.particle[i][1] = -1
            self.neighbour[i,0] = 0

        #https://www.youtube.com/watch?v=D2M8jTtKi44&feature=youtu.be
        #scatter pos to grid
        for i in pos:
            indexV    = ti.cast((pos[i] - self.min_boundary[0])*self.invSpacing , ti.i32)
            index     = self.get_cell_index(indexV) 
            
            if index != -1:
                ti.atomic_add(self.grid[index] , 1)
                self.particle[i][0] = index
            #else:
            #    print("exceed domain",i, indexV)
        
    @ti.kernel
    def sum_step(self, offset:ti.i32,step:ti.i32):
        for i in range(self.cell_num):
            if ( (i+1+offset)%(step*2) == 0 ) & (i>offset):
                prev_index = i-step
                if prev_index>=0:
                    #print(i,prev_index,self.grid[prev_index],self.grid[i], self.grid[prev_index]+self.grid[i])
                    self.grid[i] += self.grid[prev_index]


    def inclusive_scan(self):
        step = 1
        offset = 0
        #https://developer.nvidia.com/gpugems/gpugems3/part-vi-gpu-computing/chapter-39-parallel-prefix-sum-scan-cuda

        #fisrt reduce
        while step < self.cell_num:
            self.sum_step(offset, step)
            step = step*2
        
   
        #offset and reduce
        step   = int(step / 2)
        offset = int(step / 2)
        while step > 1 :
            self.sum_step(int(offset), int(step/2))
            step  = step/2
            offset = offset/2


    @ti.kernel
    def sort_particle(self, particle_num:ti.i32):
        for i in range(particle_num):
            index = self.particle[i][0]
            offset = ti.atomic_add(self.grid[index] , -1)-1
            self.particle[offset][1] = i


 
    @ti.kernel
    def query_neighbour(self, particle_num:ti.i32):
        for i in range(particle_num):
            grid_index     = self.particle[i][0]

            for s in range(-1,2):
                for j in range(-1,2):
                    for k in range(-1,2):
                        neighbour_cell_index = self.get_neighbour_cell_index(grid_index, s,j,k)

                        cell_start = self.grid[neighbour_cell_index]
                        cell_end   = self.grid[neighbour_cell_index+1]
                        m = cell_start
                        while m < cell_end:
                            self.neighbour[i,0] +=  1
                            self.neighbour[i,m] =  self.particle[m][1] 
                            m+=1 

    def export_debug_grid(self, pos, particle_num):
        # debug info
        pos_np = pos.to_numpy()
        index_np = self.particle.to_numpy()
        fo = open("particle.txt", "w")

        for i in range(particle_num):
            index = index_np[i,1]
            print ("%d %d %d %f %f %f" %   (i,index_np[i, 0],index,pos_np[index,0],pos_np[index,1],pos_np[index,2]), file = fo)

        fo.close()


        deltanp = (self.domainMaxNp - self.domainMinNp) / self.blocknp
        fo = open("vol-grid.obj", "w")
        vertex_index = 1
        volume_np = self.grid.to_numpy()

        min_v3 = [0.0,0.0,0.0]
        max_v3 = [0.0,0.0,0.0]
        for i in range(self.blocknp[0,0]):
            for j in range(self.blocknp[0,1]):
                for k in range(self.blocknp[0,2]):
                    index = self.blocknp[0,1]*self.blocknp[0,2] * i + self.blocknp[0,2] * j + k

                    offset = volume_np[index] 
                    if index    > 0:
                        offset =  volume_np[index]  - volume_np[index-1]   

                    if offset> 0:
                        min_v3[0] = self.domainMinNp[0,0] + deltanp[0,0]*float(i)
                        max_v3[0] = self.domainMinNp[0,0] + deltanp[0,0]*float(i+1)
                        min_v3[1] = self.domainMinNp[0,1] + deltanp[0,1]*float(j)
                        max_v3[1] = self.domainMinNp[0,1] + deltanp[0,1]*float(j+1)
                        min_v3[2] = self.domainMinNp[0,2] + deltanp[0,2]*float(k)
                        max_v3[2] = self.domainMinNp[0,2] + deltanp[0,2]*float(k+1)                  

                        print ("v %f %f %f" %   (min_v3[0], min_v3[1], min_v3[2]), file = fo)
                        print ("v %f %f %f" %   (min_v3[0], min_v3[1], max_v3[2]), file = fo)
                        print ("v %f %f %f" %   (max_v3[0], min_v3[1], max_v3[2]), file = fo)
                        print ("v %f %f %f" %   (max_v3[0], min_v3[1], min_v3[2]), file = fo)
                        print ("v %f %f %f" %   (min_v3[0], max_v3[1], min_v3[2]), file = fo)
                        print ("v %f %f %f" %   (min_v3[0], max_v3[1], max_v3[2]), file = fo)
                        print ("v %f %f %f" %   (max_v3[0], max_v3[1], max_v3[2]), file = fo)
                        print ("v %f %f %f" %   (max_v3[0], max_v3[1], min_v3[2]), file = fo)

                        print ("f %d %d %d %d" %   (vertex_index+0, vertex_index+1, vertex_index+2, vertex_index+3), file = fo)
                        print ("f %d %d %d %d" %   (vertex_index+4, vertex_index+5, vertex_index+6, vertex_index+7), file = fo)
                        print ("f %d %d %d %d" %   (vertex_index+0, vertex_index+1, vertex_index+5, vertex_index+4), file = fo)
                        print ("f %d %d %d %d" %   (vertex_index+2, vertex_index+3, vertex_index+7, vertex_index+6), file = fo)
                        print ("f %d %d %d %d" %   (vertex_index+1, vertex_index+2, vertex_index+6, vertex_index+5), file = fo)
                        print ("f %d %d %d %d" %   (vertex_index+0, vertex_index+4, vertex_index+7, vertex_index+3), file = fo)
                        vertex_index += 8
        fo.close()



    @ti.func
    def get_cell_index(self, index):
        #https://en.wikipedia.org/wiki/Z-order_curve
        #spacial 
        ret = -1
        if (index.x >= 0) and (index.x < self.blockSize[0].x) and \
           (index.y >= 0) and (index.y < self.blockSize[0].y) and \
           (index.z >= 0) and (index.z < self.blockSize[0].z):

            ret = self.blockSize[0].y*self.blockSize[0].z * index.x + self.blockSize[0].z * index.y + index.z
        return ret
    

    @ti.func
    def get_neighbour_cell_index(self, index, i,j,k):
        x = int( index // (self.blockSize[0].z*self.blockSize[0].y) )
        y = int( (index // self.blockSize[0].z) %self.blockSize[0].y)
        z = int( index % self.blockSize[0].z)
        #print(index, i,j,k,x,y,z)
        return self.get_cell_index(ti.Vector([x+i,y+j,z+k]))

