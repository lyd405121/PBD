import sys
import os
sys.path.append("accel")
import taichi as ti
import math
import numpy as np
import pywavefront
import SceneData as SCD
import UtilsFunc as UF
import taichi_glsl as ts
import LBvh as LBvh
import queue
import math
from heapq import * 

MAX_STACK_SIZE =  32

##


@ti.data_oriented
class Mesh2Particle:
    def __init__(self):
        self.maxboundarynp           = np.ones(shape=(1,3), dtype=np.float32)
        self.minboundarynp           = np.ones(shape=(1,3), dtype=np.float32)
        self.deltanp                 = np.ones(shape=(1,3), dtype=np.float32)
        for i in range(3):
            self.maxboundarynp[0, i] = -UF.INF_VALUE
            self.minboundarynp[0, i] = UF.INF_VALUE

        self.min_boundary    = ti.Vector.field(3, dtype=ti.f32, shape=(1))
        self.max_boundary    = ti.Vector.field(3, dtype=ti.f32, shape=(1))
        self.delta           = ti.Vector.field(3, dtype=ti.f32, shape=(1))
        self.particle_num    = ti.field(dtype=ti.i32, shape=(1))


        self.vertex_cpu            = []
        self.vertex                = ti.Vector.field(3, dtype=ti.f32)


        self.volume_map            = ti.field(dtype=ti.i32)
        self.sdf                   = ti.field(dtype=ti.f32)
        self.sdf_normal            = ti.Vector.field(3, dtype=ti.f32)
        self.edge                  = ti.field(dtype=ti.i32)  

        self.stack                 = ti.field( dtype=ti.i32)
        self.vertex_count          = 0


    def load_obj(self, filename, space):
        find_pos = filename.find('/')+1
        self.filename = filename[find_pos: len(filename)]
        self.model_filename = filename

        scene = pywavefront.Wavefront(filename)
        scene.parse() 

        for name in scene.materials:
            ######process vert#########
            num_vert = len(scene.materials[name].vertices)
            v_format = scene.materials[name].vertex_format
            
            inner_index = 0
            while inner_index < num_vert:
                
                vertex   = [0.0,0.0,0.0]
                if v_format == 'T2F_V3F':
                    for i in range(3):
                        vertex[i] = scene.materials[name].vertices[inner_index+2+i]
                    inner_index += 5
                    

                if v_format == 'T2F_N3F_V3F':
                    for i in range(3):
                        vertex[i] = scene.materials[name].vertices[inner_index+5+i]
                    inner_index += 8

                if v_format == 'N3F_V3F':
                    for i in range(3):
                        vertex[i] = scene.materials[name].vertices[inner_index+3+i]
                    inner_index += 6

                if v_format== 'V3F':
                    for i in range(3):
                        vertex[i] = scene.materials[name].vertices[inner_index+i]
                    inner_index += 3   

                for k in range(3):
                    self.maxboundarynp[0, k]   = max(vertex[k], self.maxboundarynp[0, k])
                    self.minboundarynp[0, k]   = min(vertex[k], self.minboundarynp[0, k])

                self.vertex_count += 1
                self.vertex_cpu.append(vertex)

        print("***************vertex:%d *******************"%(self.vertex_count))
        self.vertex_np        = np.zeros(shape=(self.vertex_count, 3), dtype=np.float32)
        for i in range(self.vertex_count):
            for j in range(3):
                self.vertex_np[i,j] = self.vertex_cpu[i][j]

        self.space = space
        print("***************bounding***********************")
        self.bvh = LBvh.Bvh((self.vertex_count//3), self.minboundarynp, self.maxboundarynp)

        for i in range(3):
            self.minboundarynp[0,i]   -= self.space*2.0
            self.maxboundarynp[0,i]   += self.space*2.0
        boundary_size = self.maxboundarynp-self.minboundarynp

        self.max_dim = 0
        for i in range(3):
            self.max_dim = max(self.max_dim, int(boundary_size[0,i] / self.space)+1)

        #for i in range(3):
        #    self.deltanp[0,i]   = boundary_size[0,i] / float(self.max_dim)

        for i in range(3):
            self.deltanp[0,i]   = self.space
            self.maxboundarynp[0,i] = self.minboundarynp[0,i] + self.space*self.max_dim

        print(self.minboundarynp, self.maxboundarynp, self.deltanp, self.max_dim)

        ti.root.dense(ti.i, self.vertex_count    ).place(self.vertex)
        ti.root.dense(ti.ijk, [self.max_dim, self.max_dim, MAX_STACK_SIZE ] ).place(self.stack  )
        ti.root.dense(ti.ijk, [self.max_dim, self.max_dim, self.max_dim] ).place(self.volume_map )
        ti.root.dense(ti.ijk, [self.max_dim, self.max_dim, self.max_dim] ).place(self.sdf )
        ti.root.dense(ti.ijk, [self.max_dim, self.max_dim, self.max_dim] ).place(self.edge )
        ti.root.dense(ti.ijk, [self.max_dim, self.max_dim, self.max_dim] ).place(self.sdf_normal )
         
        

    def build(self):
        self.max_boundary.from_numpy(self.maxboundarynp)
        self.min_boundary.from_numpy(self.minboundarynp)
        self.delta.from_numpy(self.deltanp)
        self.vertex.from_numpy(self.vertex_np)
        self.bvh.setup_vertex(self.vertex, 0)

        self.voxelize()
        self.make_sdf()
    
        self.write_volume()
        #self.write_sdf()
        self.write_bvh()


    def export(self):
        find_pos = self.filename.find('.')+1
        filename = self.filename[0: find_pos]

        fo = open("model/"+filename+"rigid", "w")
        print ("source ", self.model_filename, file = fo)
        print ("space %f" %   (self.space), file = fo)
        print ("num %d" %   (self.particle_num.to_numpy()[0]), file = fo)

        volume_np = self.volume_map.to_numpy()
        sdf_normal_np = self.sdf_normal.to_numpy()
        for i in range(self.max_dim):
            for j in range(self.max_dim):
                for k in range(self.max_dim):
                    if volume_np[i,j,k] > 0:
                        print("p %f %f %f %f %f %f %f" %   (self.minboundarynp[0,0]+(float(i)+0.5)* self.deltanp[0,0], self.minboundarynp[0,1]+(float(j)+0.5)* self.deltanp[0,1], \
                            self.minboundarynp[0,2]+(float(k)+0.5)* self.deltanp[0,2], sdf_normal_np[i,j,k,0], sdf_normal_np[i,j,k,1], sdf_normal_np[i,j,k,2], self.sdf_np[i,j,k]
                            ), file = fo)
                
        fo.close()





    def cross(self, left, right):
        return [left[1] * right[2] - left[2] * right[1] , left[2] * right[0]  - left[0] *right[2] , left[0] * right[1]  - left[1] * right[0] ]


    def length(self, v):
        return math.sqrt(v[0]*v[0] + v[1]*v[1] + v[2]*v[2]) 


    def minus(self, left, right):
        return [left[0]-right[0], left[1]-right[1], left[2]-right[2]]


    def add(self, left, right):
        return [left[0]+right[0], left[1]+right[1], left[2]+right[2]]



    def mul(self, left, right):
        return [left[0]*right[0], left[1]*right[1], left[2]*right[2]]




    def write_bvh(self):
        fo = open("bvh-"+self.filename, "w")
        vertex_index = 1
        for i in range(self.bvh.node_count):
            is_leaf = int(self.bvh.compact_node_np[i][0]) & 0x0001
            if is_leaf == 0:
                min_v3 = [self.bvh.compact_node_np[i][3],self.bvh.compact_node_np[i][4],self.bvh.compact_node_np[i][5]]
                max_v3 = [self.bvh.compact_node_np[i][6],self.bvh.compact_node_np[i][7],self.bvh.compact_node_np[i][8]]
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


    def write_volume(self):
        fo = open("vol-"+self.filename, "w")
        vertex_index = 1
        volume_np = self.volume_map.to_numpy()
        min_v3 = [0.0,0.0,0.0]
        max_v3 = [0.0,0.0,0.0]
        for i in range(self.max_dim):
            for j in range(self.max_dim):
                for k in range(self.max_dim):
                    if volume_np[i,j,k] > 0:
                        min_v3[0] = self.minboundarynp[0,0] + self.deltanp[0,0]*float(i)
                        max_v3[0] = self.minboundarynp[0,0] + self.deltanp[0,0]*float(i+1)
                        min_v3[1] = self.minboundarynp[0,1] + self.deltanp[0,1]*float(j)
                        max_v3[1] = self.minboundarynp[0,1] + self.deltanp[0,1]*float(j+1)
                        min_v3[2] = self.minboundarynp[0,2] + self.deltanp[0,2]*float(k)
                        max_v3[2] = self.minboundarynp[0,2] + self.deltanp[0,2]*float(k+1)                  

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



    def write_sdf(self):
        sdf_normal_np = self.sdf_normal.to_numpy()
        volume_np = self.volume_map.to_numpy()
        #print(sdf_normal_np)
        fo = open("sdf-"+self.filename, "w")
        for i in range(self.max_dim):
            for j in range(self.max_dim):
                for k in range(self.max_dim):
                    if volume_np[i,j,k] != 0:
                        print ("%f %f %f %f" % (self.sdf_np[i,j,k],sdf_normal_np[i,j,k][0],sdf_normal_np[i,j,k][1],sdf_normal_np[i,j,k][2]), file = fo)
        fo.close()

    ############algrithm##############

    @ti.func
    def intersect_prim(self, origin, direction, primitive_id):
        hit_t     = UF.INF_VALUE
        hit_pos   = ti.Vector([0.0, 0.0, 0.0])


        hit_t, u,v = self.intersect_tri(origin, direction, primitive_id)
        if hit_t < UF.INF_VALUE:
            ver_index  = 3* primitive_id
            
            a = 1.0 - u-v
            b = u
            c = v

            v1 = self.vertex[ver_index+0] 
            v2 = self.vertex[ver_index+1]
            v3 = self.vertex[ver_index+2]
            hit_pos  = a*v1 + b*v2 + c*v3  

        return hit_t, hit_pos

    @ti.func
    def intersect_tri(self, origin, direction, primitive_id):
        # https://www.scratchapixel.com/lessons/3d-basic-rendering/ray-tracing-rendering-a-triangle/ray-triangle-intersection-geometric-solution
        t = UF.INF_VALUE
        u = 0.0
        v = 0.0
        
        vertex_id = 3 * primitive_id
        v0 = self.vertex[vertex_id+0] 
        v1 = self.vertex[vertex_id+1]
        v2 = self.vertex[vertex_id+2]
        E1 = v1 - v0
        E2 = v2 - v0

        P = direction.cross(E2)
        det = E1.dot(P)

        T = ti.Vector([0.0, 0.0, 0.0]) 
        if( det > 0.0 ):
            T = origin - v0
        else:
            T = v0 - origin
            det = -det

        if( det > 0.0 ):
            u = T.dot(P)
            if (( u >= 0.0) & (u <= det )):
                Q = T.cross(E1)
                v = direction.dot(Q)
                if((v >= 0.0) & (u + v <= det )):
                    t = E2.dot(Q)
                    fInvDet = 1.0 / det
                    t *= fInvDet
                    u *= fInvDet
                    v *= fInvDet
        return t,u,v

    # to cal detail intersect
    @ti.func
    def closet_hit(self, origin, direction, stack, i,j, MAX_SIZE):
        hit_t       = UF.INF_VALUE
        hit_pos     = ti.Vector([0.0, 0.0, 0.0]) 
        hit_prim    = -1

        stack[i,j, 0]  = 0
        stack_pos       = 0
        while (stack_pos >= 0) & (stack_pos < MAX_SIZE):
            #pop
            node_index = stack[i, j, stack_pos]
            stack_pos  = stack_pos-1

            if UF.get_compact_node_type(self.bvh.compact_node, node_index) == SCD.IS_LEAF:
                prim_index          = UF.get_compact_node_prim(self.bvh.compact_node, node_index)
                t, pos        = self.intersect_prim(origin, direction, prim_index)
                if ( t < hit_t ) & (t > 0.0):
                    hit_t       = t
                    hit_pos     = pos
                    hit_prim    = prim_index
            else:
                min_v,max_v = UF.get_compact_node_min_max(self.bvh.compact_node, node_index)
                if UF.slabs(origin, direction,min_v,max_v) == 1:
                    left_node  = node_index+1
                    right_node = UF.get_compact_node_offset(self.bvh.compact_node, node_index)
                    #push
                    stack_pos              += 1
                    stack[i, j, stack_pos] = left_node
                    stack_pos              += 1
                    stack[i, j, stack_pos] = right_node

        if stack_pos == MAX_SIZE:
            print("overflow, need larger stack")

        return  hit_t, hit_pos, hit_prim




    def sample(self, i, j, k):
        i = max(0, min(i, self.max_dim-1))
        j = max(0, min(j, self.max_dim-1))
        k = max(0, min(k, self.max_dim-1))
        return self.volume_map_np[i,j,k]

    @ti.kernel
    def voxelize(self):
        
        #Z轴
        for i,j in ti.ndrange(self.max_dim,self.max_dim):
            origin      = self.delta[0] * ti.Vector([float(i)+0.5 ,float(j)+0.5, 0.0]) + self.min_boundary[0]

            dir         = ti.Vector([0.0,0.0,1.0])
            
            inside      = 0
            while 1 :
                hit_t, hit_pos, hit_prim = self.closet_hit(origin, dir, self.stack, i,j, MAX_STACK_SIZE)
                if hit_prim <0:
                    break
                
                zpos = origin.z + hit_t
                zhit = (zpos-self.min_boundary[0][2])/self.delta[0][2]
                
                cur_z = int(ts.floor((origin.z - self.min_boundary[0][2]) /self.delta[0][2] +0.5 ))
                zend  = int(min(zhit+0.5, self.max_dim-1))
                
                while cur_z < zend:
                    if  inside:
                        self.volume_map[i,j, int(cur_z)] = 1
                        self.particle_num[0] += 1
                    else:
                        self.volume_map[i,j, int(cur_z)] = 0

                    #if (i==13)&(j==19):
                    #    print(i,j,cur_z, zend, origin, self.min_boundary[0] + self.delta[0]*float(cur_z))

                    cur_z += 1
                origin  = hit_pos + ti.Vector([0.0, 0.0,UF.EPS])
                inside = 1-inside

    @ti.func
    def SampleSdf(self, i,j,k):
        i = max(0, min(i, self.max_dim-1))
        j = max(0, min(j, self.max_dim-1))
        k = max(0, min(k, self.max_dim-1))
        return self.sdf[i,j,k]


    @ti.kernel
    def SampleSdfGrad(self):
        for i,j,k in self.sdf_normal:
            x0 = max(i-1, 0)
            x1 = min(i+1, self.max_dim-1)
            y0 = max(j-1, 0)
            y1 = min(j+1, self.max_dim-1)
            z0 = max(k-1, 0)
            z1 = min(k+1, self.max_dim-1)

            dx = (self.SampleSdf(x1,j,k)-self.SampleSdf(x0,j,k)) * float(self.max_dim)*0.5 
            dy = (self.SampleSdf(i,y1,k)-self.SampleSdf(i,y0,k)) * float(self.max_dim)*0.5 
            dz = (self.SampleSdf(i,j,z1)-self.SampleSdf(i,j,z0)) * float(self.max_dim)*0.5 

            #if ts.isnan(dx) or ts.isnan(dx) or ts.isnan(dx):
            
            self.sdf_normal[i,j,k] = ti.Vector([dx,dy,dz]).normalized(0.0001)
            if ts.isnan(self.sdf_normal[i,j,k].x):
                print(x0,y0,z0,dx,dy,dz)


    def make_sdf(self):
        self.volume_map_np = self.volume_map.to_numpy()
        self.sdf_np = self.sdf.to_numpy()
        self.heap = []

        for i in range(self.max_dim):
            for j in range(self.max_dim):
                for k in range(self.max_dim):

                    center  = (self.sample(i,j,k)!=0)
                    minDist = UF.INF_VALUE

                    #print("*****************************")
                    for r  in range(i-1, i+2):
                        for s  in range(j-1, j+2):
                            for t  in range(k-1, k+2):
                                if ( (self.sample(r,s,t) !=0)!= center):
                                    dx = i-r
                                    dy = j-s
                                    dz = k-t
                                    minDist = min(math.sqrt(float(dx*dx + dy*dy + dz*dz))*0.5, minDist) 
                                    #print(dx,dy,dz,minDist)
                    #print(i,j,k,minDist, self.sample(i,j,k))
                    self.sdf_np[i,j,k] = UF.INF_VALUE
                    if minDist != UF.INF_VALUE:
                        heappush(self.heap , (minDist,i,j,k,i,j,k))
        
        if len(self.heap  )==0:
            return
        # 0 1 2 3 4  5  6 
        # d i j k si sj sk
        while len(self.heap ) > 0:
            c = heappop(self.heap ) 
            if self.sdf_np[c[1],c[2],c[3] ] == UF.INF_VALUE:
                self.sdf_np[c[1],c[2],c[3] ] = c[0]
                xmin = max(c[1] -1, 0)
                ymin = max(c[2] -1, 0)
                zmin = max(c[3] -1, 0)
                xmax = min(c[1] +1, self.max_dim-1)
                ymax = min(c[2] +1, self.max_dim-1)
                zmax = min(c[3] +1, self.max_dim-1)

                for x  in range(xmin, xmax+1):
                    for y in range(ymin, ymax+1):
                        for z in range(zmin, zmax+1):
                            if (x != c[1]) &(y != c[2]) & (z != c[3]) & (self.sdf_np[x,y,z] == UF.INF_VALUE):
                                dx = x - c[4]
                                dy = y - c[5]
                                dz = z - c[6]
                                d  = math.sqrt(dx*dx + dy*dy + dz*dz) + self.sdf_np[c[4],c[5],c[6] ]
                                heappush(self.heap , (d, x,y,z,c[4], c[5], c[6]))

        scale = 1.0 / float(self.max_dim)
        for i  in range(self.max_dim):
            for j in range(self.max_dim):
                for k in range(self.max_dim):
                    if self.volume_map_np[i,j,k] == 0:
                        self.sdf_np[i,j,k] *= scale
                    else:
                        self.sdf_np[i,j,k] *= -scale
        self.sdf.from_numpy(self.sdf_np)
        self.SampleSdfGrad()