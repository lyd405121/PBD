from ast import If
#from errno import EL
import sys
#from typing_extensions import Self
sys.path.append("accel")
import taichi as ti
import numpy as np
import SceneData as SCD
import UtilsFunc as UF
import Mesh
import math
import LBvh as LBvh
import UniformGrid as Grid

MAX_STACK_SIZE =  32
GROUND_INDEX   = 8888
USE_ONE_SIDE_PARTICLE = 0
@ti.data_oriented
class Simulation:
    def __init__(self):
        self.radius                 = 0.0
        self.space                  = 0.0
        self.source_filename        = []
        self.num_of_object          = 0
        self.mesh_count             = 0
        self.particle_num           = 0
        self.static_shape_count     = 0
        self.rigid_num              = 0
        self.rigid_par_num          = 0

        self.maxboundarynp          = np.ones(shape=(1,3), dtype=np.float32)
        self.minboundarynp          = np.ones(shape=(1,3), dtype=np.float32)

        self.scene_maxnp          = np.ones(shape=(1,3), dtype=np.float32)
        self.scene_minnp          = np.ones(shape=(1,3), dtype=np.float32)


        self.centrenp               = np.ones(shape=(1,3), dtype=np.float32)            
        self.sizenp                 = np.ones(shape=(1,3), dtype=np.float32)  
        for i in range(3):
            self.maxboundarynp[0, i] = -UF.INF_VALUE
            self.minboundarynp[0, i] = UF.INF_VALUE
            self.scene_maxnp[0, i]   = 50.0
            self.scene_minnp[0, i]   = -50.0
        self.scene_minnp[0, 1]   = -0.1

        self.shape_cpu              = []
        self.shape_pos_cpu          = []
        self.shape_quat_cpu         = []
        self.shape_scale_cpu        = []
        self.shape                  = ti.Vector.field(SCD.SHA_VEC_SIZE, dtype=ti.f32)  


        self.rigidCoefficients_cpu  = []
        self.rigidOffsets_cpu       = []
        self.rigidIndices_cpu       = []
        self.phases_cpu             = []
        self.velocity_cpu           = []
        self.position_cpu           = []
        self.sdf_normal_cpu        = []
        self.mass_cpu               = []
        self.type_cpu               = []

        self.rigidCoefficients      = ti.field( dtype=ti.f32) 
        self.rigidOffsets           = ti.field( dtype=ti.i32) 
        self.sdf_normal             = ti.Vector.field(4, dtype=ti.f32)  
        self.rigidIndices           = ti.field( dtype=ti.i32) 
        self.rigidSum               = ti.field(dtype=ti.f32) 

        self.phases                 = ti.field( dtype=ti.i32) 
        self.mass                   = ti.field( dtype=ti.f32)
        self.velocity               = ti.Vector.field(3, dtype=ti.f32)  
        self.position               = ti.Vector.field(3, float)
        self.positionNext           = ti.Vector.field(3, float)
        self.positionZero           = ti.Vector.field(3, dtype=ti.f32)
        self.positionDelta          = ti.Vector.field(3, dtype=ti.f32)  
        self.color                  = ti.Vector.field(4, dtype=ti.f32)
        self.stack                  = ti.field( dtype=ti.i32)
        self.type                   = ti.field( dtype=ti.i32)

        self.collision_id              = ti.field( dtype=ti.i32)
        self.rigidT      = ti.Vector.field(3, dtype=ti.f32)  
        self.rigidR      = ti.Matrix.field(3, 3, dtype=ti.f32)  
        self.rigidCentre = ti.Vector.field(3, dtype=ti.f32)  
        self.rigidCentreZero = ti.Vector.field(3, dtype=ti.f32)  


        self.gravity                = ti.Vector.field(3, dtype=ti.f32, shape=(1)) 
        self.deltaT                 = ti.field(dtype=ti.f32, shape=(1)) 
        self.invDeltaT              = ti.field(dtype=ti.f32, shape=(1)) 
        self.frame                  = ti.field(dtype=ti.i32, shape=(1)) 



        self.frame_cpu              = np.zeros(shape=(1), dtype=np.int32)  
        self.gravity_np             = np.zeros(shape=(1,3), dtype=np.float32)  
        self.deltaT_np              = np.zeros(shape=(1), dtype=np.float32)  
        self.invDeltaT_np           = np.zeros(shape=(1), dtype=np.float32) 

    def load_particle_file(self, filename, rigidCoff, velocity, quat, is_rigid, offset,mass):
        fo = open(filename, "r")
        lines = fo.readlines()
        for line in lines:
            line = line.strip()
            one = line.split(' ')

            if(one[0] == "source"):
                self.source_filename.append(one[1])
            elif(one[0] == "space"):
                self.space = float(one[1])
                self.radius = self.space *0.5
            elif(one[0] == "num"):
                continue
            else:
                if is_rigid:
                    self.rigidIndices_cpu.append(len(self.position_cpu))
                    self.sdf_normal_cpu.append([float(one[4]),float(one[5]),float(one[6]),float(one[7])])
                    self.type_cpu.append(UF.PARTICLE_RIGID)
                else:
                    self.type_cpu.append(UF.PARTICLE_FLUID)

                quat = self.quat_normalize(quat)
                pos = [float(one[1]),float(one[2]),float(one[3])]
                self.position_cpu.append(self.transform(pos, quat,offset ))
                
                self.phases_cpu.append(self.num_of_object)
                self.velocity_cpu.append(velocity)
                self.mass_cpu.append(mass)
                self.particle_num += 1
        


        if is_rigid:               
            self.rigidOffsets_cpu.append(len(self.rigidIndices_cpu ))
            self.rigidCoefficients_cpu.append(rigidCoff)
            self.rigid_num              += 1
        self.num_of_object+=1
        fo.close()

    
    def add_static_shape(self, shape, pos, quat, scale):
        quat = self.quat_normalize(quat)
        self.shape_cpu.append(shape)
        if shape.getType() !=  UF.SHAPE_MESH:
            shape.setID(self.static_shape_count)
        self.shape_pos_cpu.append(pos)
        self.shape_quat_cpu.append(quat)
        self.shape_scale_cpu.append(scale)
        self.static_shape_count     += 1
    
    def add_static_mesh(self, filename, pos, quat, scale):
        self.mesh                   = Mesh.mesh()
        id,min_v3,max_v3= self.mesh.load_obj(filename)
        shape = SCD.Shape()
        shape.setType(UF.SHAPE_MESH)
        shape.setID(id)
        shape.setMin([min_v3[0,0],min_v3[0,1],min_v3[0,2]])
        shape.setMax([max_v3[0,0],max_v3[0,1],max_v3[0,2]])
        self.add_static_shape(shape, pos, quat, scale)
        self.mesh_count  += 1

    def quat_normalize(self, quat):
        d = 0.0
        for i in range(4):
            d += quat[i]*quat[i]
        d = 1.0 / math.sqrt(d)
        return [quat[0]*d,quat[1]*d,quat[2]*d,quat[3]*d]

    def cross(self, l, r):
        return [l[1] * r[2] - l[2] * r[1],l[2] * r[0] - l[0] * r[2],l[0] * r[1] - l[1] * r[0]]

    def transform(self, vec, quat, offset):
        qVec = [quat[0], quat[1], quat[2]]
        cross1 = self.cross(qVec, vec)
        cross2 = self.cross(qVec, cross1)
        ret = [0.0,0.0,0.0]
        for i in range(3):
            ret[i] = vec[i] + (cross1[i] * quat[3] + cross2[i]) * 2.0 +offset[i]
        return ret
    
    def build(self):
        self.rigidCoefficients_np       = np.ones(shape=(self.rigid_num), dtype=np.float32)
        self.rigidOffsets_np            = np.ones(shape=(self.rigid_num), dtype=np.int32)
        self.rigidIndices_np            = np.ones(shape=(self.rigid_num), dtype=np.int32)
        self.phases_np                  = np.ones(shape=(self.particle_num), dtype=np.int32)
        self.velocity_np                = np.ones(shape=(self.particle_num,3), dtype=np.float32)
        self.position_np                = np.ones(shape=(self.particle_num,3), dtype=np.float32)
        self.mass_np                    = np.ones(shape=(self.particle_num), dtype=np.float32)
        self.type_np                    = np.ones(shape=(self.particle_num), dtype=np.int32)
        self.sdf_normal_np              = np.ones(shape=(self.particle_num,4), dtype=np.float32)

        self.shape_np                   = np.ones(shape=(self.static_shape_count,SCD.SHA_VEC_SIZE), dtype=np.float32)
        self.shape_pos_np               = np.ones(shape=(self.static_shape_count,3), dtype=np.float32)
        self.shape_quat_np              = np.ones(shape=(self.static_shape_count,4), dtype=np.float32)
        self.shape_scale_np             = np.ones(shape=(self.static_shape_count,3), dtype=np.float32)


        for i in range( self.rigid_num):
            self.rigidCoefficients_np[i] = self.rigidCoefficients_cpu[i]
            self.rigidOffsets_np[i]      = self.rigidOffsets_cpu[i]


        for i in range( self.rigid_num):
            self.rigidIndices_np[i]      = self.rigidIndices_cpu[i]


        for i in range( self.particle_num):
            self.phases_np[i] = self.phases_cpu[i]
            self.mass_np[i] = self.mass_cpu[i]
            self.type_np[i] = self.type_cpu[i]
            for j in range(3):
                self.position_np[i,j] = self.position_cpu[i][j]
                self.velocity_np[i,j] = self.velocity_cpu[i][j]
            for j in range(4):
                self.sdf_normal_np[i,j] = self.sdf_normal_cpu[i][j]


        for i in range( self.static_shape_count):
            self.shape_cpu[i].fillStruct(self.shape_np, i)
            id    = self.shape_cpu[i].getID()

            minv3, maxv3 =self.shape_cpu[i].getMinMax(self.shape_pos_cpu[id], self.shape_quat_cpu[id], self.shape_scale_cpu[id])
            for j in range(3):
                id = self.shape_cpu[i].getID()
                self.shape_pos_np[id,j]     = self.shape_pos_cpu[id][j]
                self.shape_quat_np[id,j]    = self.shape_quat_cpu[id][j]
                self.shape_scale_np[id,j]   = self.shape_scale_cpu[id][j]
                self.maxboundarynp[0,j]     = max(self.maxboundarynp[0,j], maxv3[j])
                self.minboundarynp[0,j]     = min(self.minboundarynp[0,j], minv3[j])
            self.shape_quat_np[i,3]     = self.shape_quat_cpu[id][3]

        self.centrenp = (self.maxboundarynp+self.minboundarynp) * 0.5
        self.sizenp   = (self.maxboundarynp-self.minboundarynp)

        print("*****particle:%d staic-shape:%d mesh:%d**************"%(self.particle_num, self.static_shape_count, self.mesh_count))
        print("****min:",self.minboundarynp, "***max:",self.maxboundarynp ,"******")

        if self.mesh_count > 0:
            self.mesh.setup_data()

        if self.static_shape_count> 0:
            self.static_bvh = LBvh.Bvh(self.static_shape_count, self.minboundarynp, self.maxboundarynp)



        self.grid = Grid.UniformGrid( self.space, self.particle_num*2, self.scene_minnp, self.scene_maxnp)

        

        ti.root.dense(ti.i,  self.rigid_num ).place(self.rigidCoefficients)
        ti.root.dense(ti.i,  self.rigid_num ).place(self.rigidOffsets)
        ti.root.dense(ti.i,  self.rigid_num ).place(self.rigidT)
        ti.root.dense(ti.i,  self.rigid_num ).place(self.rigidR)
        ti.root.dense(ti.i,  self.rigid_num ).place(self.rigidCentre)
        ti.root.dense(ti.i,  self.rigid_num ).place(self.rigidCentreZero)
        ti.root.dense(ti.i,  self.rigid_num ).place(self.rigidIndices)
        ti.root.dense(ti.i,  self.rigid_num ).place(self.rigidSum)

        ti.root.dense(ti.i,  self.particle_num ).place(self.phases)
        ti.root.dense(ti.i,  self.particle_num ).place(self.velocity)
        ti.root.dense(ti.i,  self.particle_num ).place(self.mass)
        ti.root.dense(ti.i,  self.particle_num ).place(self.position)
        ti.root.dense(ti.i,  self.particle_num ).place(self.positionNext)
        ti.root.dense(ti.i,  self.particle_num ).place(self.positionZero)
        ti.root.dense(ti.i,  self.particle_num ).place(self.positionDelta)
        ti.root.dense(ti.i,  self.particle_num ).place(self.collision_id)
        ti.root.dense(ti.i,  self.particle_num ).place(self.type)
        ti.root.dense(ti.i,  self.particle_num ).place(self.sdf_normal)

        ti.root.dense(ti.i,  self.particle_num ).place(self.color)
        ti.root.dense(ti.i,  self.static_shape_count ).place(self.shape)
        ti.root.dense(ti.ij, [self.particle_num, MAX_STACK_SIZE] ).place(self.stack  )

        if len(self.shape_pos_cpu) > 0:
            self.shape_pos              = ti.Vector.field(3, dtype=ti.f32)
            self.shape_quat             = ti.Vector.field(4, dtype=ti.f32)
            self.shape_scale            = ti.Vector.field(3, dtype=ti.f32)
            ti.root.dense(ti.i,  self.static_shape_count ).place(self.shape_pos)
            ti.root.dense(ti.i,  self.static_shape_count ).place(self.shape_quat)
            ti.root.dense(ti.i,  self.static_shape_count ).place(self.shape_scale)

        self.rigidCoefficients.from_numpy(self.rigidCoefficients_np)
        self.rigidOffsets.from_numpy(self.rigidOffsets_np)
        self.sdf_normal.from_numpy(self.sdf_normal_np)
        self.rigidIndices.from_numpy(self.rigidIndices_np)
        self.phases.from_numpy(self.phases_np)
        self.mass.from_numpy(self.mass_np)
        self.velocity.from_numpy(self.velocity_np)
        self.position.from_numpy(self.position_np)
        self.positionZero.from_numpy(self.position_np)
        self.type.from_numpy(self.type_np)

        self.deltaT_np[0]    =  0.15
        self.invDeltaT_np[0] =  1.0/self.deltaT_np[0] 
        self.gravity_np[0,1] = -0.2
        self.gravity.from_numpy(self.gravity_np )
        self.deltaT.from_numpy(self.deltaT_np )
        self.invDeltaT.from_numpy(self.invDeltaT_np )

        self.shape.from_numpy(self.shape_np)
        if len(self.shape_pos_cpu) > 0:
            self.shape_pos.from_numpy(self.shape_pos_np)
            self.shape_quat.from_numpy(self.shape_quat_np)
            self.shape_scale.from_numpy(self.shape_scale_np)
        else:
            print("no static nboundary") 

        if self.mesh_count > 0:
            self.mesh.build()

        if self.static_shape_count> 0:
            self.static_bvh.setup_shape(self.shape, self.shape_pos, self.shape_quat, self.shape_scale) 
        #self.grid.export_debug_grid(self.position, self.particle_num)




    @ti.func
    def intersect_shape(self, pos, shape_id, i):
        shape_type = UF.get_shape_type(self.shape, shape_id)
        hit_pos    = pos

        shape_pos   = self.shape_pos[shape_id]
        shape_scale = self.shape_scale[shape_id]
        shape_quat  = self.shape_quat[shape_id]

        #print(shape_pos,shape_scale,shape_quat)

        delta_pos   = pos - shape_pos
        delta_dis   = delta_pos.norm()
        

        if shape_type == UF.SHAPE_SPHERE :
            #only support shpere
            r = UF.get_shape_radius(self.shape, shape_id) * shape_scale[0]
            if delta_dis < (self.radius+r):
                delta_pos  =  delta_pos / delta_dis
                hit_pos = shape_pos + delta_pos * (self.radius+r)
                 
                
        elif shape_type == UF.SHAPE_QUAD:
            if delta_dis < self.radius:
                v1 = UF.transform_vec(UF.get_shape_v1(self.shape, shape_id), ti.Vector([0.0,0.0,0.0]),shape_quat,shape_scale)
                v2 = UF.transform_vec(UF.get_shape_v2(self.shape, shape_id), ti.Vector([0.0,0.0,0.0]),shape_quat,shape_scale)
                normal = v1.cross(v2)
                ndotd = normal.dot(delta_pos)
                if abs(ndotd) < self.radius:
                    hit_pos = pos + normal * ndotd

        elif shape_type == UF.SHAPE_MESH:
            mesh_id = UF.get_shape_id(self.shape, shape_id)
            hit_pos = self.mesh.interset_mesh(mesh_id, pos, self.radius, shape_pos, shape_quat, shape_scale, self.stack, i, MAX_STACK_SIZE)
            

        return  hit_pos


    @ti.kernel
    def sim_prepare(self):
        for i in range(self.rigid_num):
            self.rigidCentre[i] = ti.Vector([0.0,0.0,0.0])
            self.rigidSum[i]     = 0.0

        for i in range(self.particle_num):
            self.color[i] = ti.Vector([0.5,0.5,0.5,1.0])
            self.collision_id[i] = -1

    @ti.kernel
    def get_centre(self, pos:ti.template(), centre:ti.template()):
        for i in range(self.particle_num):
            phase = self.phases[i] 
            centre[phase] += pos[i] * self.mass[i] 
            self.rigidSum[phase] += self.mass[i] 

        for i in range(self.rigid_num):
            centre[i] = centre[i] / self.rigidSum[i]

    @ti.kernel
    def shape_match(self):
        #https://igl.ethz.ch/projects/ARAP/svd_rot.pdf
        #https://blog.csdn.net/kfqcome/article/details/9358853

        for i in range(self.rigid_num):
            self.rigidR[i] = ti.Matrix([[0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]])
            self.rigidT[i] = ti.Vector([0.0,0.0,0.0])

        for i in range(self.particle_num):
            phase = self.phases[i] 
            x     = self.positionZero[i]-self.rigidCentreZero[phase]
            y     = self.positionNext[i]-self.rigidCentre[phase]

            #cal S
            self.rigidR[phase] += self.mass[i] * x.outer_product(y)
            
            #print(i, phase)
            #if  i==0:
            #    print(self.rigidCentreZero[phase], self.rigidCentre[phase])


        for i in range(self.rigid_num):
            U,S,V = ti.svd(self.rigidR[i])
            M = V@U.transpose()

            self.rigidR[i] = V@ti.Matrix([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, M.determinant()]])@U.transpose()
            self.rigidT[i] = self.rigidCentre[i] - self.rigidR[i] @ self.rigidCentreZero[i]

            #self.rigidR[i] = ti.Matrix([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]])
            #self.rigidT[i] = self.rigidCentre[i] - self.rigidCentreZero[i]
    
        for i in range(self.particle_num):
            phase = self.phases[i] 
            
            self.positionNext[i] =   self.rigidR[phase] @ self.positionZero[i]  + self.rigidT[phase] 


    # to cal detail intersect
    @ti.kernel
    def get_contact(self):
        for i in range(self.particle_num):
            particle_pos         = self.positionNext[i]
            self.stack[i, 0]  = 0
            stack_pos    = 0
            phase = self.phases[i]
            
            self.positionDelta[i] = ti.Vector([0.0,0.0,0.0])

            '''
            #static mesh test
            while (stack_pos >= 0) & (stack_pos < MAX_STACK_SIZE):
                #pop
                node_index = self.stack[i, stack_pos]
                stack_pos  = stack_pos-1

                offset     = UF.get_compact_node_offset(self.static_bvh.compact_node, node_index)
   
                if offset < 0:
                    shape_index          = UF.get_compact_node_prim(self.static_bvh.compact_node, node_index)
                    self.positionDelta[i] = self.intersect_shape(particle_pos, shape_index, i)-particle_pos
 
                    if self.positionDelta[i].norm() > 0.001:
                        self.color[i] = ti.Vector([1.0,0.0,0.0,1.0])
                        self.collision_id[phase] = shape_index
                else:
                    min_v,max_v = UF.get_compact_node_min_max(self.static_bvh.compact_node, node_index)

                                    
                    if UF.point_aabb_dis(particle_pos, min_v,max_v) < self.radius:
                        left_node  = node_index+1
                        right_node = offset
                        #push
                        stack_pos              += 1
                        self.stack[i,  stack_pos] = left_node
                        stack_pos              += 1
                        self.stack[i,  stack_pos] = right_node
            '''

                
            if stack_pos == MAX_STACK_SIZE:
                print("overflow, need larger stack")
            

            #ground test
            if self.positionNext[i].y < self.radius:
                self.positionDelta[i].y = self.radius - self.positionNext[i].y 
                self.collision_id[i] = GROUND_INDEX
                self.color[i] = ti.Vector([1.0,0.0,0.0,1.0])
  

            #neighbour test
            neighbour_count = self.grid.neighbour[i,0]                  
            while neighbour_count >= 0:
                neighbour_index = self.grid.neighbour[i,neighbour_count] 
                if (neighbour_index != i) & (phase != self.phases[neighbour_index]):
                    xj  = self.positionNext[neighbour_index]
                    xij = xj - particle_pos
                    d   = xij.norm() 
                    if d < self.radius*2.0:
                        self.color[i] = ti.Vector([0.0,1.0,0.0,1.0])
                        wi = 1.0 / self.mass[i]
                        wj = 1.0 / self.mass[neighbour_index]

                        if USE_ONE_SIDE_PARTICLE:
                            nij = ti.Vector([self.sdf_normal[i].x,self.sdf_normal[i].y,self.sdf_normal[i].z])
                            if abs(self.sdf_normal[i].w) < abs(self.sdf_normal[neighbour_index].w):
                                nij = -ti.Vector([self.sdf_normal[neighbour_index].x,self.sdf_normal[neighbour_index].y,self.sdf_normal[neighbour_index].z])

                            nij = self.rigidR[phase] @ nij
                            condi = xij.dot(nij)
                            if condi > 0.0:
                                self.positionDelta[i] +=  -wi/(wi+wj) * (self.radius*2.0 - d ) * xij
                            else:
                                self.positionDelta[i] +=  -wi/(wi+wj) * (self.radius*2.0 - d ) * (xij-2.0*condi*nij)                     
                        else:

                            self.positionDelta[i] += -wi/(wi+wj) * (self.radius*2.0 - d ) * xij




                neighbour_count-=1

        for i in range(self.particle_num):
            self.positionNext[i] += self.positionDelta[i]


    @ti.kernel
    def get_color(self):
        for i in range(self.particle_num):
            grid_index     = self.grid.particle[i][0]
            self.color[i]  = ti.Vector([0.2,0.2,0.2,1.0])
            for s in range(-1,2):
                for j in range(-1,2):
                    for k in range(-1,2):
                        neighbour_cell_index = self.grid.get_neighbour_cell_index(grid_index, s,j,k)
                        if  (neighbour_cell_index != -1):
                            particle_start = self.grid.grid[neighbour_cell_index]
                            particle_end   = self.grid.grid[neighbour_cell_index+1]
                            self.color[i] += ti.Vector([0.1,0.1,0.1,0.0]) * (particle_end- particle_start) 


    def init_sim(self):
        self.frame_cpu[0]  = 0
        self.get_centre(self.positionZero,self.rigidCentreZero)

    def sim_one_frame(self):
        self.frame_cpu[0]       += 1
        self.frame.from_numpy(self.frame_cpu )  

        if self.frame_cpu[0] < 2000:

            #prepare
            self.sim_prepare()
            self.update_gravity()


            #substep
            for i in range(3):
                if(Grid.DEBUG_MODE):
                    self.grid.build_grid(self.positionNext, Grid.DEBUG_PARTICLE_NUM) 
                else:
                    self.grid.build_grid(self.positionNext, self.particle_num) 
                self.get_contact()

            self.get_centre(self.positionNext,self.rigidCentre)
            self.shape_match()

            #update final vel
            self.update_vel()

    @ti.kernel
    def update_gravity(self):
        for i in self.position:
            self.velocity[i]    += self.gravity[0] * self.deltaT[0]
            self.positionNext[i] = self.position[i]  + self.velocity[i] * self.deltaT[0]

    @ti.kernel
    def update_vel(self):
        for i in self.position:
            #phase = self.phases[i]
            if(self.type[i] == UF.PARTICLE_RIGID):
                
                self.velocity[i] = (self.positionNext[i] - self.position[i])*self.invDeltaT[0]
                #this is sleeping speed, prevent tiny move
                for j in range(3):
                    if abs(self.velocity[i][j]) < 0.05*self.deltaT[0]:
                        self.velocity[i][j] = 0.0
                #this is vel damping
                if self.collision_id[i] >= 0:
                    self.velocity[i] =  ti.Vector([0.0,0.0,0.0])
                
            else:
                self.velocity[i] = (self.positionNext[i] - self.position[i])*self.invDeltaT[0]
            self.position[i] = self.positionNext[i]

        
    def export(self):
        fo_bvh      = open("static_bvh-sim.obj", "w")
        fo_particle = open("par-sim.obj", "w")
        vertex_index_bvh = 1
        vertex_index_par = 1
        for i in range(self.static_bvh.node_count):
            is_leaf = int(self.static_bvh.compact_node_np[i][0]) & 0x0001
            index   = int(self.static_bvh.compact_node_np[i][1])
            min_v3 = [self.static_bvh.compact_node_np[i][2],self.static_bvh.compact_node_np[i][3],self.static_bvh.compact_node_np[i][4]]
            max_v3 = [self.static_bvh.compact_node_np[i][5],self.static_bvh.compact_node_np[i][6],self.static_bvh.compact_node_np[i][7]]

            if is_leaf == 1:
                id    = self.shape_cpu[index].getID()
                print ("v %f %f %f %f %f %f" %   (self.shape_pos_cpu[id][0], self.shape_pos_cpu[id][1], self.shape_pos_cpu[id][2],0.0,0.0,0.0), file = fo_particle)
                vertex_index_par += 1
            else:
                print ("v %f %f %f" %   (min_v3[0], min_v3[1], min_v3[2]), file = fo_bvh)
                print ("v %f %f %f" %   (min_v3[0], min_v3[1], max_v3[2]), file = fo_bvh)
                print ("v %f %f %f" %   (max_v3[0], min_v3[1], max_v3[2]), file = fo_bvh)
                print ("v %f %f %f" %   (max_v3[0], min_v3[1], min_v3[2]), file = fo_bvh)
                print ("v %f %f %f" %   (min_v3[0], max_v3[1], min_v3[2]), file = fo_bvh)
                print ("v %f %f %f" %   (min_v3[0], max_v3[1], max_v3[2]), file = fo_bvh)
                print ("v %f %f %f" %   (max_v3[0], max_v3[1], max_v3[2]), file = fo_bvh)
                print ("v %f %f %f" %   (max_v3[0], max_v3[1], min_v3[2]), file = fo_bvh)
                
                print ("f %d %d %d %d" %   (vertex_index_bvh+0, vertex_index_bvh+1, vertex_index_bvh+2, vertex_index_bvh+3), file = fo_bvh)
                print ("f %d %d %d %d" %   (vertex_index_bvh+4, vertex_index_bvh+5, vertex_index_bvh+6, vertex_index_bvh+7), file = fo_bvh)
                print ("f %d %d %d %d" %   (vertex_index_bvh+0, vertex_index_bvh+1, vertex_index_bvh+5, vertex_index_bvh+4), file = fo_bvh)
                print ("f %d %d %d %d" %   (vertex_index_bvh+2, vertex_index_bvh+3, vertex_index_bvh+7, vertex_index_bvh+6), file = fo_bvh)
                print ("f %d %d %d %d" %   (vertex_index_bvh+1, vertex_index_bvh+2, vertex_index_bvh+6, vertex_index_bvh+5), file = fo_bvh)
                print ("f %d %d %d %d" %   (vertex_index_bvh+0, vertex_index_bvh+4, vertex_index_bvh+7, vertex_index_bvh+3), file = fo_bvh)
                vertex_index_bvh += 8
        fo_bvh.close()
        fo_particle.close()