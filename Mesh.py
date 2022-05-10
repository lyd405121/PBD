import sys
sys.path.append("accel")
import taichi as ti
import numpy as np
import pywavefront
import SceneData as SCD
import UtilsFunc as UF
import LBvh as LBvh



@ti.data_oriented
class mesh:
    def __init__(self):
        self.mesh_vertex_cpu        = []
        self.mesh_vertex_offset     = []
        self.max_v3                 = []
        self.min_v3                 = []
        self.bvh_list               = []

        self.compact_node           = ti.Vector.field(SCD.CPNOD_VEC_SIZE, dtype=ti.f32)
        self.vertex                 = ti.Vector.field(3, dtype=ti.f32)
        self.offset                 = ti.field(dtype=ti.i32)

        self.node_count             = 0
        self.mesh_num               = 0
        self.vertex_count           = 0

        self.mesh_dict = {}


    def load_obj(self, filename):
        
        id = self.mesh_dict.get(filename)

        if id != None:
            return id,self.min_v3[id],self.max_v3[id]
        self.mesh_dict[filename] = self.mesh_num

        scene = pywavefront.Wavefront(filename)
        scene.parse() 

        vertex_count = 0
        self.mesh_vertex_offset.append(len(self.mesh_vertex_cpu))

        max_v3          = np.ones(shape=(1,3), dtype=np.float32)
        min_v3          = np.ones(shape=(1,3), dtype=np.float32)
        for i in range(3):
            max_v3[0, i] = -UF.INF_VALUE
            min_v3[0, i] = UF.INF_VALUE

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
                    max_v3[0,k]   = max(vertex[k], max_v3[0,k])
                    min_v3[0,k]   = min(vertex[k], min_v3[0,k])

                vertex_count += 1
                self.mesh_vertex_cpu.append(vertex)

        self.mesh_num       += 1
        self.vertex_count   += vertex_count-3
        self.max_v3.append(max_v3)
        self.min_v3.append(min_v3)
        return self.mesh_num-1,min_v3,max_v3


    def setup_data(self):
        self.offset_np            = np.ones(shape=(self.mesh_num), dtype=np.int32)

        for i in range(self.mesh_num):
            prim_count   = 0
            if i != self.mesh_num-1:
                prim_count = (self.mesh_vertex_offset[i+1]-self.mesh_vertex_offset[i])//3
            else:
                prim_count = (self.vertex_count-self.mesh_vertex_offset[i])//3
            self.bvh_list.append( LBvh.Bvh(prim_count, self.min_v3[i], self.max_v3[i]))
            self.offset_np[i] = self.node_count 
            self.node_count  += self.bvh_list[i].node_count
            
            
        ti.root.dense(ti.i, self.node_count    ).place(self.compact_node)
        ti.root.dense(ti.i, self.vertex_count  ).place(self.vertex)
        ti.root.dense(ti.i, self.mesh_num).place(self.offset )


    def build(self):
        #fo      = open("test-mesh.obj", "w")
        vertexnp = self.vertex.to_numpy()
        for i in range(self.vertex_count):
            #print ("v",end = ' ', file = fo)
            for j in range(3):
                vertexnp[i,j] = self.mesh_vertex_cpu[i][j]
                #print ((vertexnp[i,j]), end = ' ',file = fo)
            #print(file = fo)

        #for i in range(self.vertex_count//3):    
        #    print('f',3*i+1,3*i+2,3*i+3,file = fo)

        self.vertex.from_numpy(vertexnp)
        self.offset.from_numpy(self.offset_np)

        nodenp   = self.compact_node.to_numpy()
        for i in range(self.mesh_num):
            self.bvh_list[i].setup_vertex(self.vertex, self.mesh_vertex_offset[i])
            nodenp_i = self.bvh_list[i].compact_node.to_numpy()
            for j in range(self.bvh_list[i].node_count):
                for k in range(SCD.CPNOD_VEC_SIZE):
                    nodenp[self.offset_np[i]+j,k] = nodenp_i[j,k]
        self.compact_node.from_numpy(nodenp)
        #fo.close()

    @ti.func
    def spher_intersect_tri(self, P, r, prim_id, shape_pos, shape_quat, shape_scale):
        # hhttps://github.com/gszauer/GamePhysicsCookbook/blob/master/Code/Geometry3D.cpp
        ret     = 0
        hit_pos = P
        #print(vertex_id)
        A = UF.transform_vec(self.vertex[3*prim_id+0], shape_pos, shape_quat, shape_scale)  
        B = UF.transform_vec(self.vertex[3*prim_id+1], shape_pos, shape_quat, shape_scale) 
        C = UF.transform_vec(self.vertex[3*prim_id+2], shape_pos, shape_quat, shape_scale) 

        AB = B - A
        AC = C - A
        AP = P - A
        N  = AB.cross(AC).normalized()

        NdotAP = AP.dot(N)
        SP    = N * NdotAP
        
        if SP.norm() < r:
            S     = P - SP
            SA = A-S
            SB = B-S
            SC = C-S

            normSBC = SB.cross(SC).normalized()
            normSCA = SC.cross(SA).normalized()
            normSAB = SA.cross(SB).normalized()

            if (normSBC.dot(normSCA) > 0.0) and(normSBC.dot(normSAB) > 0.0 ):
                hit_pos = S
                ret = 1
        return ret,hit_pos




    # to cal detail intersect
    @ti.func
    def interset_mesh(self, bvh_id, pos, r, shape_pos, shape_quat, shape_scale, stack, i, MAX_SIZE):

        hit_pos      = pos
        stack[i, 0]  = self.offset[bvh_id]
        stack_pos    = 0

        while (stack_pos >= 0) & (stack_pos < MAX_SIZE):
            #pop
            node_index = stack[i,  stack_pos]
            stack_pos  = stack_pos-1

            offset     = UF.get_compact_node_offset(self.compact_node, node_index)
            if offset < 0:
                prim_index          = UF.get_compact_node_prim(self.compact_node, node_index)
                ret,hit_pos  = self.spher_intersect_tri(pos, r, prim_index,shape_pos, shape_quat, shape_scale)
                if ret == 1:
                    break
            else:
                min_sv,max_sv = UF.get_compact_node_min_max(self.compact_node, node_index)
                min_v,max_v = UF.transform_aabb(min_sv, max_sv, shape_pos, shape_quat, shape_scale)
                
                if UF.point_aabb_dis(pos, min_v,max_v) < r:
                    left_node  = node_index+1
                    right_node = offset
                    #push
                    stack_pos              += 1
                    stack[i,  stack_pos] = left_node
                    stack_pos              += 1
                    stack[i,  stack_pos] = right_node
                    #print(sphere_min,sphere_max,min_v,max_v)

        if stack_pos == MAX_SIZE:
            print("mesh overflow, need larger stack")

        return  hit_pos


