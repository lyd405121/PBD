import taichi as ti
import SceneData as SCD
import UtilsFunc as UF
import time


IS_LEAF           = 1


@ti.data_oriented
class Bvh:
    def __init__(self, primitive_count, min_boundary, max_boundary):
        
        self.primitive_count = primitive_count
        self.minboundarynp   = min_boundary
        self.maxboundarynp   = max_boundary

        self.min_boundary    = ti.Vector.field(3, dtype=ti.f32, shape=(1))
        self.max_boundary    = ti.Vector.field(3, dtype=ti.f32, shape=(1))


        self.radix_count_zero = ti.field(dtype=ti.i32, shape=[1])
        self.radix_offset     = ti.Vector.field(2, dtype=ti.i32)
 
        self.morton_code_s    = ti.Vector.field(2, dtype=ti.i32)
        self.morton_code_d    = ti.Vector.field(2, dtype=ti.i32)
 
        self.bvh_node         = ti.Vector.field(SCD.NOD_VEC_SIZE, dtype=ti.f32)
        self.compact_node     = ti.Vector.field(SCD.CPNOD_VEC_SIZE, dtype=ti.f32)
        self.bvh_done         = ti.field(dtype=ti.i32, shape=[1])
        self.leaf_node_count  = 0


        self.node_count      = self.primitive_count*2-1

        self.primitive_pot   = (self.get_pot_num(self.primitive_count)) << 1
        self.primitive_bit   = self.get_pot_bit(self.primitive_pot)

        ti.root.dense(ti.i, self.primitive_count ).place(self.morton_code_s)
        ti.root.dense(ti.i, self.primitive_count ).place(self.morton_code_d)
        ti.root.dense(ti.i, self.primitive_pot   ).place(self.radix_offset)


        ti.root.dense(ti.i, self.node_count).place(self.bvh_node )
        ti.root.dense(ti.i, self.node_count).place(self.compact_node )


    ########################host function#####################################

    def start_timer(self):
        self.start_time = time.time()* 1000


    def end_timer(self, func_name):
        end_time = time.time()* 1000
        print(func_name, "spend: {:.2f}".format(end_time - self.start_time))


    def get_pot_num(self, num):

        if num <=1 :
            return 1
        else:

            m = 1
            while m<num:
                m=m<<1
            return m >>1


    def get_pot_bit(self, num):
        m   = 1
        cnt = 0
        while m<num:
            m=m<<1
            cnt += 1
        return cnt


    def blelloch_scan_host(self, mask, move):
        self.radix_sort_predicate(mask, move)
        for i in range(1, self.primitive_bit+1):
            self.blelloch_scan_reduce(1<<i)
        for i in range(self.primitive_bit+1, 0, -1):
            self.blelloch_scan_downsweep(1<<i)



    def radix_sort_host(self):

        for i in range(30):
            mask   = 0x00000001 << i
            self.blelloch_scan_host(mask, i)
            self.radix_sort_fill(mask, i)
        #self.print_morton_reslut()


    def print_morton_reslut(self):
        tmp = self.morton_code_s.to_numpy()
        for i in range(0, self.primitive_count):
            if i > 0:
                if tmp[i,0] < tmp[i-1,0]:
                    print(i, tmp[i,:], tmp[i-1,:], "!!!!!!!!!!wrong!!!!!!!!!!!!")
                elif tmp[i,0] == tmp[i-1,0]:
                    print(i, tmp[i,:], tmp[i-1,:], "~~~~~~equal~~~~~~~~~~~~~")
                else:
                    print(i, tmp[i,:], tmp[i-1,:])
            else:
                print(i, tmp[i,:])  

    def flatten_tree(self, bvh_node, index):
        retOffset    = self.offset
        self.offset += 1

        is_leaf = int(bvh_node[index, 0]) & 0x0001
        left    = int(bvh_node[index, 1])
        right   = int(bvh_node[index, 2])

        self.compact_node_np[retOffset][0] = bvh_node[index, 0]
        for i in range(6):
            self.compact_node_np[retOffset][3+i] = bvh_node[index, 5+i]

        if is_leaf != SCD.IS_LEAF:
            self.flatten_tree( bvh_node, left)
            self.compact_node_np[retOffset][1] = -1
            self.compact_node_np[retOffset][2] = self.flatten_tree(bvh_node, right)
        else:
            self.compact_node_np[retOffset][1] = bvh_node[index, 4]
            self.compact_node_np[retOffset][2] = -1
            if self.node_count == 1:
                self.compact_node_np[retOffset][1] = -1
        return retOffset


    def build_compact_node(self, bvh_node):
        self.offset = 0
        self.flatten_tree(bvh_node,0)
        self.compact_node.from_numpy(self.compact_node_np)

    ############algrithm##############
    @ti.func
    def determineRange(self, idx):
        l_r_range = ti.cast(ti.Vector([0, self.primitive_count-1]), ti.i32)

        if idx != 0:
            self_code = self.morton_code_s[idx][0]
            l         = idx-1
            r         = idx+1
            l_code    = self.morton_code_s[l][0]
            r_code    = self.morton_code_s[r][0]

            if  (l_code == self_code ) & (r_code == self_code) :
                l_r_range[0] = idx

                while  idx < self.primitive_count-1:
                    idx += 1
                    
                    if(idx >= self.primitive_count-1):
                        break

                    if (self.morton_code_s[idx][0] != self.morton_code_s[idx+1][0]):
                        break
                l_r_range[1] = idx 
            else:
                L_delta   = UF.common_upper_bits(self_code, l_code)
                R_delta   = UF.common_upper_bits(self_code, r_code)

                d = -1
                if R_delta > L_delta:
                    d = 1
                delta_min = min(L_delta, R_delta)
                l_max = 2
                delta = -1
                i_tmp = idx + d * l_max

                if ( (0 <= i_tmp) &(i_tmp < self.primitive_count)):
                    delta = UF.common_upper_bits(self_code, self.morton_code_s[i_tmp][0])


                while delta > delta_min:
                    l_max <<= 1
                    i_tmp = idx + d * l_max
                    delta = -1
                    if ( (0 <= i_tmp) & (i_tmp < self.primitive_count)):
                        delta = UF.common_upper_bits(self_code, self.morton_code_s[i_tmp][0])

                l = 0
                t = l_max >> 1

                while(t > 0):
                    i_tmp = idx + (l + t) * d
                    delta = -1
                    if ( (0 <= i_tmp) & (i_tmp < self.primitive_count)):
                        delta = UF.common_upper_bits(self_code, self.morton_code_s[i_tmp][0])
                    if(delta > delta_min):
                        l += t
                    t >>= 1

                l_r_range[0] = idx
                l_r_range[1] = idx + l * d
                if(d < 0):
                    tmp          = l_r_range[0]
                    l_r_range[0] = l_r_range[1]
                    l_r_range[1] = tmp 

        return l_r_range
        
    @ti.func
    def findSplit(self, first, last):
        first_code = self.morton_code_s[first][0]
        last_code  = self.morton_code_s[last][0]
        split = first
        if (first_code != last_code):
            delta_node = UF.common_upper_bits(first_code, last_code)

            stride = last - first
            while 1:
                stride = (stride + 1) >> 1
                middle = split + stride
                if (middle < last):
                    delta = UF.common_upper_bits(first_code, self.morton_code_s[middle][0])
                    if (delta > delta_node):
                        split = middle
                if stride <= 1:
                    break
        return split



    @ti.kernel
    def radix_sort_predicate(self,  mask: ti.i32, move: ti.i32):
        for i in self.radix_offset:
            if i < self.primitive_count:
                self.radix_offset[i][1]       = (self.morton_code_s[i][0] & mask ) >> move
                self.radix_offset[i][0]       = 1-self.radix_offset[i][1]
                ti.atomic_add(self.radix_count_zero[0], self.radix_offset[i][0]) 
            else:
                self.radix_offset[i][0]       = 0
                self.radix_offset[i][1]       = 0

 
    @ti.kernel
    def blelloch_scan_reduce(self, mod: ti.i32):
        for i in self.radix_offset:
            if (i+1)%mod == 0:
                prev_index = i - (mod>>1)
                self.radix_offset[i][0] += self.radix_offset[prev_index][0]
                self.radix_offset[i][1]+= self.radix_offset[prev_index][1]

    @ti.kernel
    def blelloch_scan_downsweep(self, mod: ti.i32):
        for i in self.radix_offset:

            if mod == (self.primitive_pot*2):
                self.radix_offset[self.primitive_pot-1] = ti.Vector([0,0])
            elif (i+1)%mod == 0:
                prev_index = i - (mod>>1)
                if prev_index >= 0:
                    tmpV   = self.radix_offset[prev_index]
                    self.radix_offset[prev_index] = self.radix_offset[i]
                    self.radix_offset[i] += tmpV

    @ti.kernel
    def radix_sort_fill(self,  mask: ti.i32, move: ti.i32):
        for i in self.morton_code_s:
            condition = (self.morton_code_s[i][0] & mask ) >> move
            
            if condition == 1:
                offset = self.radix_offset[i][1] + self.radix_count_zero[0]
                self.morton_code_d[offset] = self.morton_code_s[i]
            else:
                offset = self.radix_offset[i][0] 
                self.morton_code_d[offset] = self.morton_code_s[i]

        for i in self.morton_code_s:
            self.morton_code_s[i]    = self.morton_code_d[i]
            self.radix_count_zero[0] = 0

    @ti.kernel
    def gen_aabb(self):
        for i in self.bvh_node:
            if (UF.get_node_has_box(self.bvh_node, i)   == 0):
                left_node,right_node   = UF.get_node_child(self.bvh_node,  i) 
                
                is_left_rdy  = UF.get_node_has_box(self.bvh_node, left_node)
                is_right_rdy = UF.get_node_has_box(self.bvh_node, right_node)

                if (is_left_rdy & is_right_rdy) > 0:
                    
                    l_min,l_max = UF.get_node_min_max(self.bvh_node, left_node)  
                    r_min,r_max = UF.get_node_min_max(self.bvh_node, right_node)  
                    UF.set_node_min_max(self.bvh_node, i, min(l_min, r_min),max(l_max, r_max))
                    self.bvh_done[0] += 1
                #print("ok", i, left_node, right_node)



    def check_build(self):
        done_prev = 0
        done_num  = 0
        while done_num < self.primitive_count-1:
            self.gen_aabb()
            done_num  = self.bvh_done.to_numpy()
            if done_num == done_prev:
                break
            done_prev = done_num
        
        if done_num != self.primitive_count-1:
            print("aabb gen error!!!!!!!!!!!!!!!!!!!%d"%done_num)
 
        bvh_node     = self.bvh_node.to_numpy()
        self.compact_node_np = self.compact_node.to_numpy()
        self.build_compact_node(bvh_node)



    def update_boundary(self):
        print("***************node:%d*******************"%(self.node_count))
        self.max_boundary.from_numpy(self.maxboundarynp)
        self.min_boundary.from_numpy(self.minboundarynp)


    @ti.func
    def gen_morton_tri(self, vertex, vindex, pindex):
        v0        = vertex[vindex]
        v1        = vertex[vindex+1]
        v2        = vertex[vindex+2]
        centre_p  = (v1 + v2 + v0) * (1.0/ 3.0)
        norm_p    = (centre_p - self.min_boundary[0])/(self.max_boundary[0] - self.min_boundary[0])
        self.morton_code_s[pindex][0] = UF.morton3D(norm_p.x, norm_p.y, norm_p.z)
        self.morton_code_s[pindex][1] = pindex


    @ti.func
    def build_interior_node(self, nindex):
        UF.set_node_type(self.bvh_node, nindex, 1-IS_LEAF)
        l_r_range   = self.determineRange(nindex)
        spilt       = self.findSplit(l_r_range[0], l_r_range[1])
 
        left_node   = spilt
        right_node  = spilt + 1

        if min(l_r_range[0], l_r_range[1]) == spilt :
            left_node  += self.primitive_count - 1
        
        if max(l_r_range[0], l_r_range[1]) == spilt + 1:
            right_node  += self.primitive_count - 1
        
        if l_r_range[0] == l_r_range[1]:
            print(l_r_range, spilt, left_node, right_node,"wrong")
        #else:
        #    print(l_r_range, spilt,left_node, right_node)

        UF.set_node_left(self.bvh_node,   nindex, left_node)
        UF.set_node_right(self.bvh_node,  nindex, right_node)
        UF.set_node_parent(self.bvh_node, left_node, nindex)
        UF.set_node_parent(self.bvh_node, right_node, nindex)

    @ti.func
    def init_leaf_node(self, nindex):
        UF.set_node_type(self.bvh_node, nindex, IS_LEAF)
        prim_index = self.morton_code_s[nindex-self.primitive_count+1][1]
        UF.set_node_prim(self.bvh_node, nindex, prim_index)
        return prim_index

    @ti.func
    def get_tri_min_max(self, vertex, vindex, nindex):
        min_v3     = ti.Vector([UF.INF_VALUE, UF.INF_VALUE, UF.INF_VALUE])
        max_v3     = ti.Vector([-UF.INF_VALUE, -UF.INF_VALUE, -UF.INF_VALUE])

        v1        = vertex[vindex]
        v2        = vertex[vindex+1]
        v3        = vertex[vindex+2]

        min_v3       = min(min(min(min_v3,v1),v2),v3)
        max_v3       = max(max(max(max_v3,v1),v2),v3)
        UF.set_node_min_max(self.bvh_node, nindex, min_v3,max_v3)




    def setup_vertex(self, vertex, offset):
        self.update_boundary()
        
        self.build_morton_vertex(vertex, offset)
        print("morton code is built")
        self.radix_sort_host()
        print("radix sort  is done")
        self.build_bvh_vertex(vertex, offset)
        print("tree build  is done")

        self.check_build()

    @ti.kernel
    def build_morton_vertex(self, vertex:ti.template(), offset:ti.int32):
        for i in ti.ndrange(self.primitive_count):
            self.gen_morton_tri(vertex, 3*i+offset, i)


    @ti.kernel
    def build_bvh_vertex(self, vertex:ti.template(), offset:ti.int32):
        
        for i in self.bvh_node:
            UF.init_bvh_node(self.bvh_node, i)
            self.bvh_done[0] = 0

        for i in self.bvh_node:
            if i >= self.primitive_count-1:
                prim_index = self.init_leaf_node(i)
                self.get_tri_min_max(vertex, 3*prim_index+offset, i)
            else:
                self.build_interior_node(i)


    @ti.func
    def gen_morton_shape(self, shape, sindex, shape_pos):
        v = shape_pos[sindex]
        norm_p    = (v - self.min_boundary[0])/(self.max_boundary[0] - self.min_boundary[0])

        self.morton_code_s[sindex][0] = UF.morton3D(norm_p.x, norm_p.y, norm_p.z)
        self.morton_code_s[sindex][1] = sindex


    @ti.func
    def get_shape_min_max(self, shape, sindex, nindex,   shape_pos, shape_quat, shape_scale):
        min_v3,max_v3  = UF.get_shape_min_max(shape, sindex,  shape_pos, shape_quat, shape_scale )
        UF.set_node_min_max(self.bvh_node, nindex, min_v3,max_v3)




    def setup_particle(self, particle_pos:ti.template(), r:ti.f32):
        self.update_boundary()
        self.build_morton_particle(particle_pos)
        print("morton code is built")
        self.radix_sort_host()
        print("radix sort  is done")
        self.build_bvh_particle(particle_pos,r)
        print("tree build  is done")

        self.check_build()


    @ti.kernel
    def build_morton_particle(self, particle_pos:ti.template()):
        for i in particle_pos:
            v = particle_pos[i]
            norm_p    = (v - self.min_boundary[0])/(self.max_boundary[0] - self.min_boundary[0])
            self.morton_code_s[i][0] = UF.morton3D(norm_p.x, norm_p.y, norm_p.z)
            self.morton_code_s[i][1] = i

    @ti.kernel
    def build_bvh_particle(self,  particle_pos:ti.template(), r:ti.f32):
        for i in self.bvh_node:
            UF.init_bvh_node(self.bvh_node, i)
            self.bvh_done[0] = 0

        for i in self.bvh_node:
            if i >= self.primitive_count-1:
                prim_index = self.init_leaf_node(i)
                minv3 = ti.Vector([-r,-r,-r]) + particle_pos[prim_index]
                maxv3 = ti.Vector([r,r,r])    + particle_pos[prim_index]
                UF.set_node_min_max(self.bvh_node, i, minv3,maxv3)
            else:
                self.build_interior_node(i)


    def setup_shape(self, shape, shape_pos:ti.template(), shape_quat:ti.template(), shape_scale:ti.template()):
        self.update_boundary()

        self.start_timer()
        self.build_morton_shape(shape, shape_pos)
        self.end_timer("gen morton code")

        self.start_timer()
        self.radix_sort_host()
        self.end_timer("radix sort")

        self.start_timer()
        self.build_bvh_shape(shape, shape_pos,shape_quat,shape_scale)
        self.end_timer("tree build")

        self.start_timer()
        self.check_build()
        self.end_timer("flat tree")

    @ti.kernel
    def build_morton_shape(self, shape:ti.template(),  shape_pos:ti.template()):
        for i in ti.ndrange(self.primitive_count):
            self.gen_morton_shape(shape, i, shape_pos)

    @ti.kernel
    def build_bvh_shape(self, shape:ti.template(), shape_pos:ti.template(), shape_quat:ti.template(), shape_scale:ti.template()):
        
        for i in self.bvh_node:
            UF.init_bvh_node(self.bvh_node, i)
            self.bvh_done[0] = 0


        for i in self.bvh_node:
            if i >= self.primitive_count-1:
                
                prim_index = self.init_leaf_node(i)
                #print(self.bvh_node[i][4])
                self.get_shape_min_max(shape, prim_index, i, shape_pos,shape_quat, shape_scale)
            else:
                self.build_interior_node(i)
                #print(i)
