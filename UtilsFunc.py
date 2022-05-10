from tkinter import CENTER
import taichi as ti
########################### Important !!!! ###############################
#struct description
  
                                            

#            32bit         | 32bit    | 32bit       | 32bit      | 32bit     |96bit  |96bit 
#bvh_node  : is_leaf axis  |left_node  right_node   parent_node  prim_index   min_v3  max_v3   11
#             1bit   2bit       

#                32bit         |32bit       |32bit |96bit  |96bit 
#compact_node  : is_leaf axis  |prim_index  offset  min_v3  max_v3   9
#                1bit   2bit


#shape     : type_f   id      param_v6                                  8 BYTE                                    
#sphere      0        id      radius    X       X       X       X       X
#quad        1        id      V1                        V2
#capsule     2        id      radius    half_h
#box         3        id      minx      miny    minz    maxx    maxy    maxz   
#mesh        4        id      minx      miny    minz    maxx    maxy    maxz
                                           
#primitive : type(0:tri 1:shape) vertexIndex(shape_index)                               2


########################### Important !!!! ###############################
AXIS_X              = 0
AXIS_Y              = 1
AXIS_Z              = 2
EPS                 = 0.00001
M_PIf               = 3.1415956
INF_VALUE           = 10000000.0

SHAPE_SPHERE        = 1
SHAPE_QUAD          = 2
SHAPE_CAPSULE       = 3
SHAPE_BOX           = 4
SHAPE_MESH          = 5

PARTICLE_RIGID      = 0
PARTICLE_CLOTH      = 1
PARTICLE_DEFORM     = 2
PARTICLE_FLUID      = 3
PARTICLE_GAS        = 4
########################### tool function ###############################

@ti.func
def get_hit_t( hit_info):
    return hit_info[0]
@ti.func
def get_hit_prim( hit_info):
    return int(hit_info[10])
@ti.func
def get_hit_pos( hit_info):
    return ti.Vector([hit_info[1], hit_info[2], hit_info[3] ])
@ti.func
def get_hit_normal( hit_info):
    return ti.Vector([hit_info[4], hit_info[5], hit_info[6] ])
@ti.func
def get_hit_uv( hit_info):
    return ti.Vector([hit_info[7], hit_info[8], hit_info[9] ])


@ti.func
def get_prim_type(primitive, index):
    return primitive[index][0]
@ti.func
def get_prim_vindex(primitive, index):
    return primitive[index][1]

    
@ti.func
def get_shape_type(shape, index):
    return int(shape[index][0])
@ti.func
def get_shape_id(shape, index):
    return int(shape[index][1])
@ti.func
def get_shape_radius(shape, index):
    return shape[index][2]
@ti.func
def get_shape_half_h(shape, index):
    return shape[index][3]

@ti.func
def get_shape_v1(shape, index):
    return ti.Vector([shape[index][2], shape[index][3], shape[index][4] ])
@ti.func
def get_shape_v2(shape, index):
    return ti.Vector([shape[index][5], shape[index][6], shape[index][7] ])


@ti.func
def quat_mul_vec3(quat, vec):
    qVec = ti.Vector([quat.x, quat.y, quat.z])
    cross1 = qVec.cross(vec)
    cross2 = qVec.cross( cross1)
    return vec + (cross1 * quat.w + cross2) * 2.0


@ti.func
def quat_to_mat(quat):
    return ti.Matrix([  [1.0 - 2.0 * quat.y * quat.y - 2.0 * quat.z * quat.z,   2.0 * quat.x * quat.y - 2.0 * quat.w * quat.z,          2.0 * quat.x * quat.z + 2.0 * quat.w * quat.y,          0.0],  \
                        [2.0 * quat.x * quat.y + 2.0 * quat.w * quat.z,         1.0 - 2.0 * quat.x * quat.x - 2.0 * quat.z * quat.z,    2.0 * quat.y * quat.z - 2.0 * quat.w * quat.x,          0.0],  \
                        [2.0 * quat.x * quat.z - 2.0 * quat.w * quat.y,         2.0 * quat.y * quat.z + 2.0 * quat.w * quat.x,          1.0 - 2.0 * quat.x * quat.x - 2.0 * quat.y * quat.y,    0.0], \
                        [0.0,                                                   0.0,                                                    0.0,                                                    1.0]])

@ti.func
def transform_vec(v, translate, quat, scale):
    return quat_mul_vec3(quat, v * scale) + translate

@ti.func
def inv_transform_vec(v, translate, quat, scale):
    return quat_mul_vec3(ti.Vector([-quat.x,quat.y,quat.z, quat.w]), v / scale)  - translate

@ti.func
def transform_aabb(minv3, maxv3, translate, quat, scale):
    v1          = transform_vec(ti.Vector([minv3.x, minv3.y, minv3.z]), translate, quat, scale)
    v2          = transform_vec(ti.Vector([minv3.x, minv3.y, maxv3.z]), translate, quat, scale)
    v3          = transform_vec(ti.Vector([minv3.x, maxv3.y, minv3.z]), translate, quat, scale)
    v4          = transform_vec(ti.Vector([minv3.x, maxv3.y, maxv3.z]), translate, quat, scale)
    v5          = transform_vec(ti.Vector([maxv3.x, minv3.y, minv3.z]), translate, quat, scale)
    v6          = transform_vec(ti.Vector([maxv3.x, minv3.y, maxv3.z]), translate, quat, scale)
    v7          = transform_vec(ti.Vector([maxv3.x, maxv3.y, minv3.z]), translate, quat, scale)
    v8          = transform_vec(ti.Vector([maxv3.x, maxv3.y, maxv3.z]), translate, quat, scale)
    minv3       = min(min(min(min(min(min(min(v1,v2),v3),v4),v5),v6),v7),v8)
    maxv3       = max(max(max(max(max(max(max(v1,v2),v3),v4),v5),v6),v7),v8)
    return minv3, maxv3

    
@ti.func
def get_shape_min_max(shape, index, shape_pos, shape_quat, shape_scale):
    stype       = int(shape[index][0])
    id          = int(shape[index][1])
    minv3       = ti.Vector([shape[index][2], shape[index][3], shape[index][4] ])
    maxv3       = ti.Vector([shape[index][5], shape[index][6], shape[index][7] ])


    if stype  == SHAPE_SPHERE:
        minv3 = ti.Vector([-shape[index][2],-shape[index][2],-shape[index][2]])
        maxv3 = ti.Vector([shape[index][2],shape[index][2],shape[index][2]])
    elif stype  == SHAPE_QUAD:
        v1 = ti.Vector([shape[index][4], shape[index][5], shape[index][6] ]) 
        v2 = ti.Vector([shape[index][4], shape[index][5], shape[index][6] ]) 

        a = v1+v2
        b = v1-v2
        c = -v1-v2
        d = v2-v1
        minv3 = min(min(min(min(minv3,a),b),c),d)
        maxv3 = max(max(max(max(maxv3,a),b),c),d)


    elif stype  == SHAPE_CAPSULE:
        minv3 = ti.Vector([-shape[index][2],    -shape[index][3]-shape[index][2],   -shape[index][2]])
        maxv3 = ti.Vector([shape[index][2],     shape[index][3]+shape[index][2],    shape[index][2]])

    minv3, maxv3 = transform_aabb(minv3, maxv3, shape_pos[id], shape_quat[id], shape_scale[id])

    return minv3, maxv3 

#            32bit         | 32bit    | 32bit       | 32bit      | 32bit     |96bit  |96bit 
#bvh_node  : is_leaf axis  |left_node  right_node   parent_node  prim_index   min_v3  max_v3   11
#             1bit   2bit     
@ti.func
def init_bvh_node(bvh_node, index):
    bvh_node[index][0]  = -1.0
    bvh_node[index][1]  = -1.0
    bvh_node[index][2]  = -1.0
    bvh_node[index][3]  = -1.0
    bvh_node[index][4]  = -1.0
    bvh_node[index][5]  = INF_VALUE
    bvh_node[index][6]  = INF_VALUE
    bvh_node[index][7]  = INF_VALUE
    bvh_node[index][8]  = -INF_VALUE
    bvh_node[index][9]  = -INF_VALUE
    bvh_node[index][10] = -INF_VALUE
@ti.func
def set_node_type(bvh_node, index, type):
    bvh_node[index][0] = float(int(bvh_node[index][0]) & (0xfffe | type))

@ti.func
def set_node_axis(bvh_node, index, axis):
    axis = axis<<1
    bvh_node[index][0] =float(int(bvh_node[index][0]) & (0xfff9 | type))

@ti.func
def set_node_prim_size(bvh_node, index, size):
    bvh_node[index][0] =float(int(bvh_node[index][0]) & (0x0007 | size))

@ti.func
def set_node_left(bvh_node, index, left):
    bvh_node[index][1]  = float(left)
@ti.func
def set_node_right(bvh_node, index, right):
    bvh_node[index][2]  = float(right)
@ti.func
def set_node_parent(bvh_node, index, parent):
    bvh_node[index][3]  = float(parent)
@ti.func
def set_node_prim(bvh_node, index, prim):
    bvh_node[index][4]  = float(prim)
@ti.func
def set_node_min_max(bvh_node, index, minv,maxv):
    bvh_node[index][5]  = minv[0]
    bvh_node[index][6]  = minv[1]
    bvh_node[index][7]  = minv[2]
    bvh_node[index][8]  = maxv[0]
    bvh_node[index][9]  = maxv[1]
    bvh_node[index][10] = maxv[2]


@ti.func
def get_node_type(bvh_node, index):
    return  int(bvh_node[index][0]) & 0x0001 
@ti.func
def get_node_axis(bvh_node, index):
    return  int(bvh_node[index][0]) & 0x0006 

@ti.func
def get_node_child(bvh_node, index):
    return int(bvh_node[index][1]),int(bvh_node[index][2])
@ti.func
def get_node_parent(bvh_node, index):
    return int(bvh_node[index][3])
@ti.func
def get_node_prim(bvh_node, index):
    return int(bvh_node[index][4])

@ti.func
def get_node_min_max(bvh_node, index):
    return ti.Vector([bvh_node[index][5], bvh_node[index][6], bvh_node[index][7] ]),ti.Vector([bvh_node[index][8], bvh_node[index][9], bvh_node[index][10] ])
@ti.func
def get_node_has_box(bvh_node, index):
    return (bvh_node[index][5]  <= bvh_node[index][8]) & (bvh_node[index][6]  <= bvh_node[index][9]) & (bvh_node[index][7]  <= bvh_node[index][10])

#                32bit         |32bit       |32bit |96bit  |96bit 
#compact_node  : is_leaf axis  |prim_index  offset  min_v3  max_v3   9
#                1bit   2bit
@ti.func
def get_compact_node_type(bvh_node, index):
    return  int(bvh_node[index][0]) & 0x0001 
@ti.func
def get_compact_node_axis(bvh_node, index):
    return  int(bvh_node[index][0]) & 0x0006 
@ti.func
def get_compact_node_prim_size(bvh_node, index):
    return  int(bvh_node[index][0]) & 0xfff8  
@ti.func
def get_compact_node_prim(bvh_node, index):
    return int(bvh_node[index][1])
@ti.func
def get_compact_node_offset(bvh_node, index):
    return int(bvh_node[index][2])
@ti.func
def get_compact_node_min_max(bvh_node, index):
    return ti.Vector([bvh_node[index][3], bvh_node[index][4], bvh_node[index][5] ]),ti.Vector([bvh_node[index][6], bvh_node[index][7], bvh_node[index][8] ])




###################################################################


############algrithm##############
@ti.func
def inverse_transform(dir, N):
    Normal   = N.normalized()
    Binormal = ti.Vector([0.0, 0.0, 0.0])
    if (abs(Normal.x) > abs(Normal.z)):
        Binormal.x = -Normal.y
        Binormal.y = Normal.x
        Binormal.z = 0.0
    else:
        Binormal.x = 0.0
        Binormal.y = -Normal.z
        Binormal.z = Normal.y
    Binormal = Binormal.normalized()
    Tangent  = Binormal.cross(Normal).normalized()
    return dir.x*Tangent + dir.y*Binormal + dir.z*Normal



@ti.func
def inside_box(p, min_b, max_b):
    ret  = -2
    for i in ti.static(range(3)):
        if  (p[i] >  min_b[i]) and (p[i] < max_b[i]) : 
            ret += 1
    return ret

@ti.func
def intersect_box(min_v, max_v, min_b, max_b):
    return inside_box(min_v, min_b, max_b) | inside_box(max_v, min_b, max_b)


@ti.func
def point_aabb_dis(p, min_b, max_b):
    #https://www.iquilezles.org/www/articles/distfunctions/distfunctions.htm
    centre = (min_b + max_b)*0.5
    p   = p - centre
    b   = max_b-centre
    q = abs(p) - b

    sdf = (max(q,0.0)) + min(max(q.x,max(q.y,q.z)),0.0)
    return sdf.norm()


@ti.func
def slabs(origin, direction, minv, maxv):
    # most effcient algrithm for ray intersect aabb 
    # en vesrion: https://www.researchgate.net/publication/220494140_An_Efficient_and_Robust_Ray-Box_Intersection_Algorithm
    # cn version: https://zhuanlan.zhihu.com/p/138259656

    
    ret  = 1
    tmin = 0.0
    tmax = INF_VALUE
    
    for i in ti.static(range(3)):
        if abs(direction[i]) < 0.000001:
            if ( (origin[i] < minv[i]) | (origin[i] > maxv[i])):
                ret = 0
        else:
            ood = 1.0 / direction[i] 
            t1 = (minv[i] - origin[i]) * ood 
            t2 = (maxv[i] - origin[i]) * ood
            if(t1 > t2):
                temp = t1 
                t1 = t2
                t2 = temp 
            if(t1 > tmin):
                tmin = t1
            if(t2 < tmax):
                tmax = t2 
            if(tmin > tmax) :
                ret=0
    return ret
    
@ti.func
def max_component( v):
    return max(v.z, max(v[0], v.y) )
@ti.func
def min_component( v):
    return min(v.z, min(v[0], v.y) )
    

@ti.func
def expandBits( x):
    '''
    # nvidia blog : https://developer.nvidia.com/blog/thinking-parallel-part-iii-tree-construction-gpu/
    v = ( (v * 0x00010001) & 0xFF0000FF)
    v = ( (v * 0x00000101) & 0x0F00F00F)
    v = ( (v * 0x00000011) & 0xC30C30C3)
    v = ( (v * 0x00000005) & 0x49249249)
    taichi can not handle it, so i change that to bit operate
    '''
    x = (x | (x << 16)) & 0x030000FF
    x = (x | (x <<  8)) & 0x0300F00F
    x = (x | (x <<  4)) & 0x030C30C3
    x = (x | (x <<  2)) & 0x09249249
    return x


@ti.func
def common_upper_bits(lhs, rhs) :
    x    = lhs ^ rhs
    ret  = 32


    while x > 0:
        x  = x>>1
        ret  -=1
        #print(ret, lhs, rhs, x, find, ret)
    #print(x)
    return ret

@ti.func
def morton3D(x, y, z):
    x = min(max(x * 1024.0, 0.0), 1023.0)
    y = min(max(y * 1024.0, 0.0), 1023.0)
    z = min(max(z * 1024.0, 0.0), 1023.0)
    xx = expandBits(ti.cast(x, dtype = ti.i32))
    yy = expandBits(ti.cast(y, dtype = ti.i32))
    zz = expandBits(ti.cast(z, dtype = ti.i32))
    #return zz  | (yy << 1) | (xx<<2)
    code = xx  | (yy << 1) | (zz<<2)
    if code == 0:
        print("morton3D",x,y,z)
    return code