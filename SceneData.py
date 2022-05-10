
import numpy as np

########################### Important !!!! ###############################
#struct description

#shape     : type_f   id      param_v6                                  8 BYTE                                    
#sphere      1        id      radius    X       X       X       X       X
#quad        2        id      V1                        V2
#capsule     3        id      radius    half_h
#box         4        id      minx      miny    minz    maxx    maxy    maxz    
#convex      5        id      minx      miny    minz    maxx    maxy    maxz
#mesh        6        id      minx      miny    minz    maxx    maxy    maxz  
                                           

#vertex    : pos_v3   normal_v3 tex_v3                                                  9
        

#            32bit         | 32bit    | 32bit       | 32bit      | 32bit     |96bit  |96bit 
#bvh_node  : is_leaf axis  |left_node  right_node   parent_node  prim_index   min_v3  max_v3   11
#             1bit   2bit       

#                32bit         |32bit       |32bit |96bit  |96bit 
#compact_node  : is_leaf axis  |prim_index  offset  min_v3  max_v3   9
#                1bit   2bit      

########################### Important !!!! ###############################
import UtilsFunc as UF
PRI_VEC_SIZE    = 3
SHA_VEC_SIZE    = 8
NOD_VEC_SIZE    = 11
CPNOD_VEC_SIZE  = 9
IS_LEAF         = 1


PRIMITIVE_NONE  = 0
PRIMITIVE_TRI   = 1
PRIMITIVE_SHAPE = 2


class Shape:
    def __init__(self):
        self.param     = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]

    def getType(self):
        return int(self.param[0])

    def setType(self, stype):
        self.param[0] = stype

    def setID(self, id):
        self.param[1] = id

    def getID(self):
        return int(self.param[1])


    def setRadius(self, radius):
        self.param[2] = radius

    def setHalfH(self, H):
        self.param[3] = H


    def setV1(self, V1):
        self.param[2] = V1[0]
        self.param[3] = V1[1]
        self.param[4] = V1[2]

    def setV2(self, V2):
        self.param[5] = V2[0]
        self.param[6] = V2[1]
        self.param[7] = V2[2]


    def setMin(self, V1):
        self.param[2] = V1[0]
        self.param[3] = V1[1]
        self.param[4] = V1[2]

    def setMax(self, V2):
        self.param[5] = V2[0]
        self.param[6] = V2[1]
        self.param[7] = V2[2]


    def cross(self, v1, v2):
        return [v1[1] * v2[2] - v1[2] * v2[1], v1[2] * v2[0] - v1[0] * v2[2], v1[0]* v2[1] - v1[1] * v2[0]]

    def quat_mul_vec3(self, quat, vec):
        qVec = [quat[0], quat[1], quat[2]]
        cross1 = self.cross(qVec, vec)
        cross2 = self.cross(qVec, cross1)

        ret = [0.0,0.0,0.0]
        for i in range(3):
            ret[i] = vec[i] + (cross1[i] * quat[3]+ cross2[i]) * 2.0
        return ret


    def transform_vec(self, v, translate, quat, scale):
        quat_vec = self.quat_mul_vec3(quat, [v[0]*scale[0], v[1]*scale[1], v[2]*scale[2]])
        return [quat_vec[0]+translate[0],quat_vec[1]+translate[1],quat_vec[2]+translate[2]]

    
    def transform_aabb(self, minv3, maxv3,translate, quat, scale):

        v1          = self.transform_vec([minv3[0], minv3[1], minv3[2]], translate, quat, scale)
        v2          = self.transform_vec([minv3[0], minv3[1], maxv3[2]], translate, quat, scale)
        v3          = self.transform_vec([minv3[0], maxv3[1], minv3[2]], translate, quat, scale)
        v4          = self.transform_vec([minv3[0], maxv3[1], maxv3[2]], translate, quat, scale)
        v5          = self.transform_vec([maxv3[0], minv3[1], minv3[2]], translate, quat, scale)
        v6          = self.transform_vec([maxv3[0], minv3[1], maxv3[2]], translate, quat, scale)
        v7          = self.transform_vec([maxv3[0], maxv3[1], minv3[2]], translate, quat, scale)
        v8          = self.transform_vec([maxv3[0], maxv3[1], maxv3[2]], translate, quat, scale)
        minv3       = min(min(min(min(min(min(min(v1,v2),v3),v4),v5),v6),v7),v8)
        maxv3       = max(max(max(max(max(max(max(v1,v2),v3),v4),v5),v6),v7),v8)
        return minv3, maxv3

    def getMinMax(self,translate, quat, scale):

        stype       = int(self.param[0])
        minv3       = [self.param[2], self.param[3], self.param[4] ]
        maxv3       = [self.param[5], self.param[6], self.param[7] ]

        if (stype  == UF.SHAPE_SPHERE):
            minv3 = [-self.param[2], -self.param[2],  -self.param[2]]
            maxv3 = [self.param[2],   self.param[2],  self.param[2] ]
        elif stype  == UF.SHAPE_QUAD:
            v1 = [self.param[2], self.param[3], self.param[4] ]
            v2 = [self.param[5], self.param[6], self.param[7] ]

            a = v1+v2
            b = v1-v2
            c = -v1-v2
            d = v2-v1

            for k in range(3):
                minv3[k] = min(min(min(min(minv3[k],a[k]),b[k]),c[k]),d[k])
                maxv3[k] = max(max(max(max(maxv3[k],a[k]),b[k]),c[k]),d[k])
        elif stype  == UF.SHAPE_CAPSULE:
            minv3 = [-self.param[2],-self.param[3]-self.param[2],-self.param[2]]
            maxv3 = [self.param[2],self.param[3]+self.param[2], self.param[2]]
        print(minv3, maxv3)
        minv3, maxv3 = self.transform_aabb(minv3, maxv3,translate, quat, scale)
        print(minv3, maxv3)
        return minv3, maxv3

    def fillStruct(self, np_data, index):
        #print(np_data, self.param)
        for i in range(SHA_VEC_SIZE):
            np_data[index, i] = self.param[i]



class Primitive:
    def __init__(self):
        self.type                = 0
        self.vertex_shape_index  = 0

    def fillStruct(self, np_data, index):
        np_data[index, 0] = self.type
        np_data[index, 1] = self.vertex_shape_index


#             32bit                     | 32bit    | 32bit       | 32bit      | 32bit     |96bit  |96bit 
#bvh_node  :  is_leaf axis    prim_size |left_node  right_node   parent_node  prim_index   min_v3  max_v3   11
#             1bit   2bit     29 bit      

#                32bit                   |32bit               |96bit  |96bit 
#compact_node  : is_leaf axis prim_size  |prim_index /offset  min_v3  max_v3   8
#                1bit   2bit  29 bit     

class Bounds:
    def __init__(self):
        self.min_v3              = [np.Infinity,np.Infinity,np.Infinity]
        self.max_v3              = [-np.Infinity,-np.Infinity,-np.Infinity]

    def Merge(self, v):
        for k in range(3):
            self.min_v3[k] = min(self.min_v3[k], v[k])
            self.max_v3[k] = max(self.max_v3[k], v[k])

    def MergeBox(self, b):
        self.Merge(b.min_v3)
        self.Merge(b.max_v3)


    def GetSurfaceArea(self):
        e1 = self.max_v3[0] - self.min_v3[0]
        e2 = self.max_v3[1] - self.min_v3[1]
        e3 = self.max_v3[2] - self.min_v3[2]
        return 2.0*(e1*e2+e2*e3+e3*e1)

class BVHNode:
    def __init__(self):
        self.is_leaf             = 0
        self.axis                = 0
        self.left_node           = 0
        self.right_node          = 0
        self.parent_node         = 0
        self.prim_index          = 0
        self.min_v3              = [np.Infinity,np.Infinity,np.Infinity]
        self.max_v3              = [-np.Infinity,-np.Infinity,-np.Infinity] 