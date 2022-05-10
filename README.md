# Positon based dynamic with taichi

- The code implements one of nvidia's paper [PBD](https://mmacklin.com/uppfrta_preprint.pdf)

- The code use [Taichi](https://github.com/taichi-dev/taichi) programming language

---

## How to run 

- First config your anaconda workspace, and open the anaconda prompt
  
- Second you need install dependency  
  ```python
  pip install -r requirements.txt
  ```

- Last you type 
  ```python
  ti Examle.py
  ```

---

## How to generate voxel file

- First copy your obj file to "model" , for example "Bearings.obj"
  
- Second you need to modify the code in GenVoxel.py
  ```python
  obj_name = "my"
  ```

- Then you type 
  ```python
  ti GenVoxel.py
  ```

- Last you will get a voxel file like below:


<figure class="half">
    <img src="image/bearings.png" height="200">
    <img src="image/bearings-vol.png" height="200">
    <img src="image/bearings-bvh.png" height="200">
    <div>&emsp;&emsp;&emsp;&emsp;origin obj file&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp; voxel file&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;bvh file</div>
</figure>


## Some example

- Rigidbody Simulation
  
![image](image/taichi.gif)

