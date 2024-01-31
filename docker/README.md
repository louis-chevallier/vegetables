# landmarks

![image](./facial_landmarks_68markup.png)

-  ![](./images/facial_landmarks_68markup.png)

# transfer de topo

chaque topo est décrite dans un folder comme : <dataroot>/models/flm0

le code contenu dans le repo https://github.com/louis-chevallier/Deformation-Transfer-for-Triangle-Meshes permet de calculer la transformation d'un meshA vers un meshB etant donnée une transformation T calculée avant.
   Convert(meshA, T) -> meshB

la transformation T est calculée a partir de 2 mesh Source et Target qui sont annotés avec des landmarks.
   ComputeTransform(Source, Target, LM) -> T

Pour Smoke, la création d'une nouvelle topo, topoN :

- récupérer le mesh neutre de topoN
- convertir par convert, chaque vector shape et chque vecteur expression
- convertir la UVMap

# todo


- topo morgan
- sourcil, haussement 
- loss eye ( couleur dans la texture cacluée a partir de params = direction, couleur iris/blanc)
- loss bouche/dent 
- cheveux
- decimation : done
- test sur rig
- ajouter  estimation focale


## Avatar

- améliorer la texture générée - elle est floue
- ajouter l'intérieur de la bouche et les yeux
- ajouter les cheveux

## animation deep

- mieux reconstruire les expressions un peu extremes
  avec un dataset synthétique ??
- stabiliser la pose estimée ( schéma recurrent ??)



## cop

| module                               | usage           | license             |
|--------------------------------------|-----------------|---------------------|
| SixDRep | pose estimation |                                             MIT|
| Bisenet | face segmentation mask |                                      MIT|
| FAN | face alignment  (face detection and landmarks):  (FAN pip module)|        BSD|
| SPIGA | face landmarks |                                                                 BSD 3|
| pipnet | landmark option|                                               no licence, https://github.com/jhb86253817/PIPNet|
| trimesh||                                                                 MIT|
| pytorch||                                                                 BSD|
| pytorch3D||                                                               BSD|
| DeformationTransfer |  (transfer topo |                                    MIT|
| Insightface | face reco  ( embedding) |                                     MIT|
| open3D | mesh processing|                                                                  MIT|
| FLAME | 3D face model||                                                                   
| ---- D3DFACS | face scans database, pas utilisé | non commercial only |
| ---- Model |   ( modèle géométrique + stats ) replacé par morgan et |          Creative Commons Attribution license|
| ---- Texture + stats | remplacé par texture apprise sur celeba |           Creative Common BY-NC-SA 4.0 : non commercial only|
| ---- code |  https://github.com/HavenFeng/photometric_optimization/tree/master/models |  MIT |



# Documentation

## Entry points

for each entry point, there is a target in the makefile

for detailed doc, type python <the tool.py> --help

for each tool, --help option gives details.

### estimate an avatar from a set of image given a model
'''
photometric_fitting.py
'''

- images a extracted from a video
- images are from a rig


### Check a model
'''
checkmodel.py
'''

- display a synthetic with an average texture
- left click shows the vertex number
- can change the model parameters ( pose, shape, blendshape, albedo )

### web server
'''
server.py
'''
Starts a web server exposing several functions ( http requests )

| role                                 | tag             |
|--------------------------------------|-----------------|
| send a configuration,   for the time being, send required quality level               | /config         |
| upload a video, the reocnstruction is launched automatically                       | /upload         |
| get progress status, read the reconstruction status = progress, number of uploaded photos and resut file ( objpath )
  if the calculation is finished ( objpath != "") , the 3D model is loaded from the server ( .obj, .mtl files)                  |  /progress      |  
| retreive the avatar | get file |
| upload a new photo | /uploadPhoto  |
| reset the uploaded photos collection | /resetImageCounter  |
| launch a reconstruction from photos | /runOnPhoto |
| configure the reconstructor | /config with parameters : x-active: 1 or 0 : to start or stop the reconstructor, x_quality : to change the quality of the reocnstruction ( and calcultaion time ) |

'''      

### create new model
'''
registerLM.py
'''

- by decimation from a pre existing model
- by topology transfer given a model and a new mesh of a face ( providing annotation = markers)

### train a deep network
'''
deep.py
'''

the deep net out an avatar from an image directly

### create 3D annotation for a mesh

'''
annot_mesh.py
'''

To add 3D markers on the surface off a 3D mesh

- markers are stored in a json file
- limitation : markers a bound to sit on vertices ( todo : anywhere on triangle using barycenter coef)

- left button drag : rotate the mesh
- middle button drag : translate the mesh
- ctrl left button on a mark : select the mark
- ctrl left button :
  - a mark has been selected : move the mark
  - no mark  selected : create a new mark ( at the end of the list of marks)
- wheel : make the mesh closer/further
- tab : iterate on the list of marks ( in the title : number of the mark , corresponding vertex number), and *select* the current mark 
- 'c' : clear the list of marks
- 'l' : load the marks from the json file
- 's' : save the marks to the json file
- 'd' : duplicate the current mark
- 'r' : delete the current mark from the list
- '+/-' : increase/decrease the size of the marks

### 

# Config hardware

- CPU core i7 3.5G
- RAM 32Giga ou plus
- GPU : RTX3060 ( ie un carte GPU NVidia récente avec au moins 12Giga de Ram )
- Disques :
  - HD1 : SSD 200G systeme 
  - HD2 : 2 Tera
  - HD3 : 1 Tera
- Système : Linux Mint 20.2 