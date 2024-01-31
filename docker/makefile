
export DATE:=$(shell date +%Y-%m-%d_%Hh%Mm%Ss)
HOST=$(shell hostname)
SHELL=bash

removetrailingslash = ${1:/=}
upfolder = $(dir $(call removetrailingslash,${1}))
isnotgit = $(subst git,,$(call removetrailingslash,$(notdir $(call removetrailingslash,$(1)))))
goupuntilgit = $(if $(call isnotgit,$(1)),$(call goupuntilgit,$(call upfolder,$(1))),$(1))
MVDIR := $(dir $(abspath $(lastword $(MAKEFILE_LIST))))
GIT ?=$(call goupuntilgit,$(MVDIR))

DATAROOT?=/mnt/hd1/data/smoke_data/runtime_data
PTAG?=_
HDROOT=hd2

#export QT_QPA_PLATFORM=offscreen 

#include $(GIT)/utils/inc.mk
#ENVCONDA=source ./buildenv.sh; buildtheenv666

OTAG=
ORTHO?=True
NSEV?=False

LR?=0.005
CYCLES?=1.5
SMODEL?=smk0.copy.decime
SHAPEP?=100
EXPP?=50

#$(warning $(shell env))

startWWW :
	ssh -X -q D9468 -l chevallierl "cd $$PWD; DEBUG=True GGPU=0 INSIDE_EMACS=${INSIDE_EMACS} make fit1"

qt :
	CUDA_VISIBLE_DEVICES=0 python qt.py

hpose :
	python qt.py

attrs :
	python attributeclassif.py --loadcache True --celebapath=$(CLBP) --cachedir=./small --load_attribute=True

start :
	CUDA_VISIBLE_DEVICES=0 MINEKOLEVEL=0 python registerLM.py --dataroot=$(DATAROOT)   --smoke_model_name=$(SMODEL)  --newtex=True  --tag=$@  2>&1 | tee trcs/$(@)$(TAG)_$(DATE)_$(ID).trc
	SMODEL=$(SMODEL).ntex  make seq

ggg :
	python grid_sample.py
#seq
#server
#server
#fix_morgan
#start : rig-gael
#help seq 
#transfer
#register_copy
#transfer

celebatex :
	CUDA_VISIBLE_DEVICES=0 MINEKOLEVEL=0 python celeba_tex.py --tqdm=no --dataroot=$(DATAROOT)  --smoke_model_name=$(SMODEL)  --tag=$@  2>&1 | tee trcs/$(@)$(TAG)_$(DATE)_$(ID).trc

fix_morgan :
	-mkdir ./results
	python fix_morgan.py --obj=annotations/morgan_neutral_color.obj --left_eye=/mnt/hd1/data/smoke_data/Morgan/left_eye_v2.obj --right_eye=/mnt/hd1/data/smoke_data/Morgan/right_eye_v2.obj --dataroot=$(DATAROOT) --smoke_model_name=smk0.copy.decime --out_obj=results/morgan_processed2.obj 

fix_smoke :
	python fix_smoke.py --obj_eyes_color="/mnt/hd1/data/smoke_data/runtime_data/models/smk0.copy.decime/head_template_mesh_eyes_color.obj" --obj_eyesballs=/mnt/hd1/data/smoke_data/runtime_data/models/smk0.copy.decime/head_template_mesh_eyes.obj  --obj=/mnt/hd1/data/smoke_data/runtime_data/models/smk0.copy.decime/head_template_mesh.obj --out=smoke_fixed.obj --annot=annotations/smk0.copy.decime.json   --dataroot=$(DATAROOT) --smoke_model_name=smk0.copy.decime 

#rigs
#basel

basel :
	python basel.py
#register_copy

#annot

#

#transfer

transfer :
#	CUDA_VISIBLE_DEVICES=0 MINEKOLEVEL=0 python registerLM.py --dataroot=$(DATAROOT)  --decime=True --smoke_model_name=smk0.copy --copy=True  --tag=$@  2>&1 | tee trcs/$(@)$(TAG)_$(DATE)_$(ID).trc

	CUDA_VISIBLE_DEVICES=0 MINEKOLEVEL=0 python registerLM.py --tqdm=no --dataroot=$(DATAROOT) --fake=True --source="/mnt/hd1/data/smoke_data/runtime_data/models/smk0.copy	.decime/head_template_mesh.obj" --source_annot=annotations/smk0.copy.decime.json --target=results/morgan_processed2.obj --target_annot=annotations/morgan2.json --smoke_model_name=smk0.copy.decime --transfer=True --out=results/morgan_transfered2.obj  --tag=$@  2>&1 | tee trcs/$(@)$(TAG)_$(DATE)_$(ID).trc

check :
	CUDA_VISIBLE_DEVICES=0 python checkmodel.py --orthographic=True --ui=True --smoke_model_name=smk0.copy.decime.transfer --generate_test_sequence=True --video_out=test_video_smk0_decime_transfer.mp4 &
	CUDA_VISIBLE_DEVICES=0 python checkmodel.py --orthographic=True --ui=True --smoke_model_name=smk0.copy.decime --video_out=test_video_smk0_decime.mp4

#	python annot_mesh.py --objs=/mnt/hd1/data/smoke_data/runtime_data/models/smk0.copy/head_template_mesh.obj --json=source.json
#	python annot_mesh.py --objs=/mnt/hd2/users/louis/dev/git/Deformation-Transfer-for-Triangle-Meshes/models/highpoly/horse-poses/horse-reference.obj --json=source.json
#	python annot_mesh.py --objs=zzz/target.obj --json=target.json

#check_regression
#rig
#checkmodel
#register_copy 
#checkmodel

bench : 
	CUDA_VISIBLE_DEVICES=0 python bench_pytorch3D.py

checkmodel :
#QT_QPA_PLATFORM=offscreen 
	CUDA_VISIBLE_DEVICES=0 python checkmodel.py --orthographic=True --ui=True --smoke_model_name=smk0.copy.decime --video_out=test_video_smk0.mp4
#	CUDA_VISIBLE_DEVICES=0 python checkmodel.py --orthographic=False --smoke_model_name=smk0 --video_out=test_video_smk0.mp4
#	CUDA_VISIBLE_DEVICES=0 python checkmodel.py --smoke_model_name=decime  --video_out=test_video_decime.mp4


#viewer
#register_copy seq
#seq
#register_copy
#seq demo
#viewer
#demo seq
#server
#python shiftedloss.py
#seq
#gan
#texture
#demo

texture :
	python texture.py

#demo
#deep
#viewer
#dir2

gan :
	python generator.py --data=/media/louis/USBSTICK128/celeba

#/mnt/hd1/data/training_images/celeba
#


annot_mesh :
#	python annot_mesh.py --objs=/mnt/hd2/data/topos/morgan/head_frozen.obj
#	python annot_mesh.py --objs=/mnt/hd1/data/smoke_data/basel/averageMeshNew.obj
#	python annot_mesh.py --objs=/mnt/hd1/data/smoke_data/Morgan/morgan_neutral.obj

	python annot_mesh.py --objs=/mnt/hd1/data/smoke_data/runtime_data/models/smk0.copy.decime/head_template_mesh.obj --json=./annotations/smk0.copy.decime.json &
#	python annot_mesh.py --objs=results/morgan_processed.obj  --json annotations/morgan.json
	python annot_mesh.py --objs=results/morgan_processed2.obj  --json annotations/morgan2.json

annotA :
	python annot_mesh.py --objs=/mnt/hd1/data/smoke_data/runtime_data/models/smk0.decime.decime/head_template_mesh.obj  --json=/mnt/hd1/data/smoke_data/runtime_data/models/smk0.decime.decime/marks.json

annotB : 
	python annot_mesh.py --obj=/mnt/hd1/data/smoke_data/basel/averageMeshNew.obj --json=/mnt/hd1/data/smoke_data/basel/marks.json

#,ppp/scene.obj


yyy :
	python yyy.py

DATAPTH=/mnt/$(HDROOT)/data
DATAPTH=/media/louis/hyperX/data
CLBP=/tmp
CLBP=$(DATAPTH)/celeba

fit1 :
	echo $(SHELL)
	source ./buildenv.sh; buildtheenv444; CUDA_VISIBLE_DEVICES=0 make fit

learnuv :
	python learnuv.py

seg :
	python face_segmentation.py  2>&1 | tee  trcs/$(@)$(TAG)_$(DATE)_$(ID).trc

%.run :
	python $*.py  2>&1 | tee  trcs/$(@)$(TAG)_$(DATE)_$(ID).trc

# pour faire des stats sur la pose, exp, cam ... a partir de celeba
#
# /mnt/hd1/data/synthetic
dir :
	QT_QPA_PLATFORM=offscreen CUDA_VISIBLE_DEVICES=0 MINEKOLEVEL=0 python iterative_fit.py --shuffle_dir=False --shape_params=$(SHAPEP) --expression_params=$(EXPP) --geo_details=True --tqdm=no --cycles=$(CYCLES) --tag $(@)_1 --dosavedebug=False --create_videos=False --showloss=False --celebapath=$(CLBP) --numbers=40000-80000 --fddetect=yes  --loadcache True --display=False  --optimizer=AdamW --orthographic=$(ORTHO) --e_lr=$(LR) --subsample=0.1 --tqdm=no  --smoke_model_name=$(SMODEL) --dataroot=$(DATAROOT)  --dir /mnt/hd2/data/celeba/img_align_celeba   2>&1 | tee trcs/$(@)$(TAG)_$(DATE)_$(ID).trc

fit_celeba :
	python iterative_fit.py --tqdm=no  --tag $(@) --fddetect=no --loadcache=True --celebapath=/mnt/hd1/data/training_images --cachedir=/mnt/hd1/data/pckls --display=False --image=celeba/img_align_celeba/000015.jpg 2>&1 | tee trcs/$(@)$(TAG)_$(DATE)_$(ID).trc

fit2 :
	python iterative_fit.py --fddetect=yes --loadcache=False --image /mnt/hd1/data/hairs_face/hairs.png  2>&1 | tee  $(@)$(TAG)_$(DATE)_$(ID).trc


fit3 : 
	QT_QPA_PLATFORM=offscreen CUDA_LAUNCH_BLOCKINGXX=1 python iterative_fit.py --pckl=params.pckl --fddetect=yes --loadcache=False --image test_femme.jpg

fit :
#	python iterative_fit.py 00002 cuda --dir input_images   2>&1 | tee  $(@)$(TAG)_$(DATE)_$(ID).trc
#	python iterative_fit.py --loadcache True --image 00001.png,00002.png,00003.png,00004.png,00005.png,00006.png,00007.png,00008.png     2>&1 | tee  $(@)$(TAG)_$(DATE)_$(ID).trc
#	python iterative_fit.py --tqdm=yes  --tag $(@) --fddetect=yes --cachedir  salle00_40000.pckl --loadcache True --image salle0/f_0200.png   2>&1 | tee trcs/$(@)$(TAG)_$(DATE)_$(ID).trc

#	python iterative_fit.py --tqdm=yes  --tag $(@)  --celebapath=$(CLBP) --fddetect=yes --cachedir  Celeba_0_999999.pckl --loadcache True --image img_align_celeba/067548.jpg 2>&1 | tee trcs/$(@)$(TAG)_$(DATE)_$(ID).trc
#	python iterative_fit.py --tqdm=no  --tag $(@)  --celebapath=$(CLBP) --fddetect=yes --cachedirX  /media/louis/hyperX/data/pckl_pipnet --loadcache True --display=False --dir /mnt/hd2/data/celeba/img_align_celeba 2>&1 | tee trcs/$(@)$(TAG)_$(DATE)_$(ID).trc

#	python iterative_fit.py --tqdm=no  --tag $(@)  --celebapath=$(CLBP) --fddetect=yes  --loadcache True --display=False --dir /mnt/hd1/data/synthetic 2>&1 | tee trcs/$(@)$(TAG)_$(DATE)_$(ID).trc
	python iterative_fit.py --tqdm=no  --tag $(@)  --shape_params=299 --expression_params=99 --celebapath=$(CLBP) --fddetect=yes  --loadcache True --display=False --dir /mnt/hd1/data/synthetic 2>&1 | tee trcs/$(@)$(TAG)_$(DATE)_$(ID).trc

#	-CUDA_LAUNCH_BLOCKING_XX=1 python iterative_fit.py --tqdm=no  --tag $(@)  --celebapath=$(CLBP) --fddetect=yes --cachedirX  /media/louis/hyperX/data/pckl_pipnet --loadcache True --display=False --number 61 --dir /mnt/hd2/data/sequence 2>&1 | tee trcs/$(@)$(TAG)_$(DATE)_$(ID).trc
#	-CUDA_LAUNCH_BLOCKING_XX=1 python iterative_fit.py --tqdm=no  --tag $(@)  --celebapath=$(CLBP) --fddetect=yes --cachedirX  /media/louis/hyperX/data/pckl_pipnet --loadcache True --display=False --number 100 --dir /mnt/hd2/data/louis-rot-horiz 2>&1 | tee trcs/$(@)$(TAG)_$(DATE)_$(ID).trc

dir0 :
	convert input_images/000060.jpg -rotate -10 imset/out.png
	 CUDA_VISIBLE_DEVICES=0 MINEKOLEVEL=0 python iterative_fit.py --fddetect True --loadcache False  --dir  ./imset   2>&1 | tee  trcs/$(@)$(TAG)_$(DATE)_$(ID).trc

dir1 :
	python iterative_fit.py --fddetect True --loadcache False  --dir  /home/wp24b/data/training/la-meziere-louis/salle1_trim_short   2>&1 | tee trcs/$(@)$(TAG)_$(DATE)_$(ID).trc

viewer : 
	CUDA_VISIBLE_DEVICES=0 MINEKOLEVEL=0 python viewer.py --smoke_model_name=decime --model=/mnt/hd2/data/results/f_0001.pckl  2>&1 | tee trcs/$(@)$(TAG)_$(DATE)_$(ID).trc

dir2 :
	CUDA_VISIBLE_DEVICES=0 MINEKOLEVEL=0 python iterative_fit.py --fddetect True --loadcache False --dir /mnt/hd2/data/sequence 2>&1 | tee trcs/$(@)$(TAG)_$(DATE)_$(ID).trc



TEST_SEQ?=/mnt/hd1/data/sequences/bruno/enroll/enroll.mp4
TEST_SEQ?=/mnt/hd1/data/sequences/louis.mp4
TEST_SEQ?=/mnt/hd1/data/sequences/louis_1.mp4



seq :
	EXT_PARAMS=--text_params=12 TT=bruno TEST_SEQ=/mnt/hd1/data/smoke_data/sequences/bruno/enroll/enroll.mp4 make seqA
#	TT=louis TEST_SEQ=/mnt/hd1/data/sequences/louis_1.mp4 make seqA
#	TT=gael TEST_SEQ=/home/louis/Desktop/gael make seqA



seqA :
#	-SEQ=/mnt/hd2/data/sequence NUMS=19,55,79 make seq1
#	-SEQ=/mnt/hd2/data/louis-rot-horiz NUMS=91,04,123 make seq1
#	-SEQ=$(TEST_SEQ) NUMS=5,50,100,140,177,250,333,400 make seq1
	-SEQ=$(TEST_SEQ) OPT=AdamW CYCLES=1.5 LR=0.005 NUMS=5,100,177,333,400 make seq1
#	-SEQ=$(TEST_SEQ) OPT=LBFGS CYCLES=1.9 LR=0.01 NUMS=5,100,177,333,400 make seq1
#	-SEQ=$(TEST_SEQ) OPT=AdamW CYCLES=1.9 LR=0.01 NUMS=5,100,177,333,400 make seq1
#	-SEQ=$(TEST_SEQ) OPT=Adam CYCLES=0.9 NUMS=5,100,177,333,400 make seq1
#	-SEQ=$(TEST_SEQ) OPT=AdamW CYCLES=0.9  LR=0.04 NUMS=5,100,177,333,400 make seq1
#	-SEQ=$(TEST_SEQ) OPT=Adam CYCLES=0.9  LR=0.04 NUMS=5,100,177,333,400 make seq1

xxx1 :
	-SEQ=$(TEST_SEQ) OPT=Adagrad LR=0.04  NUMS=5,100,177,333,400 make seq1
	-SEQ=$(TEST_SEQ) OPT=Adamax NUMS=5,100,177,333,400 make seq1
	-SEQ=$(TEST_SEQ) OPT=Adamax LR=0.04  NUMS=5,100,177,333,400 make seq1
	-SEQ=$(TEST_SEQ) OPT=AdamW NUMS=5,100,177,333,400 make seq1
	-SEQ=$(TEST_SEQ) OPT=AdamW LR=0.04  NUMS=5,100,177,333,400 make seq1

	-SEQ=$(TEST_SEQ) OPT=Adam NUMS=5,100,177,333,400 make seq1
	-SEQ=$(TEST_SEQ) OPT=Adam LR=0.01 NUMS=5,100,177,333,400 make seq1
	-SEQ=$(TEST_SEQ) OPT=Adam LR=0.02 NUMS=5,100,177,333,400 make seq1
	-SEQ=$(TEST_SEQ) OPT=Adam LR=0.02 CYCLES=0.5 NUMS=5,100,177,333,400 make seq1
	-SEQ=$(TEST_SEQ) OPT=Adam LR=0.04 NUMS=5,100,177,333,400 make seq1

	-SEQ=$(TEST_SEQ) OPT=NAdam NUMS=5,100,177,333,400 make seq1
#	-SEQ=$(TEST_SEQ) OPT=Adagrad NUMS=5,100,177,333,400 make seq1
#	-SEQ=$(TEST_SEQ) OPT=Adadelta NUMS=5,100,177,333,400 make seq1
	-SEQ=$(TEST_SEQ) OPT=Adadelta LR=0.04  NUMS=5,100,177,333,400 make seq1
	-SEQ=$(TEST_SEQ) OPT=NAdam LR=0.04  NUMS=5,100,177,333,400 make seq1
#	-SEQ=$(TEST_SEQ) OPT=RAdam NUMS=5,100,177,333,400 make seq1
#	-SEQ=$(TEST_SEQ) OPT=RMSprop NUMS=5,100,177,333,400 make seq1
#	-SEQ=$(TEST_SEQ) OPT=LBFGS NUMS=5,100,177,333,400 make seq1


#$(warning $(LR))



test_deploy :
	DATAROOT=$(PWD)/data_models/mnt/hd1/data/smoke_data SEQ=data_test/mnt/hd1/data/smoke_data/sequences/bruno/enroll/enroll.mp4 OPT=AdamW CYCLES=1.9 LR=0.01 NUMS=5,100,177,333,400  make seq1

ETAG=E_$(DATE)_$(ID)_$(TT)__$(OPT)_lr_$(LR)_cyc_$(CYCLES)_model_$(SMODEL)_shap_$(SHAPEP)_exp_$(EXPP)_ortho_$(ORTHO)

seq1 :
#	-CUDA_VISIBLE_DEVICES=0 MINEKOLEVEL=0 python iterative_fit.py  --geo_details=False --seq=$(SEQ) --cycles=$(CYCLES) --numbersX=$(NUMS) --optimizer=$(OPT) --e_lr=$(LR) --subsample=0.1 --tqdm=no  --smoke_model_name=decime --dataroot=$(DATAROOT) --tag=tag_$(TT)_$(OPT)_$(LR)_$(CYCLES) 2>&1 | tee trcs/$(@)$(TAG)_$(DATE)_$(ID).trc
#	-CUDA_VISIBLE_DEVICES=0 MINEKOLEVEL=0 python iterative_fit.py  --bin_size=20  --geo_details=False --seq=$(SEQ) --cycles=$(CYCLES) --numbersX=$(NUMS) --optimizer=$(OPT) --e_lr=$(LR) --subsample=0.1 --tqdm=no  --smoke_model_name=decime --dataroot=$(DATAROOT) --tag=tag_$(TT)_$(OPT)_$(LR)_$(CYCLES) 2>&1 | tee trcs/$(@)$(TAG)_$(DATE)_$(ID).trc
#	-QT_QPA_PLATFORM=offscreen CUDA_VISIBLE_DEVICES=0 MINEKOLEVEL=0 python iterative_fit.py  --bin_sizeX=30 --max_faces_per_binX=40460 --geo_details=False --seq=$(SEQ) --cycles=$(CYCLES) --numbersX=$(NUMS) --optimizer=$(OPT) --e_lr=$(LR) --subsample=0.1 --tqdm=no  --smoke_model_name=decime --dataroot=$(DATAROOT) --tag=tag_decime_$(TT)_$(OPT)_$(LR)_$(CYCLES) 2>&1 | tee trcs/decime_$(@)$(TAG)_$(DATE)_$(ID)_$(TT)_$(OPT)_$(LR)_$(CYCLES).trc
	-QT_QPA_PLATFORM=offscreen CUDA_LAUNCH_BLOCKINGXX=1 CUDA_VISIBLE_DEVICES=0 MINEKOLEVEL=0 python iterative_fit.py  --shape_params=$(SHAPEP) --expression_params=$(EXPP) --geo_details=True --geo_displacements=False --seq=$(SEQ) --cycles=$(CYCLES) --numbersX=$(NUMS) --optimizer=$(OPT) --orthographic=$(ORTHO) --e_lr=$(LR) --subsample=0.1 --tqdm=no  --smoke_model_name=$(SMODEL) --dataroot=$(DATAROOT) $(EXT_PARAMS) --tag=$@_$(PTAG)_$(ETAG) 2>&1 | tee trcs/smk0$(@)$(TAG)_$(ETAG).trc
	-(cd debug/$@_$(PTAG)_$(ETAG);  make -f ../../makefile video)
	-echo $@_$(PTAG)_$(ETAG)


rigfolders=$(wildcard /mnt/hd1/data/smoke_data/rig_data/last/*Bruno*)
rigfolders=$(wildcard /mnt/hd1/data/smoke_data/rig_data/subset)

#RIGD?=/mnt/hd1/data/smoke_data/rig_data/data_2022_09_27/2022-09-28_07-51_Bernard
RIGD?=/mnt/hd1/data/smoke_data/rig_data/last/2022-09-28_07-51_Bernard
rigs :
	$(foreach var,$(rigfolders),make rig RIGD=$(var);)

video :
	ffmpeg -i %09d.jpg -vb 20M train.mp4
	ffmpeg -i final_%09d.jpg -vb 20M final.mp4
#	ffmpeg -i train.mp4 -i final.mp4 "concat

exp2 :
	ORTHO=False SMODEL=smk0.copy.decime make rigs seq
#	ORTHO=False SMODEL=smk0.copy make rigs seq
#	ORTHO=True SMODEL=smk0.copy.decime make rigs seq

rigs1 :
	-ORTHO=False NSEV=False SMODEL=smk0.copy.decime make rigs
	-ORTHO=False NSEV=False SMODEL=smk0.copy make rigs
	-ORTHO=False NSEV=False SHAPEP=200 EXP=50 SMODEL=smk0.copy make rigs

rig :
#	python iterative_fit.py --fddetect True --loadcache False --dir /home/wp24b/data/training/la-meziere-louis/salle1_trim_short   2>&1 | tee trcs/$(@)$(TAG)_$(DATE)_$(ID).trc

#	 CUDA_VISIBLE_DEVICES=0 MINEKOLEVEL=0 python iterative_fit.py  --seq=True --loadcache True --subsample=0.1 --cachedir _mnt_hd1_data_downloads_louis0_999999.pckl --tqdm=yes  2>&1 | tee trcs/$(@)$(TAG)_$(DATE)_$(ID).trc

#	 QT_QPA_PLATFORM=offscreen CUDA_VISIBLE_DEVICES=0 MINEKOLEVEL=0 python iterative_fit.py --tag=$@_$(notdir $(RIGD))  --cycles=$(CYCLES) --optimizer=AdamW --e_lr=0.01 --rig=$(RIGD) --smoke_model_name=smk0.copy --dataroot=$(DATAROOT) 2>&1 | tee trcs/$(@)$(TAG)_$(DATE)_$(ID).trc
#	 affinity2=1 QT_QPA_PLATFORM=offscreen CUDA_VISIBLE_DEVICES=0 MINEKOLEVEL=0 python iterative_fit.py --tag=$@_aff2_$(notdir $(RIGD))_decime --geo_details=True --cycles=2.1 --optimizer=AdamW --e_lr=0.005 --rig=$(RIGD) --smoke_model_name=decime --dataroot=$(DATAROOT) 2>&1 | tee trcs/$(@)_decime_$(TAG)_$(DATE)_$(ID).trc
#	 affinity2=1 QT_QPA_PLATFORM=offscreen CUDA_VISIBLE_DEVICES=0 MINEKOLEVEL=0 python iterative_fit.py --tag=$@_aff2_$(notdir $(RIGD))_decime --geo_details=True --cycles=3.1 --optimizer=AdamW --e_lr=0.001 --rig=$(RIGD) --smoke_model_name=decime --dataroot=$(DATAROOT) 2>&1 | tee trcs/$(@)_decime_$(TAG)_$(DATE)_$(ID).trc
#	 affinity3=1 QT_QPA_PLATFORM=offscreen CUDA_VISIBLE_DEVICES=0 MINEKOLEVEL=0 python iterative_fit.py --tag=$@_aff3_$(notdir $(RIGD))_decime --geo_details=True --cycles=2.1 --optimizer=AdamW --e_lr=0.005 --rig=$(RIGD) --smoke_model_name=decime --dataroot=$(DATAROOT) 2>&1 | tee trcs/$(@)_decime_$(TAG)_$(DATE)_$(ID).trc

	 QT_QPA_PLATFORM=offscreen CUDA_VISIBLE_DEVICES=0 MINEKOLEVEL=0 python iterative_fit.py --normalizeSEV=$(NSEV) --tag=$@_$(PTAG)_aff3_phases2_$(notdir $(RIGD))_$(ETAG)_$(SMODEL) --shape_params=$(SHAPEP) --expression_params=$(EXPP) --geo_details=True --geo_displacements=False --cycles=$(CYCLES) --optimizer=AdamW --e_lr=$(LR) --orthographic=$(ORTHO) --rig=$(RIGD) --smoke_model_name=$(SMODEL) --dataroot=$(DATAROOT) --notodisp=0 2>&1 | tee trcs/$(@)_$(TAG)_$(ETAG).trc
	(cd debug/$@_$(PTAG)_aff3_phases2_$(notdir $(RIGD))_$(ETAG)_$(SMODEL); make -f ../../makefile video)

#	 CUDA_VISIBLE_DEVICES=0 MINEKOLEVEL=0 python iterative_fit.py  --rig=/mnt/hd2/data/sequence --loadcacheX True --subsample=0.1 --cachedirX _mnt_hd1_data_downloads_louis0_999999.pckl --tqdm=yes  2>&1 | tee trcs/$(@)$(TAG)_$(DATE)_$(ID).trc

#	 CUDA_VISIBLE_DEVICES=0 MINEKOLEVEL=0 python iterative_fit.py  --rig=/mnt/hd1/data/sequence_image_bruno_disit/49_Images_S --loadcacheX True --subsample=0.1 --cachedirX _mnt_hd1_data_downloads_louis0_999999.pckl --tqdm=yes  2>&1 | tee trcs/$(@)$(TAG)_$(DATE)_$(ID).trc

#CUDA_LAUNCH_BLOCKING=0

rig-gael :
	CUDA_LAUNCH_BLOCKING=1 CYCLES=0.01 PTAG=gael RIGD=/mnt/hd1/data/smoke_data/rig_data/gs/2022-09-20_15-33_selection  make rig

rig-gael2 :
	CUDA_LAUNCH_BLOCKING=1 CYCLES=1. PTAG=gael2 RIGD=/home/louis/Desktop/gael make rig

smat :
	TT=smat TEST_SEQ=/mnt/hd1/data/smoke_data/stills/smat2 make seqA

seq_deep :
	CUDA_VISIBLE_DEVICES=0 MINEKOLEVEL=0 python deep_fit.py   --modelin=$(MODEL) --seq=$(SEQ) --cycles=$(CYCLES) --numbersX=$(NUMS) --optimizer=$(OPT) --e_lr=$(LR) --subsample=0.1 --tqdm=no --dataroot=$(DATAROOT) --tag=tag_$(TT)_$(OPT)_$(LR)_$(CYCLES) 2>&1 | tee trcs/$(@)$(TAG)_$(DATE)_$(ID).trc


MODEL=debug/tag_2022-05-07_17_39_22.539575/model_0010.model
MODEL=debug/tag_2022-05-08_00_35_30.114454/model_0027.model
MODEL=debug/tag_2022-05-08_12_12_00.302008/model_0028.model
MODEL=/mnt/hd1/users/louis/dev/git/cara/flame2/flame2/photometric_optimization/debug/tag_2022-05-08_12_12_00.302008/model_0028.model
CACHEDIR=/mnt/$(HDROOT)/data/pckls2
CACHEDIR=/media/louis/hyperX/data/pckls2
CACHEDIR=/media/louis/hyperX/data/pckls_tiny
CACHEDIR=/media/louis/hyperX/data/pckl_pipnet
CACHEDIR=/media/louis/hyperX/data/pckl_pose

#CACHEDIR=/mnt/hd2/users/louis/dev/git/cara/flame2/flame2/photometric_optimization/tutu
#MODEL=debug/tag_2022-05-27_11_26_58.119196/model_0100.model
MODEL=debug/tag_2022-06-13_18_23_34.536090/model_0071.model
MODEL=debug/tag_2022-06-14_12_08_38.139651/model_0104.model
MODEL=debug/tag_2022-06-14_16_12_31.700951/model_0129.model
#MODEL=./last_model.model
#CACHEDIR=./small

MODEL=debug/tag_2022-09-10_17_21_52.028796/model_0020.model
MODEL=debug/tag_2022-09-10_19_35_23.119690/model_0025.model
MODEL=debug/tag_2022-09-11_21_55_52.770784/model_0045.model
MODEL=debug/tag_2022-09-22_19_54_59.077055/model_0075.model
MODELMN=debug/tag_2022-09-20_17_39_01.127070/model_0026.model

MODEL=/mnt/hd1/data/smoke_data/runtime_data/models/deep/model_0045.model

MODEL=debug/_2023-02-01_19_47_06.016639/model_0064.model
MODEL=debug/deep1_2023-02-02_17_34_24.160029/model_0069.model

deep :
	CUDA_VISIBLE_DEVICES=0 MINEKOLEVEL=0 python deep_fit.py   --tag=$@ --workers=4 --warpingfactor=0. --visall=False --maxEpoch=150 --lrK=0.0001 --backbone=mobilenet_v3  --rigidEpoch=5 --coarseEpoch=10 --tqdm=False --batch_size=24 --fddetect False --modelin=$(MODELMN)  --loadcache True --celebapath=/mnt/hd1/data/smoke_data/training_images --cachedir=/mnt/hd1/data/smoke_data/pckls_celeba --startEpochX=19 --train=True 2>&1 | tee  trcs/$(@)$(TAG)_$(DATE)_$(ID).trc

deep1 :
	CUDA_VISIBLE_DEVICES=0 MINEKOLEVEL=0 python deep_fit.py   --tag=$@ --dataroot=$(DATAROOT) --workers=4 --warpingfactor=0. --visall=False --maxEpoch=150 --lrK=0.0001 --backbone=resnet50 --backbone_frozenX=True  --imageX=img_align_celeba/067548.jpg --subsetX=50,53  --rigidEpoch=5 --coarseEpoch=10 --tqdm=False --batch_size=12 --fddetect False --modelin=$(MODEL) --size_image=256 --cropped_size=256  --loadcache False --celebapath=/mnt/hd1/data/smoke_data/training_images --cachedir=/mnt/hd1/data/smoke_data/pckls --startEpochX=19 --train=True 2>&1 | tee  trcs/$(@)$(TAG)_$(DATE)_$(ID).trc

predict :
	CUDA_VISIBLE_DEVICES=0 MINEKOLEVEL=0 python deep_fit.py  --tag=$@ --modelin=$(MODEL) --predict=test_femme.jpg  --train=False 2>&1 | tee  trcs/$(@)$(TAG)_$(DATE)_$(ID).trc

demo :
#	CUDA_VISIBLE_DEVICES=0 MINEKOLEVEL=0 python demos.py --backboneX=mobilenet_v3 --modelin=$(MODEL) --avatar=_mnt_hd2_data_louis-rot-horiz.pckl

#	CUDA_VISIBLE_DEVICES=0 MINEKOLEVEL=0 python demos.py --backboneX=mobilenet_v3 --modelin=$(MODEL) --anim=/mnt/hd1/data/sequences/bruno/anim/anim.mp4  --avatar=_mnt_hd1_data_sequences_bruno_enroll.pckl

	CUDA_VISIBLE_DEVICES=0 MINEKOLEVEL=0 python demos.py --backboneX=mobilenet_v3 --modelin=$(MODEL) --anim=/mnt/hd1/data/sequences/bruno/anim/anim.mp4  --avatar=debug/tag_bruno_AdamW_0.01_1.9_2022-10-30_13_27_42.485185/_mnt_hd1_data_sequences_bruno_enroll_enroll.mp4.pckl

register_copy :  
	CUDA_VISIBLE_DEVICES=0 MINEKOLEVEL=0 python registerLM.py --dataroot=$(DATAROOT)  --fix=True --copy=True  --tag=$@  2>&1 | tee trcs/$(@)$(TAG)_$(DATE)_$(ID).trc
	CUDA_VISIBLE_DEVICES=0 MINEKOLEVEL=0 python registerLM.py --dataroot=$(DATAROOT)  --decime=True --smoke_model_name=smk0.copy --copy=True  --tag=$@  2>&1 | tee trcs/$(@)$(TAG)_$(DATE)_$(ID).trc
	CUDA_VISIBLE_DEVICES=0 MINEKOLEVEL=0 python registerLM.py --dataroot=$(DATAROOT)  --smoke_model_name=smk0.copy.decime --decime=True --copy=True  --tag=$@  2>&1 | tee trcs/$(@)$(TAG)_$(DATE)_$(ID).trc
	CUDA_VISIBLE_DEVICES=0 MINEKOLEVEL=0 python registerLM.py --dataroot=$(DATAROOT)  --smoke_model_name=smk0.copy.decime --create_json=True --tag=$@  2>&1 | tee trcs/$(@)$(TAG)_$(DATE)_$(ID).trc


lmfix :
	CUDA_VISIBLE_DEVICES=0 MINEKOLEVEL=0 python registerLM.py --dataroot=$(DATAROOT)  --smoke_model_name=smk0 --copy=True --tag=$@  2>&1 | tee trcs/$(@)$(TAG)_$(DATE)_$(ID).trc


register :  
#	CUDA_VISIBLE_DEVICES=0 MINEKOLEVEL=0 python registerLM.py --image_size=512 --cropped_size=512 --tqdm=False   2>&1 | tee trcs/$(@)$(TAG)_$(DATE)_$(ID).trc
#	pylint registerLM.py
#	mypy --ignore-missing-imports registerLM.py
#	pycodestyle registerLM.py
	CUDA_VISIBLE_DEVICES=0 MINEKOLEVEL=0 python registerLM.py --size_image=512 --cropped_size=512   --decime=True --smoke_model_name=smk0.copy --tqdm=False --tag=$@  2>&1 | tee trcs/$(@)$(TAG)_$(DATE)_$(ID).trc
	CUDA_VISIBLE_DEVICES=0 MINEKOLEVEL=0 python registerLM.py --size_image=512 --cropped_size=512   --decime=True --smoke_model_name=smk0.copy.decime --tqdm=False --tag=$@  2>&1 | tee trcs/$(@)$(TAG)_$(DATE)_$(ID).trc

register_train :
#	CUDA_VISIBLE_DEVICES=0 MINEKOLEVEL=0 python registerLM.py  --smoke_model_name=smk0.decime --train=True  --tag=$@ 2>&1 | tee trcs/$(@)$(TAG)_$(DATE)_$(ID).trc
	CUDA_VISIBLE_DEVICES=0 MINEKOLEVEL=0 python registerLM.py  --smoke_model_name=smk0.decime.decime --train=True  --tag=$@ 2>&1 | tee trcs/$(@)$(TAG)_$(DATE)_$(ID).trc

register1 :  
#	CUDA_VISIBLE_DEVICES=0 MINEKOLEVEL=0 python registerLM.py --image_size=512 --cropped_size=512 --tqdm=False   2>&1 | tee trcs/$(@)$(TAG)_$(DATE)_$(ID).trc
#	pylint registerLM.py
#	mypy --ignore-missing-imports registerLM.py
#	pycodestyle registerLM.py
	CUDA_VISIBLE_DEVICES=0 MINEKOLEVEL=0 python registerLM.py --size_image=512 --cropped_size=512  --num_images=2000 --sequence=True --epochs=200 --tqdm=False --tag=$@ --lmmodelX=debug/register_2022-06-26_20_18_22.697661/model_0069.model 2>&1 | tee trcs/$(@)$(TAG)_$(DATE)_$(ID).trc

register2 :  
#	CUDA_VISIBLE_DEVICES=0 MINEKOLEVEL=0 python registerLM.py --image_size=512 --cropped_size=512 --tqdm=False   2>&1 | tee trcs/$(@)$(TAG)_$(DATE)_$(ID).trc
#	pylint registerLM.py
#	mypy --ignore-missing-imports registerLM.py
#	pycodestyle registerLM.py
	CUDA_VISIBLE_DEVICES=0 MINEKOLEVEL=0 python registerLM.py --train=True


transfert :
	python transfert.py

ifeq (0,1)

endif


CELEBA=/mnt/$(HDROOT)/data/downloads/img_align_celeba
CELEBA=/mnt/$(HDROOT)/data/celeba
CELEBA=/media/louis/hyperX/data/celeba/

qtp :
	python qtp.py

cache0 : 
	python iterative_fit.py --buildcache_from ./FFHQ 2>&1 | tee $(@)$(TAG)_$(DATE)_$(ID).trc

cache00 : 
	python iterative_fit.py  --buildcache_from ./sample 2>&1 | tee $(@)$(TAG)_$(DATE)_$(ID).trc

cache10 : 
	python iterative_fit.py --buildcache_from /home/wp24b/data/training/la-meziere-louis/salle1_trim_short  2>&1 | tee $(@)$(TAG)_$(DATE)_$(ID).trc

fini :
	spd-say -l fr "j'ai fini"

cache1111 : 
	-python iterative_fit.py   --buildcache_from $(CELEBA) --celebapath=$(CELEBA) --cachedir  $(CACHEDIR) --tqdm=no
	ls -l /mnt/hd1/data/pckls
	make fini

TRAINING_IMAGES=/mnt/hd1/data/training_images

cache1 : 
	-python process.py --buildcache_from $(TRAINING_IMAGES) --celebapath=$(TRAINING_IMAGES) --cachedir  $(CACHEDIR) --tqdm=yes

cachelouis :
	python iterative_fit.py --tqdm=yes   --buildcache_from /mnt/$(HDROOT)/data/downloads/louis 2>&1 | tee  trcs/$(@)$(TAG)_$(DATE)_$(ID).trc

cache5 : 
	$(SET)  python iterative_fit.py  --buildcache_from ./FFHQ 2>&1 | tee  trcs/$(@)$(TAG)_$(DATE)_$(ID).trc
	mv __FFHQ0_999999.pckl xxx

cache6 :
	$(SET)  python iterative_fit.py --dev cuda --devprocess cuda --to 40000 --buildcache_from input_images


vox :
	$(SET)  python vox.py --frm 0 --to 5000  --buildvox /home/scratch01/chevallierl/dev/mp4 2>&1 | tee  trcs/$(@)$(TAG)_$(DATE)_$(ID).trc

vox2 :
	$(SET)  python vox.py --frm 5000 --to 10000  --buildvox /home/scratch01/chevallierl/dev/mp4 2>&1 | tee  trcs/$(@)$(TAG)_$(DATE)_$(ID).trc

vox3 :
	$(SET)  python vox.py --frm 10000 --to 15000  --buildvox /home/scratch01/chevallierl/dev/mp4 2>&1 | tee  trcs/$(@)$(TAG)_$(DATE)_$(ID).trc

vox4 :
	$(SET)  python vox.py --frm 15000 --to 20000  --buildvox /home/scratch01/chevallierl/dev/mp4 2>&1 | tee  trcs/$(@)$(TAG)_$(DATE)_$(ID).trc

vox5 :
	$(SET)  python vox.py --frm 200000 --to 25000  --buildvox /home/scratch01/chevallierl/dev/mp4 2>&1 | tee  trcs/$(@)$(TAG)_$(DATE)_$(ID).trc

scrapped1 :
	$(SET)  python iterative_fit.py              --to 100000 --buildcache_from /home/wp02/Data/Faces/scrapped/ 2>&1 | tee  trcs/$(@)$(TAG)_$(DATE)_$(ID).trc
scrapped2 :
	$(SET)  python iterative_fit.py --frm 100000 --to 200000 --buildcache_from /home/wp02/Data/Faces/scrapped/ 2>&1 | tee  trcs/$(@)$(TAG)_$(DATE)_$(ID).trc
scrapped3 :
	$(SET)  python iterative_fit.py --frm 200000 --to 300000 --buildcache_from /home/wp02/Data/Faces/scrapped/ 2>&1 | tee  trcs/$(@)$(TAG)_$(DATE)_$(ID).trc
scrapped4 :
	$(SET)  python iterative_fit.py --frm 300000 --to 400000 --buildcache_from /home/wp02/Data/Faces/scrapped/ 2>&1 | tee  trcs/$(@)$(TAG)_$(DATE)_$(ID).trc
scrapped5 :
	$(SET)  python iterative_fit.py --frm 400000 --buildcache_from /home/wp02/Data/Faces/scrapped/ 2>&1 | tee  trcs/$(@)$(TAG)_$(DATE)_$(ID).trc


grille :
	make cache1.grille cache2.grille cache3.grille cache4.grille 
#	make vox.grille vox2.grille vox3.grille  vox4.grille vox5.grille 
#	make scrapped1.grille scrapped2.grille scrapped3.grille scrapped4.grille scrapped5.grille

test : 
	python iterative_fit.py  --buildcache_from ./GGG 2>&1 | tee  trcs/$(@)$(TAG)_$(DATE)_$(ID).trc
	mv __GGG0_999999.pckl xxx
	python iterative_fit.py  --buildcache_from ./FFHQ 2>&1 | tee  trcs/$(@)$(TAG)_$(DATE)_$(ID).trc
	mv __FFHQ0_999999.pckl xxx

	-rm grid_*.png
	python test.py --loadcache False --dir  ./FFHQ   2>&1 | tee  trcs/$(@)$(TAG)_$(DATE)_$(ID).trc
	python test.py --loadcache True --cachedir xxx --dir  ./FFHQ   2>&1 | tee  trcs/$(@)$(TAG)_$(DATE)_$(ID).trc
	python test.py --loadcache False --fddetect=True --dir  ./FFHQ   2>&1 | tee  trcs/$(@)$(TAG)_$(DATE)_$(ID).trc
	python test.py --loadcache True --fddetect=False --cachedir /home/wp24b/data/pckls  2>&1 | tee  trcs/$(@)$(TAG)_$(DATE)_$(ID).trc


test1 :
	python iterative_fit.py  --buildcache_from ./GGG 2>&1 | tee  $(@)$(TAG)_$(DATE)_$(ID).trc
	mv __GGG0_999999.pckl GGG
	python test.py --loadcache True --cachedir  ./GGG   2>&1 | tee  $(@)$(TAG)_$(DATE)_$(ID).trc

test2:
	python test.py --loadcache True --filter $(CELEBA)/000323.jpg --cachedir  /home/wp24b/data/pckls-celeba   2>&1 | tee  trcs/$(@)$(TAG)_$(DATE)_$(ID).trc

test3 :
	python iterative_fit.py  --buildcache_from ./KKK 2>&1 | tee  trcs/$(@)$(TAG)_$(DATE)_$(ID).trc
	mv __KKK0_999999.pckl KKK
	python test.py --loadcache True --cachedir  ./KKK   2>&1 | tee  trcs/$(@)$(TAG)_$(DATE)_$(ID).trc


lzw : 
	python lzw.py

gen :
	which python 
	python expe1.py   2>&1 | tee  trcs/$(@)$(TAG)_$(DATE)_$(ID).trc

xxx2 :
	date | notqdm

server : doc
#	echo $(DATAROOT) ; CUDA_VISIBLE_DEVICES=0 MINEKOLEVEL=0 python server2.py  --help=True
	echo $(DATAROOT) ; PORT=8081 QT_QPA_PLATFORM=offscreen CUDA_VISIBLE_DEVICES=0 MINEKOLEVEL=0 python server2.py  --shape_params=$(SHAPEP) --expression_params=$(EXPP) --geo_details=True  --geo_displacements=False --seq=$(SEQ) --cycles=1.5 --dry=True  --optimizer=AdamW   --orthographic=$(ORTHO) --e_lr=0.05   --subsample=0.1 --tqdm=no --smoke_model_name=$(SMODEL) --dataroot=$(DATAROOT)  --dosavedebug=False --create_videos=True --dataroot=$(DATAROOT) --tag=tag_$@_$(TT)_$(OPT)_$(LR)_$(CYCLES) 2>&1 | tee trcs/$(@)$(TAG)_$(DATE)_$(ID).trc

server1 : doc
#	echo $(DATAROOT) ; CUDA_VISIBLE_DEVICES=0 MINEKOLEVEL=0 python server2.py  --help=True
	echo $(DATAROOT) ; QT_QPA_PLATFORM=offscreen CUDA_VISIBLE_DEVICES=0 MINEKOLEVEL=0 python server2.py  --shape_params=$(SHAPEP) --expression_params=$(EXPP) --geo_details=True  --geo_displacements=False --seq=$(SEQ) --cycles=1.5 --dry=True  --optimizer=AdamW   --orthographic=$(ORTHO) --e_lr=0.05   --subsample=0.1 --tqdm=no --smoke_model_name=$(SMODEL) --dataroot=$(DATAROOT)  --dosavedebug=False --create_videos=True --dataroot=$(DATAROOT) --tag=tag_$@_$(TT)_$(OPT)_$(LR)_$(CYCLES) 2>&1 | tee trcs/$(@)$(TAG)_$(DATE)_$(ID).trc

server_docker :
	DATAROOT=/data/runtime_data CUDA_VISIBLE_DEVICES=0 source ./buildenv.sh; testtorchf; DATAROOT=/data/runtime_data CUDA_VISIBLE_DEVICES=0 make server

test_docker :
	CUDA_VISIBLE_DEVICES=0 source ./buildenv.sh; testtorchf; 


board :
	tensorboard --logdir ./runs --bind_all --port 6091 2>&1 > /dev/null  &

# changer l'endroit ou sont stockée les images etc
# https://linuxconfig.org/how-to-move-docker-s-default-var-lib-docker-to-another-directory-on-ubuntu-debian-linux

# ajouter le plugin nvidia a docker
#https://www.howtogeek.com/devops/how-to-use-an-nvidia-gpu-with-docker-containers/

# ajouter le runtime nvidia
#https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/user-guide.html

# https://forums.linuxmint.com/viewtopic.php?t=287553
run_docker :
	sudo docker run --rm --gpus all jitdeep/cara:v0 /bin/bash -c "cd /code/cara/smoke/smoke/face_fit/; source buildenv.sh ; nvidia-smi; testtorchf"
	sudo docker run --rm --gpus all jitdeep/cara:v0 /bin/bash -c "cd /code/cara/smoke/smoke/face_fit/; source buildenv.sh ; build_python_env_docker install  2>&1 | tee trace_building_env.txt"
	sudo docker run --rm --gpus all jitdeep/cara:v0 /bin/bash -c "cd /code/cara/smoke/smoke/face_fit/; source buildenv.sh ; nvidia-smi;  testtorchf"
#	sudo docker run --rm --gpus all jitdeep/cara:v0 /bin/bash -c "nvidia-smi; cd /code/cara/smoke/smoke/face_fit/; make server_docker"

build_docker_image :
	date
	make build_docker_image1 2>&1 | tee trcs/build_docker.trc
	date
	make run_docker 2>&1 | tee trcs/run_docker.trc


build_docker_image1 :
	sudo ls makefile
	make build_docker_image1_copy
	sudo docker build --rm --progress=auto -t jitdeep/cara:v0 .
	sudo docker login -u jitdeep -p VKr9rjLpkeB2nXx
	sudo docker push jitdeep/cara:v0

build_docker_image_test :
	(cd docker_test; sudo docker build --progress=auto -t jitdeep/test:v0 .)
#	sudo docker login -u jitdeep -p VKr9rjLpkeB2nXx
	(sudo docker push jitdeep/test:v0)


build_docker_image1_copy :
#	gdown --fuzzy https://drive.google.com/file/d/1TN2Z3Ebw_eW0NtZLZK1dbheWAFvDFOP4/view?usp=share_link
#	gunzip runtime_data.tar.gz
#	tar xf runtime_data.tar
	-rm -fr ./runtime_data
	-mkdir ./runtime_data
	-mkdir ./runtime_data/models ./runtime_data/sequences
	cp -r $(DATAROOT)/arcface  ./runtime_data
	cp -r $(DATAROOT)/enroll.mp4  ./runtime_data
	cp -r $(DATAROOT)/models/smk0.copy $(DATAROOT)/models/face-parsing $(DATAROOT)/models/6DRepNet ./runtime_data/models
	cp -r $(DATAROOT)/models/smk0.copy.decime $(DATAROOT)/models/face-parsing $(DATAROOT)/models/6DRepNet ./runtime_data/models
	cp -r $(DATAROOT)/models/epoch59.pth ./runtime_data/models
	cp -r $(DATAROOT)/arcface  ./runtime_data
#	wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2004/x86_64/cuda-keyring_1.0-1_all.deb

# running locally :
# list images :
# sudo docker images

# run image
# sudo docker run  d917119f83cf /bin/bash -c "cd /code/cara/smoke/smoke/face_fit/; make server_docker"

# on qarnot ( image has been pushed on docker hub by build):
run_qarnot :
	python emova.py


check_avatar :
	QT_QPA_PLATFORM=offscreen python checkavatar.py --pickle params.pckl

check_regression_t :
	make check_regression  2>&1 | tee  trcs/$(@)$(TAG)_$(DATE)_$(ID).trc

help :
	python deep_fit.py --doc=True
	python iterative_fit.py --doc=True
	python annot_mesh.py --doc=True
	python registerLM.py --doc=True
	python checkmodel.py --doc=True


check_regression :
	-make fit3
	-CYCLES=0.01 SEQ=/mnt/hd1/data/smoke_data/sequences/bruno/enroll/enroll.mp4 OPT=AdamW LR=0.01 NUMS=5,100,177,333,400 make seq1
	-QT_QPA_PLATFORM=offscreen CUDA_VISIBLE_DEVICES=0 python checkmodel.py --generate_test_sequence=True --orthographic=True --ui=False --annotate=True --dev=cuda --devprocess=cuda --smoke_model_name=smk0.copy --video_out=test_video_smk0.mp4
	-QT_QPA_PLATFORM=offscreen CUDA_VISIBLE_DEVICES=0 python checkmodel.py --generate_test_sequence=True --orthographic=True --ui=False --annotate=True --dev=cuda --devprocess=cuda --smoke_model_name=smk0.copy.decime --video_out=test_video_smk0.mp4
	-CYCLES=0.01 make rig
	-QT_QPA_PLATFORM=offscreen CUDA_VISIBLE_DEVICES=0 MINEKOLEVEL=0 python deep_fit.py --dataroot=$(DATAROOT) --workers=4 --warpingfactor=0. --visall=False --maxEpoch=1 --lrK=0.0001 --backbone=resnet50  --subset=5,8  --rigidEpoch=5 --coarseEpoch=10 --tqdm=False --batch_size=12 --fddetect False --size_image=256 --cropped_size=256  --loadcache False --celebapath=/mnt/hd1/data/smoke_data/training_images --cachedir=/mnt/hd1/data/smoke_data/pckls --train=True 2>&1 | tee  trcs/$(@)$(TAG)_$(DATE)_$(ID).trc

	-QT_QPA_PLATFORM=offscreen make register_copy register
	-QT_QPA_PLATFORM=offscreen make check_avatar


yy :
	QT_QPA_PLATFORM=offscreen CUDA_VISIBLE_DEVICES=0 python checkmodel.py --generate_test_sequence=False --orthographic=True --ui=False --annotate=True --dev=cuda --devprocess=cuda --smoke_model_name=smk0.copy.decime.decime --video_out=test_video_smk0.mp4


install :
#	source ./buildenv.sh; test_versions install 2>&1 | tee trcs/install.trc
#	source ./buildenv.sh; test_versions install 2>&1 | tee trcs/install.trc
	source ./buildenv.sh; python_env install  2>&1 | tee trcs/install_env.trc

doc :
	pandoc README.md -t latex -o www/test/readme.pdf


#####################"
run_t :
	-make check_regression fix_morgan transfer
	-make  seq
	-SMODEL=smk0.copy.decime.transfer make seq
	-make build_docker_image1

run :
	make run_t  2>&1 | tee trcs/run.trc

rrr :
	CUDA_VISIBLE_DEVICES=0 MINEKOLEVEL=0 python deep_fit.py  --modelin=debug/_2023-02-01_19_47_06.016639/model_0064.model   --train=False --predict=/mnt/hd1/data/smoke_data/rig_data/gs/2022-09-20_15-33/Front_01.JPG  --tag=$@ 

seqxx :
	SMODEL=smk0.copy.decime.transfer  TT=bruno TEST_SEQ=/mnt/hd1/data/smoke_data/sequences/bruno/enroll/enroll.mp4 make seqA

ftf :
	make fix_morgan
	python annot_mesh.py --use_3D_location=True --objs=results/morgan_processed2.obj  --json annotations/morgan2.json
	make transfer seqxx
	MODEL=debug/_2023-02-01_19_47_06.016639/model_0064.model make deep1

nothing :
	echo

checkcompo :
	QT_QPA_PLATFORM=offscreen CUDA_VISIBLE_DEVICES=0 python checkmodel.py --smoke_model_name=smk0
	QT_QPA_PLATFORM=offscreen CUDA_VISIBLE_DEVICES=0 python checkmodel.py --smoke_model_name=smk0.copy
	QT_QPA_PLATFORM=offscreen CUDA_VISIBLE_DEVICES=0 python checkmodel.py --smoke_model_name=smk0.copy.decime.decime
	QT_QPA_PLATFORM=offscreen CUDA_VISIBLE_DEVICES=0 python checkmodel.py --smoke_model_name=smk0.copy.decime.transfer

