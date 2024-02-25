export DATE:=$(shell date +%Y-%m-%d_%Hh%Mm%Ss)
export HOST=$(shell hostname)
SHELL=bash
export GITINFO=$(shell git log --pretty=format:"%h - %an, %ar : %s" -1)

train :
	CUDA_AVAILABLE_DEVICES=0 python -c 'import train; train.train("/mnt/hd1/data")' 2>&1 | tee trcs/$(@)_$(DATE).trc

start :
	mkdir -p trcs
	CUDA_AVAILABLE_DEVICES=0 python -c 'import train; train.train("/mnt/NUC/data/vegetables", train_dir="/mnt/hd1/data")' 2>&1 | tee trcs/$(@)_$(DATE).trc
#	CUDA_AVAILABLE_DEVICES=0 python -c 'import train; train.test("/mnt/NUC/data/vegetables", train_dir="/mnt/hd1/data")'
#	CUDA_AVAILABLE_DEVICES=0 python -c 'import train; train.predict("/mnt/NUC/data/vegetables")'
#	CUDA_AVAILABLE_DEVICES=0 python -c 'import server; server.go("/mnt/NUC/data/vegetables")'

$(warning $(GITINFO))

server :
	CUDA_AVAILABLE_DEVICES=0  python -c 'import server; server.go("/mnt/hd1/data")'

test :
	CUDA_AVAILABLE_DEVICES=0 python -c 'import train; train.test("/mnt/NUC/data/vegetables", train_dir="/mnt/hd1/data")'


run :
	date
	source ${HOME}/scripts/.bashrc; spy; pyenv; make server_nuc

server_nuc :
	python -c 'import server; server.go("/media/usb-seagate2/data/vegetables")'

commit_push :
	git config --global user.email 'louis.chevallier@gmail.com'
	git config --global user.name 'louis chevallier'
	git commit -a -m 'train+save+onnx+server+test+predict'
	git remote set-url origin https://louis-chevallier:"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"@github.com/louis-chevallier/vegetables.git
#	git remote set-url origin https://louis-chevallier:"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"@github.com/louis-chevallier/vegetables.git
	git push --set-upstream origin main

key :
# https://docs.cherrypy.dev/en/latest/deploy.html
	openssl genrsa -out privkey.pem 2048
	openssl req -new -x509 -days 365 -key privkey.pem -out cert.pem
