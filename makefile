
start :
#	python -c 'import train; train.train("/mnt/hd1/data")'
#	python -c 'import train; train.test("/mnt/hd1/data")'



commit_push :
	git config --global user.email 'louis.chevallier@gmail.com'
	git config --global user.name 'louis chevallier'
	git commit -a -m 'train+save+onnx+server+test+predict'




