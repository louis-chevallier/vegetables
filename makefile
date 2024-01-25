
start :
#	python -c 'import train; train.train("/mnt/hd1/data")'
#	python -c 'import train; train.test("/mnt/hd1/data")'
	python -c 'import train; train.predict("/mnt/hd1/data")'



commit_push :
	git config --global user.email 'louis.chevallier@gmail.com'
	git config --global user.name 'louis chevallier'
	git commit -a -m 'train+save+onnx+server+test+predict'

	git remote set-url origin https://louis-chevallier:"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"@github.com/louis-chevallier/vegetables.git
	git push --set-upstream origin main



