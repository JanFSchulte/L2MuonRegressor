import subprocess

for lRate in [0.01,0.001,0.0001]:
	for epochs in [100,1000]:
		for activation in ["linear","sigmoid"]:
			subprocess.call(["sbatch --gpus=1 --mem=8G -N1 -n1 --time=24:00:00  submit_training.sh %f %d %s"%(lRate,epochs,activation)],shell=True)

