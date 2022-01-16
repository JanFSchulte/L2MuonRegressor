#!/bin/sh
#SBATCH --job-name=L2MuonDNN #Job name
#SBATCH --mail-type=FAIL # Mail events (NONE, BEGIN, END, FAIL, ALL)
#SBATCH --mail-user=schul105@purdue.edu # Where to send mail	
#SBATCH --output=/depot/cms/users/schul105/L2MuonRegressor/test-%A.out	# Name output file 

LRATE=$1
EPOCHS=$2
ACT=$3

pwd; date; hostname


#   Run fewzz job  DY 1D in M   PI bkg
#   $1 == input parameter (working directory)
mydir=/depot/cms/users/schul105/L2MuonRegressor
cd $mydir
echo "Working in "`pwd`

./set_env.sh
python3 training_regressorForL2Muons_v3.py $LRATE $EPOCHS $ACT

date

