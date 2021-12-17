#source /cvmfs/cms.cern.ch/cmsset_default.sh
#export SCRAM_ARCH="slc7_amd64_gcc820"
#export CMSSW_VERSION="CMSSW_11_1_2"

source /cvmfs/cms.cern.ch/cmsset_default.sh

#cmsrel "$CMSSW_VERSION"
#cd "$CMSSW_VERSION/src"

#cmsenv
#scram b

module load anaconda/5.3.1-py37
source activate hmumu_coffea
source setup_proxy.sh

