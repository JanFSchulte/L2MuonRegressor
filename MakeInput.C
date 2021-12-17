#define MakeInput_cxx
#include "MakeInput.h"
#include <TH2.h>
#include <TStyle.h>
#include <TCanvas.h>

void MakeInput::Loop()
{
//   In a ROOT session, you can do:
//      root> .L MakeInput.C
//      root> MakeInput t
//      root> t.GetEntry(12); // Fill t data members with entry number 12
//      root> t.Show();       // Show values of entry 12
//      root> t.Show(16);     // Read and show values of entry 16
//      root> t.Loop();       // Loop on all entries
//

//     This is the loop skeleton where:
//    jentry is the global entry number in the chain
//    ientry is the entry number in the current Tree
//  Note that the argument to GetEntry must be:
//    jentry for TChain::GetEntry
//    ientry for TTree::GetEntry and TBranch::GetEntry
//
//       To read only selected branches, Insert statements like:
// METHOD1:
//    fChain->SetBranchStatus("*",0);  // disable all branches
//    fChain->SetBranchStatus("branchname",1);  // activate branchname
// METHOD2: replace line
//    fChain->GetEntry(jentry);       //read all branches
//by  b_branchname->GetEntry(ientry); //read only this branch
   if (fChain == 0) return;

   std::ofstream myfile;
   myfile.open ("data/inputfile_regressorTraining.csv");
   TString Var_names[26] = {"seg1_x", "seg1_y", "seg1_z", "seg1_r", "seg1_lay", "seg2_x", "seg2_y", "seg2_z", "seg2_r", "seg2_lay", "seg3_x", "seg3_y", "seg3_z", "seg3_r", "seg3_lay", "seg4_x", "seg4_y", "seg4_z", "seg4_r", "seg4_lay", "phiBend12", "phiBend23", "phiBend34", "L2_pt", "L2_eta", "L2_phi"};
   TH1F * seg_vars[26];
   for(int i=0;i<26;i++){
     
     myfile << Var_names[i]+",";
   }
   myfile<<"\n";
   Long64_t nentries = fChain->GetEntriesFast();
   char buffer [50000];
   Long64_t nbytes = 0, nb = 0;
   for (Long64_t jentry=0; jentry<nentries;jentry++) {
      Long64_t ientry = LoadTree(jentry);
      if (ientry < 0) break;
      nb = fChain->GetEntry(jentry);   nbytes += nb;
      // if (Cut(ientry) < 0) continue;
      float seg1_x, seg1_y, seg1_z, seg1_r, seg1_lay, seg2_x, seg2_y, seg2_z, seg2_r, seg2_lay, seg3_x, seg3_y, seg3_z, seg3_r, seg3_lay, seg4_x, seg4_y, seg4_z, seg4_r, seg4_lay, phiBend12, phiBend23, phiBend34 = -999;
      for(int iL2=0; iL2<nL2s; iL2++){ 
	for(int iSeg=0;iSeg<5;iSeg++){
	  if(segment_L2ID[iSeg]!=iL2) continue;
	  if(iSeg==0){
	    seg1_x = segment_globalX[iSeg];
	    seg1_y = segment_globalY[iSeg];
	    seg1_z = segment_globalZ[iSeg];
	    seg1_r = segment_globalR[iSeg];
	    seg1_lay = segment_layerID[iSeg];
	    phiBend12 = segment_phiBend[iSeg];
	  }
	  if(iSeg==1){
	    seg2_x = segment_globalX[iSeg];
	    seg2_y = segment_globalY[iSeg];
	    seg2_z = segment_globalZ[iSeg];
	    seg2_r = segment_globalR[iSeg];
	    seg2_lay = segment_layerID[iSeg];
	    phiBend23 = segment_phiBend[iSeg];
	  }
	  if(iSeg==2){
	    seg3_x = segment_globalX[iSeg];
	    seg3_y = segment_globalY[iSeg];
	    seg3_z = segment_globalZ[iSeg];
	    seg3_r = segment_globalR[iSeg];
	    seg3_lay = segment_layerID[iSeg];
	    phiBend34 = segment_phiBend[iSeg];
	  }
	  if(iSeg==3){
	    seg4_x = segment_globalX[iSeg];
	    seg4_y = segment_globalY[iSeg];
	    seg4_z = segment_globalZ[iSeg];
	    seg4_r = segment_globalR[iSeg];
	    seg4_lay = segment_layerID[iSeg];
	  }

	}
	sprintf (buffer, "%f, %f, %f, %f, %f, %f, %f, %f, %f, %f, %f, %f, %f, %f, %f, %f, %f, %f, %f, %f, %f, %f, %f, %f, %f, %f\n",seg1_x, seg1_y, seg1_z, seg1_r, seg1_lay, seg2_x, seg2_y, seg2_z, seg2_r, seg2_lay, seg3_x, seg3_y, seg3_z, seg3_r, seg3_lay, seg4_x, seg4_y, seg4_z, seg4_r, seg4_lay, phiBend12, phiBend23, phiBend34, l2_pt[iL2], l2_eta[iL2], l2_phi[iL2]);
	myfile << buffer;
      }
      for(int iSeg=0;iSeg<100;iSeg++){
	if(l2_pt[segment_L2ID[iSeg]]>10 && fabs(l2_eta[segment_L2ID[iSeg]])<0.8 && !(segment_globalX[iSeg]==0 && segment_globalY[iSeg]==0 && segment_globalZ[iSeg]==0)){                                                                                             
	  std::cout<<"For "<<segment_L2ID[iSeg]<<"th L2, with pt:"<<l2_pt[segment_L2ID[iSeg]]<<", eta:"<<l2_eta[segment_L2ID[iSeg]]<<", phi:"<<l2_phi[segment_L2ID[iSeg]]<<std::endl;                                                                                
	  std::cout<<"For "<<segment_L2ID[iSeg]<<"th L2, the gen pt:"<<gen_pt[segment_L2ID[iSeg]]<<", eta:"<<gen_eta[segment_L2ID[iSeg]]<<", phi:"<<gen_phi[segment_L2ID[iSeg]]<<std::endl;                                                                          
	  std::cout<<"For "<<iSeg<<"th seg of "<< segment_L2ID[iSeg]<<"L2, with layerID:"<<segment_layerID[iSeg]<<", global X, Y, Z, R:"<<segment_globalX[iSeg]<<", "<<segment_globalY[iSeg]<<", "<<segment_globalZ[iSeg]<<", "<<segment_globalR[iSeg]<<std::endl;   
        }                                                                                                                          
      }                                                                                                                            
      std::cout<<"_______________________________________________________________________"<<std::endl;

   }
   myfile.close();
}
