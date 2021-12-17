//////////////////////////////////////////////////////////
// This class has been automatically generated on
// Wed Dec 15 12:36:40 2021 by ROOT version 6.22/09
// from TTree t/
// found on file: segments.root
//////////////////////////////////////////////////////////

#ifndef MakeInput_h
#define MakeInput_h

#include <TROOT.h>
#include <TChain.h>
#include <TFile.h>

// Header file for the classes stored in the TTree if any.

class MakeInput {
public :
   TTree          *fChain;   //!pointer to the analyzed TTree or TChain
   Int_t           fCurrent; //!current Tree number in a TChain

// Fixed size dimensions of array or collections stored in the TTree if any.

   // Declaration of leaf types
   Float_t         gen_pt[100];
   Float_t         gen_eta[100];
   Float_t         gen_phi[100];
   Int_t           gen_charge[100];
   Int_t           nL2s;
   Int_t           nSegments;
   Float_t         l2_pt[100];
   Float_t         l2_eta[100];
   Float_t         l2_phi[100];
   Int_t           segment_L2ID[100];
   Int_t           segment_layerID[100];
   Float_t         segment_globalX[100];
   Float_t         segment_globalY[100];
   Float_t         segment_globalZ[100];
   Float_t         segment_globalR[100];
   Float_t         segment_phi[100];
   Float_t         segment_deltaPhi[100];
   Float_t         segment_phiBend[100];

   // List of branches
   TBranch        *b_gen_pt;   //!
   TBranch        *b_gen_eta;   //!
   TBranch        *b_gen_phi;   //!
   TBranch        *b_gen_charge;   //!
   TBranch        *b_nL2s;   //!
   TBranch        *b_nSegments;   //!
   TBranch        *b_l2_pt;   //!
   TBranch        *b_l2_eta;   //!
   TBranch        *b_l2_phi;   //!
   TBranch        *b_segment_L2ID;   //!
   TBranch        *b_segment_layerID;   //!
   TBranch        *b_segment_globalX;   //!
   TBranch        *b_segment_globalY;   //!
   TBranch        *b_segment_globalZ;   //!
   TBranch        *b_segment_globalR;   //!
   TBranch        *b_segment_phi;   //!
   TBranch        *b_segment_deltaPhi;   //!
   TBranch        *b_segment_phiBend;   //!

   MakeInput(TTree *tree=0);
   virtual ~MakeInput();
   virtual Int_t    Cut(Long64_t entry);
   virtual Int_t    GetEntry(Long64_t entry);
   virtual Long64_t LoadTree(Long64_t entry);
   virtual void     Init(TTree *tree);
   virtual void     Loop();
   virtual Bool_t   Notify();
   virtual void     Show(Long64_t entry = -1);
};

#endif

#ifdef MakeInput_cxx
MakeInput::MakeInput(TTree *tree) : fChain(0) 
{
// if parameter tree is not specified (or zero), connect the file
// used to generate this class and read the Tree.
   if (tree == 0) {
      TFile *f = (TFile*)gROOT->GetListOfFiles()->FindObject("data/segments.root");
      if (!f || !f->IsOpen()) {
         f = new TFile("data/segments.root");
      }
      TDirectory * dir = (TDirectory*)f->Get("data/segments.root:/SegmentAnalyzer");
      dir->GetObject("t",tree);

   }
   Init(tree);
}

MakeInput::~MakeInput()
{
   if (!fChain) return;
   delete fChain->GetCurrentFile();
}

Int_t MakeInput::GetEntry(Long64_t entry)
{
// Read contents of entry.
   if (!fChain) return 0;
   return fChain->GetEntry(entry);
}
Long64_t MakeInput::LoadTree(Long64_t entry)
{
// Set the environment to read one entry
   if (!fChain) return -5;
   Long64_t centry = fChain->LoadTree(entry);
   if (centry < 0) return centry;
   if (fChain->GetTreeNumber() != fCurrent) {
      fCurrent = fChain->GetTreeNumber();
      Notify();
   }
   return centry;
}

void MakeInput::Init(TTree *tree)
{
   // The Init() function is called when the selector needs to initialize
   // a new tree or chain. Typically here the branch addresses and branch
   // pointers of the tree will be set.
   // It is normally not necessary to make changes to the generated
   // code, but the routine can be extended by the user if needed.
   // Init() will be called many times when running on PROOF
   // (once per file to be processed).

   // Set branch addresses and branch pointers
   if (!tree) return;
   fChain = tree;
   fCurrent = -1;
   fChain->SetMakeClass(1);

   fChain->SetBranchAddress("gen_pt", gen_pt, &b_gen_pt);
   fChain->SetBranchAddress("gen_eta", gen_eta, &b_gen_eta);
   fChain->SetBranchAddress("gen_phi", gen_phi, &b_gen_phi);
   fChain->SetBranchAddress("gen_charge", gen_charge, &b_gen_charge);
   fChain->SetBranchAddress("nL2s", &nL2s, &b_nL2s);
   fChain->SetBranchAddress("nSegments", &nSegments, &b_nSegments);
   fChain->SetBranchAddress("l2_pt", l2_pt, &b_l2_pt);
   fChain->SetBranchAddress("l2_eta", l2_eta, &b_l2_eta);
   fChain->SetBranchAddress("l2_phi", l2_phi, &b_l2_phi);
   fChain->SetBranchAddress("segment_L2ID", segment_L2ID, &b_segment_L2ID);
   fChain->SetBranchAddress("segment_layerID", segment_layerID, &b_segment_layerID);
   fChain->SetBranchAddress("segment_globalX", segment_globalX, &b_segment_globalX);
   fChain->SetBranchAddress("segment_globalY", segment_globalY, &b_segment_globalY);
   fChain->SetBranchAddress("segment_globalZ", segment_globalZ, &b_segment_globalZ);
   fChain->SetBranchAddress("segment_globalR", segment_globalR, &b_segment_globalR);
   fChain->SetBranchAddress("segment_phi", segment_phi, &b_segment_phi);
   fChain->SetBranchAddress("segment_deltaPhi", segment_deltaPhi, &b_segment_deltaPhi);
   fChain->SetBranchAddress("segment_phiBend", segment_phiBend, &b_segment_phiBend);
   Notify();
}

Bool_t MakeInput::Notify()
{
   // The Notify() function is called when a new file is opened. This
   // can be either for a new TTree in a TChain or when when a new TTree
   // is started when using PROOF. It is normally not necessary to make changes
   // to the generated code, but the routine can be extended by the
   // user if needed. The return value is currently not used.

   return kTRUE;
}

void MakeInput::Show(Long64_t entry)
{
// Print contents of entry.
// If entry is not specified, print current entry
   if (!fChain) return;
   fChain->Show(entry);
}
Int_t MakeInput::Cut(Long64_t entry)
{
// This function may be called from Loop.
// returns  1 if entry is accepted.
// returns -1 otherwise.
   return 1;
}
#endif // #ifdef MakeInput_cxx
