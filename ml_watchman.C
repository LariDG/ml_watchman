//
//  ml_watchman
//
//  Created by Larisa Dorman-Gajic
//


#include "TTree.h"
#include "TBranch.h"
#include "TString.h"
#include "TSystem.h"
#include "TROOT.h"
#include "TStopwatch.h"
#include "TF1.h"
#include "TMath.h"
#include <iostream>
#include <fstream>
#include <iomanip>


using namespace std;

int ml_watchman() {
    
    TFile * input = new TFile("","READONLY");
    TTree * regTree = (TTree*) input -> Get("data");
    
    
    
    ofstream csvfile;
    csvfile.open ("merged_electrons_fit_wbls_3_baseline.csv");

    
    
    int gtid; regTree->SetBranchAddress("gtid", &gtid);
    double mc_energy; regTree->SetBranchAddress("mc_energy", &mc_energy);
    double mcx; regTree->SetBranchAddress("mcx", &mcx);
    double mcy; regTree->SetBranchAddress("mcy", &mcy);
    double mcz; regTree->SetBranchAddress("mcz", &mcz);
    double pe; regTree->SetBranchAddress("pe", &pe);
    double x; regTree->SetBranchAddress("x", &x);
    double y; regTree->SetBranchAddress("y", &y);
    double z; regTree->SetBranchAddress("z", &z);
    double n100_prev; regTree->SetBranchAddress("n100_prev", &n100_prev);
    double dxmcx; regTree->SetBranchAddress("dxmcx", &dxmcx);
    double dymcy; regTree->SetBranchAddress("dymcy", &dymcy);
    double dzmcz; regTree->SetBranchAddress("dzmcz", &dzmcz);
    double drmcr; regTree->SetBranchAddress("drmcr", &drmcr);
    double closestPMT_prev; regTree->SetBranchAddress("closestPMT_prev", &closestPMT_prev);
    
    
    
    csvfile << "gtid " << "mc_energy " << "mcx " << "mcy " << "mcz " << "pe " << "x " << "y " << "z " << "n_100 " << "dxmcx " << "dymcy " << "dzmcz " << "drmcr " << "closestPMT_prev " << "reco_wall_r " << "reco_wall_z " << "true_wall_r " << "true_wall_z " << "\n";
    
    for (Long64_t ievt=0; ievt<regTree->GetEntries();ievt++) {
        regTree -> GetEntry(ievt);
       // if(closestPMT_prev <= 500) {
       //     continue;
       // }
        if(drmcr < 0) {
            continue;
        }
        
        double reco_vtx_r = x*x + y*y;
        double reco_wall_r = 10000-TMath::Sqrt(recoVtxR);
        double reco_wall_z = 10000-TMath::Abs(z);
        
        double true_vtx_r = mcx+mcx + mcy*mcy;
        double true_wall_r = 10000-TMath::Sqrt(trueVtxR);
        double true_wall_z = 10000-TMath::Abs(mcz);
        
        
        
        csvfile << gtid << " " << mc_energy << " " << mcx << " " << mcy << " " << mcz << " " << pe << " " << x << " " << y << " " << z << " " << n100_prev << " " << dxmcx << " " << dymcy << " " << dzmcz << " " << drmcr << " " << closestPMT_prev << " " << reco_wall_r << " " << reco_wall_z << " " << true_wall_r << " " << true_wall_z << "\n" ;
      

    };

    csvfile.close();
    return 0;
}

