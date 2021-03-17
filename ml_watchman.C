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
    
    TFile * input = new TFile("b_elect_flat_1_10_MeV_3_WbLS.root","READONLY");
    TTree * regTree = (TTree*) input -> Get("data");
    
    
    
    ofstream mc_data;
    mc_data.open ("elect_flat_1_10_MeV_3_WbLS_mc.csv");
    
    ofstream reco_data;
    reco_data.open ("elect_flat_1_10_MeV_3_WbLS_reco.csv");
    
    ofstream energy_data;
    energy_data.open ("mc_energy_vs_n100.csv");
    
    
    int gtid; regTree->SetBranchAddress("gtid", &gtid);
    double mc_energy; regTree->SetBranchAddress("mc_energy", &mc_energy);
    double mcx; regTree->SetBranchAddress("mcx", &mcx);
    double mcy; regTree->SetBranchAddress("mcy", &mcy);
    double mcz; regTree->SetBranchAddress("mcz", &mcz);
    double pe; regTree->SetBranchAddress("pe", &pe);
    double x; regTree->SetBranchAddress("x", &x);
    double y; regTree->SetBranchAddress("y", &y);
    double z; regTree->SetBranchAddress("z", &z);
    //int ncherenkovhit; regTree->SetBranchAddress("ncherenkovhit", &ncherenkovhit);
    double n100; regTree->SetBranchAddress("n100", &n100);
    //double n100_prev; regTree->SetBranchAddress("n100_prev", &n100_prev);
    //double n9_prev; regTree->SetBranchAddress("n9_prev", &n9_prev);
    double closestPMT; regTree->SetBranchAddress("closestPMT", &closestPMT);
    //double closestPMT_prev; regTree->SetBranchAddress("closestPMT_prev", &closestPMT_prev);
    
    
    mc_data << "gtid," << "mcx," << "mcy," << "mcz," << "mc_energy," << "true_wall_r," << "true_wall_z," << "\n";
    
    reco_data << "gtid," << "x," << "y," << "z," << "pe," << "closestPMT," << "n100," << "reco_wall_r," << "reco_wall_z," << "\n";
    
    
    for (Long64_t ievt=0; ievt<regTree->GetEntries();ievt++) {
        regTree -> GetEntry(ievt);

        if((x < -5000) or (x > 5000) or (y < -5000) or (y > 5000) or (y < -5000) or (y > 5000) or (closestPMT < 500)) {
            continue;
        }
        
        double reco_vtx_r = x*x + y*y;
        double reco_wall_r = 10000-TMath::Sqrt(reco_vtx_r);
        double reco_wall_z = 10000-TMath::Abs(z);
        
        double true_vtx_r = mcx*mcx + mcy*mcy;
        double true_wall_r = 10000-TMath::Sqrt(true_vtx_r);
        double true_wall_z = 10000-TMath::Abs(mcz);
        
        if((true_wall_r < 5000) or (true_wall_z < 5000)) {
            continue;
        }
        
        mc_data << gtid << "," << mcx << "," << mcy << "," << mcz << "," << mc_energy << "," << closestPMT << "," << true_wall_r << "," << true_wall_z << "\n";
        
        reco_data << gtid << "," << x << "," << y << "," << z << "," << pe << "," << closestPMT << "," << n100 << "," << reco_wall_r << "," << reco_wall_z << "\n";
        

    };

    mc_data.close();
    reco_data.close();
    energy_data.close();
    
    return 0;
}

