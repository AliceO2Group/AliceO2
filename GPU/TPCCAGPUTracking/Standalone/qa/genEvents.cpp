#include <iostream>
#include <fstream>
#include <string.h>

#include "Rtypes.h"
#include "AliHLTTPCCAClusterData.h"
#include "AliHLTTPCCAMCInfo.h"
#include "AliHLTTPCClusterMCData.h"
#include "AliHLTTPCCAParam.h"
#include "AliHLTTPCGMPhysicalTrackModel.h"
#include "AliHLTTPCGMPropagator.h"
#include "AliHLTTPCCAStandaloneFramework.h"
#include "AliHLTTPCGMMerger.h"
 
#include "../cmodules/qconfig.h"
#include "TRandom.h"
#include "TMath.h"
#include "TH1F.h"
#include "TFile.h"
#include "TCanvas.h"
#include "TPad.h"

#include "include.h"

#include <iostream>
#include <iomanip>
#include <limits>

using namespace std;

struct GenCluster
{
  int fSector;
  int fRow; 
  int fMCID;
  float fX;
  float fY;
  float fZ;
  unsigned int fId;
};

const double kTwoPi = TMath::TwoPi();//2.*kPi;
const double kSliceDAngle = kTwoPi/18.;
const double kSliceAngleOffset = kSliceDAngle/2;

int GetSlice( double GlobalPhi )
{
  double phi = GlobalPhi;
  //  std::cout<<" GetSlice: phi = "<<phi<<std::endl;

  if( phi >= kTwoPi ) phi -= kTwoPi;
  if( phi < 0 ) phi+= kTwoPi; 
  return (int) ( phi / kSliceDAngle );
}

int GetDSlice( double LocalPhi )
{
  return GetSlice( LocalPhi + kSliceAngleOffset );
}

double GetSliceAngle( int iSlice )
{
  return kSliceAngleOffset + iSlice*kSliceDAngle;
}

int RecalculateSlice( AliHLTTPCGMPhysicalTrackModel &t, int &iSlice )
{
  double phi = atan2( t.GetY(), t.GetX() );
  //  std::cout<<" recalculate: phi = "<<phi<<std::endl;
  int dSlice = GetDSlice( phi );

  if( dSlice == 0 ) return 0; // nothing to do

  //  std::cout<<" dSlice = "<<dSlice<<std::endl;
  double dAlpha = dSlice*kSliceDAngle;
  // rotate track on angle dAlpha

  t.Rotate(dAlpha);
  
  iSlice+=dSlice;
  if( iSlice>=18 ) iSlice-=18;
  return 1;
}

double GetGaus( double sigma )
{
  double x = 0;
  do{
    x = gRandom->Gaus(0.,sigma);
    if( fabs(x)<=3.5*sigma ) break;
  } while(1);
  return x;
}

static TH1F* hClusterError[3][2]={{0,0},{0,0},{0,0}};

void InitEventGenerator()
{
  const char* rows[3] = {"0-63","128-159","64-127"};
  for( int i=0; i<3; i++){
    for( int j=0; j<2; j++){
      char name[1024], title[1024];
 
      sprintf(name, "clError%s%d", (j==0 ?"Y" :"Z"), i);

      sprintf(title, "Cluster %s Error for rows %s",(j==0 ?"Y" :"Z"), rows[i]);

      hClusterError[i][j] = new TH1F(name, title, 1000, 0., .7 );
      hClusterError[i][j]->GetXaxis()->SetTitle("Cluster Error [cm]");
    }
  }
}

void FinishEventGenerator()
{
  TFile* tout = new TFile("generator.root", "RECREATE");
  TCanvas *c = new TCanvas("ClusterErrors","Cluste rErrors",0,0,700,700.*2./3.);
  c->Divide(3,2);
  int ipad=1;
  for( int j=0; j<2; j++){
    for( int i=0; i<3; i++){  
      c->cd(ipad++);
      int k=i;
      if( i==1 ) k=2;
      if( i==2 ) k = 1;
      if( tout ) hClusterError[k][j]->Write();
      gPad->SetLogy();
      hClusterError[k][j]->Draw();
      //delete hClusterError[i][j];
      //hClusterError[i][j]=0;
    }
  }
  c->Print("plots/clusterErrors.pdf");
  delete c;
  if (tout){
    tout->Close();
    delete tout;
  }
}

int GenerateEvent(const AliHLTTPCCAParam& sliceParam, char* filename)
{
  static int iEvent = -1;
  iEvent++;
  if( iEvent==0 ){
    gRandom->SetSeed(configStandalone.seed);
  }

  std::ofstream out;
  out.open(filename, std::ofstream::binary);
  if (out.fail())
    {
      printf("Error opening file\n");
      return(1);
    }


  int nTracks = configStandalone.configEG.numberOfTracks; //Number of MC tracks, must be at least as large as the largest fMCID assigned above
  cout<<"NTracks "<<nTracks<<endl;
  std::vector<AliHLTTPCCAMCInfo> mcInfo(nTracks);
  memset(mcInfo.data(), 0, nTracks * sizeof(mcInfo[0]));

  //double Bz = sliceParam.ConstBz();
  //std::cout<<"Bz[kG] = "<<sliceParam.BzkG()<<std::endl;

  AliHLTTPCGMPropagator prop;
  {
    prop.SetToyMCEventsFlag( kTRUE );
    AliHLTTPCCAStandaloneFramework &hlt = AliHLTTPCCAStandaloneFramework::Instance();
    const AliHLTTPCGMMerger &merger = hlt.Merger();
    prop.SetPolynomialField( merger.pField() );	  
  }
  
  //  const double kCLight = 0.000299792458;
  //Bz*=kCLight;

  std::vector<GenCluster> vClusters;
  int clusterId = 0; //Here we count up the cluster ids we fill (must be unique).
  //gRandom->SetSeed(0);
  //unsigned int seed = gRandom->GetSeed();
  
  for (int itr = 0; itr < nTracks; itr++){
    // std::cout<<"Track "<<itr<<":"<<std::endl;
    //gRandom->SetSeed(seed);   

    mcInfo[itr].fPID = -100; //-100: Unknown / other, 0: Electron, 1, Muon, 2: Pion, 3: Kaon, 4: Proton      
    mcInfo[itr].fCharge = 1;
    mcInfo[itr].fPrim = 1; //Primary particle
    mcInfo[itr].fPrimDaughters = 0; //Primary particle with daughters in the TPC
    mcInfo[itr].fX = 0; //Position of MC track at entry of TPC / first hit in the TPC
    mcInfo[itr].fY = 0;
    mcInfo[itr].fZ = 0;
    mcInfo[itr].fPx = 0; //Momentum of MC track at that position
    mcInfo[itr].fPy = 0;
    mcInfo[itr].fPz = 0;
    
    AliHLTTPCGMPhysicalTrackModel t;
    double dphi = kTwoPi/nTracks;
    double phi = kSliceAngleOffset + dphi*itr;
    double eta = gRandom->Uniform(-1.5,1.5);

    double theta = 2*TMath::ATan(1./TMath::Exp(eta));
    double lambda = theta-TMath::Pi()/2;
    //double theta = gRandom->Uniform(-60,60)*TMath::Pi()/180.;
    double pt = .08*std::pow(10,gRandom->Uniform(0,2.2));
    
    double q = 1.;
    int iSlice = GetSlice( phi );
    phi = phi - GetSliceAngle( iSlice );

    //std::cout<<"phi = "<<phi<<std::endl;
    double x0 = cos(phi);
    double y0 = sin(phi);
    double z0 = tan(lambda);
    t.Set( x0, y0, z0, pt*x0, pt*y0, pt*z0, q);

    if( RecalculateSlice(t,iSlice) !=0 ){
      std::cout<<"Initial slice wrong!!!"<<std::endl;
      //exit(0);
    }
    
    for( int iRow=0; iRow<sliceParam.NRows(); iRow++ ){
      //if( iRow>=50 ) break; //SG!!!
      float xRow = sliceParam.RowX( iRow );	
      // transport to row
      int err = 0;
      for( int itry=0; itry<1; itry++ ){ 
	float B[3];
	prop.GetBxByBz( GetSliceAngle( iSlice ), t.GetX(), t.GetY(), t.GetZ(), B );
	float dLp=0;
	err = t.PropagateToXBxByBz( xRow, B[0], B[1], B[2], dLp );
	if( err ){
	  std::cout<<"Can not propagate to x = "<<xRow<<std::endl;
	  t.Print();	  
	  break;
	}
	if( fabs(t.GetZ())>=250. ){ 
	  std::cout<<"Can not propagate to x = "<<xRow<<": Z outside the volume"<<std::endl;
	  t.Print();	  
	  err = -1;
	  break;
	}
	// rotate track coordinate system to current sector
	int isNewSlice = RecalculateSlice(t,iSlice);
	if( !isNewSlice ) break;
	else{
	  std::cout<<"track "<<itr<<": new slice "<<iSlice<<" at row "<<iRow<<std::endl;
	}
      }
      if( err ) break;
      //std::cout<<" track "<<itr<<": Slice "<<iSlice<<" row "<<iRow<<" params :"<<std::endl;
      //t.Print();
      // track at row iRow, slice iSlice
      if( iRow==0 ){ // store MC track at first row
	//std::cout<<std::setprecision( 20 );
	//std::cout<<"track "<<itr<<": x "<<t.X()<<" y "<<t.Y()<<" z "<<t.Z()<<std::endl;
	AliHLTTPCGMPhysicalTrackModel tg(t); // global coordinates	
	tg.Rotate( - GetSliceAngle( iSlice ));

	mcInfo[itr].fPID = 2; // pion
	mcInfo[itr].fCharge = 3*q;
	mcInfo[itr].fX = tg.GetX(); //Position of MC track at entry of TPC / first hit in the TPC
	mcInfo[itr].fY = tg.GetY();
	mcInfo[itr].fZ = tg.GetZ();	
	mcInfo[itr].fPx = tg.GetPx(); //Momentum of MC track at that position
	mcInfo[itr].fPy = tg.GetPy();
	mcInfo[itr].fPz = tg.GetPz();
	// std::cout<<" mc Z = "<<tg.GetZ()<<std::endl;
      }
      
      // create cluster
      GenCluster c;
      float sigmaY = 0.3; 
      float sigmaZ = 0.5; 
      const int rowType = iRow < 64 ? 0 : iRow < 128 ? 2 : 1;
      t.UpdateValues();
      sliceParam.GetClusterErrors2( rowType, t.GetZ(), t.GetSinPhi(), t.GetDzDs(), sigmaY, sigmaZ );
      sigmaY = std::sqrt(sigmaY);
      sigmaZ = std::sqrt(sigmaZ);
      hClusterError[rowType][0]->Fill(sigmaY);
      hClusterError[rowType][1]->Fill(sigmaZ);
      //std::cout<<sigmaY<<" "<<sigmaY<<std::endl;
      //if( sigmaY > 0.5 ) sigmaY = 0.5;
      //if( sigmaZ > 0.5 ) sigmaZ = 0.5;
      c.fSector = (t.GetZ()>=0.) ?iSlice :iSlice+18;
      c.fRow = iRow; 
      c.fMCID = itr;
      c.fX = t.GetX();
      c.fY = t.GetY() + GetGaus(sigmaY);
      c.fZ = t.GetZ() + GetGaus(sigmaZ);
      c.fId = clusterId++;
      vClusters.push_back(c);
    } // iRow
  } // itr
  
  std::vector<AliHLTTPCClusterMCLabel> labels; 

  for (int iSector = 0;iSector < 36;iSector++) //HLT Sector numbering, sectors go from 0 to 35, all spanning all rows from 0 to 158.
    {
      int nNumberOfHits = 0;
      for( unsigned int i=0; i<vClusters.size(); i++ ) if( vClusters[i].fSector==iSector ) nNumberOfHits++;
      //For every sector we first have to fill the number of hits in this sector to the file
      out.write((char*) &nNumberOfHits, sizeof(nNumberOfHits));
      
      AliHLTTPCCAClusterData::Data* clusters = new AliHLTTPCCAClusterData::Data[nNumberOfHits]; 
      int icl=0;
      for( unsigned int i=0; i<vClusters.size(); i++ ){
	GenCluster &c = vClusters[i];
	if( c.fSector==iSector ){
	  clusters[icl].fId = c.fId;
	  clusters[icl].fRow = c.fRow; //We fill one hit per TPC row
	  clusters[icl].fX = c.fX;
	  clusters[icl].fY = c.fY;
	  clusters[icl].fZ = c.fZ;
	  clusters[icl].fAmp = 100; //Arbitrary amplitude
	  icl++;
	  AliHLTTPCClusterMCLabel clusterLabel;
	  for (int j = 0;j < 3;j++){
	    clusterLabel.fClusterID[j].fMCID = -1;
	    clusterLabel.fClusterID[j].fWeight = 0;
	  }
	  clusterLabel.fClusterID[0].fMCID = c.fMCID;
	  clusterLabel.fClusterID[0].fWeight = 1;
	  labels.push_back( clusterLabel);	  
	}
      }      
      out.write((char*) clusters, sizeof(clusters[0]) * nNumberOfHits);
      delete clusters;
    }
  
  //Create vector with cluster MC labels, clusters are counter from 0 to clusterId in the order they have been written above. No separation in slices.

  out.write((const char*) labels.data(), labels.size() * sizeof(labels[0]));
  labels.clear();
  
  out.write((const char*) &nTracks, sizeof(nTracks));
  out.write((const char*) mcInfo.data(), nTracks * sizeof(mcInfo[0]));
  mcInfo.clear();
  
  out.close();
  return(0);
}
