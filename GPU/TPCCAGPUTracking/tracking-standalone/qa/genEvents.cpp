#include <iostream>
#include <fstream>
#include <string.h>

#include "AliHLTTPCCAClusterData.h"
#include "AliHLTTPCCAMCInfo.h"
#include "AliHLTTPCClusterMCData.h"
#include "AliHLTTPCCAParam.h"
#include "AliHLTTPCGMPhysicalTrackModel.h"
#include "AliHLTTPCGMPropagator.h"

#include "../cmodules/qconfig.h"

#include "include.h"

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

const double kTwoPi = 2.*kPi;
const double kSliceDAngle = kTwoPi/18;
const double kSliceAngleOffset = kSliceDAngle/2;

int GetSlice( double GlobalPhi )
{
  double phi = GlobalPhi;
  if( phi > kTwoPi ) phi -= kTwoPi;
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
  phi += kSliceAngleOffset;
  if( phi >= kTwoPi ) phi -= kTwoPi;
  int dSlice = (int) ( phi / kSliceDAngle );

  if( dSlice == 0 ) return 0; // nothing to do

  std::cout<<" dSlice = "<<dSlice<<std::endl;
  double dAlpha = dSlice*kSliceDAngle;
  // rotate track on angle dAlpha

  t.Rotate(dAlpha);
  
  iSlice+=dSlice;
  if( iSlice>=18 ) iSlice-=18;
  return 1;
}


int GenerateEvent(const AliHLTTPCCAParam& sliceParam, char* filename)
{
  std::ofstream out;
  out.open(filename, std::ofstream::binary);
  if (out.fail())
    {
      printf("Error opening file\n");
      return(1);
    }


  int nTracks = configStandalone.configEG.numberOfTracks; //Number of MC tracks, must be at least as large as the largest fMCID assigned above
  std::vector<AliHLTTPCCAMCInfo> mcInfo(nTracks);
  memset(mcInfo.data(), 0, nTracks * sizeof(mcInfo[0]));

  float Bz = configStandalone.solenoidBz;
  std::cout<<"Bz[kG] = "<<Bz<<std::endl;

  const double kCLight = 0.000299792458;
  Bz*=kCLight;

  std::vector<GenCluster> vClusters;
  int clusterId = 0; //Here we count up the cluster ids we fill (must be unique).

  for (int itr = 0; itr < nTracks; itr++){
    std::cout<<"Track "<<itr<<":"<<std::endl;

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
    double theta = 30.*kPi/180.;
    double pt = 1.;
    double q = 1.;
    int iSlice = GetSlice( phi );
    phi = phi - GetSliceAngle( iSlice );
    double x0 = cos(phi);
    double y0 = sin(phi);
    double z0 = tan(theta);
    t.Set( x0, y0, z0, pt*x0, pt*y0, pt*z0, q);

    if( RecalculateSlice(t,iSlice) !=0 ){
      std::cout<<"Initial slice wrong!!!"<<std::endl;
      exit(0);
    }
    
    for( int iRow=0; iRow<sliceParam.NRows(); iRow++ ){
      float xRow = sliceParam.RowX( iRow );	
      // transport to row
      int err = 0;
      for( int itry=0; itry<1; itry++ ){ 
	float tmp=0;
	err = t.PropagateToXBzLight( xRow, Bz, tmp );
	if( fabs(t.GetZ())>=250. ) err = 1;
	if( err ){
	  std::cout<<"Can not propagate to x = "<<xRow<<std::endl;
	  t.Print();	  
	  break;
	}
	// rotate track coordinate system to current sector
	int isNewSlice = RecalculateSlice(t,iSlice);
	if( !isNewSlice ) break;
	else{
	  std::cout<<" new slice "<<iSlice<<" at row "<<iRow<<std::endl;
	}
      }
      if( err ) break;
      //std::cout<<" track "<<itr<<": Slice "<<iSlice<<" row "<<iRow<<" params :"<<std::endl;
      //t.Print();
      // track at row iRow, slice iSlice
      if( iRow==0 ){ // store MC track at first row
	AliHLTTPCGMPhysicalTrackModel tg(t); // global coordinates
	tg.RotateLight( - GetSliceAngle( iSlice ));
	mcInfo[itr].fPID = 2; // pion
	mcInfo[itr].fX = tg.GetX(); //Position of MC track at entry of TPC / first hit in the TPC
	mcInfo[itr].fY = tg.GetY();
	mcInfo[itr].fZ = tg.GetZ();
	mcInfo[itr].fPx = tg.GetPx(); //Momentum of MC track at that position
	mcInfo[itr].fPy = tg.GetPy();
	mcInfo[itr].fPz = tg.GetPz();
	std::cout<<" mc Z = "<<tg.GetZ()<<std::endl;
      }
      
      // create cluster
      GenCluster c;	
      c.fSector = (t.GetZ()>=0.) ?iSlice :iSlice+18;
      c.fRow = iRow; 
      c.fMCID = itr;
      c.fX = t.GetX();
      c.fY = t.GetY();
      c.fZ = t.GetZ();
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
