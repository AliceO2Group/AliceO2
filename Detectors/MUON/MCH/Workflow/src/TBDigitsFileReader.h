// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

#ifndef O2_MCH_TBDIGITSFILEREADER_H_
#define O2_MCH_TBDIGITSFILEREADER_H_


#include <memory>
#include <fstream>

#include "MCHBase/Digit.h"

using namespace o2::mch;
using namespace std;

namespace o2 {

namespace mch {

struct TBCluster
{
  char        fDir;        //direction : x or y (121) ('0' if not fill)
  int         fNhits;      //numbers of hited strips
  int         fXNhits;     //numbers of hit pads inside a cluster, in the x direction
  int         fYNhits;     //numbers of hit pads inside a cluster, in the y direction
  double      fCharge;     //total charge
  double      fChargemax;    //charge of the highest pad
  double      fXclus;        //x position using the COG of charges
  double      fYclus;        //y position using the COG of charges
  double      fXmat;         //x position using the Mathieson fit
  double      fYmat;         //y position using the Mathieson fit
  double      fYmaterror;    //
  double      fXmaterror;    //
  double      fChi2mat;      //Chi2 of Mathieson fit
  double      fChmat;        //Total fraction of Mathieson charge
  double      fYcenter;      //center of the pad closer to COG
  double      fXcenter;      //center of the pad closer to COG
  double      fInversepitch;    //inverse half-pitch
  double      fK3x;             //Mathieson K3X
  double      fK3y;             //Mathieson K3Y
  double      fK3yrec;          //Mathieson K3Y fitted
  double      fPadyMean;        //Mean Padx
  double      fPadxMean;        //Mean Pady
  double      fPadySigma;       //Sigma Padx
  double      fPadxSigma;       //Sigma Pady
  double      fTimeMean;        //Mean time of the padhits of the cluster
  double      fTimeSigma;       //Sigma time of the padhits of the cluster
  double      fSizeMean;        //Mean size of the padhits of the cluster
  double      fSizeSigma;       //Sigma size of the padhits of the cluster
};

class TBDigitsFileReader
{
public:
  TBDigitsFileReader();

  void init(std::string inputFileName);

  bool readDigitsFromFile();

  ssize_t getNumberOfDigits();
  void storeDigits(void* bufferPtr);

  std::vector<TBCluster>& getClusters() { return clusters; }

  void get_trk_pos(int de, float& x, float& y)
  {
    x = trkx[de];
    y = trky[de];
  }

private:
  std::ifstream mInputFile;
  std::vector< std::unique_ptr<Digit> > digits;
  std::vector<TBCluster> clusters;
  float trkx[1500];
  float trky[1500];
  int fEvent, fSiEvent;
};

}
}



std::ostream& operator<<(std::ostream& stream, const o2::mch::TBCluster& c);

#endif
