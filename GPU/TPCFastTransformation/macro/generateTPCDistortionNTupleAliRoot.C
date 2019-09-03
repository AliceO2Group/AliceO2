// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

/// \file  generateTPCDistortionNTupleAliRoot.C
/// \brief A developer macro for generating TPC distortion ntuple to test the TPCFastTransformation class
///        Works only with AliRoot, not with O2
///
/// \author  Sergey Gorbunov <sergey.gorbunov@cern.ch>
///

/*
  Run the macro:
    uncomment the first #define

 aliroot
  .x initTPCcalibration.C("alien://Folder=/alice/data/2015/OCDB",246984,1)
   gSystem->Load("libAliTPCFastTransformation")
  .x generateTPCDistortionNTupleAliRoot.C+
*/

// A developer code.
// It is hidden inside #ifdef in order to avoid an automatic compilation during the O2 build
// Uncomment the #define for compiling the code

//#define FASTTRANSFORM_DEVELOPING

#if defined(FASTTRANSFORM_DEVELOPING)

#include "AliTPCcalibDB.h"
#include "AliTPCRecoParam.h"
#include "Riostream.h"
#include "TStopwatch.h"
#include "TFile.h"
#include "TNtuple.h"

#define GPUCA_ALIROOT_LIB

#include "TPCFastTransform.h"
#include "TPCFastTransformManager.h"
#include "TPCFastTransformQA.h"
#include "AliHLTTPCGeometry.h"

using namespace std;
using namespace GPUCA_NAMESPACE::gpu;

int generateTPCDistortionNTupleAliRoot()
{
  AliTPCcalibDB* tpcCalib = AliTPCcalibDB::Instance();
  if (!tpcCalib) {
    cerr << "AliTPCcalibDB does not exist" << endl;
    return -1;
  }
  AliTPCTransform* origTransform = tpcCalib->GetTransform();

  AliTPCRecoParam* recoParam = origTransform->GetCurrentRecoParamNonConst();
  if (!recoParam) {
    std::cout << "TPCFastTransformManager::Init: No TPC Reco Param set in transformation" << std::endl;
    return -1;
  }

  UInt_t timeStamp = origTransform->GetCurrentTimeStamp();

  TPCFastTransformManager manager;
  TPCFastTransform fastTransform;

  int err = manager.create(fastTransform, origTransform, timeStamp);

  if (err != 0) {
    cerr << "Cannot create fast transformation object from AliTPCcalibDB, TPCFastTransformManager returns  " << err << endl;
    return -1;
  }

  const TPCFastTransformGeo& geo = fastTransform.getGeometry();

  recoParam->SetUseTOFCorrection(kFALSE);

  cout << " generate NTuple " << endl;

  TFile* f = new TFile("tpcDistortionNTuple.root", "RECREATE");
  TNtuple* nt = new TNtuple("dist", "dist", "slice:row:su:sv:dx:du:dv");

  int nSlices = 1; //fastTransform.getNumberOfSlices();
  //for( int slice=0; slice<nSlices; slice++){
  for (int slice = 0; slice < 1; slice++) {
    const TPCFastTransformGeo::SliceInfo& sliceInfo = geo.getSliceInfo(slice);

    for (int row = 0; row < geo.getNumberOfRows(); row++) {

      float x = geo.getRowInfo(row).x;
      const int nKnots = 101;
      for (int knotU = 0; knotU < nKnots; knotU++) {
        float su = knotU / (double)(nKnots - 1);

        for (int knotV = 0; knotV < nKnots; knotV++) {
          float sv = knotV / (double)(nKnots - 1);

          //for (float su = 0.; su <= 1.; su += 0.01) {
          //for (float sv = 0.; sv <= 1.; sv += 0.01) {

          float u, v, y = 0, z = 0;
          geo.convScaledUVtoUV(slice, row, su, sv, u, v);

          // nominal x,y,z coordinates of the knot (without distortions and time-of-flight correction)
          geo.convUVtoLocal(slice, u, v, y, z);

          // row, pad, time coordinates of the knot
          float vertexTime = 0.f;
          float pad = 0.f, time = 0.f;
          fastTransform.convUVtoPadTime(slice, row, u, v, pad, time, vertexTime);

          // original TPC transformation (row,pad,time) -> (x,y,z) without time-of-flight correction
          float ox = 0, oy = 0, oz = 0;
          {
            int sector = 0, secrow = 0;
            AliHLTTPCGeometry::Slice2Sector(slice, row, sector, secrow);
            int is[] = {sector};
            double xx[] = {static_cast<double>(secrow), pad, time};
            origTransform->Transform(xx, is, 0, 1);
            ox = xx[0];
            oy = xx[1];
            oz = xx[2];
          }

          // convert to u,v
          float ou = 0, ov = 0;
          geo.convLocalToUV(slice, oy, oz, ou, ov);

          // distortions in x,u,v:
          float dx = ox - x;
          float du = ou - u;
          float dv = ov - v;

          cout << slice << " " << row << " " << su << " " << sv << " " << dx << " " << du << " " << dv << endl;
          nt->Fill(slice, row, su, sv, dx, du, dv);
        }
      }
    }
  }
  nt->Write();
  f->Write();
  recoParam->SetUseTOFCorrection(kTRUE);

  return 0;
}

#endif //FASTTRANSFORM_DEVELOPING
