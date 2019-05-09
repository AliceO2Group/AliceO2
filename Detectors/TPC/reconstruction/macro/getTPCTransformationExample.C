// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

/*
  Load the macro:
  gSystem->Load("libO2TPCFastTransformation");
  gSystem->Load("libTPCReconstruction");
  .x getTPCTransformationExample.C++
*/
#if !defined(__CLING__) || defined(__ROOTCLING__)

#include "TPCReconstruction/TPCFastTransformHelperO2.h"
#include "TH1F.h"
#include "TStyle.h"

#endif

using namespace o2::TPC;
using namespace o2::gpu;

void spaceChargeCorrection(const double XYZ[3], double dXdYdZ[3])
{
  dXdYdZ[0] = 1.;
  dXdYdZ[1] = 2.;
  dXdYdZ[2] = 3.;
}

void getTPCTransformationExample()
{

  o2::TPC::TPCFastTransformHelperO2::instance()->setSpaceChargeCorrection(spaceChargeCorrection);

  std::unique_ptr<TPCFastTransform> fastTransform(TPCFastTransformHelperO2::instance()->create(0));

  TH1F* hist = new TH1F("h", "h", 100, -1.e-4, 1.e-4);

  double statDiff = 0., statN = 0.;

  for (int slice = 0; slice < fastTransform->getNumberOfSlices(); slice += 1) {
    std::cout << "slice " << slice << " ... " << std::endl;

    const TPCFastTransform::SliceInfo& sliceInfo = fastTransform->getSliceInfo(slice);

    for (int row = 0; row < fastTransform->getNumberOfRows(); row++) {

      int nPads = fastTransform->getRowInfo(row).maxPad + 1;

      for (int pad = 0; pad < nPads; pad += 10) {

        for (float time = 0; time < 1000; time += 30) {

          fastTransform->setApplyDistortionFlag(0);
          float x0=0., y0=0., z0=0.;
          int err0 = fastTransform->Transform(slice, row, pad, time, x0, y0, z0);

          fastTransform->setApplyDistortionFlag(1);
          float x1=0., y1=0., z1=0.;
          int err1 = fastTransform->Transform(slice, row, pad, time, x1, y1, z1);

          if (err0 != 0 || err1 != 0) {
            std::cout << "can not transform!!" << std::endl;
            continue;
          }

          // local 2 global

          float x0g = x0 * sliceInfo.cosAlpha - y0 * sliceInfo.sinAlpha;
          float y0g = x0 * sliceInfo.sinAlpha + y0 * sliceInfo.cosAlpha;
          float z0g = z0;

          float x1g = x1 * sliceInfo.cosAlpha - y1 * sliceInfo.sinAlpha;
          float y1g = x1 * sliceInfo.sinAlpha + y1 * sliceInfo.cosAlpha;
          float z1g = z1;

          //cout<<x0<<" "<<y0<<" "<<z0<<" "<<x0g<<" "<<y0g<<" "<<z0g<<endl;
          //cout<<x1<<" "<<y1<<" "<<z1<<" "<<x1g<<" "<<y1g<<" "<<z1g<<endl;

          // compare the original correction to the difference ( transformation with correction - transformation without correction )

          double xyz[3] = { x0g, y0g, z0g };
          double d[3] = { 0, 0, 0 };
          spaceChargeCorrection(xyz, d);

          hist->Fill((x1g - x0g) - d[0]);
          hist->Fill((y1g - y0g) - d[1]);
          hist->Fill((z1g - z0g) - d[2]);

          //std::cout << (x1g-x0g) - d[0]<<" "<< (y1g-y0g) - d[1]<<" "<< (z1g-z0g) - d[2]<<std::endl;
        }
      }
    }
  }
  std::cout << "draw.." << std::endl;
  gStyle->SetOptStat("emruo");
  hist->Draw();
}
