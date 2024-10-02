// Copyright 2019-2020 CERN and copyright holders of ALICE O2.
// See https://alice-o2.web.cern.ch/copyright for details of the copyright holders.
// All rights not expressly granted are reserved.
//
// This software is distributed under the terms of the GNU General Public
// License v3 (GPL Version 3), copied verbatim in the file "COPYING".
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

///
/// @file   CorrectdEdxDistortions.cxx
/// @author Matthias Kleiner, mkleiner@ikf.uni-frankfurt.de
///

#include "TPCCalibration/CorrectdEdxDistortions.h"
#include "TPCFastTransform.h"
#include "CommonUtils/TreeStreamRedirector.h"

o2::tpc::CorrectdEdxDistortions::~CorrectdEdxDistortions() = default;

o2::tpc::CorrectdEdxDistortions::CorrectdEdxDistortions() = default;

void o2::tpc::CorrectdEdxDistortions::setStreamer(const char* debugRootFile)
{
  mStreamer = std::make_unique<o2::utils::TreeStreamRedirector>(debugRootFile, "recreate");
};

void o2::tpc::CorrectdEdxDistortions::setSCCorrFromFile(const char* scAvgFile, const char* scDerFile, const float lumi)
{
  auto avg = o2::gpu::TPCFastTransform::loadFromFile(scAvgFile, "ccdb_object");
  auto der = o2::gpu::TPCFastTransform::loadFromFile(scDerFile, "ccdb_object");
  if (!avg || !der) {
    LOGP(warn, "Couldnt load all sc correction objects from local file");
    return;
  }
  avg->rectifyAfterReadingFromFile();
  der->rectifyAfterReadingFromFile();
  setCorrectionMaps(avg, der, lumi);
}

void o2::tpc::CorrectdEdxDistortions::setCorrectionMaps(o2::gpu::TPCFastTransform* avg, o2::gpu::TPCFastTransform* der)
{
  if (!avg || !der) {
    LOGP(warn, "Nullptr detected in setting the correction maps");
    return;
  }
  mCorrAvg = std::unique_ptr<o2::gpu::TPCFastTransform>(new o2::gpu::TPCFastTransform);
  mCorrAvg->cloneFromObject(*avg, nullptr);
  mCorrAvg->rectifyAfterReadingFromFile();

  mCorrDer = std::unique_ptr<o2::gpu::TPCFastTransform>(new o2::gpu::TPCFastTransform);
  mCorrDer->cloneFromObject(*der, nullptr);
  mCorrDer->rectifyAfterReadingFromFile();

  mCorrAvg->setApplyCorrectionOn();
  mCorrDer->setApplyCorrectionOn();
}

void o2::tpc::CorrectdEdxDistortions::setCorrectionMaps(o2::gpu::TPCFastTransform* avg, o2::gpu::TPCFastTransform* der, const float lumi)
{
  setCorrectionMaps(avg, der);
  setLumi(lumi);
}

void o2::tpc::CorrectdEdxDistortions::setLumi(float lumi)
{
  if (!mCorrAvg || !mCorrDer) {
    LOGP(warn, "Nullptr detected in accessing the correction maps");
    return;
  }
  const float lumiAvg = mCorrAvg->getLumi();
  const float lumiDer = mCorrDer->getLumi();
  mScaleDer = (lumi - lumiAvg) / lumiDer;
  LOGP(info, "Setting mScaleDer: {} for inst lumi: {}  avg lumi: {}  deriv. lumi: {}", mScaleDer, lumi, lumiAvg, lumiDer);
}

float o2::tpc::CorrectdEdxDistortions::getCorrection(const float time, unsigned char sector, unsigned char padrow, int pad) const
{
  //
  // Get the corrections at the previous and next padrow and interpolate them to the start and end position of current pad
  // Calculate from corrected position the radial distortions and compare the effective length of distorted electrons with pad length
  //

  // localY of current pad
  const float ly = mTPCGeometry.LinearPad2Y(sector, padrow, pad);

  // get correction at "pad + 0.5*padlength" pos1 and dont extrapolate/interpolate across GEM gaps
  const int row1 = ((padrow == mTPCGeometry.EndIROC() - 1) || (padrow == mTPCGeometry.EndOROC1() - 1) || (padrow == mTPCGeometry.EndOROC2() - 1)) ? padrow : std::clamp(padrow + 1, 0, GPUCA_ROW_COUNT - 1);

  float lxT_1 = 0;
  float lyT_1 = 0;
  float lzT_1 = 0;
  mCorrAvg->Transform(sector, row1, pad, time, lxT_1, lyT_1, lzT_1, 0, mCorrDer.get(), nullptr, mScaleDer, 0, 1);

  // correct for different localY position of pads
  lyT_1 += ly - mTPCGeometry.LinearPad2Y(sector, row1, pad);

  // get radius of upper pad
  const float r_1_f = std::sqrt(lxT_1 * lxT_1 + lyT_1 * lyT_1);

  // get correction at "pad - 0.5*padlength" pos0 and dont extrapolate/interpolate across GEM gaps
  const int row0 = ((padrow == mTPCGeometry.EndIROC()) || (padrow == mTPCGeometry.EndOROC1()) || (padrow == mTPCGeometry.EndOROC2())) ? padrow : std::clamp(padrow - 1, 0, GPUCA_ROW_COUNT - 1);

  // check if previous pad row has enough pads
  const unsigned char pad0 = std::clamp(static_cast<int>(pad), 0, mTPCGeometry.NPads(row0) - 1);
  float lxT_0 = 0;
  float lyT_0 = 0;
  float lzT_0 = 0;
  mCorrAvg->Transform(sector, row0, pad0, time, lxT_0, lyT_0, lzT_0, 0, mCorrDer.get(), nullptr, mScaleDer, 0, 1);

  // correct for different localY position of pads
  lyT_0 += ly - mTPCGeometry.LinearPad2Y(sector, row0, pad0);

  // get radius of lower pad
  const float r_0_f = std::sqrt(lxT_0 * lxT_0 + lyT_0 * lyT_0);

  // effective radial length of electrons
  const float dr_f = r_1_f - r_0_f;

  // position of upper and lower pad edge
  const float x_0 = padrow - 0.5;
  const float x_1 = padrow + 0.5;

  // interpolate corrections to upper and lower pad edge
  const int deltaRow = (row1 - row0);
  const float d_StartPad = (r_0_f * (row1 - x_0) + r_1_f * (x_0 - row0)) / deltaRow;
  const float d_EndPad = (r_0_f * (row1 - x_1) + r_1_f * (x_1 - row0)) / deltaRow;
  const float scCorr = (d_EndPad - d_StartPad) / mTPCGeometry.PadHeight(padrow);

  // check if corrected position is still reasonable
  const bool isOk = ((lxT_1 < mLX0Min) || (lxT_0 < mLX1Min) || (scCorr < mScCorrMin) || (scCorr > mScCorrMax)) ? false : true;

  // store debug informations
  if (mStreamer) {
    const float lx = mTPCGeometry.Row2X(padrow);

    // original correction
    float lxT = 0;
    float lyT = 0;
    float lzT = 0;
    mCorrAvg->Transform(sector, padrow, pad, time, lxT, lyT, lzT, 0, mCorrDer.get(), nullptr, mScaleDer, 0, 1);

    (*mStreamer) << "tree"
                 << "sector=" << sector
                 << "padrow=" << padrow
                 << "row0=" << row0
                 << "row1=" << row1
                 << "pad0=" << pad0
                 << "pad=" << pad
                 << "time=" << time
                 << "lx=" << lx
                 << "lxT=" << lxT
                 << "lyT=" << lyT
                 << "lzT=" << lzT
                 << "lxT_0=" << lxT_0
                 << "lxT_1=" << lxT_1
                 << "ly=" << ly
                 << "lyT_0=" << lyT_0
                 << "lyT_1=" << lyT_1
                 << "lzT_0=" << lzT_0
                 << "lzT_1=" << lzT_1
                 << "d_StartPad=" << d_StartPad
                 << "d_EndPad=" << d_EndPad
                 << "r_0_f=" << r_0_f
                 << "r_1_f=" << r_1_f
                 << "scCorr=" << scCorr
                 << "isOk=" << isOk
                 << "\n";
  }

  return isOk ? scCorr : 1;
}
