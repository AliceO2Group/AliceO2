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
/// @file   CorrectdEdxDistortions.h
/// @author Matthias Kleiner, mkleiner@ikf.uni-frankfurt.de
///

#ifndef ALICEO2_TPC_CORRECTDEDXDISTORTIONS_H
#define ALICEO2_TPC_CORRECTDEDXDISTORTIONS_H

#include "GPUTPCGeometry.h"
#include <memory>

// forward declare classes
namespace o2::gpu
{
class TPCFastTransform;
}

namespace o2::utils
{
class TreeStreamRedirector;
}

namespace o2::tpc
{

class CorrectdEdxDistortions
{
 public:
  /// Default constructor
  CorrectdEdxDistortions();

  /// Default destructor
  ~CorrectdEdxDistortions();

  /// setting the space-charge corrections from local file
  /// \param scAvgFile average space-charge correction map
  /// \param scDerFile derivative space-charge correction map
  /// \param lumi luminosity which is used for scaling the correction map
  void setSCCorrFromFile(const char* scAvgFile, const char* scDerFile, const float lumi = -1);

  /// setting the space-charge corrections
  /// \param avg average space-charge correction map
  /// \param der derivative space-charge correction map
  /// \param lumi luminosity which is used for scaling the correction map
  void setCorrectionMaps(o2::gpu::TPCFastTransform* avg, o2::gpu::TPCFastTransform* der, const float lumi);

  /// setting the space-charge corrections
  /// \param avg average space-charge correction map
  /// \param der derivative space-charge correction map
  void setCorrectionMaps(o2::gpu::TPCFastTransform* avg, o2::gpu::TPCFastTransform* der);

  /// \param lumi luminosity which is used for scaling the correction map
  void setLumi(float lumi);

  /// enable the debug streamer
  void setStreamer(const char* debugRootFile = "debug_sc_corrections.root");

  /// \return getting the correction value for the cluster charge
  /// \param time true time information of the cluster
  /// \param sector TPC sector of the cluster
  /// \param padrow global padrow
  /// \param pad pad number in the padrow
  float getCorrection(const float time, unsigned char sector, unsigned char padrow, int pad) const;

  /// set minimum allowed lx correction at lower pad
  void setMinLX0(const float minLX) { mLX0Min = minLX; }

  /// set minimum allowed lx correction upper lower pad
  void setMinLX1(const float minLX) { mLX1Min = minLX; }

  /// set minimum allowed correction value
  void setMinCorr(const float minCorr) { mScCorrMin = minCorr; }

  /// set maximum allowed correction value
  void setMaxCorr(const float maxCorr) { mScCorrMax = maxCorr; }

 private:
  float mScaleDer = 0;
  std::unique_ptr<o2::gpu::TPCFastTransform> mCorrAvg{nullptr};        ///< correction map for average distortions
  std::unique_ptr<o2::gpu::TPCFastTransform> mCorrDer{nullptr};        ///< correction map for distortions
  o2::gpu::GPUTPCGeometry mTPCGeometry;                                ///< geometry information of TPC
  std::unique_ptr<o2::utils::TreeStreamRedirector> mStreamer{nullptr}; ///< debug streamer
  float mLX0Min{82};                                                   ///< minimum allowed lx position after correction at position of the bottom pad
  float mLX1Min{82};                                                   ///< minimum allowed lx position after correction at position of the top pad
  float mScCorrMin{0.7};                                               ///< minimum allowed correction values
  float mScCorrMax{1.6};                                               ///< maximum allowed correction values
};
} // namespace o2::tpc

#endif
