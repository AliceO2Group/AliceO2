// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

/// \file CalibTimeSlewingParamFT0.h
/// \brief Class to store the output of the matching to FT0 for calibration

#ifndef ALICEO2_FT0CALIBTIMESLEWING_H
#define ALICEO2_FT0CALIBTIMESLEWING_H

#include <vector>
#include <array>
#include <TGraph.h>
#include <TH2F.h>
#include <TTree.h>
#include <TGraph.h>
#include "Rtypes.h"
#include "FairLogger.h"
#include "FT0Calibration/FT0CalibrationInfoObject.h"
#include "FT0Base/Geometry.h"
namespace o2::ft0
{
static constexpr int NCHANNELS = o2::ft0::Geometry::Nchannels;
class FT0CalibTimeSlewing
{
 public:
  static constexpr int HISTOGRAM_RANGE_X = 4000;
  static constexpr unsigned int NUMBER_OF_HISTOGRAM_BINS_X = HISTOGRAM_RANGE_X / 4;
  static constexpr int HISTOGRAM_RANGE_Y = 200;
  static constexpr unsigned int NUMBER_OF_HISTOGRAM_BINS_Y = HISTOGRAM_RANGE_Y;
  FT0CalibTimeSlewing();

  FT0CalibTimeSlewing(const FT0CalibTimeSlewing& source) = default;

  FT0CalibTimeSlewing& operator=(const FT0CalibTimeSlewing& source) = default;

  float getChannelOffset(int channel, int amplitude) const;

  const TGraph getGraph(int channel) const { return mTimeSlewing[channel]; }
  std::array<TGraph, NCHANNELS> getGraphs() const { return mTimeSlewing; }
  void fillGraph(int channel, TH2F* histo);

  float getSigmaPeak(int channel) const { return mSigmaPeak[channel]; }
  void setSigmaPeak(int channel, float value) { mSigmaPeak[channel] = value; }

  ///< perform all initializations
  void init();
  void mergeFilesWithTree();
  void fillHistos(TTree* tr);
  TH2F* getTimeAmpHist(int channel) { return mTimeAmpHist[channel]; };
  void setSingleFileName(std::string name) { mSingleFileName = name; }
  void setMergedFileName(std::string name) { mMergedFileName = name; }
  void setNfiles(int nfiles) { mNfiles = nfiles; };

  FT0CalibTimeSlewing& operator+=(const FT0CalibTimeSlewing& other);

 private:
  // FT0 channel calibrations
  std::array<TGraph, NCHANNELS> mTimeSlewing; ///< array of TGraph wirh time -amplitude for each channel
  std::array<float, NCHANNELS> mSigmaPeak;    ///< array with the sigma of the peak
  TFile* mMergedFile;                         // file  with merged tree
  TH2F* mTimeAmpHist[NCHANNELS];              //historgams time vs amplitude
  int mNfiles;                                // number of files with stored Tree with CalibrationInfoObject
  std::string mSingleFileName;
  std::string mMergedFileName;

  ClassDefNV(FT0CalibTimeSlewing, 1); // class for FT0 time slewing params
};
} // namespace o2::ft0
#endif
