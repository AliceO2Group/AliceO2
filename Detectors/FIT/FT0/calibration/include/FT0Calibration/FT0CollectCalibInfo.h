// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

/// \brief Class to collect info for FT0 calibration
/// \author Alla.Maevskaya@cern.ch

#ifndef ALICEO2_FT0_FT0COLLECTCALIBINFO_
#define ALICEO2_FT0_FT0COLLECTCALIBINFOFT0_

#include <Rtypes.h>
#include <array>
#include <vector>
#include <string>
#include <TStopwatch.h>
#include <TParameter.h>
#include "FT0Calibration/FT0CalibrationInfoObject.h"
#include "FT0Base/Geometry.h"

class TTree;

namespace o2
{
namespace ft0
{
class FT0CollectCalibInfo
{
  using Geo = o2::ft0::Geometry;

 public:
  static constexpr int MAXNUMBEROFHITS = 256;

  FT0CollectCalibInfo() : mMinTimestamp("minTimestamp", -1), mMaxTimestamp("maxTimestamp", -1) {}

  ///< collect the CalibInfo for the FT0 channels
  void run();

  ///< perform all initializations
  void init();

  ///< set tree/chain containing FT0 calib info
  void setInputTreeFT0CalibInfo(TTree* tree) { mTreeFT0CalibInfo = tree; }

  ///< set output tree to write matched tracks
  void setOutputTree(TTree* tr) { mOutputTree = tr; }

  ///< set input branch names for the input from the tree
  void setFT0CalibInfoBranchName(const std::string& nm) { mFT0CalibInfoBranchName = nm; }
  void setOutputBranchName(const std::string& nm) { mOutputBranchName = nm; }

  ///< get input branch names for the input from the tree
  const std::string& getFT0CalibInfoBranchName() const { return mFT0CalibInfoBranchName; }
  const std::string& getOutputBranchName() const { return mOutputBranchName; }

  ///< get the min/max timestamp for following calibration of LHCPhase
  const TParameter<int>& getMinTimestamp() const { return mMinTimestamp; }
  const TParameter<int>& getMaxTimestamp() const { return mMaxTimestamp; }

  ///< print settings
  void print() const;

 private:
  void attachInputTrees();
  bool loadFT0CalibInfo();

  ///< add CalibInfoFT0 for a specific channel
  void addHit(o2::ft0::FT0CalibrationInfoObject& FT0CalibrationInfoObject);

  ///< fill the output tree
  void fillTree();

  //================================================================

  // Data members

  bool mInitDone = false; ///< flag init already done
  int mCurrFT0InfoTreeEntry = -1;

  ///========== Parameters to be set externally, e.g. from CCDB ====================

  // to be done later

  TTree* mTreeFT0CalibInfo = nullptr; ///< input tree with Calib infos

  TTree* mOutputTree = nullptr; ///< output tree for matched tracks

  ///>>>------ these are input arrays which should not be modified by the calibration code
  //           since this info is provided by external device
  std::vector<o2::ft0::FT0CalibrationInfoObject>* mFT0CalibInfo = nullptr; ///< input FT0 calib info
  /// <<<-----
  std::vector<o2::ft0::FT0CalibrationInfoObject> mFT0CollectedCalibInfo[Geo::Nchannels]; ///< output FT0 calibration info
  std::vector<o2::ft0::FT0CalibrationInfoObject>* mFT0CalibInfoOut = nullptr;            ///< this is the pointer to the CalibInfo of a specific channel that we need to fill the output tree

  std::string mFT0CalibInfoBranchName = "FT0CalibInfo";    ///< name of branch containing input FT0 calib infos
  std::string mOutputBranchName = "FT0CollectedCalibInfo"; ///< name of branch containing output

  TStopwatch mTimerTot;
  TStopwatch mTimerDBG;

  TParameter<int> mMinTimestamp; ///< minimum timestamp over the hits that we collect; we will need it at calibration time to
                                 ///< book the histogram for the LHCPhase calibration

  TParameter<int> mMaxTimestamp; ///< maximum timestamp over the hits that we collect; we will need it at calibration time to
                                 ///< book the histogram for the LHCPhase calibration

  ClassDefNV(FT0CollectCalibInfo, 1);
};
} // namespace ft0
} // namespace o2

#endif
