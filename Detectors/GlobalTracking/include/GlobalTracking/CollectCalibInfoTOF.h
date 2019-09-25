// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

/// \file MatchTOF.h
/// \brief Class to perform TOF calibration
/// \author Francesco.Noferini@cern.ch, Chiara.Zampolli@cern.ch

#ifndef ALICEO2_GLOBTRACKING_COLLECTCALIBINFOTOF_
#define ALICEO2_GLOBTRACKING_COLLECTCALIBINFOTOF_

#include <Rtypes.h>
#include <array>
#include <vector>
#include <string>
#include <TStopwatch.h>
#include <TParameter.h>
#include "DataFormatsTOF/CalibInfoTOF.h"
#include "DataFormatsTOF/CalibInfoTOFshort.h"
#include "TOFBase/Geo.h"

class TTree;

namespace o2
{

namespace globaltracking
{
class CollectCalibInfoTOF
{
  using Geo = o2::tof::Geo;

 public:
  static constexpr int MAXNUMBEROFHITS = 256;

  CollectCalibInfoTOF() : mMinTimestamp("minTimestamp", -1), mMaxTimestamp("maxTimestamp", -1) {}

  ///< collect the CalibInfo for the TOF channels
  void run();

  ///< perform all initializations
  void init();

  ///< set tree/chain containing TOF calib info
  void setInputTreeTOFCalibInfo(TTree* tree) { mTreeTOFCalibInfo = tree; }

  ///< set output tree to write matched tracks
  void setOutputTree(TTree* tr) { mOutputTree = tr; }

  ///< set input branch names for the input from the tree
  void setTOFCalibInfoBranchName(const std::string& nm) { mTOFCalibInfoBranchName = nm; }
  void setOutputBranchName(const std::string& nm) { mOutputBranchName = nm; }

  ///< get input branch names for the input from the tree
  const std::string& getTOFCalibInfoBranchName() const { return mTOFCalibInfoBranchName; }
  const std::string& getOutputBranchName() const { return mOutputBranchName; }

  ///< get the min/max timestamp for following calibration of LHCPhase
  const TParameter<int>& getMinTimestamp() const { return mMinTimestamp; }
  const TParameter<int>& getMaxTimestamp() const { return mMaxTimestamp; }

  ///< print settings
  void print() const;

 private:
  void attachInputTrees();
  bool loadTOFCalibInfo();

  ///< add CalibInfoTOF for a specific channel
  void addHit(o2::dataformats::CalibInfoTOF& calibInfoTOF);

  ///< fill the output tree
  void fillTree();

  //================================================================

  // Data members

  bool mInitDone = false; ///< flag init already done
  int mCurrTOFInfoTreeEntry = -1;

  ///========== Parameters to be set externally, e.g. from CCDB ====================

  // to be done later

  TTree* mTreeTOFCalibInfo = nullptr; ///< input tree with Calib infos

  TTree* mOutputTree = nullptr; ///< output tree for matched tracks

  ///>>>------ these are input arrays which should not be modified by the calibration code
  //           since this info is provided by external device
  std::vector<o2::dataformats::CalibInfoTOF>* mTOFCalibInfo = nullptr; ///< input TOF calib info
  /// <<<-----
  std::vector<o2::dataformats::CalibInfoTOFshort> mTOFCollectedCalibInfo[Geo::NCHANNELS]; ///< output TOF calibration info
  std::vector<o2::dataformats::CalibInfoTOFshort>* mTOFCalibInfoOut = nullptr;            ///< this is the pointer to the CalibInfo of a specific channel that we need to fill the output tree

  std::string mTOFCalibInfoBranchName = "TOFCalibInfo";    ///< name of branch containing input TOF calib infos
  std::string mOutputBranchName = "TOFCollectedCalibInfo"; ///< name of branch containing output

  TStopwatch mTimerTot;
  TStopwatch mTimerDBG;

  TParameter<int> mMinTimestamp; ///< minimum timestamp over the hits that we collect; we will need it at calibration time to
                                 ///< book the histogram for the LHCPhase calibration

  TParameter<int> mMaxTimestamp; ///< maximum timestamp over the hits that we collect; we will need it at calibration time to
                                 ///< book the histogram for the LHCPhase calibration

  ClassDefNV(CollectCalibInfoTOF, 1);
};
} // namespace globaltracking
} // namespace o2

#endif
