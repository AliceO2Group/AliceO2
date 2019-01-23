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

#ifndef ALICEO2_GLOBTRACKING_CALIBTOF_
#define ALICEO2_GLOBTRACKING_CALIBTOF_

#include <Rtypes.h>
#include <array>
#include <vector>
#include <string>
#include <TStopwatch.h>
#include "ReconstructionDataFormats/CalibInfoTOF.h"
#include "CommonDataFormat/EvIndex.h"
#include "CommonUtils/TreeStreamRedirector.h"
#include "TOFBase/Geo.h"
#include "ReconstructionDataFormats/PID.h"
#include "TH1F.h"

class TTree;

namespace o2
{

namespace globaltracking
{
class CalibTOF
{
  using Geo = o2::tof::Geo;

 public:
  ///< perform matching for provided input
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

  ///< print settings
  void print() const;

  TH1F *getLHCphaseHisto() {return mHistoLHCphase;}

  /*
#ifdef _ALLOW_DEBUG_TREES_
  enum DebugFlagTypes : UInt_t {
    MatchTreeAll = 0x1 << 1, ///< produce matching candidates tree for all candidates
  };
  ///< check if partucular flags are set
  bool isDebugFlag(UInt_t flags) const { return mDBGFlags & flags; }

  ///< get debug trees flags
  UInt_t getDebugFlags() const { return mDBGFlags; }

  ///< set or unset debug stream flag
  void setDebugFlag(UInt_t flag, bool on = true);

  ///< set the name of output debug file
  void setDebugTreeFileName(std::string name)
  {
    if (!name.empty()) {
      mDebugTreeFileName = name;
    }
  }

  ///< get the name of output debug file
  const std::string& getDebugTreeFileName() const { return mDebugTreeFileName; }

  ///< fill matching debug tree
  void fillTOFmatchTree(const char* tname, int cacheTOF, int sectTOF, int plateTOF, int stripTOF, int padXTOF, int padZTOF, int cacheeTrk, int crossedStrip, int sectPropagation, int platePropagation, int stripPropagation, int padXPropagation, int padZPropagation, float resX, float resZ, float res, o2::dataformats::TrackTPCITS& trk, float intLength, float intTimePion, float timeTOF);
  void fillTOFmatchTreeWithLabels(const char* tname, int cacheTOF, int sectTOF, int plateTOF, int stripTOF, int padXTOF, int padZTOF, int cacheeTrk, int crossedStrip, int sectPropagation, int platePropagation, int stripPropagation, int padXPropagation, int padZPropagation, float resX, float resZ, float res, o2::dataformats::TrackTPCITS& trk, int TPClabelTrackID, int TPClabelEventID, int TPClabelSourceID, int ITSlabelTrackID, int ITSlabelEventID, int ITSlabelSourceID, int TOFlabelTrackID0, int TOFlabelEventID0, int TOFlabelSourceID0, int TOFlabelTrackID1, int TOFlabelEventID1, int TOFlabelSourceID1, int TOFlabelTrackID2, int TOFlabelEventID2, int TOFlabelSourceID2, float intLength, float intTimePion, float timeTOF);
  void dumpWinnerMatches();
#endif
  */

 private:
  // objects needed for calibration
  TH1F *mHistoLHCphase = nullptr;

  void attachInputTrees();
  bool loadTOFCalibInfo();

  int doCalib(int flag, int channel=0); // flag: 0=LHC phase, 1=channel offset+problematic(return value), 2=time-slewing

  //================================================================

  // Data members

  bool mInitDone = false; ///< flag init already done
  int mCurrTOFInfoTreeEntry = -1;

  ///========== Parameters to be set externally, e.g. from CCDB ====================

  // to be done later

  TTree* mTreeTOFCalibInfo = nullptr; ///< input tree with Calib infos

  TTree* mOutputTree = nullptr; ///< output tree for matched tracks

  ///>>>------ these are input arrays which should not be modified by the matching code
  //           since this info is provided by external device
  std::vector<o2::dataformats::CalibInfoTOF>* mTOFCalibInfo = nullptr; ///< input TOF matching info
  /// <<<-----

  std::string mTOFCalibInfoBranchName = "TOFCalibInfo";   ///< name of branch containing input TOF calib infos
  std::string mOutputBranchName = "TOFCalibParam";        ///< name of branch containing output

#ifdef _ALLOW_DEBUG_TREES_
  std::unique_ptr<o2::utils::TreeStreamRedirector> mDBGOut;
  UInt_t mDBGFlags = 0;
  std::string mDebugTreeFileName = "dbg_calibTOF.root"; ///< name for the debug tree file
#endif

  TStopwatch mTimerTot;
  TStopwatch mTimerDBG;
  ClassDefNV(CalibTOF, 1);
};
} // namespace globaltracking
} // namespace o2

#endif
