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
#include "TOFBase/Geo.h"
#include "TH1F.h"
#include "TH1D.h"
#include "TH2F.h"
#include "TF1.h"
#include "TFile.h"

class TTree;

namespace o2
{

namespace globaltracking
{
class CalibTOF
{
  using Geo = o2::tof::Geo;
  
 public:
  static constexpr int NSTRIPPERSTEP = 9;
  static constexpr int NPADSPERSTEP = Geo::NPADS * NSTRIPPERSTEP;

  ///< calibrate using the provided input
  void run(int flag);

  ///< perform all initializations
  void init();

  ///< set tree/chain containing TOF calib info
  void setInputTreeTOFCollectedCalibInfo(TTree* tree) { mTreeCollectedCalibInfoTOF = tree; }

  ///< set output tree to write calibration objects
  void setOutputTree(TTree* tr) { mOutputTree = tr; }

  ///< set input branch names for the input from the tree
  void setCollectedCalibInfoTOFBranchName(const std::string& nm) { mCollectedCalibInfoTOFBranchName = nm; }
  void setOutputBranchName(const std::string& nm) { mOutputBranchName = nm; }

  ///< get input branch names for the input from the tree
  const std::string& getCollectedCalibInfoTOFBranchName() const { return mCollectedCalibInfoTOFBranchName; }
  const std::string& getOutputBranchName() const { return mOutputBranchName; }

  ///< print settings
  void print() const;

  TH1F *getLHCphaseHisto() {return mHistoLHCphase;}
  TH1F *getChOffsetHisto(int ipad){return mHistoChOffsetTemp[ipad];}
  TH2F *getChTimeSleewingHisto(int ipad){return mHistoChTimeSleewingTemp[ipad];};
  TH2F *getChTimeSleewingHistoAll(){return mHistoChTimeSleewingAll;};

 private:
  void fillLHCphaseCalibInput(); // we will fill the input for the LHC phase calibration
  void doLHCPhaseCalib(); // calibrate with respect LHC phase
  void fillChannelCalibInput(float offset, int ipad); // we will fill the input for the channel-level calibration
  void fillChannelTimeSleewingCalib(float offset, int ipad);// we will fill the input for the channel-time-sleewing calibration
  void doChannelLevelCalibration(int flag,int ipad); // calibrate single channel from histos
  void resetChannelLevelHistos(int flag); // reset signle channel histos
 

  // objects needed for calibration
  TH1F *mHistoLHCphase = nullptr;
  TH1F *mHistoChOffsetTemp[NPADSPERSTEP];  // to fill all pads of a strip simultaneosly 
  TH2F *mHistoChTimeSleewingTemp[NPADSPERSTEP];  // to fill all pads of a strip simultaneosly 
  TH2F *mHistoChTimeSleewingAll; // time sleewing all channels

  TH1D *mProjTimeSlewingTemp; // temporary histo for time slewing

  void attachInputTrees();
  bool loadTOFCollectedCalibInfo(int increment = 1);

  int doCalib(int flag, int channel = -1); // flag: 0=LHC phase, 1=channel offset+problematic(return value), 2=time-slewing

  //================================================================

  // Data members

  bool mInitDone = false; ///< flag init already done
  int mCurrTOFInfoTreeEntry = -1;

  ///========== Parameters to be set externally, e.g. from CCDB ====================

  // to be done later

  TTree* mTreeCollectedCalibInfoTOF = nullptr; ///< input tree with Calib infos

  TTree* mOutputTree = nullptr; ///< output tree for matched tracks

  ///>>>------ these are input arrays which should not be modified by the matching code
  //           since this info is provided by external device
  std::vector<o2::dataformats::CalibInfoTOF>* mCalibInfoTOF = nullptr; ///< input TOF matching info
  /// <<<-----

  std::string mCollectedCalibInfoTOFBranchName = "TOFCollectedCalibInfo";   ///< name of branch containing input TOF calib infos
  std::string mOutputBranchName = "TOFCalibParam";        ///< name of branch containing output
  // output calibration
  float mLHCphase=0; ///< outputt LHC phase in ps
  float mLHCphaseErr=0; ///< outputt LHC phase in ps
  int mNChannels=Geo::NCHANNELS;      // needed to give the size to the branches of channels
  float mCalibChannelOffset[Geo::NCHANNELS]; ///< output TOF channel offset in ps
  float mCalibChannelOffsetErr[Geo::NCHANNELS]; ///< output TOF channel offset in ps

  // previous calibration read from CCDB
  float mInitialCalibChannelOffset[Geo::NCHANNELS]; ///< output TOF channel offset in ps


  TF1 *mFuncLHCphase = nullptr;
  TF1 *mFuncChOffset = nullptr;

  TStopwatch mTimerTot;
  TStopwatch mTimerDBG;
  ClassDefNV(CalibTOF, 1);
};
} // namespace globaltracking
} // namespace o2

#endif
