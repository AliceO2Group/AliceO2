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
#include "ReconstructionDataFormats/CalibInfoTOFshort.h"
#include "TOFBase/Geo.h"
#include "TH1F.h"
#include "TH1D.h"
#include "TH2F.h"
#include "TF1.h"
#include "TFile.h"
#include "TGraphErrors.h"

class TTree;

namespace o2
{

namespace globaltracking
{
class CalibTOF
{
  using Geo = o2::tof::Geo;
  
 public:
  static constexpr int NSTRIPSPERSTEP = 13;  // we chose this number because we process per sector, and 
                                            // each sector has 91 = 13x7 strips
  static constexpr int NPADSPERSTEP = Geo::NPADS * NSTRIPSPERSTEP;
  static constexpr int NSTEPSPERSECTOR = 91 / NSTRIPSPERSTEP;
  enum {kLHCphase = 1,
	kChannelOffset = 2,
	kChannelTimeSlewing = 4}; // enum to define which calibration we will do
  
  ///< constructor
  CalibTOF();
  
  ///< destructor
  ~CalibTOF(); 

  ///< calibrate using the provided input
  void run(int flag, int sector = -1);
  void fillOutput();

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

  TH2F *getLHCphaseHisto() {return mHistoLHCphase;}
  TH2F *getChTimeSlewingHistoAll() {return mHistoChTimeSlewingAll;};
  void setMinTimestamp(int minTimestamp) {mMinTimestamp = minTimestamp;}
  void setMaxTimestamp(int maxTimestamp) {mMaxTimestamp = maxTimestamp;}

  TGraphErrors *processSlewing(TH2F *histo, Bool_t forceZero,TF1 *fitFunc);
  Int_t FitPeak(TF1 *fitFunc, TH1 *h, Float_t startSigma, Float_t nSigmaMin, Float_t nSigmaMax, const char *debuginfo="", TH2 *hdbg=nullptr);


  void setDebugMode(Int_t flag = kTRUE) {mDebugMode = flag;}
  Int_t getDebugMode() const {return mDebugMode;}

 private:
  Int_t mDebugMode = 0; // >0= time slewing extra plot, >1= problematic fits stored 

  void fillLHCphaseCalibInput(std::vector<o2::dataformats::CalibInfoTOFshort>* calibinfotof); // we will fill the input for the LHC phase calibration
  void doLHCPhaseCalib(); // calibrate with respect LHC phase
  void fillChannelCalibInput(std::vector<o2::dataformats::CalibInfoTOFshort>* calibinfotof, float offset, int ipad, TH1F* histo, std::vector<o2::dataformats::CalibInfoTOFshort>* calibTimePad); // we will fill the input for the channel-level calibration
  void fillChannelTimeSlewingCalib(float offset, int ipad, TH2F* histo, std::vector<o2::dataformats::CalibInfoTOFshort>* calibTimePad);// we will fill the input for the channel-time-slewing calibration
  int doChannelCalibration(int ipad, TH1F* histo, TF1* func); // calibrate single channel from histos --> return > 0 if prolematic
  void resetChannelLevelHistos(TH1F* histoOffset[NPADSPERSTEP], TH2F* histoTimeSlewing, std::vector<o2::dataformats::CalibInfoTOFshort>* calibTimePad[NPADSPERSTEP]); // reset signle channel histos
 

  // objects needed for calibration
  TH2F *mHistoLHCphase = nullptr;
  TH2F *mHistoChTimeSlewingAll; // time slewing all channels

  TH1D *mProjTimeSlewingTemp; // temporary histo for time slewing

  void attachInputTrees();
  bool loadTOFCollectedCalibInfo(TTree *localTree, int &currententry, int increment = 1);


  //================================================================

  // Data members

  bool mInitDone = false; ///< flag init already done

  ///========== Parameters to be set externally, e.g. from CCDB ====================

  // to be done later

  TTree* mTreeCollectedCalibInfoTOF = nullptr; ///< input tree with Calib infos

  TTree* mOutputTree = nullptr; ///< output tree for matched tracks

  std::string mCollectedCalibInfoTOFBranchName = "TOFCollectedCalibInfo";   ///< name of branch containing input TOF calib infos
  std::string mOutputBranchName = "TOFCalibParam";        ///< name of branch containing output
  // output calibration
  int mNLHCphaseIntervals = 0;  ///< Number of measurements for the LHCPhase
  float mLHCphase[1000]; ///< outputt LHC phase in ps
  float mLHCphaseErr[1000]; ///< outputt LHC phase in ps
  int mLHCphaseStartInterval[1000]; ///< timestamp from which the LHCphase measurement is valid
  int mLHCphaseEndInterval[1000];   ///< timestamp until which the LHCphase measurement is valid
  int mNChannels = Geo::NCHANNELS;      // needed to give the size to the branches of channels
  float mCalibChannelOffset[Geo::NCHANNELS]; ///< output TOF channel offset in ps
  float mCalibChannelOffsetErr[Geo::NCHANNELS]; ///< output TOF channel offset in ps (negative error as flag for problematic channels)

  // previous calibration read from CCDB
  float mInitialCalibChannelOffset[Geo::NCHANNELS]; ///< initial calibrations read from the OCDB (the calibration process will do a residual calibration with respect to those)


  TF1 *mFuncLHCphase = nullptr;

  int mMinTimestamp = 0;   ///< minimum timestamp over the hits that we collect; we need it to
                           ///< book the histogram for the LHCPhase calibration
  
  int mMaxTimestamp = 1;   ///< maximum timestamp over the hits that we collect; we need it to
                           ///< book the histogram for the LHCPhase calibration

  ClassDefNV(CalibTOF, 1);
};
} // namespace globaltracking
} // namespace o2

#endif
