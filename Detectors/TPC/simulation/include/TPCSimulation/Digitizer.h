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

/// \file Digitizer.h
/// \brief Definition of the ALICE TPC digitizer
/// \author Andi Mathis, TU München, andreas.mathis@ph.tum.de

#ifndef ALICEO2_TPC_Digitizer_H_
#define ALICEO2_TPC_Digitizer_H_

#include "TPCSimulation/DigitContainer.h"
#include "TPCSimulation/PadResponse.h"
#include "TPCSimulation/Point.h"
#include "TPCSpaceCharge/SpaceCharge.h"

#include "TPCBase/Mapper.h"

#include <cmath>

using std::vector;

class TTree;
class TH3;

namespace o2
{
namespace tpc
{

class DigitContainer;

/// \class Digitizer
/// This is the digitizer for the ALICE GEM TPC.
/// It is the main class and steers all relevant physical processes for the signal formation in the detector.
/// -# Transformation of energy deposit of the incident particle to a number of primary electrons
/// -# Drift and diffusion of the primary electrons while moving in the active volume towards the readout chambers
/// (ElectronTransport)
/// -# Amplification of the electrons in the stack of four GEM foils (GEMAmplification)
/// -# Induction of the signal on the pad plane, including a spread of the signal due to the pad response (PadResponse)
/// -# Shaping and further signal processing in the Front-End Cards (SampaProcessing)
/// The such created Digits and then sorted in an intermediate Container (DigitContainer) and after processing of the
/// full event/drift time summed up
/// and sorted as Digits into a vector which is then passed further on

class Digitizer
{
 public:
  using SC = SpaceCharge<double, 129, 129, 180>;

  /// Default constructor
  Digitizer() = default;

  /// Destructor
  ~Digitizer() = default;

  Digitizer(const Digitizer&) = delete;
  Digitizer& operator=(const Digitizer&) = delete;

  /// Initializer
  void init();

  /// Process a single hit group
  /// \param hits Container with TPC hit groups
  /// \param eventID ID of the event to be processed
  /// \param sourceID ID of the source to be processed
  void process(const std::vector<o2::tpc::HitGroup>& hits, const int eventID,
               const int sourceID = 0);

  /// Flush the data
  /// \param digits Container for the digits
  /// \param labels Container for the MC labels
  /// \param commonModeOutput Output container for the common mode
  /// \param finalFlush Flag whether the whole container is dumped
  void flush(std::vector<o2::tpc::Digit>& digits,
             o2::dataformats::MCTruthContainer<o2::MCCompLabel>& labels,
             std::vector<o2::tpc::CommonMode>& commonModeOutput, bool finalFlush = false);

  /// Set the sector to be processed
  /// \param sec Sector to be processed
  void setSector(Sector sec)
  {
    mSector = sec;
    mDigitContainer.reset();
  }

  /// Set the start time of the first event
  /// \param time Time of the first event
  void setStartTime(double time);

  /// Set mOutputDigitTimeOffset
  void setOutputDigitTimeOffset(double offset) { mOutputDigitTimeOffset = offset; }

  /// Set the time of the event to be processed
  /// \param time Time of the event
  void setEventTime(double time) { mEventTime = time; }

  /// Switch for triggered / continuous readout
  /// \param isContinuous - false for triggered readout, true for continuous readout
  static void setContinuousReadout(bool isContinuous) { mIsContinuous = isContinuous; }

  /// Option to retrieve triggered / continuous readout
  static bool isContinuousReadout() { return mIsContinuous; }

  /// Enable the use of space-charge distortions and provide space-charge density histogram as input
  /// \param distortionType select the type of space-charge distortions (constant or realistic)
  /// \param hisInitialSCDensity optional space-charge density histogram to use at the beginning of the simulation
  /// \param nZSlices number of grid points in z, must be (2**N)+1
  /// \param nPhiBins number of grid points in phi
  /// \param nRBins number of grid points in r, must be (2**N)+1
  void setUseSCDistortions(SC::SCDistortionType distortionType, const TH3* hisInitialSCDensity);
  /// Enable the use of space-charge distortions and provide SpaceCharge object as input
  /// \param spaceCharge unique pointer to spaceCharge object
  void setUseSCDistortions(SC* spaceCharge);

  /// Enable the use of space-charge distortions by providing global distortions and global corrections stored in a ROOT file
  /// The storage of the values should be done by the methods provided in the SpaceCharge class
  /// \param TFile file containing distortions and corrections
  void setUseSCDistortions(TFile& finp);

 private:
  DigitContainer mDigitContainer;    ///< Container for the Digits
  std::unique_ptr<SC> mSpaceCharge;  ///< Handler of space-charge distortions
  Sector mSector = -1;               ///< ID of the currently processed sector
  double mEventTime = 0.f;           ///< Time of the currently processed event
  double mOutputDigitTimeOffset = 0; ///< Time of the first IR sampled in the digitizer
  // FIXME: whats the reason for hving this static?
  static bool mIsContinuous;      ///< Switch for continuous readout
  bool mUseSCDistortions = false; ///< Flag to switch on the use of space-charge distortions
  ClassDefNV(Digitizer, 1);
};
} // namespace tpc
} // namespace o2

#endif // ALICEO2_TPC_Digitizer_H_
