// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
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
#include "TPCSimulation/SpaceCharge.h"

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
  /// \param finalFlush Flag whether the whole container is dumped
  void flush(std::vector<o2::tpc::Digit>& digits,
             o2::dataformats::MCTruthContainer<o2::MCCompLabel>& labels, bool finalFlush = false);

  /// Set the sector to be processed
  /// \param sec Sector to be processed
  void setSector(Sector sec)
  {
    mSector = sec;
    mDigitContainer.reset();
  }

  /// Set the start time of the first event
  /// \param time Time of the first event
  void setStartTime(TimeBin time) { mDigitContainer.setStartTime(time); }

  /// Set the time of the event to be processed
  /// \param time Time of the event
  void setEventTime(float time) { mEventTime = time; }

  /// Switch for triggered / continuous readout
  /// \param isContinuous - false for triggered readout, true for continuous readout
  static void setContinuousReadout(bool isContinuous) { mIsContinuous = isContinuous; }

  /// Option to retrieve triggered / continuous readout
  static bool isContinuousReadout() { return mIsContinuous; }

  /// Enable the use of space-charge distortions
  /// \param distortionType select the type of space-charge distortions (constant or realistic)
  /// \param hisInitialSCDensity optional space-charge density histogram to use at the beginning of the simulation
  /// \param nZSlices number of grid points in z, must be (2**N)+1
  /// \param nPhiBins number of grid points in phi
  /// \param nRBins number of grid points in r, must be (2**N)+1
  void enableSCDistortions(SpaceCharge::SCDistortionType distortionType, TH3* hisInitialSCDensity, int nZSlices, int nPhiBins, int nRBins);

 private:
  DigitContainer mDigitContainer;                   ///< Container for the Digits
  std::unique_ptr<SpaceCharge> mSpaceChargeHandler; ///< Handler of space-charge distortions
  Sector mSector = -1;                              ///< ID of the currently processed sector
  float mEventTime = 0.f;                           ///< Time of the currently processed event
  // FIXME: whats the reason for hving this static?
  static bool mIsContinuous;      ///< Switch for continuous readout
  bool mUseSCDistortions = false; ///< Flag to switch on the use of space-charge distortions

  ClassDefNV(Digitizer, 1);
};
} // namespace tpc
} // namespace o2

#endif // ALICEO2_TPC_Digitizer_H_
