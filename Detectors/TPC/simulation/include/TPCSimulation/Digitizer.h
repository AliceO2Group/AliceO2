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

#include "TPCBase/Mapper.h"
#include "Steer/HitProcessingManager.h"

#include <cmath>

using std::vector;

class TTree;

namespace o2
{
namespace TPC
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
  Digitizer();

  /// Destructor
  ~Digitizer();

  /// Initializer
  void init();

  /// Steer conversion of points to digits
  /// \param sector Sector to be processed
  /// \param hits Container with TPC hit groups
  /// \param eventID ID of the processed event
  /// \param eventTime Time of the bunch crossing of the processed event
  /// \return digits container
  DigitContainer* Process(const Sector& sector, const std::vector<o2::TPC::HitGroup>& hits, int eventID,
                          float eventTime);

  /// Steer conversion of points to digits
  /// \param sector Sector to be processed
  /// \param hits Container with sorted TPC hit groups
  /// \param hitids Container with additional information which hit groups to process
  /// \param context Container with event information
  /// \return digits container
  DigitContainer* Process2(const Sector& sector, const std::vector<std::vector<o2::TPC::HitGroup>*>& hits,
                           const std::vector<o2::TPC::TPCHitGroupID>& hitids, const o2::steer::RunContext& context);

  /// Process a single hit group
  /// \param inputgroup Hit group to be processed
  /// \param sector Sector to be processed
  /// \param eventTime Time of the event to be processed
  /// \param eventID ID of the event to be processed
  void ProcessHitGroup(const HitGroup& inputgroup, const Sector& sector, const float eventTime, const int eventID,
                       const int sourceID = 0);

  DigitContainer* getDigitContainer() const { return mDigitContainer; }

  /// Switch for triggered / continuous readout
  /// \param isContinuous - false for triggered readout, true for continuous readout
  static void setContinuousReadout(bool isContinuous) { mIsContinuous = isContinuous; }

 private:
  Digitizer(const Digitizer&);
  Digitizer& operator=(const Digitizer&);

  DigitContainer* mDigitContainer; ///< Container for the Digits
  static bool mIsContinuous;       ///< Switch for continuous readout

  ClassDefNV(Digitizer, 1);
};
}
}

#endif // ALICEO2_TPC_Digitizer_H_
