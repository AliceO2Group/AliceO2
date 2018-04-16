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

#include <cmath>

using std::vector;

class TTree;

namespace o2
{
namespace TPC
{

class DigitContainer;

/// Debug output
typedef struct {
  float CRU;
  float time;
  float row;
  float pad;
  float nElectrons;
} GEMRESPONSE;
static GEMRESPONSE GEMresponse;

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
  /// \param points Container with TPC points
  /// \return digits container
  DigitContainer* Process(const std::vector<o2::TPC::HitGroup>& hits, int eventID, float eventTime);

  DigitContainer* getDigitContainer() const { return mDigitContainer; }

  /// Enable the debug output after application of the PRF
  /// Can be set via DigitizerTask::setDebugOutput("PRFdebug")
  static void setPRFDebug() { mDebugFlagPRF = true; }

  /// Switch for triggered / continuous readout
  /// \param isContinuous - false for triggered readout, true for continuous readout
  static void setContinuousReadout(bool isContinuous) { mIsContinuous = isContinuous; }

 private:
  Digitizer(const Digitizer&);
  Digitizer& operator=(const Digitizer&);

  DigitContainer* mDigitContainer; ///< Container for the Digits

  std::unique_ptr<TTree> mDebugTreePRF; ///< Output tree for the output after the PRF
  static bool mDebugFlagPRF;            ///< Flag for debug output after the PRF
  static bool mIsContinuous;            ///< Switch for continuous readout

  ClassDefNV(Digitizer, 1);
};
}
}

#endif // ALICEO2_TPC_Digitizer_H_
