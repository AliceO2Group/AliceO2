// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

#ifndef ALICEO2_FDD_DIGITIZER_H
#define ALICEO2_FDD_DIGITIZER_H

#include "CommonDataFormat/InteractionRecord.h"
#include "DataFormatsFDD/Digit.h"
#include "DataFormatsFDD/MCLabel.h"
#include "FDDSimulation/Detector.h"
#include "SimulationDataFormat/MCTruthContainer.h"
#include "SimulationDataFormat/MCCompLabel.h"
#include "FDDSimulation/DigitizationParameters.h"

namespace o2
{
namespace fdd
{
class Digitizer
{
 public:
  Digitizer(const DigitizationParameters& params, Int_t mode = 0) : mMode(mode), parameters(params) { initParameters(); };
  ~Digitizer() = default;

  void process(const std::vector<o2::fdd::Hit>* hits, o2::fdd::Digit* digit);

  void initParameters();
  void SetEventTime(double value) { mEventTime = value; }
  void SetEventID(Int_t id) { mEventID = id; }
  void SetSrcID(Int_t id) { mSrcID = id; }
  void SetInteractionRecord(uint16_t bc, uint32_t orbit)
  {
    mIntRecord.bc = bc;
    mIntRecord.orbit = orbit;
  }
  const o2::InteractionRecord& GetInteractionRecord() const { return mIntRecord; }
  o2::InteractionRecord& GetInteractionRecord(o2::InteractionRecord& src) { return mIntRecord; }
  void SetInteractionRecord(const o2::InteractionRecord& src) { mIntRecord = src; }
  uint32_t GetOrbit() const { return mIntRecord.orbit; }
  uint16_t GetBC() const { return mIntRecord.bc; }

  void SetTriggers(o2::fdd::Digit* digit);
  Int_t SimulateLightYield(Int_t pmt, Int_t nPhot);

  void init();
  void finish();

  void setMCLabels(o2::dataformats::MCTruthContainer<o2::fdd::MCLabel>* mclb) { mMCLabels = mclb; }

 private:
  // digit info
  // parameters
  Int_t mMode;  //triggered or continuos
  o2::InteractionRecord mIntRecord; // Interaction record (orbit, bc)
  Int_t mEventID;
  Int_t mSrcID;        // signal, background or QED
  Double_t mEventTime; // timestamp

  DigitizationParameters parameters;

  o2::dataformats::MCTruthContainer<o2::fdd::MCLabel>* mMCLabels = nullptr;

  ClassDefNV(Digitizer, 1);
};
} // namespace fdd
} // namespace o2

#endif
