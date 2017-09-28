// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

/// \file DigitPad.h
/// \brief Definition of the Pad container
/// \author Andi Mathis, TU München, andreas.mathis@ph.tum.de
/// \author Jens Wiechula, Jens.Wiechula@ikf.uni-frankfurt.de

#ifndef ALICEO2_TPC_DigitPad_H_
#define ALICEO2_TPC_DigitPad_H_

#include <map>

#include "FairRootManager.h"
#include "SimulationDataFormat/MCTruthContainer.h"
#include "SimulationDataFormat/MCCompLabel.h"
#include <TClonesArray.h>
#include "SimulationDataFormat/LabelContainer.h"

// temporary include
#include <iostream>

#define EXTLABELS

namespace o2 {
namespace TPC {

/// \class DigitPad
/// This is the fifth and lowest class of the intermediate Digit Containers, in which all incoming electrons from the hits are sorted into after amplification
/// The structure assures proper sorting of the Digits when later on written out for further processing.
/// This class holds the individual pad containers and is contained within the Row Container.

class DigitPad{
  public:

    /// Constructor
    /// \param mPad Pad ID
    DigitPad(int mPad);

    /// Destructor
    ~DigitPad() = default;

    /// Resets the container
    void reset();

    /// Get the Pad ID
    /// \return Pad ID
    int getPad() const {return mPad;}

    /// Get the accumulated charge on that pad
    /// \return Accumulated charge
    float getChargePad() const {return mChargePad;}

    /// Add digit to the time bin container
    /// \param hitID MC Hit ID
    /// \param charge Charge of the digit
    void setDigit(size_t hitID, float charge);

    /// Fill output TClonesArray
    /// \param output Output container
    /// \param mcTruth MC Truth container
    /// \param debug Optional debug output container
    /// \param cru CRU ID
    /// \param timeBin Time bin
    /// \param row Row ID
    /// \param pad pad ID
    /// \param commonMode Common mode value of that specific ROC
    void fillOutputContainer(TClonesArray *output, o2::dataformats::MCTruthContainer<o2::MCCompLabel> &mcTruth, TClonesArray *debug, int cru, int timeBin, int row, int pad, float commonMode = 0.f);

  private:

    /// Compare two MC labels regarding trackID, eventID and sourceID
    /// \param label1 MC label 1
    /// \param label2 MC label 2
    /// \return true, if trackID, eventID and sourceID are the same
    bool compareMClabels(const MCCompLabel &label1, const MCCompLabel &label2) const;

    float                  mChargePad;   ///< Total accumulated charge on that pad for a given time bin
    unsigned char          mPad;         ///< Pad of the ADC value
#ifdef EXTLABELS
    unsigned int           mId;          ///< An integer id for this digit (can be combined with mPad to not waste memory??)
#else
    std::vector<std::pair<MCCompLabel, int>> mMClabel; ///< vector to accumulate the MC labels
#endif
    // TODO: optimize this treatment, for example by using a structure like this
    // struct MCIDValue {
    //   unsigned int eventId : 15; // 32k event Id possible
    //   unsigned int trackId: 17; // 128K tracks possible
    //   unsigned int occurences : 32; // 4G occurrences possible
    // }
    // std::vector<MCID> mMCID;

    // a global memory space where we keep intermediate monte carlo labels
    // (this avoids having to use a std::vector<> member inside each digit which has at leas 24bytes overhead)
    static o2::dataformats::LabelContainer<std::pair<MCCompLabel, int>, false> sLabels;
    static unsigned int sID; // a global id counter
};

inline
DigitPad::DigitPad(int pad)
  : mChargePad(0.),
    mPad(pad),
#ifdef EXTLABELS
    mId(sID++)
#else
    mMClabel()
#endif
{
  //  if(sID == 0) std::cerr << "OVERFLOW\n";
}

inline 
void DigitPad::setDigit(size_t trackID, float charge)
{
  static FairRootManager *mgr = FairRootManager::Instance();
  bool isKnown = false;
  MCCompLabel tempLabel(trackID, mgr->GetEntryNr());
  mChargePad += charge;
#ifndef EXTLABELS
  for(auto &mcLabel : mMClabel) {
    if(compareMClabels(tempLabel, mcLabel.first)) {
      ++mcLabel.second;
      isKnown=true;
    }
  }
  if(!isKnown) mMClabel.emplace_back(tempLabel, 1);
#else
  // same using global labelview container
  isKnown = false;
  for(auto &mcLabel : sLabels.getLabels(mId)) {
    if(compareMClabels(tempLabel, mcLabel.first)) {
       ++mcLabel.second;
       isKnown=true;
     }
  }
  if(!isKnown) sLabels.addLabel(mId, std::make_pair(tempLabel, 1));
#endif
}

inline
void DigitPad::reset()
{
  std::cerr << " RESETING PAD \n";
  // FIXME: We have to think about what this means for the global label container
  // likely all pads will be reset at once and we can clear the whole container
  mChargePad = 0;
#ifndef EXTLABELS
  mMClabel.clear();
#endif
}

inline
bool DigitPad::compareMClabels(const MCCompLabel &label1, const MCCompLabel &label2) const
{
  return (label1.getEventID() == label2.getEventID() && label1.getTrackID() == label2.getTrackID() && label1.getSourceID() == label2.getSourceID());
}

  
}
}

#endif // ALICEO2_TPC_DigitPad_H_
