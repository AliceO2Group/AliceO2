// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

/// \file DigitContainer.h
/// \brief Definition of the Digit Container
/// \author Andi Mathis, TU München, andreas.mathis@ph.tum.de

#ifndef ALICEO2_TPC_DigitContainer_H_
#define ALICEO2_TPC_DigitContainer_H_

#include "TPCBase/Defs.h"
#include "TPCSimulation/DigitSector.h"

namespace o2 {
namespace TPC {

class Digit;
class DigitMCMetaData;
  
/// \class DigitContainer
/// This is the base class of the intermediate Digit Containers, in which all incoming electrons from the hits are sorted into after amplification
/// The structure assures proper sorting of the Digits when later on written out for further processing.
/// This class holds the CRU containers.

class DigitContainer{
  public:
    
    /// Default constructor
    DigitContainer();

    /// Destructor
    ~DigitContainer() = default;

    void setUp(const short sector, const TimeBin timeBinEvent);

    unsigned short getSectorLeft(const short sector) const;
    unsigned short getSectorRight(const short sector) const;
    bool checkNeighboursProcessed(const short sector) const;
    unsigned short getBufferPosition(const short sector);

    /// Add digit to the container
    /// \param eventID MC Event ID
    /// \param hitID MC Hit ID
    /// \param cru CRU of the digit
    /// \param row Pad row of digit
    /// \param pad Pad of digit
    /// \param timeBin Time bin of the digit
    /// \param charge Charge of the digit
    void addDigit(size_t eventID, size_t hitID, const CRU &cru, TimeBin timeBin, GlobalPadNumber globalPad, float charge);

    /// Fill output vector
    /// \param output Output container
    /// \param mcTruth MC Truth container
    /// \param debug Optional debug output container
    /// \param eventTime time stamp of the event
    /// \param isContinuous Switch for continuous readout
    void fillOutputContainer(std::vector<Digit> *output, dataformats::MCTruthContainer<MCCompLabel> &mcTruth,
                             std::vector<DigitMCMetaData> *debug, TimeBin eventTime=0, bool isContinuous=true, bool isFinal=false);

  private:
    unsigned short mSectorID;
    std::array<bool, Sector::MAXSECTOR> mSectorProcessed;
    std::array<short, Sector::MAXSECTOR> mSectorMapping;
    unsigned short mNextFreePosition;
    std::array<DigitSector, 5> mSector; ///< Container for the sector to be processed
};

inline
DigitContainer::DigitContainer()
  : mSectorID(-1),
    mSectorProcessed(),
    mSectorMapping(),
    mNextFreePosition(0),
    mSector()
{
  mSectorProcessed.fill(false);
  mSectorMapping.fill(-1);
}

}
}

#endif // ALICEO2_TPC_DigitContainer_H_
