// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

/// \file Hit.h
/// \brief Definition of the ZDC Hit class

#ifndef ALICEO2_ZDC_HIT_H_
#define ALICEO2_ZDC_HIT_H_

#include "SimulationDataFormat/BaseHits.h" // for BasicXYZEHit
#include "CommonUtils/ShmAllocator.h"

namespace o2
{
namespace zdc
{

class Hit : public o2::BasicXYZEHit<Float_t, Float_t>
{

 public:
  // Default constructor
  Hit() = default;

  /// Class Constructor
  /// \param trackID Index of MCTrack
  /// \param detID Detector ID
  /// \param parent mother of the track
  /// \param sflag true if it is a secondary
  /// \param primaryEnergy energy of the  primary [GeV]
  /// \param detID detector ID (1-ZNA, 2-ZPA, 3-ZEM, 4-ZNC, 5-ZPC)
  /// \param sectorID sector ID
  /// \param pos track position
  /// \param mom track momentum
  /// \param tof track TOF
  /// \param xImpact x,y,z of the impact of the 1st particle
  /// \param energyloss deposited energy
  /// \param nphePMC light output on common PMT
  /// \param nphePMQ light output on sector PMT
  Hit(int trackID, int parent, Bool_t sFlag, Float_t primaryEnergy, Int_t detID, Int_t sectorID,
      Vector3D<float> pos, Vector3D<float> mom, Float_t tof, Vector3D<float> xImpact, Float_t energyloss, Int_t nphePMC,
      Int_t nphePMQ);

  void setPMCLightYield(float val) { mNphePMC = val; }
  void setPMQLightYield(float val) { mNphePMQ = val; }
  void setNoNumContributingSteps(int val) { mNoContributingSteps = val; }

  int getParentID() const { return mParentID; }
  int getSector() const { return mSectorID; }
  float getPMCLightYield() const { return mNphePMC; }
  float getPMQLightYield() const { return mNphePMQ; }
  int getNumContributingSteps() const { return mNoContributingSteps; }

 private:
  Int_t mParentID;
  Bool_t mSecFlag;
  Float_t mPrimaryEnergy;
  Int_t mNoContributingSteps = 1;
  Int_t mSectorID;
  Vector3D<float> mMomentum;
  Vector3D<float> mXImpact;
  Int_t mNphePMC;
  Int_t mNphePMQ;

  ClassDefNV(Hit, 1);
};

inline Hit::Hit(int trackID, int parent, Bool_t sFlag, Float_t primaryEnergy, Int_t detID, Int_t sectorID,
                Vector3D<float> pos, Vector3D<float> mom, Float_t tof, Vector3D<float> xImpact, Float_t energyloss,
                Int_t nphePMC, Int_t nphePMQ)
  : BasicXYZEHit(pos.X(), pos.Y(), pos.Z(), tof, energyloss, trackID, detID),
    mParentID(parent),
    mSecFlag(sFlag),
    mPrimaryEnergy(primaryEnergy),
    mSectorID(sectorID),
    mMomentum(mom.X(), mom.Y(), mom.Z()),
    mXImpact(xImpact),
    mNphePMC(nphePMC),
    mNphePMQ(nphePMQ)
{
}
} // namespace zdc
} // namespace o2

#ifdef USESHM
namespace std
{
template <>
class allocator<o2::zdc::Hit> : public o2::utils::ShmAllocator<o2::zdc::Hit>
{
};
} // namespace std

#endif

#endif
