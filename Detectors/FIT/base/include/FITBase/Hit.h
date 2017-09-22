// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

#ifndef ALICEO2_FIT_HIT_H
#define ALICEO2_FIT_HIT_H

#include "SimulationDataFormat/BaseHits.h"

namespace o2
{
namespace fit
{
/// \class Hit
/// \brief TOF simulation hit information
class Hit : public o2::BasicXYZEHit<float>
{
 public:
  /// \brief Default constructor
  Hit() = default;

  /// \brief Hit constructor
  /// \param shunt
  /// \param trackID Index of the track
  /// \param detID ID of the detector segment
  /// \param pos Position vector of the point
  /// \param tof Time of the hit
  Hit(Float_t x, Float_t y, Float_t z, Float_t time, Float_t energy, Int_t trackId, Int_t detId)
    : o2::BasicXYZEHit<float>(x, y, z, time, energy, trackId, detId)
  {
  }
  /// \brief Destructor
  ~Hit() override = default;

  void PrintStream(std::ostream& stream) const;

  ClassDef(Hit, 1);

  //   std::ostream &operator<<(std::ostream &stream, const Hit &point);
};
}
}
#endif
