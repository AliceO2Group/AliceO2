// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

/// \file laserTrackGenerator
/// \brief This macro implements a simple generator for laser tracks.
///
/// The laser track definitions are loaded from file.
/// Momenta need to be rescaled to avoid crashes in geant.
/// The sign is inverted to track them from the mirror inside the active volume
/// \author Jens Wiechula, Jens.Wiechula@ikf.uni-frankfurt.de

#if !defined(__CLING__) || defined(__ROOTCLING__)
#include <array>
#include "FairGenerator.h"
#include "FairPrimaryGenerator.h"
#include "DataFormatsTPC/LaserTrack.h"
#endif

class LaserTrackGenerator : public FairGenerator
{
 public:
  LaserTrackGenerator() : FairGenerator("TPCLaserTrackGenerator")
  {
    mLaserTrackContainer.loadTracksFromFile();
  }

  Bool_t ReadEvent(FairPrimaryGenerator* primGen) override
  {

    // loop over all tracks and add them to the generator
    const auto& tracks = mLaserTrackContainer.getLaserTracks();

    // TODO: use something better instead. The particle should stop at the inner field cage of the TPC.
    //       perhaps use a custom particle with special treatment in Detector.cxx
    //const int pdgCode = 2212;
    const int pdgCode = 11;
    std::array<float, 3> xyz;
    std::array<float, 3> pxyz;

    for (const auto& track : tracks) {
      track.getXYZGlo(xyz);
      track.getPxPyPzGlo(pxyz);
      // rescale to 1TeV to avoid segfault in geant
      // change sign to propagate tracks from the mirror invards
      // the tracking, however, will give values with the original sign
      auto norm = -1000. / track.getP();
      primGen->AddTrack(pdgCode, pxyz[0] * norm, pxyz[1] * norm, pxyz[2] * norm, xyz[0], xyz[1], xyz[2]);
      printf("Add track %.2f %.2f %.2f %.2f %.2f %.2f\n", pxyz[0] * norm, pxyz[1] * norm, pxyz[2] * norm, xyz[0], xyz[1], xyz[2]);
    }

    return kTRUE;
  }

 private:
  o2::tpc::LaserTrackContainer mLaserTrackContainer;
};

FairGenerator* laserTrackGenerator()
{
  auto gen = new LaserTrackGenerator();
  return gen;
}
