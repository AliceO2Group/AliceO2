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

#include "CompareTracks.h"
#include "Histos.h"
#include <TH1F.h>
#include <TH2F.h>
#include <TMath.h>

namespace o2::mch::eval
{

void printResiduals(const TrackParam& param1, const TrackParam& param2)
{
  /// print param2 - param1
  std::cout << "{dx = " << param2.getNonBendingCoor() - param1.getNonBendingCoor()
            << ", dy = " << param2.getBendingCoor() - param1.getBendingCoor()
            << ", dz = " << param2.getZ() - param1.getZ()
            << ", dpx = " << (param2.px() - param1.px()) << " (" << 100. * (param2.px() - param1.px()) / param1.px() << "\%)"
            << ", dpy = " << (param2.py() - param1.py()) << " (" << 100. * (param2.py() - param1.py()) / param1.py() << "\%)"
            << ", dpz = " << (param2.pz() - param1.pz()) << " (" << 100. * (param2.pz() - param1.pz()) / param1.pz() << "\%)"
            << ", dcharge = " << param2.getCharge() - param1.getCharge()
            << "}" << std::endl;
}

void printCovResiduals(const TMatrixD& cov1, const TMatrixD& cov2)
{
  /// print cov2 - cov1
  TMatrixD diff(cov2, TMatrixD::kMinus, cov1);
  diff.Print();
}

int compareEvents(std::list<ExtendedTrack>& tracks1,
                  std::list<ExtendedTrack>& tracks2,
                  double precision,
                  bool printDiff,
                  bool printAll,
                  std::vector<TH1*>& trackResidualsAtFirstCluster,
                  std::vector<TH1*>& clusterClusterResiduals)
{
  /// compare the tracks between the 2 events

  int nDifferences(0);

  // first look for identical tracks in the 2 events
  for (auto& track1 : tracks1) {
    // find a track in the second event identical to track1 and not already matched
    auto itTrack2 = tracks2.begin();
    do {
      itTrack2 = find(itTrack2, tracks2.end(), track1);
    } while (itTrack2 != tracks2.end() && itTrack2->hasMatchFound() && ++itTrack2 != tracks2.end());
    if (itTrack2 != tracks2.end()) {
      track1.setMatchFound(true);
      itTrack2->setMatchFound(true);
      track1.setMatchIdentical(true);
      itTrack2->setMatchIdentical(true);
      fillClusterClusterResiduals(track1, *itTrack2, clusterClusterResiduals);
      // compare the track parameters
      bool areParamCompatible = areCompatible(track1.param(), itTrack2->param(), precision);
      bool areCovCompatible = areCompatible(track1.param().getCovariances(), itTrack2->param().getCovariances(), precision);
      if (!areParamCompatible || !areCovCompatible) {
        ++nDifferences;
      }
      if (printAll || (printDiff && !areParamCompatible)) {
        printResiduals(track1.param(), itTrack2->param());
      }
      if (printAll || (printDiff && !areCovCompatible)) {
        printCovResiduals(track1.param().getCovariances(), itTrack2->param().getCovariances());
      }
      fillTrackResiduals(track1.param(), itTrack2->param(), trackResidualsAtFirstCluster);
    }
  }

  // then look for similar tracks in the 2 events
  for (auto& track1 : tracks1) {
    // skip already matched tracks
    if (track1.hasMatchFound()) {
      continue;
    }
    // find a track in the second event similar to track1 and not already matched
    for (auto& track2 : tracks2) {
      if (!track2.hasMatchFound() && track2.isMatching(track1)) {
        track1.setMatchFound(true);
        track2.setMatchFound(true);
        fillClusterClusterResiduals(track1, track2, clusterClusterResiduals);
        // compare the track parameters
        bool areParamCompatible = areCompatible(track1.param(), track2.param(), precision);
        bool areCovCompatible = areCompatible(track1.param().getCovariances(), track2.param().getCovariances(), precision);
        if (!areParamCompatible || !areCovCompatible) {
          ++nDifferences;
        }
        if (printAll || (printDiff && !areParamCompatible)) {
          printResiduals(track1.param(), track2.param());
        }
        if (printAll || (printDiff && !areCovCompatible)) {
          printCovResiduals(track1.param().getCovariances(), track2.param().getCovariances());
        }
        fillTrackResiduals(track1.param(), track2.param(), trackResidualsAtFirstCluster);
        break;
      }
    }
  }

  // then print the missing tracks
  for (const auto& track1 : tracks1) {
    if (!track1.hasMatchFound()) {
      if (printDiff) {
        std::cout << "did not find a track in file 2 matching: " << track1 << "\n";
      }
      ++nDifferences;
    }
  }

  // and finally print the additional tracks
  for (const auto& track2 : tracks2) {
    if (!track2.hasMatchFound()) {
      if (printDiff) {
        std::cout << "did not find a track in file 1 matching: " << track2 << "\n";
      }
      ++nDifferences;
    }
  }

  return nDifferences;
}

bool areCompatible(const TrackParam& param1, const TrackParam& param2, double precision)
{
  /// compare track parameters within precision
  return (abs(param2.getNonBendingCoor() - param1.getNonBendingCoor()) <= precision &&
          abs(param2.getBendingCoor() - param1.getBendingCoor()) <= precision &&
          abs(param2.getZ() - param1.getZ()) <= precision &&
          abs(param2.px() - param1.px()) <= precision &&
          abs(param2.py() - param1.py()) <= precision &&
          abs(param2.pz() - param1.pz()) <= precision &&
          param2.getCharge() == param1.getCharge());
}

bool areCompatible(const TMatrixD& cov1, const TMatrixD& cov2, double precision)
{
  /// compare track parameters covariances (if any) within precision
  if (cov1.NonZeros() == 0 || cov2.NonZeros() == 0) {
    return true;
  }
  TMatrixD diff(cov2, TMatrixD::kMinus, cov1);
  return (diff <= precision && diff >= -precision);
}

bool isSelected(const ExtendedTrack& track)
{
  /// apply standard track selections + pDCA

  static const double sigmaPDCA23 = 80.;
  static const double sigmaPDCA310 = 54.;
  static const double nSigmaPDCA = 6.;
  static const double relPRes = 0.0004;
  static const double slopeRes = 0.0005;

  double thetaAbs = TMath::ATan(track.getRabs() / 505.) * TMath::RadToDeg();
  if (thetaAbs < 2. || thetaAbs > 10.) {
    return false;
  }

  double eta = track.P().Eta();
  if (eta < -4. || eta > -2.5) {
    return false;
  }

  double pUncorr = TMath::Sqrt(track.param().px() * track.param().px() + track.param().py() * track.param().py() + track.param().pz() * track.param().pz());
  double pDCA = pUncorr * track.getDCA();
  double sigmaPDCA = (thetaAbs < 3) ? sigmaPDCA23 : sigmaPDCA310;
  double pTot = track.P().P();
  double nrp = nSigmaPDCA * relPRes * pTot;
  double pResEffect = sigmaPDCA / (1. - nrp / (1. + nrp));
  double slopeResEffect = 535. * slopeRes * pTot;
  double sigmaPDCAWithRes = TMath::Sqrt(pResEffect * pResEffect + slopeResEffect * slopeResEffect);
  if (pDCA > nSigmaPDCA * sigmaPDCAWithRes) {
    return false;
  }

  return true;
}

void selectTracks(std::list<ExtendedTrack>& tracks)
{
  /// remove tracks that do not pass the selection criteria
  for (auto itTrack = tracks.begin(); itTrack != tracks.end();) {
    if (!isSelected(*itTrack)) {
      itTrack = tracks.erase(itTrack);
    } else {
      ++itTrack;
    }
  }
}

} // namespace o2::mch::eval
