// Copyright 2020-2022 CERN and copyright holders of ALICE O2.
// See https://alice-o2.web.cern.ch/copyright for details of the copyright holders.
// All rights not expressly granted are reserved.
//
// This software is distributed under the terms of the GNU General Public
// License v3 (GPL Version 3), copied verbatim in the file "COPYING".
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

#include "ITS3Align/MisalignmentHits.h"
#include "ITS3Base/SegmentationSuperAlpide.h"
#include "ITS3Base/ITS3Params.h"
#include "SimConfig/DigiParams.h"
#include "DetectorsBase/Propagator.h"
#include "Framework/Logger.h"

#include "Math/Factory.h"
#include "Math/UnaryOperators.h"
#include "TGeoNode.h"
#include "TGeoBBox.h"
#include "TString.h"

#include <memory>
#include <string>
#include <cstring>
#include <algorithm>

namespace o2::its3::align
{

void MisAlignmentHits::init()
{
  if (o2::its3::ITS3Params::Instance().misalignmentHitsUseProp) {
    mMethod = PropMethod::Propagator;
  } else {
    mMethod = PropMethod::Line;
  }

  mGeo = o2::its::GeometryTGeo::Instance();

  mMinimizer.reset(ROOT::Math::Factory::CreateMinimizer("Minuit2", "Migrad"));
  if (mMinimizer == nullptr) {
    LOGP(fatal, "Cannot create minimizer");
  }
  mMinimizer->SetMaxFunctionCalls(1'000'000'000);
  mMinimizer->SetStrategy(0);
  mMinimizer->SetPrintLevel(0);

  if (mMethod == PropMethod::Propagator) {
    LOGP(info, "Using propagator to find intersection");
    const auto& prefix = o2::conf::DigiParams::Instance().digitizationgeometry_prefix;
    mMCReader = std::make_unique<o2::steer::MCKinematicsReader>(prefix, o2::steer::MCKinematicsReader::Mode::kMCKine);
    mMinimizer->SetFunction(mPropagator);
  } else {
    LOGP(info, "Using local straight-line to find intersection");
    mMinimizer->SetFunction(mLine);
  }

  resetStats();

  if (auto file = o2::its3::ITS3Params::Instance().misalignmentHitsParams; file.empty()) {
    LOGP(fatal, "No parameter file specified");
  } else {
    mDeformations.init(file);
  }
}

std::optional<o2::itsmft::Hit> MisAlignmentHits::processHit(int iEvent, const o2::itsmft::Hit& hit)
{
  ++mStats[Stats::kHitTotal];

  if (!constants::detID::isDetITS3(hit.GetDetectorID())) {
    ++mStats[Stats::kHitIsOB];
    return hit;
  }
  ++mStats[Stats::kHitIsIB];

  // Set the working hits
  mCurHit = hit;
  mCurWorkingHits[WorkingHit::kEntering] = WorkingHit(iEvent, WorkingHit::kEntering, hit);
  mCurWorkingHits[WorkingHit::kExiting] = WorkingHit(iEvent, WorkingHit::kExiting, hit);

  // Do work
  if (!deformHit(WorkingHit::kEntering) || !deformHit(WorkingHit::kExiting)) {
    ++mStats[Stats::kHitDead];
    return std::nullopt;
  }
  ++mStats[Stats::kHitAlive];

  // Set the possibly new detectorIDs with mid point approximation
  auto midPointOrig = mCurWorkingHits[WorkingHit::kEntering].mPoint + (mCurWorkingHits[WorkingHit::kExiting].mPoint - mCurWorkingHits[WorkingHit::kEntering].mPoint) * 0.5;
  auto midPointDef = mCurWorkingHits[WorkingHit::kEntering].mPointDef + (mCurWorkingHits[WorkingHit::kExiting].mPointDef - mCurWorkingHits[WorkingHit::kEntering].mPointDef) * 0.5;
  const short idDef = getDetID(midPointDef), idOrig = getDetID(midPointOrig);
  if (idDef == -1) {
    return std::nullopt;
  }

  if (idDef != idOrig) {
    ++mStats[Stats::kHitMigrated];
  } else {
    ++mStats[Stats::kHitNotMigrated];
  }

  if constexpr (false) {
    /// TODO Does not yet work correctly
    /// Check if we crossed a boundary within the entering and exiting hit from the midpoint
    bool crossesBoundary{false};
    TGeoNode *nEnt{nullptr}, *nExt{nullptr};
    {
      auto dirEnt = mCurWorkingHits[WorkingHit::kEntering].mPointDef - midPointDef;
      auto stepEnt = std::min(static_cast<double>(dirEnt.R()), std::abs(dirEnt.R() - 5.e-4));
      auto dirEntU = dirEnt.Unit();
      gGeoManager->SetCurrentPoint(midPointDef.X(), midPointDef.Y(), midPointDef.Z());
      gGeoManager->SetCurrentDirection(dirEntU.X(), dirEntU.Y(), dirEntU.Z());
      nEnt = gGeoManager->FindNextBoundaryAndStep(stepEnt, false);
      if (gGeoManager->IsOnBoundary()) {
        ++mStats[Stats::kHitEntBoundary];
        crossesBoundary = true;
      }
    }
    {
      auto dirExt = midPointDef - mCurWorkingHits[WorkingHit::kEntering].mPointDef;
      auto stepExt = std::min(static_cast<double>(dirExt.R()), std::abs(dirExt.R() - 5.e-4));
      auto dirExtU = dirExt.Unit();
      gGeoManager->SetCurrentPoint(midPointDef.X(), midPointDef.Y(), midPointDef.Z());
      gGeoManager->SetCurrentDirection(dirExtU.X(), dirExtU.Y(), dirExtU.Z());
      nExt = gGeoManager->FindNextBoundaryAndStep(stepExt, false);
      if (gGeoManager->IsOnBoundary()) {
        ++mStats[Stats::kHitExtBoundary];
        crossesBoundary = true;
      }
    }

    if (crossesBoundary && nEnt != nullptr && nExt != nullptr) {
      if (nEnt != nExt) {
        return std::nullopt;
      } else {
        ++mStats[Stats::kHitSameBoundary]; // indicates that the step size is too large and we end up in the mother volume; just pretend that his fine for now
      }
    }
    ++mStats[Stats::kHitNoBoundary];
  }

  // Get new postion
  mCurHit.SetPosStart(mCurWorkingHits[WorkingHit::kEntering].mPointDef);
  mCurHit.SetPos(mCurWorkingHits[WorkingHit::kExiting].mPointDef);
  mCurHit.SetDetectorID(idDef);

  ++mStats[Stats::kHitSuccess];
  return mCurHit;
}

bool MisAlignmentHits::deformHit(WorkingHit::HitType t)
{
  auto& wHit = mCurWorkingHits[t];

  mMinimizer->Clear(); // clear for next iteration
  constexpr double minStep{1e-5};
  constexpr double zMargin{4.0};
  constexpr double phiMargin{0.4};
  if (mMethod == PropMethod::Line) {
    prepareLineMethod(t);
    mMinimizer->SetVariable(0, "t", 0.0, minStep); // this is left as a free parameter on since t is very small since start and end of hit are close
  } else {
    if (!preparePropagatorMethod(t)) {
      return false;
    }
    mMinimizer->SetVariable(0, "r", mPropagator.mTrack.getX(), minStep); // this is left as a free parameter on since t is very small since start and end of hit are close
  }
  mMinimizer->SetLimitedVariable(1, "phiStar", wHit.mPhi, minStep,
                                 std::max(static_cast<double>(wHit.mPhiBorder1), static_cast<double>(wHit.mPhi) - phiMargin),
                                 std::min(static_cast<double>(wHit.mPhiBorder2), static_cast<double>(wHit.mPhi) + phiMargin));
  mMinimizer->SetLimitedVariable(2, "zStar", wHit.mPoint.Z(), minStep,
                                 std::max(static_cast<double>(-constants::segment::lengthSensitive / 2.f), static_cast<double>(wHit.mPoint.Z()) - zMargin),
                                 std::min(static_cast<double>(constants::segment::lengthSensitive / 2.f), static_cast<double>(wHit.mPoint.Z()) + zMargin));

  mMinimizer->Minimize(); // perform the actual minimization

  auto ss = mMinimizer->Status();
  if (ss == 1) {
    ++mStats[Stats::kMinimizerCovPos];
  } else if (ss == 2) {
    ++mStats[Stats::kMinimizerHesse];
  } else if (ss == 3) {
    ++mStats[Stats::kMinimizerEDM];
  } else if (ss == 4) {
    ++mStats[Stats::kMinimizerLimit];
  } else if (ss == 5) {
    ++mStats[Stats::kMinimizerOther];
  } else {
    ++mStats[Stats::kMinimizerConverged];
  }

  if (ss == 0 || ss == 1) { // for Minuit2 0=ok, 1=ok with pos. forced hesse
    ++mStats[Stats::kMinimizerStatusOk];
    if (mMinimizer->MinValue() < 2e-4) { // within 2 um considering the pixel pitch this good enough
      ++mStats[Stats::kMinimizerValueOk];
    } else {
      ++mStats[Stats::kMinimizerValueBad];
      return false;
    }
  } else {
    ++mStats[Stats::kMinimizerStatusBad];
    return false;
  }

  // Valid solution found; calculate new position on ideal geo
  wHit.recalculateIdeal(static_cast<float>(mMinimizer->X()[1]), static_cast<float>(mMinimizer->X()[2]));

  return true;
}

short MisAlignmentHits::getDetID(const o2::math_utils::Point3D<float>& point)
{
  // Do not modify the path, I do not know if this is needed but lets be safe
  gGeoManager->PushPath();
  auto id = getDetIDFromCords(point);
  gGeoManager->PopPath();
  return id;
}

short MisAlignmentHits::getDetIDFromCords(const o2::math_utils::Point3D<float>& point)
{
  // retrive if any the node which constains the point
  const auto node = gGeoManager->FindNode(point.X(), point.Y(), point.Z());
  if (node == nullptr) {
    ++mStats[Stats::kFindNodeFailed];
    return -1;
  }
  ++mStats[Stats::kFindNodeSuccess];

  // check if this node is a sensitive volume
  const std::string path = gGeoManager->GetPath();
  if (path.find(o2::its::GeometryTGeo::getITS3SensorPattern()) == std::string::npos) {
    ++mStats[Stats::kProjNonSensitive];
    return -1;
  }
  ++mStats[Stats::kProjSensitive];

  return getDetIDFromPath(path);
}

short MisAlignmentHits::getDetIDFromPath(const std::string& path) const
{
  static const std::regex pattern{R"(/cave_1/barrel_1/ITSV_2/ITSUWrapVol0_1/ITS3Layer(\d+)_(\d+)/ITS3CarbonForm(\d+)_(\d+)/ITS3Chip(\d+)_(\d+)/ITS3Segment(\d+)_(\d+)/ITS3RSU(\d+)_(\d+)/ITS3Tile(\d+)_(\d+)/ITS3PixelArray(\d+)_(\d+))"};
  if (std::smatch matches; std::regex_search(path, matches, pattern)) {
    if (matches.size() == 15) {
      int iLayer = std::stoi(matches[1]);
      int iCarbonForm = std::stoi(matches[4]);
      int iSegment = std::stoi(matches[8]);
      int iRSU = std::stoi(matches[10]);
      int iTile = std::stoi(matches[12]);
      return mGeo->getChipIndex(iLayer, iCarbonForm, 0, iSegment, iRSU, iTile);
    } else {
      LOGP(fatal, "Path did not contain expected number of matches ({})!", matches.size());
    }
  } else {
    LOGP(fatal, "Path was not matched ({})!", path);
  }
  __builtin_unreachable();
}

void MisAlignmentHits::printStats() const
{
  auto makeFraction = [&](Stats n, Stats d) -> float { return static_cast<float>(mStats[n]) / static_cast<float>(mStats[d] + mStats[n]); };
  LOGP(info, "Processed {} Hits (IB:{}; OB:{}) ({:.2f}%):", mStats[Stats::kHitTotal], mStats[Stats::kHitIsIB], mStats[Stats::kHitIsOB], makeFraction(Stats::kHitIsIB, Stats::kHitIsOB));
  LOGP(info, "  - Minimizer Status: {} ok {} bad ({:.2f}%)", mStats[Stats::kMinimizerStatusOk], mStats[Stats::kMinimizerStatusBad], makeFraction(Stats::kMinimizerStatusOk, Stats::kMinimizerStatusBad));
  LOGP(info, "  - Minimizer Value: {} ok {} bad ({:.2f}%)", mStats[Stats::kMinimizerValueOk], mStats[Stats::kMinimizerValueBad], makeFraction(Stats::kMinimizerValueOk, Stats::kMinimizerValueBad));
  LOGP(info, "  - Minimizer Detailed: {} Converged {} pos. forced Hesse ({:.2f}%)", mStats[Stats::kMinimizerConverged], mStats[Stats::kMinimizerHesse], makeFraction(Stats::kMinimizerConverged, Stats::kMinimizerHesse));
  LOGP(info, "  - Minimizer Detailed: {} EDM {} call limit {} other ({:.2f}%)", mStats[Stats::kMinimizerEDM], mStats[Stats::kMinimizerLimit], mStats[Stats::kMinimizerOther], makeFraction(Stats::kMinimizerEDM, Stats::kMinimizerLimit));
  LOGP(info, "  - FindNode: {} ok {} failed", mStats[Stats::kFindNodeSuccess], mStats[Stats::kFindNodeFailed]);
  LOGP(info, "  - IsSensitve: {} yes {} no ({:.2f}%)", mStats[Stats::kProjSensitive], mStats[Stats::kProjNonSensitive], makeFraction(Stats::kProjSensitive, Stats::kProjNonSensitive));
  LOGP(info, "  - IsAlive: {} yes {} no ({:.2f}%)", mStats[Stats::kHitAlive], mStats[Stats::kHitDead], makeFraction(Stats::kHitAlive, Stats::kHitDead));
  LOGP(info, "  - HasMigrated: {} yes {} no ({:.2f}%)", mStats[Stats::kHitMigrated], mStats[Stats::kHitNotMigrated], makeFraction(Stats::kHitMigrated, Stats::kHitNotMigrated));
  // LOGP(info, "  - Crosses Boundary: {} entering {} exiting {} same {} no", mStats[Stats::kHitEntBoundary], mStats[Stats::kHitExtBoundary], mStats[Stats::kHitSameBoundary], mStats[Stats::kHitNoBoundary]);
  if (mMethod == PropMethod::Propagator) {
    LOGP(info, " - Propagator: {} null track {} null pdg", mStats[Stats::kPropTrackNull], mStats[Stats::kPropPDGNull]);
  }
  LOGP(info, "  --> Good Hits {} ({:.2f}%)", mStats[Stats::kHitSuccess], makeFraction(Stats::kHitSuccess, Stats::kHitIsIB));
}

void MisAlignmentHits::prepareLineMethod(WorkingHit::HitType from)
{
  // Set the starint point and radius
  // always start from the entering hit that way t is always pos. defined
  mLine.mStart = mCurWorkingHits[WorkingHit::kEntering].mPoint;
  mLine.mRadius = mCurWorkingHits[from].mRadius;
  mLine.mSensorID = mCurWorkingHits[from].mSensorID;
  mLine.mPhiTot = mCurWorkingHits[from].mPhiBorder2 - mCurWorkingHits[from].mPhiBorder1;
  mLine.mPhi1 = mCurWorkingHits[from].mPhiBorder1;
  // Calculate the direction vector
  mLine.mD[0] = mCurWorkingHits[WorkingHit::kExiting].mPoint.X() - mCurWorkingHits[WorkingHit::kEntering].mPoint.X();
  mLine.mD[1] = mCurWorkingHits[WorkingHit::kExiting].mPoint.Y() - mCurWorkingHits[WorkingHit::kEntering].mPoint.Y();
  mLine.mD[2] = mCurWorkingHits[WorkingHit::kExiting].mPoint.Z() - mCurWorkingHits[WorkingHit::kEntering].mPoint.Z();
}

double MisAlignmentHits::StraightLine::DoEval(const double* x) const
{
  const double t = x[0];
  const double phi = x[1];
  const double z = x[2];
  const double nphi = std::clamp((phi - mPhi1) * 2.0 / mPhiTot - 1.0, -1.0, 1.0);
  const double nz = std::clamp((z - (-constants::segment::lengthSensitive / 2.0)) * 2.0 / constants::segment::lengthSensitive - 1.0, -1.0, 1.0);

  /// Find the point along the line given current t
  double xline = mStart.X() + t * mD[0],
         yline = mStart.Y() + t * mD[1],
         zline = mStart.Z() + t * mD[2];

  // Find the point of the deformed geometry given a certain phi' and z'
  double xideal = mRadius * std::cos(phi), yideal = mRadius * std::sin(phi),
         zideal = z;
  const auto [dx, dy, dz] = mMis->getDeformation(mSensorID, nphi, nz);
  double xdef = xideal + dx, ydef = yideal + dy, zdef = zideal + dz;

  // Minimize the euclidean distance of the line point and the deformed point
  return std::hypot(xline - xdef, yline - ydef, zline - zdef);
}

bool MisAlignmentHits::preparePropagatorMethod(WorkingHit::HitType from)
{
  mPropagator.mRadius = mCurWorkingHits[from].mRadius;
  mPropagator.mSensorID = mCurWorkingHits[from].mSensorID;
  mPropagator.mPhiTot = mCurWorkingHits[from].mPhiBorder2 - mCurWorkingHits[from].mPhiBorder1;
  mPropagator.mPhi1 = mCurWorkingHits[from].mPhiBorder1;
  const auto mcTrack = mMCReader->getTrack(mCurWorkingHits[from].mEvent, mCurWorkingHits[from].mTrackID);
  if (mcTrack == nullptr) {
    ++mStats[Stats::kPropTrackNull];
    return false;
  }
  const std::array<float, 3> xyz{(float)mcTrack->GetStartVertexCoordinatesX(), (float)mcTrack->GetStartVertexCoordinatesY(), (float)mcTrack->GetStartVertexCoordinatesZ()},
    pxyz{(float)mcTrack->GetStartVertexMomentumX(), (float)mcTrack->GetStartVertexMomentumY(), (float)mcTrack->GetStartVertexMomentumZ()};
  const TParticlePDG* pPDG = TDatabasePDG::Instance()->GetParticle(mcTrack->GetPdgCode());
  if (pPDG == nullptr) {
    ++mStats[Stats::kPropPDGNull];
    return false;
  }
  mPropagator.mTrack = o2::track::TrackPar(xyz, pxyz, TMath::Nint(pPDG->Charge() / 3), false);
  mPropagator.mBz = o2::base::Propagator::Instance()->getNominalBz();
  return true;
}

double MisAlignmentHits::Propagator::DoEval(const double* x) const
{
  const double r = x[0];
  const double phi = x[1];
  const double z = x[2];
  const double nphi = (phi - mPhi1) * 2.0 / mPhiTot - 1.0;
  const double nz = (z - (-constants::segment::lengthSensitive / 2.0)) * 2.0 / constants::segment::lengthSensitive - 1.0;

  auto trc = mTrack;
  if (!trc.propagateTo(r, mBz)) {
    return 999;
  }
  const auto glo = trc.getXYZGlo();

  // Find the point of the deformed geometry given a certain phi' and z'
  double xideal = mRadius * std::cos(phi), yideal = mRadius * std::sin(phi),
         zideal = z;
  const auto [dx, dy, dz] = mMis->getDeformation(mSensorID, nphi, nz);
  double xdef = xideal + dx, ydef = yideal + dy, zdef = zideal + dz;

  // Minimize the euclidean distance of the propagator point and the deformed point
  return std::hypot(glo.X() - xdef, glo.Y() - ydef, glo.Z() - zdef);
}

} // namespace o2::its3::align
