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

/// \file ClusterFinderOriginal.cxx
/// \brief Definition of a class to reconstruct clusters with the original MLEM algorithm
///
/// The original code is in AliMUONClusterFinderMLEM and associated classes.
/// It has been re-written in an attempt to simplify it without changing the results.
///
/// \author Philippe Pillot, Subatech

#include "MCHClustering/ClusterFinderOriginal.h"

#include <algorithm>
#include <cstring>
#include <iterator>
#include <limits>
#include <numeric>
#include <set>
#include <stdexcept>
#include <string>

#include <TH2I.h>
#include <TAxis.h>
#include <TMath.h>
#include <TRandom.h>

#include <FairLogger.h>

#include "MCHBase/MathiesonOriginal.h"
#include "MCHBase/ResponseParam.h"
#include "MCHClustering/ClusterizerParam.h"
#include "PadOriginal.h"
#include "ClusterOriginal.h"

namespace o2
{
namespace mch
{

//_________________________________________________________________________________________________
ClusterFinderOriginal::ClusterFinderOriginal()
  : mMathiesons(std::make_unique<MathiesonOriginal[]>(2)),
    mPreCluster(std::make_unique<ClusterOriginal>())
{
  /// default constructor
}

//_________________________________________________________________________________________________
ClusterFinderOriginal::~ClusterFinderOriginal() = default;

//_________________________________________________________________________________________________
void ClusterFinderOriginal::init(bool run2Config)
{
  /// initialize the clustering for run2 or run3 data

  mPreClusterFinder.init();

  if (run2Config) {

    // function to reinterpret digit ADC as calibrated charge
    mADCToCharge = [](uint32_t adc) {
      float charge(0.);
      std::memcpy(&charge, &adc, sizeof(adc));
      return static_cast<double>(charge);
    };

    // minimum charge of pad, pixel and cluster
    mLowestPadCharge = 4.f * 0.22875f;
    mLowestPixelCharge = mLowestPadCharge / 12.;
    mLowestClusterCharge = 2. * mLowestPadCharge;

    // Mathieson function for station 1
    mMathiesons[0].setPitch(0.21);
    mMathiesons[0].setSqrtKx3AndDeriveKx2Kx4(0.7000);
    mMathiesons[0].setSqrtKy3AndDeriveKy2Ky4(0.7550);

    // Mathieson function for other stations
    mMathiesons[1].setPitch(0.25);
    mMathiesons[1].setSqrtKx3AndDeriveKx2Kx4(0.7131);
    mMathiesons[1].setSqrtKy3AndDeriveKy2Ky4(0.7642);

  } else {

    // minimum charge of pad, pixel and cluster
    mLowestPadCharge = ClusterizerParam::Instance().lowestPadCharge;
    mLowestPixelCharge = mLowestPadCharge / 12.;
    mLowestClusterCharge = 2. * mLowestPadCharge;

    // Mathieson function for station 1
    mMathiesons[0].setPitch(ResponseParam::Instance().pitchSt1);
    mMathiesons[0].setSqrtKx3AndDeriveKx2Kx4(ResponseParam::Instance().mathiesonSqrtKx3St1);
    mMathiesons[0].setSqrtKy3AndDeriveKy2Ky4(ResponseParam::Instance().mathiesonSqrtKy3St1);

    // Mathieson function for other stations
    mMathiesons[1].setPitch(ResponseParam::Instance().pitchSt2345);
    mMathiesons[1].setSqrtKx3AndDeriveKx2Kx4(ResponseParam::Instance().mathiesonSqrtKx3St2345);
    mMathiesons[1].setSqrtKy3AndDeriveKy2Ky4(ResponseParam::Instance().mathiesonSqrtKy3St2345);
  }
}

//_________________________________________________________________________________________________
void ClusterFinderOriginal::deinit()
{
  /// deinitialize the clustering
  mPreClusterFinder.deinit();
}

//_________________________________________________________________________________________________
void ClusterFinderOriginal::reset()
{
  /// reset the list of reconstructed clusters and associated digits
  mClusters.clear();
  mUsedDigits.clear();
}

//_________________________________________________________________________________________________
void ClusterFinderOriginal::findClusters(gsl::span<const Digit> digits)
{
  /// reconstruct the clusters from the list of digits of one precluster
  /// reconstructed clusters and associated digits are added to the internal lists

  // skip preclusters with only 1 digit
  if (digits.size() < 2) {
    return;
  }

  // set the Mathieson function to be used
  mMathieson = (digits[0].getDetID() < 300) ? &mMathiesons[0] : &mMathiesons[1];

  // reset the current precluster being processed
  resetPreCluster(digits);

  // try to simplify the precluster by removing pads if possible (sent back to preclustering)
  std::vector<int> removedDigits{};
  simplifyPreCluster(removedDigits);

  if (mPreCluster->multiplicity() > 1) {

    // extract clusters from the precluster
    int iNewCluster = mClusters.size();
    processPreCluster();

    if (mClusters.size() > iNewCluster) {

      // copy the digits associated to the new clusters (if any) in the list of used digits
      int iFirstNewDigit = mUsedDigits.size();
      for (const auto& pad : *mPreCluster) {
        if (pad.isReal()) {
          mUsedDigits.emplace_back(digits[pad.digitIndex()]);
        }
      }
      int nNewDigits = mUsedDigits.size() - iFirstNewDigit;

      // give the new clusters a unique ID, make them point to these digits then set their resolution
      for (; iNewCluster < mClusters.size(); ++iNewCluster) {
        mClusters[iNewCluster].uid = Cluster::buildUniqueId(digits[0].getDetID() / 100 - 1, digits[0].getDetID(), iNewCluster);
        mClusters[iNewCluster].firstDigit = iFirstNewDigit;
        mClusters[iNewCluster].nDigits = nNewDigits;
        setClusterResolution(mClusters[iNewCluster]);
      }
    }
  }

  if (!removedDigits.empty()) {

    // load the released digits (if any) in the preclusterizer
    mPreClusterFinder.reset();
    for (auto iDigit : removedDigits) {
      mPreClusterFinder.loadDigit(digits[iDigit]);
    }

    // preclusterize and get the new preclusters and associated digits
    std::vector<PreCluster> preClusters{};
    std::vector<Digit> usedDigits{};
    int nPreClusters = mPreClusterFinder.run();
    preClusters.reserve(nPreClusters);
    usedDigits.reserve(removedDigits.size());
    mPreClusterFinder.getPreClusters(preClusters, usedDigits);

    // clusterize every new preclusters
    for (const auto& preCluster : preClusters) {
      findClusters({&usedDigits[preCluster.firstDigit], preCluster.nDigits});
    }
  }
}

//_________________________________________________________________________________________________
void ClusterFinderOriginal::resetPreCluster(gsl::span<const Digit>& digits)
{
  /// reset the precluster with the pads converted from the input digits

  mPreCluster->clear();

  mSegmentation = &mapping::segmentation(digits[0].getDetID());

  for (int iDigit = 0; iDigit < digits.size(); ++iDigit) {

    const auto& digit = digits[iDigit];
    int padID = digit.getPadID();

    double x = mSegmentation->padPositionX(padID);
    double y = mSegmentation->padPositionY(padID);
    double dx = mSegmentation->padSizeX(padID) / 2.;
    double dy = mSegmentation->padSizeY(padID) / 2.;
    double charge = mADCToCharge(digit.getADC());
    bool isSaturated = digit.isSaturated();
    int plane = mSegmentation->isBendingPad(padID) ? 0 : 1;

    if (charge <= 0.) {
      throw std::runtime_error("The precluster contains a digit with charge <= 0");
    }

    mPreCluster->addPad(x, y, dx, dy, charge, isSaturated, plane, iDigit, PadOriginal::kZero);
  }
}

//_________________________________________________________________________________________________
void ClusterFinderOriginal::simplifyPreCluster(std::vector<int>& removedDigits)
{
  /// try to simplify the precluster if possible (mostly for two-cathode preclusters)
  /// - for the preclusters that are too small, all pads are simply removed
  /// - for the others, the discarded pads (if any) are sent back to the preclustering
  /// return true if the precluster has been simplified

  // discard small clusters (leftovers from splitting or noise)
  if (mPreCluster->multiplicity() < 3 && mPreCluster->charge() < mLowestClusterCharge) {
    mPreCluster->clear();
    return;
  }

  // the following is only for two-cathode preclusters
  if (mPreCluster->multiplicity(0) == 0 || mPreCluster->multiplicity(1) == 0) {
    return;
  }

  // tag every pad that overlap with another on the other plane
  std::vector<bool> overlap(mPreCluster->multiplicity(), false);
  for (int i = 0; i < mPreCluster->multiplicity(); ++i) {
    const auto& padi = mPreCluster->pad(i);
    for (int j = i + 1; j < mPreCluster->multiplicity(); ++j) {
      const auto& padj = mPreCluster->pad(j);
      if (padi.plane() == padj.plane() || (overlap[i] && overlap[j])) {
        continue;
      }
      if (areOverlapping(padi, padj, SDistancePrecision)) {
        overlap[i] = true;
        overlap[j] = true;
      }
    }
  }

  // count the number of pads that do not overlap
  int nNotOverlapping(0);
  for (int i = 0; i < mPreCluster->multiplicity(); ++i) {
    if (!overlap[i]) {
      ++nNotOverlapping;
    }
  }

  // discard pads with no overlap (unless it is at the edge or there is only one with low charge)
  // loop over pads in decreasing index order to do not shift the indices while removing
  if (nNotOverlapping > 0) {
    const mapping::CathodeSegmentation* cathSeg[2] = {&mSegmentation->bending(), &mSegmentation->nonBending()};
    for (int i = mPreCluster->multiplicity() - 1; i >= 0; --i) {
      if (overlap[i]) {
        continue;
      }
      const auto& pad = mPreCluster->pad(i);
      if (nNotOverlapping == 1 && pad.charge() < mLowestPadCharge) {
        break; // there is only one
      }
      int cathPadIdx = cathSeg[1 - pad.plane()]->findPadByPosition(pad.x(), pad.y());
      if (!cathSeg[1 - pad.plane()]->isValid(cathPadIdx)) {
        continue;
      }
      removedDigits.push_back(pad.digitIndex());
      mPreCluster->removePad(i);
    }
  }

  // now adresses the case of large charge asymmetry between the two planes
  if (!mPreCluster->isSaturated() && mPreCluster->chargeAsymmetry() > 0.5) {

    // get the pads with minimum and maximum charge on the plane with maximum integrated charge
    int plane = mPreCluster->maxChargePlane();
    double chargeMin(std::numeric_limits<double>::max()), chargeMax(-1.);
    int iPadMin(-1), iPadMax(-1);
    for (int i = 0; i < mPreCluster->multiplicity(); ++i) {
      const auto& pad = mPreCluster->pad(i);
      if (pad.plane() == plane) {
        if (pad.charge() < chargeMin) {
          chargeMin = pad.charge();
          iPadMin = i;
        }
        if (pad.charge() > chargeMax) {
          chargeMax = pad.charge();
          iPadMax = i;
        }
      }
    }
    if (iPadMin < 0 || iPadMax < 0) {
      throw std::runtime_error("This plane should contain at least 1 pad!?");
    }

    // distance of the pad with minimum charge to the pad with maximum charge
    const auto& padMin = mPreCluster->pad(iPadMin);
    const auto& padMax = mPreCluster->pad(iPadMax);
    double dxOfMin = (padMin.x() - padMax.x()) / padMax.dx() / 2.;
    double dyOfMin = (padMin.y() - padMax.y()) / padMax.dy() / 2.;
    double distOfMin = TMath::Sqrt(dxOfMin * dxOfMin + dyOfMin * dyOfMin);

    // arrange pads of this plane according to their distance to the pad with maximum charge normalized to its size
    double precision = SDistancePrecision / 2. * TMath ::Sqrt((1. / padMax.dx() / padMax.dx() + 1. / padMax.dy() / padMax.dy()) / 2.);
    auto cmp = [precision](double dist1, double dist2) { return dist1 < dist2 - precision; };
    std::multimap<double, int, decltype(cmp)> distIndices(cmp);
    for (int i = 0; i < mPreCluster->multiplicity(); ++i) {
      if (i == iPadMax) {
        distIndices.emplace(0., i);
      } else {
        const auto& pad = mPreCluster->pad(i);
        if (pad.plane() == plane) {
          double dx = (pad.x() - padMax.x()) / padMax.dx() / 2.;
          double dy = (pad.y() - padMax.y()) / padMax.dy() / 2.;
          distIndices.emplace(TMath::Sqrt(dx * dx + dy * dy), i);
        }
      }
    }

    // try to extract from this plane the cluster centered around the pad with maximum charge
    double previousDist(std::numeric_limits<float>::max());
    double previousChargeMax(-1.);
    std::set<int, std::greater<>> padsToRemove{};
    for (const auto& distIndex : distIndices) {
      const auto& pad = mPreCluster->pad(distIndex.second);

      // try not to overstep the pad with minimum charge
      if (distIndex.first > distOfMin + precision) {
        double ddx = (pad.x() - padMax.x()) / padMax.dx() / 2. * dxOfMin;
        double ddy = (pad.y() - padMax.y()) / padMax.dy() / 2. * dyOfMin;
        if ((ddx > -precision && ddy > -precision) ||
            (TMath::Abs(ddx) > TMath::Abs(ddy) + precision && ddx > -precision) ||
            (TMath::Abs(ddy) > TMath::Abs(ddx) + precision && ddy > -precision)) {
          continue;
        }
      }

      // stop if their is a gap between this pad and the last one tagged for removal
      if (distIndex.first > previousDist + 1. + precision) {
        break;
      }

      // update the maximum charge if we reach another ring
      if (TMath::Abs(distIndex.first - previousDist) >= precision) {
        previousChargeMax = chargeMax;
      }

      // tag all pads at this distance to be removed if the maximum charge decreases
      if (pad.charge() <= previousChargeMax) {
        if (TMath::Abs(distIndex.first - previousDist) < precision) {
          chargeMax = TMath::Max(pad.charge(), chargeMax);
        } else {
          chargeMax = pad.charge();
        }
        previousDist = distIndex.first;
        padsToRemove.insert(distIndex.second);
      }
    }

    // remove the tagged pads (in decreasing index order to do not invalidate them while removing)
    for (auto iPad : padsToRemove) {
      removedDigits.push_back(mPreCluster->pad(iPad).digitIndex());
      mPreCluster->removePad(iPad);
    }
  }
}

//_________________________________________________________________________________________________
void ClusterFinderOriginal::processPreCluster()
{
  /// builds an array of pixel and extract clusters from it

  buildPixArray();
  if (mPixels.empty()) {
    return;
  }

  // switch between different clustering methods depending on the complexity of the precluster
  std::pair<int, int> nPadsXY = mPreCluster->sizeInPads(PadOriginal::kZero);
  if (nPadsXY.first < 4 && nPadsXY.second < 4) {

    // simple precluster
    processSimple();

  } else {

    // find the local maxima in the pixel array
    std::unique_ptr<TH2D> histAnode(nullptr);
    std::multimap<double, std::pair<int, int>, std::greater<>> localMaxima{};
    findLocalMaxima(histAnode, localMaxima);
    if (localMaxima.empty()) {
      return;
    }

    if (localMaxima.size() == 1 || mPreCluster->multiplicity() <= 50) {

      // precluster of reasonable size --> treat it in one piece
      process();

    } else {

      // too large precluster --> split it and treat every pieces separately
      for (const auto& localMaximum : localMaxima) {

        // select the part of the precluster that is around the local maximum
        restrictPreCluster(*histAnode, localMaximum.second.first, localMaximum.second.second);

        // treat it
        process();
      }
    }
  }
}

//_________________________________________________________________________________________________
void ClusterFinderOriginal::buildPixArray()
{
  /// build pixel array for MLEM method

  mPixels.clear();

  // pixel half size is the minimum pad half size on both plane
  std::pair<double, double> dim = mPreCluster->minPadDimensions(-1, false);
  double width[2] = {dim.first, dim.second};

  // to make sure the grid is aligned with the smallest pad in both direction
  double xy0[2] = {0., 0.};
  bool found[2] = {false, false};
  for (const auto& pad : *mPreCluster) {
    for (int ixy = 0; ixy < 2; ++ixy) {
      if (!found[ixy] && TMath::Abs(pad.dxy(ixy) - width[ixy]) < SDistancePrecision) {
        xy0[ixy] = pad.xy(ixy);
        found[ixy] = true;
      }
    }
    if (found[0] && found[1]) {
      break;
    }
  }

  // to deal with mono-cathod clusters
  int plane0 = 0, plane1 = 1;
  if (mPreCluster->multiplicity(0) == 0) {
    plane0 = 1;
  } else if (mPreCluster->multiplicity(1) == 0) {
    plane1 = 0;
  }

  // grid size is the intersect of the cluster areas on both planes
  double area[2][2] = {0.};
  mPreCluster->area(plane0, area);
  if (plane1 != plane0) {
    double area2[2][2] = {0.};
    mPreCluster->area(plane1, area2);
    area[0][0] = TMath::Max(area[0][0], area2[0][0]);
    area[0][1] = TMath::Min(area[0][1], area2[0][1]);
    area[1][0] = TMath::Max(area[1][0], area2[1][0]);
    area[1][1] = TMath::Min(area[1][1], area2[1][1]);
  }

  // abort if the areas do not intersect
  if (area[0][1] - area[0][0] < SDistancePrecision || area[1][1] - area[1][0] < SDistancePrecision) {
    return;
  }

  // adjust limits
  int nbins[2] = {0, 0};
  for (int ixy = 0; ixy < 2; ++ixy) {
    double precision = SDistancePrecision / width[ixy] / 2.;
    double dist = (area[ixy][0] - xy0[ixy]) / width[ixy] / 2.;
    area[ixy][0] = xy0[ixy] + (TMath::Nint(dist + precision) - 0.5) * width[ixy] * 2.;
    nbins[ixy] = TMath::Ceil((area[ixy][1] - area[ixy][0]) / width[ixy] / 2. - precision);
    area[ixy][1] = area[ixy][0] + nbins[ixy] * width[ixy] * 2.;
  }

  // book pixel histograms and fill them
  TH2D hCharges("Charges", "", nbins[0], area[0][0], area[0][1], nbins[1], area[1][0], area[1][1]);
  TH2I hEntries("Entries", "", nbins[0], area[0][0], area[0][1], nbins[1], area[1][0], area[1][1]);
  for (const auto& pad : *mPreCluster) {
    ProjectPadOverPixels(pad, hCharges, hEntries);
  }

  // store fired pixels with an entry from both planes if both planes are fired
  for (int i = 1; i <= nbins[0]; ++i) {
    double x = hCharges.GetXaxis()->GetBinCenter(i);
    for (int j = 1; j <= nbins[1]; ++j) {
      int entries = hEntries.GetBinContent(i, j);
      if (entries == 0 || (plane0 != plane1 && (entries < 1000 || entries % 1000 < 1))) {
        continue;
      }
      double y = hCharges.GetYaxis()->GetBinCenter(j);
      double charge = hCharges.GetBinContent(i, j);
      mPixels.emplace_back(x, y, width[0], width[1], charge);
    }
  }

  // split the pixel into 2 if there is only one
  if (mPixels.size() == 1) {
    auto& pixel = mPixels.front();
    pixel.setdx(width[0] / 2.);
    pixel.setx(pixel.x() - width[0] / 2.);
    mPixels.emplace_back(pixel.x() + width[0], pixel.y(), width[0] / 2., width[1], pixel.charge());
  }

  // sort and remove pixels with the lowest signal if there are too many
  size_t nPads = mPreCluster->multiplicity();
  if (mPixels.size() > nPads) {
    std::stable_sort(mPixels.begin(), mPixels.end(), [](const PadOriginal& pixel1, const PadOriginal& pixel2) {
      return pixel1.charge() > pixel2.charge();
    });
    mPixels.erase(mPixels.begin() + nPads, mPixels.end());
  }
}

//_________________________________________________________________________________________________
void ClusterFinderOriginal::ProjectPadOverPixels(const PadOriginal& pad, TH2D& hCharges, TH2I& hEntries) const
{
  /// project the pad over pixel histograms

  const TAxis* xaxis = hCharges.GetXaxis();
  const TAxis* yaxis = hCharges.GetYaxis();

  int iMin = TMath::Max(1, xaxis->FindBin(pad.x() - pad.dx() + SDistancePrecision));
  int iMax = TMath::Min(hCharges.GetNbinsX(), xaxis->FindBin(pad.x() + pad.dx() - SDistancePrecision));
  int jMin = TMath::Max(1, yaxis->FindBin(pad.y() - pad.dy() + SDistancePrecision));
  int jMax = TMath::Min(hCharges.GetNbinsY(), yaxis->FindBin(pad.y() + pad.dy() - SDistancePrecision));

  double charge = pad.charge();
  int entry = 1 + pad.plane() * 999;

  for (int i = iMin; i <= iMax; ++i) {
    for (int j = jMin; j <= jMax; ++j) {
      int entries = hEntries.GetBinContent(i, j);
      hCharges.SetBinContent(i, j, (entries > 0) ? TMath::Min(hCharges.GetBinContent(i, j), charge) : charge);
      hEntries.SetBinContent(i, j, entries + entry);
    }
  }
}

//_________________________________________________________________________________________________
void ClusterFinderOriginal::findLocalMaxima(std::unique_ptr<TH2D>& histAnode,
                                            std::multimap<double, std::pair<int, int>, std::greater<>>& localMaxima)
{
  /// find local maxima in pixel space for large preclusters in order to
  /// try to split them into smaller pieces (to speed up the MLEM procedure)
  /// and tag the corresponding pixels

  // create a 2D histogram from the pixel array
  double xMin(std::numeric_limits<double>::max()), xMax(-std::numeric_limits<double>::max());
  double yMin(std::numeric_limits<double>::max()), yMax(-std::numeric_limits<double>::max());
  double dx(mPixels.front().dx()), dy(mPixels.front().dy());
  for (const auto& pixel : mPixels) {
    xMin = TMath::Min(xMin, pixel.x());
    xMax = TMath::Max(xMax, pixel.x());
    yMin = TMath::Min(yMin, pixel.y());
    yMax = TMath::Max(yMax, pixel.y());
  }
  int nBinsX = TMath::Nint((xMax - xMin) / dx / 2.) + 1;
  int nBinsY = TMath::Nint((yMax - yMin) / dy / 2.) + 1;
  histAnode = std::make_unique<TH2D>("anode", "anode", nBinsX, xMin - dx, xMax + dx, nBinsY, yMin - dy, yMax + dy);
  for (const auto& pixel : mPixels) {
    histAnode->Fill(pixel.x(), pixel.y(), pixel.charge());
  }

  // find the local maxima
  std::vector<std::vector<int>> isLocalMax(nBinsX, std::vector<int>(nBinsY, 0));
  for (int j = 1; j <= nBinsY; ++j) {
    for (int i = 1; i <= nBinsX; ++i) {
      if (isLocalMax[i - 1][j - 1] == 0 && histAnode->GetBinContent(i, j) >= mLowestPixelCharge) {
        flagLocalMaxima(*histAnode, i, j, isLocalMax);
      }
    }
  }

  // store local maxima and tag corresponding pixels
  TAxis* xAxis = histAnode->GetXaxis();
  TAxis* yAxis = histAnode->GetYaxis();
  for (int j = 1; j <= nBinsY; ++j) {
    for (int i = 1; i <= nBinsX; ++i) {
      if (isLocalMax[i - 1][j - 1] > 0) {
        localMaxima.emplace(histAnode->GetBinContent(i, j), std::make_pair(i, j));
        auto itPixel = findPad(mPixels, xAxis->GetBinCenter(i), yAxis->GetBinCenter(j), mLowestPixelCharge);
        itPixel->setStatus(PadOriginal::kMustKeep);
        if (localMaxima.size() > 99) {
          break;
        }
      }
    }
    if (localMaxima.size() > 99) {
      LOG(warning) << "Too many local maxima !!!";
      break;
    }
  }
}

//_________________________________________________________________________________________________
void ClusterFinderOriginal::flagLocalMaxima(const TH2D& histAnode, int i0, int j0, std::vector<std::vector<int>>& isLocalMax) const
{
  /// flag the bin (i,j) as a local maximum or not by comparing its charge to the one of its neighbours
  /// and flag the neighbours accordingly (recursive procedure in case the charges are equal)

  int idxi0 = i0 - 1;
  int idxj0 = j0 - 1;
  int charge0 = TMath::Nint(histAnode.GetBinContent(i0, j0));
  int iMin = TMath::Max(1, i0 - 1);
  int iMax = TMath::Min(histAnode.GetNbinsX(), i0 + 1);
  int jMin = TMath::Max(1, j0 - 1);
  int jMax = TMath::Min(histAnode.GetNbinsY(), j0 + 1);

  for (int j = jMin; j <= jMax; ++j) {
    int idxj = j - 1;
    for (int i = iMin; i <= iMax; ++i) {
      if (i == i0 && j == j0) {
        continue;
      }
      int idxi = i - 1;
      int charge = TMath::Nint(histAnode.GetBinContent(i, j));
      if (charge0 < charge) {
        isLocalMax[idxi0][idxj0] = -1;
        return;
      } else if (charge0 > charge) {
        isLocalMax[idxi][idxj] = -1;
      } else if (isLocalMax[idxi][idxj] == -1) {
        isLocalMax[idxi0][idxj0] = -1;
        return;
      } else if (isLocalMax[idxi][idxj] == 0) {
        isLocalMax[idxi0][idxj0] = 1;
        flagLocalMaxima(histAnode, i, j, isLocalMax);
        if (isLocalMax[idxi][idxj] == -1) {
          isLocalMax[idxi0][idxj0] = -1;
          return;
        } else {
          isLocalMax[idxi][idxj] = -2;
        }
      }
    }
  }
  isLocalMax[idxi0][idxj0] = 1;
}

//_________________________________________________________________________________________________
void ClusterFinderOriginal::restrictPreCluster(const TH2D& histAnode, int i0, int j0)
{
  /// keep in the pixel array only the ones around the local maximum
  /// and tag the pads in the precluster that overlap with them

  // drop all pixels from the array and put back the ones around the local maximum
  mPixels.clear();
  const TAxis* xAxis = histAnode.GetXaxis();
  const TAxis* yAxis = histAnode.GetYaxis();
  double dx = xAxis->GetBinWidth(1) / 2.;
  double dy = yAxis->GetBinWidth(1) / 2.;
  double charge0 = histAnode.GetBinContent(i0, j0);
  int iMin = TMath::Max(1, i0 - 1);
  int iMax = TMath::Min(histAnode.GetNbinsX(), i0 + 1);
  int jMin = TMath::Max(1, j0 - 1);
  int jMax = TMath::Min(histAnode.GetNbinsY(), j0 + 1);
  for (int j = jMin; j <= jMax; ++j) {
    for (int i = iMin; i <= iMax; ++i) {
      double charge = histAnode.GetBinContent(i, j);
      if (charge >= mLowestPixelCharge && charge <= charge0) {
        mPixels.emplace_back(xAxis->GetBinCenter(i), yAxis->GetBinCenter(j), dx, dy, charge);
      }
    }
  }

  // discard all pads of the clusters except the ones overlapping with selected pixels
  for (auto& pad : *mPreCluster) {
    pad.setStatus(PadOriginal::kOver);
    for (const auto& pixel : mPixels) {
      if (areOverlapping(pad, pixel, SDistancePrecision)) {
        pad.setStatus(PadOriginal::kZero);
        break;
      }
    }
  }
}

//_________________________________________________________________________________________________
void ClusterFinderOriginal::processSimple()
{
  /// process simple precluster (small number of pads). In short: it computes the charge of the pixels
  /// with the MLEM algorithm and use them as a single cluster seed for the fit. It does not run
  /// the iterative reduction of pixel size and recalculation of their charge

  // add virtual pads if necessary
  addVirtualPad();

  // calculate pad-pixel coupling coefficients and pixel visibilities
  std::vector<double> coef(0);
  std::vector<double> prob(0);
  computeCoefficients(coef, prob);

  // discard "invisible" pixels
  for (int ipix = 0; ipix < mPixels.size(); ++ipix) {
    if (prob[ipix] < 0.01) {
      mPixels[ipix].setCharge(0);
    }
  }

  // run the MLEM algorithm with 15 iterations
  double qTot = mlem(coef, prob, 15);

  // abort if the total charge of the pixels is too low
  if (qTot < 1.e-4 || (mPreCluster->multiplicity() < 3 && qTot < mLowestClusterCharge)) {
    return;
  }

  // fit all pads but the ones saturated
  for (auto& pad : *mPreCluster) {
    if (!pad.isSaturated()) {
      pad.setStatus(PadOriginal::kUseForFit);
    }
  }

  // use all the pixels as a single cluster seed
  std::vector<int> pixels(mPixels.size());
  std::iota(pixels.begin(), pixels.end(), 0);

  // set the fit range based on the limits of the pixel array
  double fitRange[2][2] = {{std::numeric_limits<double>::max(), -std::numeric_limits<double>::max()},
                           {std::numeric_limits<double>::max(), -std::numeric_limits<double>::max()}};
  for (const auto& pixel : mPixels) {
    fitRange[0][0] = TMath::Min(fitRange[0][0], pixel.x() - 3. * pixel.dx());
    fitRange[0][1] = TMath::Max(fitRange[0][1], pixel.x() + 3. * pixel.dx());
    fitRange[1][0] = TMath::Min(fitRange[1][0], pixel.y() - 3. * pixel.dy());
    fitRange[1][1] = TMath::Max(fitRange[1][1], pixel.y() + 3. * pixel.dy());
  }

  // do the fit
  double fitParam[SNFitParamMax + 1] = {0.};
  fit({&pixels}, fitRange, fitParam);
}

//_________________________________________________________________________________________________
void ClusterFinderOriginal::process()
{
  /// process "standard" precluster. In short: it computes the charge of the pixels
  /// this is the core of the algorithm. In short: it computes the charge of the pixels with
  /// the MLEM algorithm then reduce their size and repeat it until the size is small enough,
  /// then it sends the result to the splitter

  // add virtual pads if necessary
  addVirtualPad();

  // number of pads to be considered in the precluster
  size_t npadOK(0);
  for (const auto& pad : *mPreCluster) {
    if (pad.status() == PadOriginal::kZero) {
      ++npadOK;
    }
  }

  // compute the limits of the histogram based on the current pixel array
  double xMin(std::numeric_limits<double>::max()), xMax(-std::numeric_limits<double>::max());
  double yMin(std::numeric_limits<double>::max()), yMax(-std::numeric_limits<double>::max());
  for (const auto& pixel : mPixels) {
    xMin = TMath::Min(xMin, pixel.x());
    xMax = TMath::Max(xMax, pixel.x());
    yMin = TMath::Min(yMin, pixel.y());
    yMax = TMath::Max(yMax, pixel.y());
  }

  std::vector<double> coef(0);
  std::vector<double> prob(0);
  std::unique_ptr<TH2D> histMLEM(nullptr);
  while (true) {

    // calculate pad-pixel coupling coefficients and pixel visibilities
    computeCoefficients(coef, prob);

    // discard "invisible" pixels
    for (int ipix = 0; ipix < mPixels.size(); ++ipix) {
      if (prob[ipix] < 0.01) {
        mPixels[ipix].setCharge(0);
      }
    }

    // run the MLEM algorithm with 15 iterations
    double qTot = mlem(coef, prob, 15);

    // abort if the total charge of the pixels is too low
    if (qTot < 1.e-4 || (npadOK < 3 && qTot < mLowestClusterCharge)) {
      return;
    }

    // create a 2D histogram from the pixel array
    double dx(mPixels.front().dx()), dy(mPixels.front().dy());
    int nBinsX = TMath::Nint((xMax - xMin) / dx / 2.) + 1;
    int nBinsY = TMath::Nint((yMax - yMin) / dy / 2.) + 1;
    histMLEM.reset(nullptr); // delete first to avoid "Replacing existing TH1: mlem (Potential memory leak)"
    histMLEM = std::make_unique<TH2D>("mlem", "mlem", nBinsX, xMin - dx, xMax + dx, nBinsY, yMin - dy, yMax + dy);
    for (const auto& pixel : mPixels) {
      histMLEM->Fill(pixel.x(), pixel.y(), pixel.charge());
    }

    // stop here if the pixel size is small enough
    if ((dx < 0.07 || dy < 0.07) && dy < dx) {
      break;
    }

    // calculate the position of the center-of-gravity around the pixel with maximum charge
    double xyCOG[2] = {0., 0.};
    findCOG(*histMLEM, xyCOG);

    // decrease the pixel size and align the array with the position of the center-of-gravity
    refinePixelArray(xyCOG, npadOK, xMin, xMax, yMin, yMax);
  }

  // discard pixels with low visibility by moving their charge to their nearest neighbour (cuts are empirical !!!)
  double threshold = TMath::Min(TMath::Max(histMLEM->GetMaximum() / 100., 2.0 * mLowestPixelCharge), 100.0 * mLowestPixelCharge);
  cleanPixelArray(threshold, prob);

  // re-run the MLEM algorithm with 2 iterations
  double qTot = mlem(coef, prob, 2);

  // abort if the total charge of the pixels is too low
  if (qTot < 2. * mLowestPixelCharge) {
    return;
  }

  // update the histogram
  for (const auto& pixel : mPixels) {
    histMLEM->SetBinContent(histMLEM->GetXaxis()->FindBin(pixel.x()), histMLEM->GetYaxis()->FindBin(pixel.y()), pixel.charge());
  }

  // split the precluster into clusters
  split(*histMLEM, coef);
}

//_________________________________________________________________________________________________
void ClusterFinderOriginal::addVirtualPad()
{
  /// add virtual pads (with small charge) to improve fit for
  /// preclusters with number of pads == 2 in x and/or y directions

  const mapping::CathodeSegmentation* cathSeg[2] = {&mSegmentation->bending(), &mSegmentation->nonBending()};

  // decide what plane to consider for x and y directions
  int iPlaneXY[2] = {1, 0}; // 0 = bending plane and 1 = non-bending plane as defined in resetPreCluster(...)

  // check if virtual pads are needed in each direction
  // if the minimum pad size in y is the same on both planes, check also the other plane
  std::pair<double, double> dim0 = mPreCluster->minPadDimensions(0, PadOriginal::kZero, true);
  std::pair<double, double> dim1 = mPreCluster->minPadDimensions(1, PadOriginal::kZero, true);
  bool sameSizeY = TMath::Abs(dim0.second - dim1.second) < SDistancePrecision;
  bool addVirtualPad[2] = {false, false};
  std::pair<int, int> nPadsXY0 = mPreCluster->sizeInPads(iPlaneXY[0], PadOriginal::kZero);
  std::pair<int, int> nPadsXY1 = mPreCluster->sizeInPads(iPlaneXY[1], PadOriginal::kZero);
  if (nPadsXY0.first == 2 && (!sameSizeY || nPadsXY1.first <= 2)) {
    addVirtualPad[0] = true;
  }
  if (nPadsXY1.second == 2 && (!sameSizeY || nPadsXY0.second <= 2)) {
    addVirtualPad[1] = true;
  }

  double chargeReduction[2] = {100., 15.};
  for (int ixy = 0; ixy < 2; ++ixy) {

    // no need to add virtual pads in this direction
    if (!addVirtualPad[ixy]) {
      continue;
    }

    // find pads with maximum and next-to-maximum charges on the plane of interest
    // find min and max dimensions of the precluster in this direction on that plane
    long iPadMax[2] = {-1, -1};
    double chargeMax[2] = {0., 0.};
    double limits[2] = {std::numeric_limits<double>::max(), -std::numeric_limits<double>::max()};
    for (int i = 0; i < mPreCluster->multiplicity(); ++i) {
      const auto& pad = mPreCluster->pad(i);
      if (pad.plane() != iPlaneXY[ixy]) {
        continue;
      }
      if (pad.charge() > chargeMax[0]) {
        chargeMax[1] = chargeMax[0];
        chargeMax[0] = pad.charge();
        iPadMax[1] = iPadMax[0];
        iPadMax[0] = i;
      } else if (pad.charge() > chargeMax[1]) {
        chargeMax[1] = pad.charge();
        iPadMax[1] = i;
      }
      double xy = pad.xy(ixy);
      if (xy < limits[0]) {
        limits[0] = xy;
      }
      if (xy > limits[1]) {
        limits[1] = xy;
      }
    }

    // try to add a virtual pad next to the max pad then, if done, next to the next-to-max pads
    // do not try to add a second virtual pad if the next-to-max charge is too low
    int n = (chargeMax[1] / chargeMax[0] < 0.5) ? 1 : 2;
    int sideDone(0);
    for (int i = 0; i < n; ++i) {

      if (iPadMax[i] < 0) {
        throw std::runtime_error("This plane should contain at least 2 pads!?");
      }

      // check if the pad is at the cluster limit and on which side to add a virtual pad
      const auto& pad = mPreCluster->pad(iPadMax[i]);
      int side(0);
      double xy = pad.xy(ixy);
      if (TMath::Abs(xy - limits[0]) < SDistancePrecision) {
        side = -1;
      } else if (TMath::Abs(xy - limits[1]) < SDistancePrecision) {
        side = 1;
      } else {
        break;
      }

      // do not add 2 virtual pads on the same side
      if (side == sideDone) {
        break;
      }

      // find the pad to add in the mapping
      double pos[2] = {pad.x(), pad.y()};
      pos[ixy] += side * (pad.dxy(ixy) + SDistancePrecision);
      pos[1 - ixy] += SDistancePrecision; // pickup always the same in case we fall at the border between 2 pads
      int cathPadIdx = cathSeg[iPlaneXY[ixy]]->findPadByPosition(pos[0], pos[1]);
      if (!cathSeg[iPlaneXY[ixy]]->isValid(cathPadIdx)) {
        break;
      }

      // add the virtual pad
      double charge = TMath::Max(TMath::Min(chargeMax[i] / chargeReduction[ixy], mLowestPadCharge), 2. * mLowestPixelCharge);
      mPreCluster->addPad(cathSeg[iPlaneXY[ixy]]->padPositionX(cathPadIdx), cathSeg[iPlaneXY[ixy]]->padPositionY(cathPadIdx),
                          cathSeg[iPlaneXY[ixy]]->padSizeX(cathPadIdx) / 2., cathSeg[iPlaneXY[ixy]]->padSizeY(cathPadIdx) / 2.,
                          charge, false, iPlaneXY[ixy], -1, pad.status());

      sideDone = side;
    }
  }
}

//_________________________________________________________________________________________________
void ClusterFinderOriginal::computeCoefficients(std::vector<double>& coef, std::vector<double>& prob) const
{
  /// Compute pad-pixel coupling coefficients and pixel visibilities needed for the MLEM algorithm

  coef.assign(mPreCluster->multiplicity() * mPixels.size(), 0.);
  prob.assign(mPixels.size(), 0.);

  int iCoef(0);
  for (const auto& pad : *mPreCluster) {

    // ignore the pads that must not be considered
    if (pad.status() != PadOriginal::kZero) {
      iCoef += mPixels.size();
      continue;
    }

    for (int i = 0; i < mPixels.size(); ++i) {

      // charge (given by Mathieson integral) on pad, assuming the Mathieson is center at pixel.
      coef[iCoef] = chargeIntegration(mPixels[i].x(), mPixels[i].y(), pad);

      // update the pixel visibility
      prob[i] += coef[iCoef];

      ++iCoef;
    }
  }
}

//_________________________________________________________________________________________________
double ClusterFinderOriginal::mlem(const std::vector<double>& coef, const std::vector<double>& prob, int nIter)
{
  /// use MLEM to update the charge of the pixels (iterative procedure with nIter iteration)
  /// return the total charge of all the pixels

  double qTot(0.);
  double maxProb = *std::max_element(prob.begin(), prob.end());
  std::vector<double> padSum(mPreCluster->multiplicity(), 0.);

  for (int iter = 0; iter < nIter; ++iter) {

    // calculate expectations, ignoring the pads that must not be considered
    int iCoef(0);
    for (int iPad = 0; iPad < mPreCluster->multiplicity(); ++iPad) {
      const auto& pad = mPreCluster->pad(iPad);
      if (pad.status() != PadOriginal::kZero) {
        iCoef += mPixels.size();
        continue;
      }
      padSum[iPad] = 0.;
      for (const auto& pixel : mPixels) {
        padSum[iPad] += pixel.charge() * coef[iCoef++];
      }
    }

    qTot = 0.;
    for (int iPix = 0; iPix < mPixels.size(); ++iPix) {

      // skip "invisible" pixel
      if (prob[iPix] < 0.01) {
        mPixels[iPix].setCharge(0.);
        continue;
      }

      double pixelSum(0.);
      double pixelNorm(maxProb);
      for (int iPad = 0; iPad < mPreCluster->multiplicity(); ++iPad) {

        // ignore the pads that must not be considered
        const auto& pad = mPreCluster->pad(iPad);
        if (pad.status() != PadOriginal::kZero) {
          continue;
        }

        // correct for pad charge overflows
        int iCoef = iPad * mPixels.size() + iPix;
        if (pad.isSaturated() && padSum[iPad] > pad.charge()) {
          pixelNorm -= coef[iCoef];
          continue;
        }

        if (padSum[iPad] > 1.e-6) {
          pixelSum += pad.charge() * coef[iCoef] / padSum[iPad];
        }
      }

      // correct the pixel charge
      if (pixelNorm > 1.e-6) {
        mPixels[iPix].setCharge(mPixels[iPix].charge() * pixelSum / pixelNorm);
        qTot += mPixels[iPix].charge();
      } else {
        mPixels[iPix].setCharge(0.);
      }
    }

    // can happen in clusters with large number of overflows - speeding up
    if (qTot < 1.e-6) {
      return qTot;
    }
  }

  return qTot;
}

//_________________________________________________________________________________________________
void ClusterFinderOriginal::findCOG(const TH2D& histMLEM, double xy[2]) const
{
  /// calculate the position of the center-of-gravity around the pixel with maximum charge

  // define the range of pixels and the minimum charge to consider
  int ix0(0), iy0(0), iz0(0);
  double chargeThreshold = histMLEM.GetBinContent(histMLEM.GetMaximumBin(ix0, iy0, iz0)) / 10.;
  int ixMin = TMath::Max(1, ix0 - 1);
  int ixMax = TMath::Min(histMLEM.GetNbinsX(), ix0 + 1);
  int iyMin = TMath::Max(1, iy0 - 1);
  int iyMax = TMath::Min(histMLEM.GetNbinsY(), iy0 + 1);

  // first only consider pixels above threshold
  const TAxis* xAxis = histMLEM.GetXaxis();
  const TAxis* yAxis = histMLEM.GetYaxis();
  double xq(0.), yq(0.), q(0.);
  bool onePixelWidthX(true), onePixelWidthY(true);
  for (int iy = iyMin; iy <= iyMax; ++iy) {
    for (int ix = ixMin; ix <= ixMax; ++ix) {
      double charge = histMLEM.GetBinContent(ix, iy);
      if (charge >= chargeThreshold) {
        xq += xAxis->GetBinCenter(ix) * charge;
        yq += yAxis->GetBinCenter(iy) * charge;
        q += charge;
        if (ix != ix0) {
          onePixelWidthX = false;
        }
        if (iy != iy0) {
          onePixelWidthY = false;
        }
      }
    }
  }

  // if all pixels used so far are aligned with iy0, add one more, with the highest charge, in y direction
  if (onePixelWidthY) {
    double xPixel(0.), yPixel(0.), chargePixel(0.);
    int ixPixel(ix0);
    for (int iy = iyMin; iy <= iyMax; ++iy) {
      if (iy != iy0) {
        for (int ix = ixMin; ix <= ixMax; ++ix) {
          double charge = histMLEM.GetBinContent(ix, iy);
          if (charge > chargePixel) {
            xPixel = xAxis->GetBinCenter(ix);
            yPixel = yAxis->GetBinCenter(iy);
            chargePixel = charge;
            ixPixel = ix;
          }
        }
      }
    }
    xq += xPixel * chargePixel;
    yq += yPixel * chargePixel;
    q += chargePixel;
    if (ixPixel != ix0) {
      onePixelWidthX = false;
    }
  }

  // if all pixels used so far are aligned with ix0, add one more, with the highest charge, in x direction
  if (onePixelWidthX) {
    double xPixel(0.), yPixel(0.), chargePixel(0.);
    for (int ix = ixMin; ix <= ixMax; ++ix) {
      if (ix != ix0) {
        for (int iy = iyMin; iy <= iyMax; ++iy) {
          double charge = histMLEM.GetBinContent(ix, iy);
          if (charge > chargePixel) {
            xPixel = xAxis->GetBinCenter(ix);
            yPixel = yAxis->GetBinCenter(iy);
            chargePixel = charge;
          }
        }
      }
    }
    xq += xPixel * chargePixel;
    yq += yPixel * chargePixel;
    q += chargePixel;
  }

  // compute the position of the centrer-of-gravity
  xy[0] = xq / q;
  xy[1] = yq / q;
}

//_________________________________________________________________________________________________
void ClusterFinderOriginal::refinePixelArray(const double xyCOG[2], size_t nPixMax, double& xMin, double& xMax, double& yMin, double& yMax)
{
  /// devide by 2 the size of the pixels in the direction where it is the highest,
  /// shift the pixels to align the array with the center of gravity around the
  /// maximum pixel charge and update the current pixel array and its limits.
  /// nPixMax is the maximum number of new pixels that can be produced.
  /// all pixels have the same charge (mLowestPadCharge) in the end

  xMin = std::numeric_limits<double>::max();
  xMax = -std::numeric_limits<double>::max();
  yMin = std::numeric_limits<double>::max();
  yMax = -std::numeric_limits<double>::max();

  // sort pixels according to the charge and move all pixels that must be kept at the beginning
  std::stable_sort(mPixels.begin(), mPixels.end(), [](const PadOriginal& pixel1, const PadOriginal& pixel2) {
    return (pixel1.status() == PadOriginal::kMustKeep && pixel2.status() != PadOriginal::kMustKeep) ||
           (pixel1.status() == pixel2.status() && pixel1.charge() > pixel2.charge());
  });
  double pixMinCharge = TMath::Min(0.01 * mPixels.front().charge(), 100. * mLowestPixelCharge);

  // define the half-size and shift of the new pixels depending on the direction of splitting
  double width[2] = {mPixels.front().dx(), mPixels.front().dy()};
  double shift[2] = {0., 0.};
  if (width[0] > width[1]) {
    width[0] /= 2.;
    shift[0] = -width[0];
  } else {
    width[1] /= 2.;
    shift[1] = -width[1];
  }

  // define additional shift to align pixels with the center of gravity (no more than new pixel size)
  double shiftToCOG[2] = {0., 0.};
  for (int ixy = 0; ixy < 2; ++ixy) {
    shiftToCOG[ixy] = mPixels.front().xy(ixy) + shift[ixy] - xyCOG[ixy];
    shiftToCOG[ixy] -= int(shiftToCOG[ixy] / width[ixy] / 2.) * width[ixy] * 2.;
  }

  int nPixels = mPixels.size();
  int nNewPixels(0);
  for (int i = 0; i < nPixels; ++i) {

    // do not exceed the maximum number of pixels expected and
    // skip pixels with charge too low unless they must be kept (this onward thanks to the ordering)
    auto& pixel = mPixels[i];
    if (nNewPixels == nPixMax || (pixel.charge() < pixMinCharge && pixel.status() != PadOriginal::kMustKeep)) {
      mPixels.erase(mPixels.begin() + i, mPixels.begin() + nPixels);
      break;
    }

    // shift half the pixel left(bottom) and toward COG and update limits
    pixel.setx(pixel.x() + shift[0] - shiftToCOG[0]);
    pixel.sety(pixel.y() + shift[1] - shiftToCOG[1]);
    pixel.setdx(width[0]);
    pixel.setdy(width[1]);
    pixel.setCharge(mLowestPadCharge);
    xMin = TMath::Min(xMin, pixel.x());
    yMin = TMath::Min(yMin, pixel.y());
    ++nNewPixels;

    // stop if the maximum number of pixels is reached: cannot add the second half-pixel
    if (nNewPixels == nPixMax) {
      xMax = TMath::Max(xMax, pixel.x());
      yMax = TMath::Max(yMax, pixel.y());
      continue;
    }

    // add the second half-pixel on the right(top) and update the limits
    mPixels.emplace_back(pixel);
    auto& pixel2 = mPixels.back();
    pixel2.setx(pixel2.x() - 2. * shift[0]);
    pixel2.sety(pixel2.y() - 2. * shift[1]);
    pixel2.setStatus(PadOriginal::kZero);
    xMax = TMath::Max(xMax, pixel2.x());
    yMax = TMath::Max(yMax, pixel2.y());
    ++nNewPixels;
  }

  // add pixels if the center of gravity is at the limit of the pixel array and update limits
  if (mPixels.size() < nPixMax && xyCOG[1] + width[1] > yMax) {
    yMax = xyCOG[1] + 2. * width[1];
    mPixels.emplace_back(mPixels.front());
    mPixels.back().setx(xyCOG[0]);
    mPixels.back().sety(yMax);
  }
  if (mPixels.size() < nPixMax && xyCOG[1] - width[1] < yMin) {
    yMin = xyCOG[1] - 2. * width[1];
    mPixels.emplace_back(mPixels.front());
    mPixels.back().setx(xyCOG[0]);
    mPixels.back().sety(yMin);
  }
  if (mPixels.size() < nPixMax && xyCOG[0] + width[0] > xMax) {
    xMax = xyCOG[0] + 2. * width[0];
    mPixels.emplace_back(mPixels.front());
    mPixels.back().setx(xMax);
    mPixels.back().sety(xyCOG[1]);
  }
  if (mPixels.size() < nPixMax && xyCOG[0] - width[0] < xMin) {
    xMin = xyCOG[0] - 2. * width[0];
    mPixels.emplace_back(mPixels.front());
    mPixels.back().setx(xMin);
    mPixels.back().sety(xyCOG[1]);
  }
}

//_________________________________________________________________________________________________
void ClusterFinderOriginal::cleanPixelArray(double threshold, std::vector<double>& prob)
{
  /// discard pixels with a charge below the given threshold by moving
  /// their charge to their nearest neighbour (to keep the total charge)
  /// update the visibility of discarded pixels to make them "invisible"

  for (int i = 0; i < mPixels.size(); ++i) {

    auto& pixel = mPixels[i];
    if (pixel.charge() >= threshold) {
      continue;
    }

    // make it "invisible"
    prob[i] = 0.;

    // no charge to move
    if (pixel.charge() <= 0.) {
      continue;
    }

    // find the nearest neighbour above the threshold
    int iNeighbour(-1);
    double distMin(std::numeric_limits<double>::max());
    for (int j = 0; j < mPixels.size(); ++j) {
      auto& pixel2 = mPixels[j];
      if (j != i && pixel2.charge() >= threshold) {
        double distX = (pixel.x() - pixel2.x()) / pixel2.dx();
        double distY = (pixel.y() - pixel2.y()) / pixel2.dy();
        double dist = distX * distX + distY * distY;
        if (dist < distMin) {
          distMin = dist;
          iNeighbour = j;
        }
      }
    }
    if (iNeighbour < 0) {
      LOG(info) << "There is no pixel above the threshold!?";
      continue;
    }

    // move the charge
    mPixels[iNeighbour].setCharge(mPixels[iNeighbour].charge() + pixel.charge());
    pixel.setCharge(0.);
  }
}

//_________________________________________________________________________________________________
int ClusterFinderOriginal::fit(const std::vector<const std::vector<int>*>& clustersOfPixels,
                               const double fitRange[2][2], double fitParam[SNFitParamMax + 1])
{
  /// fit the selected part of the precluster with up to SNFitClustersMax clusters
  /// the clusters' position seeds are determined using the clustersOfPixels (one per cluster seed)
  /// the total charge of the clusters is fixed to the total charge of all provided pixels
  /// fitRange determines the limits of the fitted clusters' positions
  /// the fitted parameters are returned in fitParam (2 for 1st cluster + 3 per other cluster)
  /// the function returns the actual number of fitted parameters (<= SNFitParamMax),
  /// which depends on the number of clusters effectively used in the fit (<= #cluster seeds)
  /// fitParam[SNFitParamMax] returns the total charge of the clusters (= charge of all pixels)

  // there must be at most SNFitClustersMax clusters fitted at a time
  if (clustersOfPixels.empty() || clustersOfPixels.size() > SNFitClustersMax) {
    throw std::runtime_error(std::string("Cannot fit ") + clustersOfPixels.size() + " clusters at a time");
  }

  // number of pads to use, number of virtual pads and average pad charge
  int nRealPadsToFit(0), nVirtualPadsToFit(0);
  double averagePadCharge(0.);
  for (const auto& pad : *mPreCluster) {
    if (pad.status() == PadOriginal::kUseForFit) {
      averagePadCharge += pad.charge();
      if (pad.isReal()) {
        ++nRealPadsToFit;
      } else {
        ++nVirtualPadsToFit;
      }
    }
  }
  // need at least 2 real pads to fit
  if (nRealPadsToFit < 2) {
    fitParam[SNFitParamMax] = 0.;
    return 0;
  }
  averagePadCharge /= nRealPadsToFit;

  // determine the clusters' position seeds ordered per decreasing charge and the overall mean position
  // as well as the total charge of all the pixels associated to the part of the precluster being fitted
  double xMean(0.), yMean(0.);
  fitParam[SNFitParamMax] = 0.;
  std::multimap<double, std::pair<double, double>, std::greater<>> xySeed{};
  for (const auto iPixels : clustersOfPixels) {
    double chargeMax(0.), xSeed(0.), ySeed(0.);
    for (auto iPixel : *iPixels) {
      const auto& pixel = mPixels[iPixel];
      double charge = pixel.charge();
      fitParam[SNFitParamMax] += charge;
      xMean += pixel.x() * charge;
      yMean += pixel.y() * charge;
      if (charge > chargeMax) {
        chargeMax = charge;
        xSeed = pixel.x();
        ySeed = pixel.y();
      }
    }
    xySeed.emplace(chargeMax, std::make_pair(xSeed, ySeed));
  }
  xMean /= fitParam[SNFitParamMax];
  yMean /= fitParam[SNFitParamMax];

  // reduce the number of clusters to fit if there are not enough pads in each direction
  auto nPadsXY = mPreCluster->sizeInPads(PadOriginal::kUseForFit);
  if (xySeed.size() > 1) {
    int max = TMath::Min(SNFitClustersMax, (nRealPadsToFit + 1) / 3);
    if (max > 1) {
      if ((nPadsXY.first < 3 && nPadsXY.second < 3) ||
          (nPadsXY.first == 3 && nPadsXY.second < 3) ||
          (nPadsXY.first < 3 && nPadsXY.second == 3)) {
        max = 1;
      }
    }
    if (xySeed.size() > max) {
      xySeed.erase(std::next(xySeed.begin(), max), xySeed.end());
    }
  }

  // prepare the initial fit parameters and limits (use clusters' position seeds if several clusters are used, mean position otherwise)
  // the last 2 parameters are fixed to the total pixel charge and the average pad charge for the part of the precluster being fitted
  double param[SNFitParamMax + 2] = {xMean, yMean, 0.6, xMean, yMean, 0.6, xMean, yMean, fitParam[SNFitParamMax], averagePadCharge};
  double parmin[SNFitParamMax] = {fitRange[0][0], fitRange[1][0], 1.e-9, fitRange[0][0], fitRange[1][0], 1.e-9, fitRange[0][0], fitRange[1][0]};
  double parmax[SNFitParamMax] = {fitRange[0][1], fitRange[1][1], 1., fitRange[0][1], fitRange[1][1], 1., fitRange[0][1], fitRange[1][1]};
  if (xySeed.size() > 1) {
    int iParam(0);
    for (const auto& seed : xySeed) {
      param[iParam++] = seed.second.first;
      param[iParam++] = seed.second.second;
      ++iParam;
    }
  }

  // try to fit with only 1 cluster, then 2 (if any), then 3 (if any) and stop if the fit gets worse
  // the fitted parameters are used as initial parameters of the corresponding cluster(s) for the next fit
  double chi2n0(std::numeric_limits<float>::max());
  int nTrials(0);
  int nParamUsed(0);
  for (int nFitClusters = 1; nFitClusters <= xySeed.size(); ++nFitClusters) {

    // number of parameters to use
    nParamUsed = 3 * nFitClusters - 1;

    // do the fit
    double chi2 = fit(param, parmin, parmax, nParamUsed, nTrials);

    // stop here if the normalized chi2 is not (significantly) smaller than in the previous fit
    int dof = TMath::Max(nRealPadsToFit + nVirtualPadsToFit - nParamUsed, 1);
    double chi2n = chi2 / dof;
    if (nParamUsed > 2 &&
        (chi2n > chi2n0 || (nFitClusters == xySeed.size() && chi2n * (1 + TMath::Min(1 - param[nParamUsed - 3], 0.25)) > chi2n0))) {
      nParamUsed -= 3;
      break;
    }
    chi2n0 = chi2n;

    // reset the clusters position to the center of the pad if there is only one in this direction
    if (nPadsXY.first == 1) {
      for (int i = 0; i < nParamUsed; ++i) {
        if (i == 0 || i == 3 || i == 6) {
          param[i] = xMean;
        }
      }
    }
    if (nPadsXY.second == 1) {
      for (int i = 0; i < nParamUsed; ++i) {
        if (i == 1 || i == 4 || i == 7) {
          param[i] = yMean;
        }
      }
    }

    // make sure the parameters are within limits and save them
    for (int i = 0; i < nParamUsed; ++i) {
      param[i] = TMath::Max(param[i], parmin[i]);
      param[i] = TMath::Min(param[i], parmax[i]);
      fitParam[i] = param[i];
    }

    // stop here if the current chi2 is too low
    if (chi2 < 0.1) {
      break;
    }
  }

  // store the reconstructed clusters if their charge is high enough
  double chargeFraction[SNFitClustersMax] = {0.};
  param2ChargeFraction(fitParam, nParamUsed, chargeFraction);
  for (int iParam = 0; iParam < nParamUsed; iParam += 3) {
    if (chargeFraction[iParam / 3] * fitParam[SNFitParamMax] >= mLowestClusterCharge) {
      mClusters.push_back({static_cast<float>(fitParam[iParam]), static_cast<float>(fitParam[iParam + 1]), 0., 0., 0., 0, 0, 0});
    }
  }

  return nParamUsed;
}

//_________________________________________________________________________________________________
double ClusterFinderOriginal::fit(double currentParam[SNFitParamMax + 2],
                                  const double parmin[SNFitParamMax], const double parmax[SNFitParamMax],
                                  int nParamUsed, int& nTrials) const
{
  /// perform the fit with a custom algorithm, using currentParam as starting parameters
  /// update currentParam with the fitted parameters and return the corresponding chi2

  // default step size in x, y and charge fraction
  static const double defaultShift[SNFitParamMax] = {0.01, 0.002, 0.02, 0.01, 0.002, 0.02, 0.01, 0.002};

  double shift[SNFitParamMax] = {0.};
  for (int i = 0; i < nParamUsed; ++i) {
    shift[i] = defaultShift[i];
  }

  // copy of current and best parameters and associated first derivatives and chi2
  double param[2][SNFitParamMax] = {{0.}, {0.}};
  double deriv[2][SNFitParamMax] = {{0.}, {0.}};
  double chi2[2] = {0., std::numeric_limits<float>::max()};
  int iBestParam(1);

  double shiftSave(0.);
  int nSimilarSteps[SNFitParamMax] = {0};
  int nLoop(0), nFail(0);

  while (true) {
    ++nLoop;

    // keep the best results from the previous step and save the new ones in the other slot
    int iCurrentParam = 1 - iBestParam;

    // get the chi2 of the fit with the current parameters
    chi2[iCurrentParam] = computeChi2(currentParam, nParamUsed);
    ++nTrials;

    // compute first and second chi2 derivatives w.r.t. each parameter
    double deriv2nd[SNFitParamMax] = {0.};
    for (int i = 0; i < nParamUsed; ++i) {
      param[iCurrentParam][i] = currentParam[i];
      currentParam[i] += defaultShift[i] / 10.;
      double chi2Shift = computeChi2(currentParam, nParamUsed);
      ++nTrials;
      deriv[iCurrentParam][i] = (chi2Shift - chi2[iCurrentParam]) / defaultShift[i] * 10;
      deriv2nd[i] = param[0][i] != param[1][i] ? (deriv[0][i] - deriv[1][i]) / (param[0][i] - param[1][i]) : 0;
      currentParam[i] -= defaultShift[i] / 10.;
    }

    // abort if we exceed the maximum number of trials (integrated over the fits with 1, 2 and 3 clusters)
    if (nTrials > 2000) {
      break;
    }

    // choose the best parameters between the current ones and the best ones from the previous step
    iBestParam = chi2[0] < chi2[1] ? 0 : 1;
    nFail = iBestParam == iCurrentParam ? 0 : nFail + 1;

    // stop here if we reached the maximum number of iterations
    if (nLoop > 150) {
      break;
    }

    double stepMax(0.), derivMax(0.);
    int iDerivMax(0);
    for (int i = 0; i < nParamUsed; ++i) {

      // estimate the shift to perform of this parameter to reach the minimum chi2
      double previousShift = shift[i];
      if (nLoop == 1) {
        // start with the default step size
        shift[i] = TMath::Sign(defaultShift[i], -deriv[iCurrentParam][i]);
      } else if (TMath::Abs(deriv[0][i]) < 1.e-3 && TMath::Abs(deriv[1][i]) < 1.e-3) {
        // stay there if the minimum is reached w.r.t. to this parameter
        shift[i] = 0;
      } else if ((deriv[iBestParam][i] * deriv[1 - iBestParam][i] > 0. &&
                  TMath::Abs(deriv[iBestParam][i]) > TMath::Abs(deriv[1 - iBestParam][i])) ||
                 TMath::Abs(deriv[0][i] - deriv[1][i]) < 1.e-3 ||
                 TMath::Abs(deriv2nd[i]) < 1.e-6) {
        // same size of shift if the first derivative increases or remain constant
        shift[i] = -TMath::Sign(shift[i], (chi2[0] - chi2[1]) * (param[0][i] - param[1][i]));
        // move faster if we already did >= 2 steps like this and the chi2 improved
        if (iBestParam == iCurrentParam) {
          if (nSimilarSteps[i] > 1) {
            shift[i] *= 2.;
          }
          ++nSimilarSteps[i];
        }
      } else {
        // adjust the shift otherwise
        shift[i] = deriv2nd[i] != 0. ? -deriv[iBestParam][i] / deriv2nd[i] : 0.;
        nSimilarSteps[i] = 0;
      }

      // maximum shift normalized to the default step size and maximum first derivative
      stepMax = TMath::Max(stepMax, TMath::Abs(shift[i]) / defaultShift[i]);
      if (TMath::Abs(deriv[iBestParam][i]) > derivMax) {
        iDerivMax = i;
        derivMax = TMath::Abs(deriv[iBestParam][i]);
      }

      // limit the shift to 10 times the default step size
      if (TMath::Abs(shift[i]) / defaultShift[i] > 10.) {
        shift[i] = TMath::Sign(10., shift[i]) * defaultShift[i];
      }

      // reset the current parameter and adjust the shift if the chi2 did not improve
      if (iBestParam != iCurrentParam) {
        nSimilarSteps[i] = 0;
        currentParam[i] = param[iBestParam][i];
        if (TMath::Abs(shift[i] + previousShift) > 0.1 * defaultShift[i]) {
          shift[i] = (shift[i] + previousShift) / 2.;
        } else {
          shift[i] /= -2.;
        }
      }

      // reduce the shift if the step is too big
      if (TMath::Abs(shift[i] * deriv[iBestParam][i]) > chi2[iBestParam]) {
        shift[i] = TMath::Sign(chi2[iBestParam] / deriv[iBestParam][i], shift[i]);
      }

      // introduce step relaxation factor
      if (nSimilarSteps[i] < 3) {
        double scMax = 1. + 4. / TMath::Max(nLoop / 2., 1.);
        if (TMath::Abs(previousShift) > 0. && TMath::Abs(shift[i] / previousShift) > scMax) {
          shift[i] = TMath::Sign(previousShift * scMax, shift[i]);
        }
      }

      // shift the current parameter and make sure we do not overstep its limits
      currentParam[i] += shift[i];
      if (currentParam[i] < parmin[i]) {
        shift[i] = parmin[i] - currentParam[i];
        currentParam[i] = parmin[i];
      } else if (currentParam[i] > parmax[i]) {
        shift[i] = parmax[i] - currentParam[i];
        currentParam[i] = parmax[i];
      }
    }

    // stop here if the minimum was found
    if (stepMax < 1. && derivMax < 2.) {
      break;
    }

    // check for small step
    if (shift[iDerivMax] == 0.) {
      shift[iDerivMax] = defaultShift[iDerivMax] / 10.;
      currentParam[iDerivMax] += shift[iDerivMax];
      continue;
    }

    // further adjustment...
    if (nSimilarSteps[iDerivMax] == 0 && derivMax > 0.5 && nLoop > 9) {
      if (deriv2nd[iDerivMax] != 0. && TMath::Abs(deriv[iBestParam][iDerivMax] / deriv2nd[iDerivMax] / shift[iDerivMax]) > 10.) {
        if (iBestParam == iCurrentParam) {
          deriv2nd[iDerivMax] = -deriv2nd[iDerivMax];
        }
        shift[iDerivMax] = -deriv[iBestParam][iDerivMax] / deriv2nd[iDerivMax] / 10.;
        currentParam[iDerivMax] += shift[iDerivMax];
        if (iBestParam == iCurrentParam) {
          shiftSave = shift[iDerivMax];
        }
      }
      if (nFail > 10) {
        currentParam[iDerivMax] -= shift[iDerivMax];
        shift[iDerivMax] = 4. * shiftSave * (gRandom->Rndm(0) - 0.5);
        currentParam[iDerivMax] += shift[iDerivMax];
      }
    }
  }

  // return the best parameters and associated chi2
  for (int i = 0; i < nParamUsed; ++i) {
    currentParam[i] = param[iBestParam][i];
  }
  return chi2[iBestParam];
}

//_________________________________________________________________________________________________
double ClusterFinderOriginal::computeChi2(const double param[SNFitParamMax + 2], int nParamUsed) const
{
  /// return the chi2 to be minimized when fitting the selected part of the precluster
  /// param[0... SNFitParamMax-1] are the cluster parameters
  /// param[SNFitParamMax] is the total pixel charge associated to this part of the precluster
  /// param[SNFitParamMax+1] is the average pad charge
  /// nParamUsed is the number of cluster parameters effectively used (= #cluster * 3 - 1)

  // get the fraction of charge carried by each cluster
  double chargeFraction[SNFitClustersMax] = {0.};
  param2ChargeFraction(param, nParamUsed, chargeFraction);

  double chi2(0.);
  for (const auto& pad : *mPreCluster) {

    // skip pads not to be used for this fit
    if (pad.status() != PadOriginal::kUseForFit) {
      continue;
    }

    // compute the expected pad charge with these cluster parameters
    double padChargeFit(0.);
    for (int iParam = 0; iParam < nParamUsed; iParam += 3) {
      padChargeFit += chargeIntegration(param[iParam], param[iParam + 1], pad) * chargeFraction[iParam / 3];
    }
    padChargeFit *= param[SNFitParamMax];

    // compute the chi2
    double delta = padChargeFit - pad.charge();
    chi2 += delta * delta / pad.charge();
  }

  return chi2 / param[SNFitParamMax + 1];
}

//_________________________________________________________________________________________________
void ClusterFinderOriginal::param2ChargeFraction(const double param[SNFitParamMax], int nParamUsed,
                                                 double fraction[SNFitClustersMax]) const
{
  /// extract the fraction of charge carried by each cluster from the fit parameters
  if (nParamUsed == 2) {
    fraction[0] = 1.;
  } else if (nParamUsed == 5) {
    fraction[0] = param[2];
    fraction[1] = TMath::Max(1. - fraction[0], 0.);
  } else {
    fraction[0] = param[2];
    fraction[1] = TMath::Max((1. - fraction[0]) * param[5], 0.);
    fraction[2] = TMath::Max(1. - fraction[0] - fraction[1], 0.);
  }
}

//_________________________________________________________________________________________________
float ClusterFinderOriginal::chargeIntegration(double x, double y, const PadOriginal& pad) const
{
  /// integrate the Mathieson over the pad area, assuming the center of the Mathieson is at (x,y)
  double xPad = pad.x() - x;
  double yPad = pad.y() - y;
  return mMathieson->integrate(xPad - pad.dx(), yPad - pad.dy(), xPad + pad.dx(), yPad + pad.dy());
}

//_________________________________________________________________________________________________
void ClusterFinderOriginal::split(const TH2D& histMLEM, const std::vector<double>& coef)
{
  /// group the pixels in clusters then group together the clusters coupled to the same pads,
  /// split them into sub-groups if they are too many, merge them if they are not coupled to enough pads
  /// and finally fit the associated pads using the (sub-)group of pixels as seed for clusters' positions

  // save the pad charges as they can be modified during the splitting
  std::vector<double> padCharges(0);
  padCharges.reserve(mPreCluster->multiplicity());
  for (const auto& pad : *mPreCluster) {
    padCharges.push_back(pad.charge());
  }

  // find clusters of pixels
  int nBinsX = histMLEM.GetNbinsX();
  int nBinsY = histMLEM.GetNbinsY();
  std::vector<std::vector<int>> clustersOfPixels{};
  std::vector<std::vector<bool>> isUsed(nBinsX, std::vector<bool>(nBinsY, false));
  for (int j = 1; j <= nBinsY; ++j) {
    for (int i = 1; i <= nBinsX; ++i) {
      if (!isUsed[i - 1][j - 1] && histMLEM.GetBinContent(i, j) >= mLowestPixelCharge) {
        // add a new cluster of pixels and the associated pixels recursively
        clustersOfPixels.emplace_back();
        addPixel(histMLEM, i, j, clustersOfPixels.back(), isUsed);
      }
    }
  }
  if (clustersOfPixels.size() > 200) {
    throw std::runtime_error("Too many clusters of pixels!");
  }

  // compute the coupling between clusters of pixels and pads (including overflows)
  std::vector<std::vector<double>> couplingClPad(clustersOfPixels.size(), std::vector<double>(mPreCluster->multiplicity(), 0.));
  for (int iPad = 0; iPad < mPreCluster->multiplicity(); ++iPad) {
    int idx0 = iPad * mPixels.size();
    for (int iCluster = 0; iCluster < clustersOfPixels.size(); ++iCluster) {
      for (auto ipixel : clustersOfPixels[iCluster]) {
        if (coef[idx0 + ipixel] >= SLowestCoupling) {
          couplingClPad[iCluster][iPad] += coef[idx0 + ipixel];
        }
      }
    }
  }

  // compute the coupling between clusters of pixels (excluding coupling via pads in overflow)
  std::vector<std::vector<double>> couplingClCl(clustersOfPixels.size(), std::vector<double>(clustersOfPixels.size(), 0.));
  for (int iPad = 0; iPad < mPreCluster->multiplicity(); ++iPad) {
    if (!mPreCluster->pad(iPad).isSaturated()) {
      for (int iCluster1 = 0; iCluster1 < clustersOfPixels.size(); ++iCluster1) {
        if (couplingClPad[iCluster1][iPad] >= SLowestCoupling) {
          for (int iCluster2 = iCluster1 + 1; iCluster2 < clustersOfPixels.size(); ++iCluster2) {
            if (couplingClPad[iCluster2][iPad] >= SLowestCoupling) {
              couplingClCl[iCluster1][iCluster2] += TMath::Sqrt(couplingClPad[iCluster1][iPad] * couplingClPad[iCluster2][iPad]);
            }
          }
        }
      }
    }
  }
  for (int iCluster1 = 0; iCluster1 < clustersOfPixels.size(); ++iCluster1) {
    for (int iCluster2 = iCluster1 + 1; iCluster2 < clustersOfPixels.size(); ++iCluster2) {
      couplingClCl[iCluster2][iCluster1] = couplingClCl[iCluster1][iCluster2];
    }
  }

  // define the fit range
  const TAxis* xAxis = histMLEM.GetXaxis();
  const TAxis* yAxis = histMLEM.GetYaxis();
  double fitRange[2][2] = {{xAxis->GetXmin() - xAxis->GetBinWidth(1), xAxis->GetXmax() + xAxis->GetBinWidth(1)},
                           {yAxis->GetXmin() - yAxis->GetBinWidth(1), yAxis->GetXmax() + yAxis->GetBinWidth(1)}};

  std::vector<bool> isClUsed(clustersOfPixels.size(), false);
  std::vector<int> coupledClusters{};
  std::vector<int> clustersForFit{};
  std::vector<const std::vector<int>*> clustersOfPixelsForFit{};
  for (int iCluster = 0; iCluster < clustersOfPixels.size(); ++iCluster) {

    // skip clusters of pixels already used
    if (isClUsed[iCluster]) {
      continue;
    }

    // fill the list of coupled clusters of pixels recursively starting from this one
    coupledClusters.clear();
    addCluster(iCluster, coupledClusters, isClUsed, couplingClCl);

    while (coupledClusters.size() > 0) {

      // select the clusters of pixels for the fit: all of them if <= SNFitClustersMax
      // or the group of maximum 3 the least coupled with the others
      clustersForFit.clear();
      if (coupledClusters.size() <= SNFitClustersMax) {
        clustersForFit.swap(coupledClusters);
      } else {
        extractLeastCoupledClusters(coupledClusters, clustersForFit, couplingClCl);
      }

      // select the associated pads for the fit
      int nSelectedPads = selectPads(coupledClusters, clustersForFit, couplingClPad);

      // abort if there are not enough pads selected, deselect pads and
      // merge the clusters of pixels selected for the fit into the others, if any
      if (nSelectedPads < 3 && coupledClusters.size() + clustersForFit.size() > 1) {
        for (auto& pad : *mPreCluster) {
          if (pad.status() == PadOriginal::kUseForFit || pad.status() == PadOriginal::kCoupled) {
            pad.setStatus(PadOriginal::kZero);
          }
        }
        if (coupledClusters.size() > 0) {
          merge(clustersForFit, coupledClusters, clustersOfPixels, couplingClCl, couplingClPad);
        }
        continue;
      }

      // do the fit
      clustersOfPixelsForFit.clear();
      for (auto iCluster : clustersForFit) {
        clustersOfPixelsForFit.push_back(&clustersOfPixels[iCluster]);
      }
      double fitParam[SNFitParamMax + 1] = {0.};
      int nParamUsed = fit(clustersOfPixelsForFit, fitRange, fitParam);

      // update the status (and possibly the charge) of selected pads
      updatePads(fitParam, nParamUsed);
    }
  }

  // restore the pad charges in case they were modified during the splitting
  int iPad(0);
  for (auto& pad : *mPreCluster) {
    pad.setCharge(padCharges[iPad++]);
  }
}

//_________________________________________________________________________________________________
void ClusterFinderOriginal::addPixel(const TH2D& histMLEM, int i0, int j0, std::vector<int>& pixels, std::vector<std::vector<bool>>& isUsed)
{
  /// add a pixel to the cluster of pixels then add recursively its neighbours,
  /// if their charge is higher than mLowestPixelCharge and excluding corners

  auto itPixel = findPad(mPixels, histMLEM.GetXaxis()->GetBinCenter(i0), histMLEM.GetYaxis()->GetBinCenter(j0), mLowestPixelCharge);
  pixels.push_back(std::distance(mPixels.begin(), itPixel));
  isUsed[i0 - 1][j0 - 1] = true;

  int iMin = TMath::Max(1, i0 - 1);
  int iMax = TMath::Min(histMLEM.GetNbinsX(), i0 + 1);
  int jMin = TMath::Max(1, j0 - 1);
  int jMax = TMath::Min(histMLEM.GetNbinsY(), j0 + 1);
  for (int j = jMin; j <= jMax; ++j) {
    for (int i = iMin; i <= iMax; ++i) {
      if (!isUsed[i - 1][j - 1] && (i == i0 || j == j0) && histMLEM.GetBinContent(i, j) >= mLowestPixelCharge) {
        addPixel(histMLEM, i, j, pixels, isUsed);
      }
    }
  }
}

//_________________________________________________________________________________________________
void ClusterFinderOriginal::addCluster(int iCluster, std::vector<int>& coupledClusters, std::vector<bool>& isClUsed,
                                       const std::vector<std::vector<double>>& couplingClCl) const
{
  /// add a cluster of pixels to the list of coupled clusters then
  /// add recursively all clusters coupled to this one if not yet used

  coupledClusters.push_back(iCluster);
  isClUsed[iCluster] = true;

  for (int iCluster2 = 0; iCluster2 < couplingClCl.size(); ++iCluster2) {
    if (!isClUsed[iCluster2] && couplingClCl[iCluster][iCluster2] >= SLowestCoupling) {
      addCluster(iCluster2, coupledClusters, isClUsed, couplingClCl);
    }
  }
}

//_________________________________________________________________________________________________
void ClusterFinderOriginal::extractLeastCoupledClusters(std::vector<int>& coupledClusters, std::vector<int>& clustersForFit,
                                                        const std::vector<std::vector<double>>& couplingClCl) const
{
  /// find the group of 1, 2 or min(3, #coupled/2) clusters of pixels the least coupled with the others
  /// and move them to the list of clusters of pixels to be used as seed for the fit

  double minCoupling(DBL_MAX);
  int leastCoupledClusters[3] = {-1, -1, -1};

  // compute the coupling of each cluster with all the others and find the least coupled
  std::vector<double> coupling1(coupledClusters.size(), 0.);
  for (int i = 0; i < coupledClusters.size(); ++i) {
    for (int j = i + 1; j < coupledClusters.size(); ++j) {
      coupling1[i] += couplingClCl[coupledClusters[i]][coupledClusters[j]];
      coupling1[j] += couplingClCl[coupledClusters[i]][coupledClusters[j]];
    }
    if (coupling1[i] < minCoupling) {
      leastCoupledClusters[0] = i;
      minCoupling = coupling1[i];
    }
  }

  if (SNFitClustersMax > 1) {

    bool tryTierce = SNFitClustersMax > 2 && coupledClusters.size() > 5;

    for (int i = 0; i < coupledClusters.size(); ++i) {
      for (int j = i + 1; j < coupledClusters.size(); ++j) {

        // look for a lower coupling with the others by grouping clusters by pair
        double coupling2 = coupling1[i] + coupling1[j] - 2. * couplingClCl[coupledClusters[i]][coupledClusters[j]];
        if (coupling2 < minCoupling) {
          leastCoupledClusters[0] = i;
          leastCoupledClusters[1] = j;
          leastCoupledClusters[2] = -1;
          minCoupling = coupling2;
        }

        // look for a lower coupling with the others by grouping clusters by tierce
        if (tryTierce) {
          for (int k = j + 1; k < coupledClusters.size(); ++k) {
            double coupling3 = coupling2 + coupling1[k] -
                               2. * (couplingClCl[coupledClusters[i]][coupledClusters[k]] +
                                     couplingClCl[coupledClusters[j]][coupledClusters[k]]);
            if (coupling3 < minCoupling) {
              leastCoupledClusters[0] = i;
              leastCoupledClusters[1] = j;
              leastCoupledClusters[2] = k;
              minCoupling = coupling3;
            }
          }
        }
      }
    }
  }

  // transfert the least coupled group of clusters to the list to be used for the fit
  // take into account the shift of indices each time a cluster is remove
  int idxShift(0);
  for (int i = 0; i < 3 && leastCoupledClusters[i] >= 0; ++i) {
    clustersForFit.push_back(coupledClusters[leastCoupledClusters[i] - idxShift]);
    coupledClusters.erase(coupledClusters.begin() + leastCoupledClusters[i] - idxShift);
    ++idxShift;
  }
}

//_________________________________________________________________________________________________
int ClusterFinderOriginal::selectPads(const std::vector<int>& coupledClusters, const std::vector<int>& clustersForFit,
                                      const std::vector<std::vector<double>>& couplingClPad)
{
  /// select pads only coupled with the clusters of pixels to be used for the fit

  int nSelectedPads(0);

  for (int iPad = 0; iPad < mPreCluster->multiplicity(); ++iPad) {

    // exclude pads already used or saturated
    auto& pad = mPreCluster->pad(iPad);
    if (pad.status() != PadOriginal::kZero || pad.isSaturated()) {
      continue;
    }

    for (int iCluster1 : clustersForFit) {

      // select pads coupled with a cluster of pixels used in the fit
      if (couplingClPad[iCluster1][iPad] >= SLowestCoupling) {
        pad.setStatus(PadOriginal::kUseForFit);
        ++nSelectedPads;

        // excluding those also coupled with another cluster of pixels not used in the fit
        for (int iCluster2 : coupledClusters) {
          if (couplingClPad[iCluster2][iPad] >= SLowestCoupling) {
            pad.setStatus(PadOriginal::kCoupled);
            --nSelectedPads;
            break;
          }
        }

        break;
      }
    }
  }

  return nSelectedPads;
}

//_________________________________________________________________________________________________
void ClusterFinderOriginal::merge(const std::vector<int>& clustersForFit, const std::vector<int>& coupledClusters,
                                  std::vector<std::vector<int>>& clustersOfPixels,
                                  std::vector<std::vector<double>>& couplingClCl,
                                  std::vector<std::vector<double>>& couplingClPad) const
{
  /// merge each cluster of pixels selected for the fit into the most coupled one among the others

  for (int iCluster1 : clustersForFit) {

    // find the cluster among the others the most coupled with this one
    double maxCoupling(-1.);
    int iMostCoupled(0);
    for (int iCluster2 : coupledClusters) {
      if (couplingClCl[iCluster1][iCluster2] > maxCoupling) {
        maxCoupling = couplingClCl[iCluster1][iCluster2];
        iMostCoupled = iCluster2;
      }
    }

    // copy the pixels of this cluster into the most coupled one
    clustersOfPixels[iMostCoupled].insert(clustersOfPixels[iMostCoupled].end(),
                                          clustersOfPixels[iCluster1].begin(), clustersOfPixels[iCluster1].end());

    // update the coupling with the other clusters
    for (int iCluster2 : coupledClusters) {
      if (iCluster2 != iMostCoupled) {
        couplingClCl[iMostCoupled][iCluster2] += couplingClCl[iCluster1][iCluster2];
        couplingClCl[iCluster2][iMostCoupled] = couplingClCl[iMostCoupled][iCluster2];
      }
    }

    // update the coupling between clusters and pads
    for (int iPad = 0; iPad < mPreCluster->multiplicity(); ++iPad) {
      if (mPreCluster->pad(iPad).status() == PadOriginal::kZero) {
        couplingClPad[iMostCoupled][iPad] += couplingClPad[iCluster1][iPad];
      }
    }
  }
}

//_________________________________________________________________________________________________
void ClusterFinderOriginal::updatePads(const double fitParam[SNFitParamMax + 1], int nParamUsed)
{
  /// discard the pads used in the fit and update the charge and status of the coupled pads

  // get the fraction of charge carried by each fitted cluster
  double chargeFraction[SNFitClustersMax] = {0.};
  if (nParamUsed > 0) {
    param2ChargeFraction(fitParam, nParamUsed, chargeFraction);
  }

  for (auto& pad : *mPreCluster) {

    if (pad.status() == PadOriginal::kUseForFit) {

      // discard the pads already used in a fit
      pad.setStatus(PadOriginal::kOver);

    } else if (pad.status() == PadOriginal::kCoupled) {

      // subtract the charge from the fitted clusters
      if (nParamUsed > 0) {
        double padChargeFit(0.);
        for (int iParam = 0; iParam < nParamUsed; iParam += 3) {
          padChargeFit += chargeIntegration(fitParam[iParam], fitParam[iParam + 1], pad) * chargeFraction[iParam / 3];
        }
        padChargeFit *= fitParam[SNFitParamMax];
        pad.setCharge(pad.charge() - padChargeFit);
      }

      // reset the pad status to further use it if its charge is high enough
      pad.setStatus((pad.charge() > mLowestPadCharge) ? PadOriginal::kZero : PadOriginal::kOver);
    }
  }
}

//_________________________________________________________________________________________________
void ClusterFinderOriginal::setClusterResolution(Cluster& cluster) const
{
  /// set the cluster resolution in both directions depending on whether its position
  /// lies on top of a fired digit in both planes or not (e.g. mono-cathode)

  if (cluster.getChamberId() < 4) {

    // do not consider mono-cathode clusters in stations 1 and 2
    cluster.ex = ClusterizerParam::Instance().defaultClusterResolution;
    cluster.ey = ClusterizerParam::Instance().defaultClusterResolution;

  } else {

    // find pads below the cluster
    int padIDNB(-1), padIDB(-1);
    bool padsFound = mSegmentation->findPadPairByPosition(cluster.x, cluster.y, padIDB, padIDNB);

    // look for these pads (if any) in the list of digits associated to this cluster
    auto itPadNB = mUsedDigits.end();
    if (padsFound || mSegmentation->isValid(padIDNB)) {
      itPadNB = std::find_if(mUsedDigits.begin() + cluster.firstDigit, mUsedDigits.end(),
                             [padIDNB](const Digit& digit) { return digit.getPadID() == padIDNB; });
    }
    auto itPadB = mUsedDigits.end();
    if (padsFound || mSegmentation->isValid(padIDB)) {
      itPadB = std::find_if(mUsedDigits.begin() + cluster.firstDigit, mUsedDigits.end(),
                            [padIDB](const Digit& digit) { return digit.getPadID() == padIDB; });
    }

    // set the cluster resolution accordingly
    cluster.ex = (itPadNB == mUsedDigits.end()) ? ClusterizerParam::Instance().badClusterResolution
                                                : ClusterizerParam::Instance().defaultClusterResolution;
    cluster.ey = (itPadB == mUsedDigits.end()) ? ClusterizerParam::Instance().badClusterResolution
                                               : ClusterizerParam::Instance().defaultClusterResolution;
  }
}

} // namespace mch
} // namespace o2
