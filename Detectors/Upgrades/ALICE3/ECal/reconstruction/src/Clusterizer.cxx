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

/// \file Clusterizer.cxx
/// \brief Class for cluster finding and unfolding
///
/// \author Evgeny Kryshen <evgeny.kryshen@cern.ch>

#include <TMath.h>
#include <TF1.h>
#include <fairlogger/Logger.h>
#include <ECalReconstruction/Clusterizer.h>
#include <ECalBase/Geometry.h>
#include <DataFormatsECal/Digit.h>
#include <DataFormatsECal/Cluster.h>

using namespace o2::ecal;

//==============================================================================
Clusterizer::Clusterizer(bool applyCorrectionZ, bool applyCorrectionE)
{
  auto& geo = Geometry::instance();
  mDigitIndices.resize(geo.getNrows(), std::vector<int>(geo.getNcols(), -1));
  mApplyCorrectionZ = applyCorrectionZ;
  mApplyCorrectionE = applyCorrectionE;

  mCrystalEnergyCorrectionPars.reserve(6);
  mCrystalEnergyCorrectionPars[0] = 0.00444;
  mCrystalEnergyCorrectionPars[1] = -1.322;
  mCrystalEnergyCorrectionPars[2] = 1.021;
  mCrystalEnergyCorrectionPars[3] = 0.0018;
  mCrystalEnergyCorrectionPars[4] = 0.;
  mCrystalEnergyCorrectionPars[5] = 0.;

  mSamplingEnergyCorrectionPars.reserve(6);
  mSamplingEnergyCorrectionPars[0] = 0.0033;
  mSamplingEnergyCorrectionPars[1] = -2.09;
  mSamplingEnergyCorrectionPars[2] = 1.007;
  mSamplingEnergyCorrectionPars[3] = 0.0667;
  mSamplingEnergyCorrectionPars[4] = -0.108;
  mSamplingEnergyCorrectionPars[5] = 0.0566;

  mCrystalZCorrectionPars.reserve(9);
  mCrystalZCorrectionPars[0] = -0.005187;
  mCrystalZCorrectionPars[1] = 0.7301;
  mCrystalZCorrectionPars[2] = -0.7382;
  mCrystalZCorrectionPars[3] = 0.;
  mCrystalZCorrectionPars[4] = 0.;
  mCrystalZCorrectionPars[5] = 0.;
  mCrystalZCorrectionPars[6] = 0.;
  mCrystalZCorrectionPars[7] = 0.;
  mCrystalZCorrectionPars[8] = 0.;

  mSamplingZCorrectionPars.reserve(9);
  mSamplingZCorrectionPars[0] = -2.137;
  mSamplingZCorrectionPars[1] = 6.400;
  mSamplingZCorrectionPars[2] = -3.342;
  mSamplingZCorrectionPars[3] = -0.1364;
  mSamplingZCorrectionPars[4] = 0.4019;
  mSamplingZCorrectionPars[5] = -0.1969;
  mSamplingZCorrectionPars[6] = 0.008223;
  mSamplingZCorrectionPars[7] = -0.02425;
  mSamplingZCorrectionPars[8] = 0.01190;

  fCrystalShowerShape = new TF1("fCrystal", "x<[1] ? [0]*exp([3]*x+[4]*x*x+[5]*x*x*x) : (x<[2] ? [0]*[6]*exp([7]*x+[8]*x*x) : [0]*[9]*exp([10]*x+[11]*x*x))", 0, 15);
  double pc[12];
  pc[0] = 1. / 13.15;
  pc[1] = 2.2;
  pc[2] = 5;
  pc[3] = 4.38969;
  pc[4] = -5.15975;
  pc[5] = 1.18978;
  pc[6] = 1.48726;
  pc[7] = -1.54621;
  pc[8] = 0.0814617;
  pc[9] = 0.0369055;
  pc[10] = -0.174372;
  pc[11] = -0.0455978;

  fCrystalShowerShape->SetParameters(pc);

  fSamplingShowerShape = new TF1("fSampling", "x<[1] ? [0]*exp([3]*x+[4]*x*x+[5]*x*x*x) : (x<[2] ? [0]*[6]*exp([7]*x+[8]*x*x) : [0]*[9]*exp([10]*x+[11]*x*x))", 0, 15);
  double ps[12];
  ps[0] = 1 / 35.6;
  ps[1] = 3.2;
  ps[2] = 6;
  ps[3] = 3.06543;
  ps[4] = -2.23235;
  ps[5] = 0.325344;
  ps[6] = 6.0733;
  ps[7] = -1.62713;
  ps[8] = 0.0965569;
  ps[9] = 0.0765706;
  ps[10] = -0.217398;
  ps[11] = -0.0204646;
  fSamplingShowerShape->SetParameters(ps);

  fCrystalRMS = new TF1("fCrystalRMS", "[0]*x*exp([1]*x+[2]*x*x+[3]*x*x*x)", 0, 2.2);
  double p[4];
  p[0] = 1.39814;
  p[1] = -6.05426;
  p[2] = 6.26678;
  p[3] = -1.97092;
  fCrystalRMS->SetParameters(p);
}

//==============================================================================
void Clusterizer::findClusters(const gsl::span<const Digit>& digits, std::vector<Cluster>& foundClusters, std::vector<Cluster>& unfoldedClusters)
{
  foundClusters.clear();
  unfoldedClusters.clear();

  // Collect list of clusters
  makeClusters(digits, foundClusters);

  // Split clusters with several local maxima if necessary
  makeUnfoldings(foundClusters, unfoldedClusters);

  // Evaluate cluster position, dispersion etc.
  evalClusters(foundClusters);
  evalClusters(unfoldedClusters);
}

//==============================================================================
void Clusterizer::addDigitToCluster(Cluster& cluster, int row, int col, const gsl::span<const Digit>& digits)
{
  auto& geo = Geometry::instance();
  if (row < 0 || row >= geo.getNrows() || col < 0 || col >= geo.getNcols()) {
    return;
  }
  int digitIndex = mDigitIndices[row][col];
  LOGP(debug, "    checking row={} and col={} digitIndex={}", row, col, digitIndex);
  if (digitIndex < 0) {
    return;
  }
  const Digit& digit = digits[digitIndex];
  if (cluster.getMultiplicity() > 0) {
    // check if new digit is in the same chamber and sector
    const Digit& digit2 = digits[cluster.getDigitIndex(0)];
    auto [sector1, ch1] = geo.getSectorChamber(digit.getTower());
    auto [sector2, ch2] = geo.getSectorChamber(digit2.getTower());
    LOGP(debug, "    checking if sector and chamber are the same: ({},{}) ({},{})", sector1, ch1, sector2, ch2);
    if (sector1 != sector2 || ch1 != ch2) {
      return;
    }
  }

  mDigitIndices[row][col] = -1;
  cluster.addDigit(digitIndex, digit.getTower(), digit.getEnergy());
  LOGP(debug, "    adding new digit at row={} and col={}", row, col);
  addDigitToCluster(cluster, row - 1, col, digits);
  addDigitToCluster(cluster, row + 1, col, digits);
  addDigitToCluster(cluster, row, col - 1, digits);
  addDigitToCluster(cluster, row, col + 1, digits);
}

//==============================================================================
void Clusterizer::makeClusters(const gsl::span<const Digit>& digits, std::vector<Cluster>& clusters)
{
  // Combine digits into cluster

  int nDigits = digits.size();

  // reset mDigitIndices
  for (auto& rows : mDigitIndices) {
    rows.assign(rows.size(), -1);
  }

  // fill mDigitIndices
  auto& geo = Geometry::instance();
  for (int i = 0; i < nDigits; i++) {
    const Digit& digit = digits[i];
    auto [row, col] = geo.globalRowColFromIndex(digit.getTower());
    bool isCrystal = geo.isCrystal(digit.getTower());
    if (isCrystal) {
      if (digit.getEnergy() < mCrystalDigitThreshold) {
        continue;
      }
    } else {
      if (digit.getEnergy() < mSamplingDigitThreshold) {
        continue;
      }
    }
    mDigitIndices[row][col] = i;
  }

  // add digit seeds to clusters and recursively add neighbours
  for (int i = 0; i < nDigits; i++) {
    const Digit& digitSeed = digits[i];
    auto [row, col] = geo.globalRowColFromIndex(digitSeed.getTower());
    if (mDigitIndices[row][col] < 0) {
      continue; // digit was already added in one of the clusters
    }
    if (digitSeed.getEnergy() < mClusteringThreshold) {
      continue;
    }
    LOGP(debug, "  starting new cluster at row={} and col={}", row, col);
    auto& cluster = clusters.emplace_back();
    addDigitToCluster(cluster, row, col, digits);
  }

  LOGP(debug, "made {} clusters from {} digits", clusters.size(), nDigits);
}

//==============================================================================
void Clusterizer::makeUnfoldings(std::vector<Cluster>& foundClusters, std::vector<Cluster>& unfoldedClusters)
{
  // Split cluster if several local maxima are found
  if (!mUnfoldClusters) {
    return;
  }

  int* maxAt = new int[mNLMMax];
  float* maxAtEnergy = new float[mNLMMax];

  for (auto& clu : foundClusters) {
    int nMax = getNumberOfLocalMax(clu, maxAt, maxAtEnergy);
    if (nMax > 1) {
      unfoldOneCluster(&clu, nMax, maxAt, maxAtEnergy, unfoldedClusters);
    } else {
      clu.setNLM(1);
      unfoldedClusters.emplace_back(clu);
    }
  }
  delete[] maxAt;
  delete[] maxAtEnergy;
}

//==============================================================================
void Clusterizer::unfoldOneCluster(Cluster* iniClu, int nMax, int* digitId, float* maxAtEnergy, std::vector<Cluster>& unfoldedClusters)
{
  // Based on MpdEmcClusterizerKI::UnfoldOneCluster by D. Peresunko
  // Performs the unfolding of a cluster with nMax overlapping showers
  // Parameters: iniClu cluster to be unfolded
  //             nMax number of local maxima found (this is the number of new clusters)
  //             digitId: index of digits, corresponding to local maxima
  //             maxAtEnergy: energies of digits, corresponding to local maxima

  // Take initial cluster and calculate local coordinates of digits
  // To avoid multiple re-calculation of same parameters
  int mult = iniClu->getMultiplicity();
  std::vector<double> x(mult);
  std::vector<double> y(mult);
  std::vector<double> z(mult);
  std::vector<double> e(mult);
  std::vector<std::vector<double>> eInClusters(mult, std::vector<double>(nMax));

  auto& geo = Geometry::instance();
  bool isCrystal = geo.isCrystal(iniClu->getDigitTowerId(0));

  for (int idig = 0; idig < mult; idig++) {
    e[idig] = iniClu->getDigitEnergy(idig);
    geo.detIdToGlobalPosition(iniClu->getDigitTowerId(idig), x[idig], y[idig], z[idig]);
  }

  // Coordinates of centers of clusters
  std::vector<double> xMax(nMax);
  std::vector<double> yMax(nMax);
  std::vector<double> zMax(nMax);
  std::vector<double> eMax(nMax);

  for (int iclu = 0; iclu < nMax; iclu++) {
    xMax[iclu] = x[digitId[iclu]];
    yMax[iclu] = y[digitId[iclu]];
    zMax[iclu] = z[digitId[iclu]];
    eMax[iclu] = e[digitId[iclu]];
  }

  std::vector<double> prop(nMax); // proportion of clusters in the current digit

  // Try to decompose cluster to contributions
  int nIterations = 0;
  bool insuficientAccuracy = true;

  while (insuficientAccuracy && nIterations < mNMaxIterations) {
    // Loop over all digits of parent cluster and split their energies between daughter clusters
    // according to shower shape
    for (int idig = 0; idig < mult; idig++) {
      double eEstimated = 0;
      for (int iclu = 0; iclu < nMax; iclu++) {
        prop[iclu] = eMax[iclu] * showerShape(std::sqrt((x[idig] - xMax[iclu]) * (x[idig] - xMax[iclu]) +
                                                        (y[idig] - yMax[iclu]) * (y[idig] - yMax[iclu])),
                                              z[idig] - zMax[iclu], isCrystal);
        eEstimated += prop[iclu];
      }
      if (eEstimated == 0.) { // numerical accuracy
        continue;
      }
      // Split energy of digit according to contributions
      for (int iclu = 0; iclu < nMax; iclu++) {
        eInClusters[idig][iclu] = e[idig] * prop[iclu] / eEstimated;
      }
    }
    // Recalculate parameters of clusters and check relative variation of energy and absolute of position
    insuficientAccuracy = false; // will be true if at least one parameter changed too much
    for (int iclu = 0; iclu < nMax; iclu++) {
      double oldX = xMax[iclu];
      double oldY = yMax[iclu];
      double oldZ = zMax[iclu];
      double oldE = eMax[iclu];
      // new energy, need for weight
      eMax[iclu] = 0;
      for (int idig = 0; idig < mult; idig++) {
        eMax[iclu] += eInClusters[idig][iclu];
      }
      xMax[iclu] = 0;
      yMax[iclu] = 0;
      zMax[iclu] = 0;
      double wtot = 0.;
      for (int idig = 0; idig < mult; idig++) {
        double w = std::max(std::log(eInClusters[idig][iclu] / eMax[iclu]) + mLogWeight, 0.);
        xMax[iclu] += x[idig] * w;
        yMax[iclu] += y[idig] * w;
        zMax[iclu] += z[idig] * w;
        wtot += w;
      }
      if (wtot > 0.) {
        xMax[iclu] /= wtot;
        yMax[iclu] /= wtot;
        zMax[iclu] /= wtot;
      }
      // Compare variation of parameters
      insuficientAccuracy += (std::abs(eMax[iclu] - oldE) > mUnfogingEAccuracy);
      insuficientAccuracy += (std::abs(xMax[iclu] - oldX) > mUnfogingXZAccuracy);
      insuficientAccuracy += (std::abs(yMax[iclu] - oldY) > mUnfogingXZAccuracy);
      insuficientAccuracy += (std::abs(zMax[iclu] - oldZ) > mUnfogingXZAccuracy);
    }
    nIterations++;
  }

  // Iterations finished, add new clusters
  for (int iclu = 0; iclu < nMax; iclu++) {
    auto& clu = unfoldedClusters.emplace_back();
    clu.setNLM(nMax);
    for (int idig = 0; idig < mult; idig++) {
      int jdigit = iniClu->getDigitIndex(idig);
      int towerId = iniClu->getDigitTowerId(idig);
      clu.addDigit(jdigit, towerId, eInClusters[idig][iclu]);
    }
  }
}

//==============================================================================
void Clusterizer::evalClusters(std::vector<Cluster>& clusters)
{
  auto& geo = Geometry::instance();
  for (auto& cluster : clusters) {
    double x = 0;
    double y = 0;
    double z = 0;
    double wtot = 0;
    double etot = cluster.getE();
    for (size_t i = 0; i < cluster.getMultiplicity(); i++) {
      float energy = cluster.getDigitEnergy(i);
      int towerId = cluster.getDigitTowerId(i);
      double xi, yi, zi;
      geo.detIdToGlobalPosition(towerId, xi, yi, zi);
      double w = std::max(0., mLogWeight + std::log(energy / etot));
      x += w * xi;
      y += w * yi;
      z += w * zi;
      wtot += w;
    }
    if (wtot != 0) {
      x /= wtot;
      y /= wtot;
      z /= wtot;
    }
    cluster.setX(x);
    cluster.setY(y);
    cluster.setZ(z);

    // cluster shape
    float chi2 = 0;
    int ndf = 0;
    float ee = cluster.getE();
    for (size_t i = 0; i < cluster.getMultiplicity(); i++) {
      float energy = cluster.getDigitEnergy(i);
      int towerId = cluster.getDigitTowerId(i);
      double xi, yi, zi;
      geo.detIdToGlobalPosition(towerId, xi, yi, zi);
      double r = std::sqrt((x - xi) * (x - xi) + (y - yi) * (y - yi) + (z - zi) * (z - zi));
      if (r > 2.2) {
        continue;
      }
      double frac = fCrystalShowerShape->Eval(r);
      double rms = fCrystalRMS->Eval(r);
      chi2 += std::pow((energy / ee - frac) / rms, 2.);
      ndf++;
    }
    cluster.setChi2(chi2 / ndf);

    // correct cluster energy and z position
    float eta = std::abs(cluster.getEta());
    bool isCrystal = geo.isCrystal(cluster.getDigitTowerId(0));
    if (mApplyCorrectionE) {
      std::vector<double>& pe = isCrystal ? mCrystalEnergyCorrectionPars : mSamplingEnergyCorrectionPars;
      ee *= pe[0] * std::pow(ee, pe[1]) + pe[2] + pe[3] * eta + pe[4] * eta * eta + pe[5] * eta * eta * eta;
      cluster.setE(ee);
    }
    if (mApplyCorrectionZ) {
      std::vector<double>& pz = isCrystal ? mCrystalZCorrectionPars : mSamplingZCorrectionPars;
      float zCor = (pz[0] + pz[1] * eta + pz[2] * eta * eta) + (pz[3] + pz[4] * eta + pz[5] * eta * eta) * ee + (pz[6] + pz[7] * eta + pz[8] * eta * eta) * ee * ee;
      cluster.setZ(z > 0 ? z - zCor : z + zCor);
    }

    // check if cluster is at the edge of detector module
    bool isEdge = 0;
    for (size_t i = 0; i < cluster.getMultiplicity(); i++) {
      int towerId = cluster.getDigitTowerId(i);
      if (geo.isAtTheEdge(towerId)) {
        isEdge = 1;
        break;
      }
    }
    cluster.setEdgeFlag(isEdge);

    LOGF(debug, "Cluster coordinates: (%6.2f,%6.2f,%6.2f)", cluster.getX(), cluster.getY(), cluster.getZ());
  }
}

//==============================================================================
int Clusterizer::getNumberOfLocalMax(Cluster& clu, int* maxAt, float* maxAtEnergy)
{
  // Based on MpdEmcClusterizerKI::GetNumberOfLocalMax by D. Peresunko
  // Calculates the number of local maxima in the cluster using LocalMaxCut as the minimum
  // energy difference between maximum and surrounding digits
  auto& geo = Geometry::instance();
  int n = clu.getMultiplicity();
  bool isCrystal = geo.isCrystal(clu.getDigitTowerId(0));
  bool* isLocalMax = new bool[n];

  for (int i = 0; i < n; i++) {
    isLocalMax[i] = false;
    float en1 = clu.getDigitEnergy(i);
    if (en1 > mClusteringThreshold) {
      isLocalMax[i] = true;
    }
  }

  for (int i = 0; i < n; i++) {
    int detId1 = clu.getDigitTowerId(i);
    float en1 = clu.getDigitEnergy(i);
    for (int j = i + 1; j < n; j++) {
      int detId2 = clu.getDigitTowerId(j);
      float en2 = clu.getDigitEnergy(j);
      if (geo.areNeighboursVertex(detId1, detId2) == 1) {
        if (en1 > en2) {
          isLocalMax[j] = false;
          // but may be digit too is not local max ?
          if (en2 > en1 - mLocalMaximumCut) {
            isLocalMax[i] = false;
          }
        } else {
          isLocalMax[i] = false;
          // but may be digitN is not local max too?
          if (en1 > en2 - mLocalMaximumCut) {
            isLocalMax[j] = false;
          }
        }
      } // if neighbours
    } // digit j
  } // digit i

  int iDigitN = 0;
  for (int i = 0; i < n; i++) {
    if (isLocalMax[i]) {
      maxAt[iDigitN] = i;
      maxAtEnergy[iDigitN] = clu.getDigitEnergy(i);
      iDigitN++;
      if (iDigitN >= mNLMMax) { // Note that size of output arrays is limited:
        LOGP(error, "Too many local maxima, cluster multiplicity {} region={}", n, isCrystal ? "crystal" : "sampling");
        return 0;
      }
    }
  }
  delete[] isLocalMax;
  return iDigitN;
}

//==============================================================================
double Clusterizer::showerShape(double dx, double dz, bool isCrystal)
{
  double x = std::sqrt(dx * dx + dz * dz);
  return isCrystal ? fCrystalShowerShape->Eval(x) : fSamplingShowerShape->Eval(x);
}
