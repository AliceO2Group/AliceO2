// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

/// \file ClusterFactory.cxx
#include <array>
#include <gsl/span>
#include "Rtypes.h"
#include "DataFormatsEMCAL/Cluster.h"
#include "DataFormatsEMCAL/Digit.h"
#include "DataFormatsEMCAL/Cell.h"
#include "DataFormatsEMCAL/AnalysisCluster.h"
#include "DataFormatsEMCAL/Constants.h"
#include "EMCALBase/Geometry.h"
#include "MathUtils/Cartesian3D.h"

#include "EMCALBase/ClusterFactory.h"

using namespace o2::emcal;

template <class InputType>
ClusterFactory<InputType>::ClusterFactory(gsl::span<const o2::emcal::Cluster> clustersContainer, gsl::span<const InputType> inputsContainer, gsl::span<const int> cellsIndices)
{
  setClustersContainer(clustersContainer);
  setCellsContainer(inputsContainer);
  setCellsIndicesContainer(cellsIndices);
}

template <class InputType>
void ClusterFactory<InputType>::reset()
{
  mClustersContainer = gsl::span<const o2::emcal::Cluster>();
  mInputsContainer = gsl::span<const InputType>();
  mCellsIndices = gsl::span<int>();
}

///
/// evaluates cluster parameters: position, shower shape, primaries ...
//____________________________________________________________________________
template <class InputType>
o2::emcal::AnalysisCluster ClusterFactory<InputType>::buildCluster(int clusterIndex) const
{
  if (clusterIndex >= mClustersContainer.size()) {
    throw ClusterRangeException(clusterIndex, mClustersContainer.size());
  }

  o2::emcal::AnalysisCluster clusterAnalysis;
  clusterAnalysis.setID(clusterIndex);

  int firstCellIndex = mClustersContainer[clusterIndex].getCellIndexFirst();
  int nCells = mClustersContainer[clusterIndex].getNCells();

  gsl::span<const int> inputsIndices = gsl::span<const int>(&mCellsIndices[firstCellIndex], nCells);

  // First calculate the index of input with maximum amplitude and get
  // the supermodule number where it sits.

  auto [inputIndMax, inputEnergyMax, cellAmp] = getMaximalEnergyIndex(inputsIndices);

  clusterAnalysis.setIndMaxInput(inputIndMax);

  clusterAnalysis.setE(cellAmp);

  mSuperModuleNumber = mGeomPtr->GetSuperModuleNumber(mInputsContainer[inputIndMax].getTower());

  clusterAnalysis.setNCells(inputsIndices.size());

  std::vector<unsigned short> cellsIdices;

  for (auto cellIndex : inputsIndices) {
    cellsIdices.push_back(cellIndex);
  }

  clusterAnalysis.setCellsIndices(cellsIdices);

  // evaluate global and local position
  evalGlobalPosition(inputsIndices, clusterAnalysis);
  evalLocalPosition(inputsIndices, clusterAnalysis);

  // evaluate shower parameters
  evalElipsAxis(inputsIndices, clusterAnalysis);
  evalDispersion(inputsIndices, clusterAnalysis);

  evalCoreEnergy(inputsIndices, clusterAnalysis);
  evalTime(inputsIndices, clusterAnalysis);

  // TODO to be added at a later stage
  //evalPrimaries(inputsIndices, clusterAnalysis);
  //evalParents(inputsIndices, clusterAnalysis);

  // TODO to be added at a later stage
  // Called last because it sets the global position of the cluster?
  // Do not call it when recalculating clusters out of standard reconstruction
  //if (!mJustCluster)
  //  evalLocal2TrackingCSTransform();

  return clusterAnalysis;
}

///
/// Calculates the dispersion of the shower at the origin of the cluster
/// in cell units
//____________________________________________________________________________
template <class InputType>
void ClusterFactory<InputType>::evalDispersion(gsl::span<const int> inputsIndices, AnalysisCluster& clusterAnalysis) const
{
  double d = 0., wtot = 0.;
  int nstat = 0;

  // Calculates the dispersion in cell units
  double etaMean = 0.0, phiMean = 0.0;

  // Calculate mean values
  for (auto iInput : inputsIndices) {

    if (clusterAnalysis.E() > 0 && mInputsContainer[iInput].getEnergy() > 0) {
      auto [nSupMod, nModule, nIphi, nIeta] = mGeomPtr->GetCellIndex(mInputsContainer[iInput].getTower());
      auto [iphi, ieta] = mGeomPtr->GetCellPhiEtaIndexInSModule(nSupMod, nModule, nIphi, nIeta);

      // In case of a shared cluster, index of SM in C side, columns start at 48 and ends at 48*2
      // C Side impair SM, nSupMod%2=1; A side pair SM nSupMod%2=0
      if (mSharedCluster && nSupMod % 2) {
        ieta += EMCAL_COLS;
      }

      double etai = (double)ieta;
      double phii = (double)iphi;
      double w = TMath::Max(0., mLogWeight + TMath::Log(mInputsContainer[iInput].getEnergy() / clusterAnalysis.E()));

      if (w > 0.0) {
        phiMean += phii * w;
        etaMean += etai * w;
        wtot += w;
      }
    }
  }

  if (wtot > 0) {
    phiMean /= wtot;
    etaMean /= wtot;
  } else {
    LOG(ERROR) << Form("Wrong weight %f\n", wtot);
  }

  // Calculate dispersion
  for (auto iInput : inputsIndices) {

    if (clusterAnalysis.E() > 0 && mInputsContainer[iInput].getEnergy() > 0) {
      auto [nSupMod, nModule, nIphi, nIeta] = mGeomPtr->GetCellIndex(mInputsContainer[iInput].getTower());
      auto [iphi, ieta] = mGeomPtr->GetCellPhiEtaIndexInSModule(nSupMod, nModule, nIphi, nIeta);

      // In case of a shared cluster, index of SM in C side, columns start at 48 and ends at 48*2
      // C Side impair SM, nSupMod%2=1; A side pair SM, nSupMod%2=0
      if (mSharedCluster && nSupMod % 2) {
        ieta += EMCAL_COLS;
      }

      double etai = (double)ieta;
      double phii = (double)iphi;
      double w = TMath::Max(0., mLogWeight + TMath::Log(mInputsContainer[iInput].getEnergy() / clusterAnalysis.E()));

      if (w > 0.0) {
        nstat++;
        d += w * ((etai - etaMean) * (etai - etaMean) + (phii - phiMean) * (phii - phiMean));
      }
    }
  }

  if (wtot > 0 && nstat > 1) {
    d /= wtot;
  } else {
    d = 0.;
  }

  clusterAnalysis.setDispersion(TMath::Sqrt(d));
}

///
/// Calculates the center of gravity in the local EMCAL-module coordinates
//____________________________________________________________________________
template <class InputType>
void ClusterFactory<InputType>::evalLocalPosition(gsl::span<const int> inputsIndices, AnalysisCluster& clusterAnalysis) const
{

  int nstat = 0;

  double dist = tMaxInCm(double(clusterAnalysis.E()));

  double clXYZ[3] = {0., 0., 0.}, clRmsXYZ[3] = {0., 0., 0.}, xyzi[3], wtot = 0., w = 0.;

  for (auto iInput : inputsIndices) {

    try {
      mGeomPtr->RelPosCellInSModule(mInputsContainer[iInput].getTower(), dist).GetCoordinates(xyzi[0], xyzi[1], xyzi[2]);
    } catch (InvalidCellIDException& e) {
      LOG(ERROR) << e.what();
      continue;
    }

    //Temporal patch, due to mapping problem, need to swap "y" in one of the 2 SM, although no effect in position calculation. GCB 05/2010
    if (mSharedCluster && mSuperModuleNumber != mGeomPtr->GetSuperModuleNumber(mInputsContainer[iInput].getTower())) {
      xyzi[1] *= -1;
    }

    if (mLogWeight > 0.0) {
      w = TMath::Max(0., mLogWeight + TMath::Log(mInputsContainer[iInput].getEnergy() / clusterAnalysis.E()));
    } else {
      w = mInputsContainer[iInput].getEnergy(); // just energy
    }

    if (w > 0.0) {
      wtot += w;
      nstat++;

      for (int i = 0; i < 3; i++) {
        clXYZ[i] += (w * xyzi[i]);
        clRmsXYZ[i] += (w * xyzi[i] * xyzi[i]);
      }
    } // w > 0
  }   // dig loop

  //  cout << " wtot " << wtot << endl;

  if (wtot > 0) {
    //    xRMS   = TMath::Sqrt(x2m - xMean*xMean);
    for (int i = 0; i < 3; i++) {
      clXYZ[i] /= wtot;

      if (nstat > 1) {
        clRmsXYZ[i] /= (wtot * wtot);
        clRmsXYZ[i] = clRmsXYZ[i] - clXYZ[i] * clXYZ[i];

        if (clRmsXYZ[i] > 0.0) {
          clRmsXYZ[i] = TMath::Sqrt(clRmsXYZ[i]);
        } else {
          clRmsXYZ[i] = 0;
        }
      } else {
        clRmsXYZ[i] = 0;
      }
    }
  } else {
    for (int i = 0; i < 3; i++) {
      clXYZ[i] = clRmsXYZ[i] = -1.;
    }
  }

  clusterAnalysis.setLocalPosition(Point3D<float>(clXYZ[0], clXYZ[1], clXYZ[2]));
}

///
/// Calculates the center of gravity in the global ALICE coordinates
//____________________________________________________________________________
template <class InputType>
void ClusterFactory<InputType>::evalGlobalPosition(gsl::span<const int> inputsIndices, AnalysisCluster& clusterAnalysis) const
{

  int i = 0, nstat = 0;

  double dist = tMaxInCm(double(clusterAnalysis.E()));

  double clXYZ[3] = {0., 0., 0.}, clRmsXYZ[3] = {0., 0., 0.}, lxyzi[3], xyzi[3], wtot = 0., w = 0.;

  for (auto iInput : inputsIndices) {

    // get the local coordinates of the cell
    try {
      mGeomPtr->RelPosCellInSModule(mInputsContainer[iInput].getTower(), dist).GetCoordinates(xyzi[0], xyzi[1], xyzi[2]);
    } catch (InvalidCellIDException& e) {
      LOG(ERROR) << e.what();
      continue;
    }

    // Now get the global coordinate
    mGeomPtr->GetGlobal(lxyzi, xyzi, mGeomPtr->GetSuperModuleNumber(mInputsContainer[iInput].getTower()));

    if (mLogWeight > 0.0) {
      w = TMath::Max(0., mLogWeight + TMath::Log(mInputsContainer[iInput].getEnergy() / clusterAnalysis.E()));
    } else {
      w = mInputsContainer[iInput].getEnergy(); // just energy
    }

    if (w > 0.0) {
      wtot += w;
      nstat++;

      for (i = 0; i < 3; i++) {
        clXYZ[i] += (w * xyzi[i]);
        clRmsXYZ[i] += (w * xyzi[i] * xyzi[i]);
      }
    }
  }

  //  cout << " wtot " << wtot << endl;

  if (wtot > 0) {
    //    xRMS   = TMath::Sqrt(x2m - xMean*xMean);
    for (i = 0; i < 3; i++) {
      clXYZ[i] /= wtot;

      if (nstat > 1) {
        clRmsXYZ[i] /= (wtot * wtot);
        clRmsXYZ[i] = clRmsXYZ[i] - clXYZ[i] * clXYZ[i];

        if (clRmsXYZ[i] > 0.0) {
          clRmsXYZ[i] = TMath::Sqrt(clRmsXYZ[i]);
        } else {
          clRmsXYZ[i] = 0;
        }
      } else {
        clRmsXYZ[i] = 0;
      }
    }
  } else {
    for (i = 0; i < 3; i++) {
      clXYZ[i] = clRmsXYZ[i] = -1.;
    }
  }

  clusterAnalysis.setGlobalPosition(Point3D<float>(clXYZ[0], clXYZ[1], clXYZ[2]));
}

///
/// evaluates local position of clusters in SM
//____________________________________________________________________________
template <class InputType>
void ClusterFactory<InputType>::evalLocalPositionFit(double deff, double mLogWeight,
                                                     double phiSlope, gsl::span<const int> inputsIndices, AnalysisCluster& clusterAnalysis) const
{
  int i = 0, nstat = 0;
  double clXYZ[3] = {0., 0., 0.}, clRmsXYZ[3] = {0., 0., 0.}, xyzi[3], wtot = 0., w = 0.;

  for (auto iInput : inputsIndices) {

    try {
      mGeomPtr->RelPosCellInSModule(mInputsContainer[iInput].getTower(), deff).GetCoordinates(xyzi[0], xyzi[1], xyzi[2]);
    } catch (InvalidCellIDException& e) {
      LOG(ERROR) << e.what();
      continue;
    }

    if (mLogWeight > 0.0) {
      w = TMath::Max(0., mLogWeight + TMath::Log(mInputsContainer[iInput].getEnergy() / clusterAnalysis.E()));
    } else {
      w = mInputsContainer[iInput].getEnergy(); // just energy
    }

    if (w > 0.0) {
      wtot += w;
      nstat++;

      for (i = 0; i < 3; i++) {
        clXYZ[i] += (w * xyzi[i]);
        clRmsXYZ[i] += (w * xyzi[i] * xyzi[i]);
      }
    }
  } //loop

  //  cout << " wtot " << wtot << endl;

  if (wtot > 0) {
    //    xRMS   = TMath::Sqrt(x2m - xMean*xMean);
    for (i = 0; i < 3; i++) {
      clXYZ[i] /= wtot;

      if (nstat > 1) {
        clRmsXYZ[i] /= (wtot * wtot);
        clRmsXYZ[i] = clRmsXYZ[i] - clXYZ[i] * clXYZ[i];

        if (clRmsXYZ[i] > 0.0) {
          clRmsXYZ[i] = TMath::Sqrt(clRmsXYZ[i]);
        } else {
          clRmsXYZ[i] = 0;
        }
      } else {
        clRmsXYZ[i] = 0;
      }
    }
  } else {
    for (i = 0; i < 3; i++) {
      clXYZ[i] = clRmsXYZ[i] = -1.;
    }
  }

  // clRmsXYZ[i] ??

  if (phiSlope != 0.0 && mLogWeight > 0.0 && wtot) {
    // Correction in phi direction (y - coords here); Aug 16;
    // May be put to global level or seperate method
    double ycorr = clXYZ[1] * (1. + phiSlope);

    //printf(" y %f : ycorr %f : slope %f \n", clXYZ[1], ycorr, phiSlope);
    clXYZ[1] = ycorr;
  }

  clusterAnalysis.setLocalPosition(Point3D<float>(clXYZ[0], clXYZ[1], clXYZ[2]));
}

///
/// Applied for simulation data with threshold 3 adc
/// Calculate efective distance (deff) and weigh parameter (w0)
/// for coordinate calculation; 0.5 GeV < esum <100 GeV.
//_____________________________________________________________________________
template <class InputType>
void ClusterFactory<InputType>::getDeffW0(const double esum, double& deff, double& w0)
{
  double e = 0.0;
  const double kdp0 = 9.25147, kdp1 = 1.16700; // Hard coded now
  const double kwp0 = 4.83713, kwp1 = -2.77970e-01, kwp2 = 4.41116;

  // No extrapolation here
  e = esum < 0.5 ? 0.5 : esum;
  e = e > 100. ? 100. : e;

  deff = kdp0 + kdp1 * TMath::Log(e);
  w0 = kwp0 / (1. + TMath::Exp(kwp1 * (e + kwp2)));
}

///
/// This function calculates energy in the core,
/// i.e. within a radius rad = mCoreRadius around the center. Beyond this radius
/// in accordance with shower profile the energy deposition
/// should be less than 2%
/// Unfinished - Nov 15,2006
/// Distance is calculate in (phi,eta) units
//______________________________________________________________________________
template <class InputType>
void ClusterFactory<InputType>::evalCoreEnergy(gsl::span<const int> inputsIndices, AnalysisCluster& clusterAnalysis) const
{

  float coreEnergy = 0.;

  if (!clusterAnalysis.getLocalPosition().Mag2()) {
    evalLocalPosition(inputsIndices, clusterAnalysis);
  }

  double phiPoint = clusterAnalysis.getLocalPosition().Phi();
  double etaPoint = clusterAnalysis.getLocalPosition().Eta();
  for (auto iInput : inputsIndices) {

    auto [eta, phi] = mGeomPtr->EtaPhiFromIndex(mInputsContainer[iInput].getTower());
    phi = phi * TMath::DegToRad();

    double distance = TMath::Sqrt((eta - etaPoint) * (eta - etaPoint) + (phi - phiPoint) * (phi - phiPoint));

    if (distance < mCoreRadius) {
      coreEnergy += mInputsContainer[iInput].getEnergy();
    }
  }
  clusterAnalysis.setCoreEnergy(coreEnergy);
}

///
/// Calculates the axis of the shower ellipsoid in eta and phi
/// in cell units
//____________________________________________________________________________
template <class InputType>
void ClusterFactory<InputType>::evalElipsAxis(gsl::span<const int> inputsIndices, AnalysisCluster& clusterAnalysis) const
{
  TString gn(mGeomPtr->GetName());

  double wtot = 0.;
  double x = 0.;
  double z = 0.;
  double dxx = 0.;
  double dzz = 0.;
  double dxz = 0.;

  std::array<float, 2> lambda;

  for (auto iInput : inputsIndices) {

    auto [nSupMod, nModule, nIphi, nIeta] = mGeomPtr->GetCellIndex(mInputsContainer[iInput].getTower());
    auto [iphi, ieta] = mGeomPtr->GetCellPhiEtaIndexInSModule(nSupMod, nModule, nIphi, nIeta);

    // In case of a shared cluster, index of SM in C side, columns start at 48 and ends at 48*2
    // C Side impair SM, nSupMod%2=1; A side pair SM, nSupMod%2=0
    if (mSharedCluster && nSupMod % 2) {
      ieta += EMCAL_COLS;
    }

    double etai = (double)ieta;
    double phii = (double)iphi;

    double w = TMath::Max(0., mLogWeight + TMath::Log(mInputsContainer[iInput].getEnergy() / clusterAnalysis.E()));
    // clusterAnalysis.E() summed amplitude of inputs, i.e. energy of cluster
    // Gives smaller value of lambda than log weight
    // w = mEnergyList[iInput] / clusterAnalysis.E(); // Nov 16, 2006 - try just energy

    dxx += w * etai * etai;
    x += w * etai;
    dzz += w * phii * phii;
    z += w * phii;

    dxz += w * etai * phii;

    wtot += w;
  }

  if (wtot > 0) {
    dxx /= wtot;
    x /= wtot;
    dxx -= x * x;
    dzz /= wtot;
    z /= wtot;
    dzz -= z * z;
    dxz /= wtot;
    dxz -= x * z;

    lambda[0] = 0.5 * (dxx + dzz) + TMath::Sqrt(0.25 * (dxx - dzz) * (dxx - dzz) + dxz * dxz);

    if (lambda[0] > 0) {
      lambda[0] = TMath::Sqrt(lambda[0]);
    } else {
      lambda[0] = 0;
    }

    lambda[1] = 0.5 * (dxx + dzz) - TMath::Sqrt(0.25 * (dxx - dzz) * (dxx - dzz) + dxz * dxz);

    if (lambda[1] > 0) { //To avoid exception if numerical errors lead to negative lambda.
      lambda[1] = TMath::Sqrt(lambda[1]);
    } else {
      lambda[1] = 0.;
    }
  } else {
    lambda[0] = 0.;
    lambda[1] = 0.;
  }

  clusterAnalysis.setM02(lambda[0] * lambda[0]);
  clusterAnalysis.setM20(lambda[1] * lambda[1]);
}

///
/// Finds the maximum energy in the cluster and computes the Summed amplitude of digits/cells
//____________________________________________________________________________
template <class InputType>
std::tuple<int, float, float> ClusterFactory<InputType>::getMaximalEnergyIndex(gsl::span<const int> inputsIndices) const
{

  float energy = 0.;
  int mid = 0;
  float cellAmp = 0;
  for (auto iInput : inputsIndices) {
    if (iInput >= mInputsContainer.size()) {
      throw CellIndexRangeException(iInput, mInputsContainer.size());
    }
    cellAmp += mInputsContainer[iInput].getEnergy();
    if (mInputsContainer[iInput].getEnergy() > energy) {
      energy = mInputsContainer[iInput].getEnergy();
      mid = iInput;
    }
  } // loop on cluster inputs

  return std::make_tuple(mid, energy, cellAmp);
}

///
/// Calculates the multiplicity of inputs with energy larger than H*energy
//____________________________________________________________________________
template <class InputType>
int ClusterFactory<InputType>::getMultiplicityAtLevel(float H, gsl::span<const int> inputsIndices, AnalysisCluster& clusterAnalysis) const
{
  int multipl = 0;
  for (auto iInput : inputsIndices) {
    if (mInputsContainer[iInput].getEnergy() > H * clusterAnalysis.E()) {
      multipl++;
    }
  }

  return multipl;
}

///
/// Time is set to the time of the input with the maximum energy
//____________________________________________________________________________
template <class InputType>
void ClusterFactory<InputType>::evalTime(gsl::span<const int> inputsIndices, AnalysisCluster& clusterAnalysis) const
{
  float maxE = 0;
  unsigned short maxAt = 0;
  for (auto iInput : inputsIndices) {
    if (mInputsContainer[iInput].getEnergy() > maxE) {
      maxE = mInputsContainer[iInput].getEnergy();
      maxAt = iInput;
    }
  }

  clusterAnalysis.setClusterTime(mInputsContainer[maxAt].getTimeStamp());
}

///
/// \param  e: energy in GeV)
/// \param  key: = 0(gamma, default); !=  0(electron)
//_____________________________________________________________________
template <class InputType>
double ClusterFactory<InputType>::tMaxInCm(const double e, const int key) const
{
  const double ca = 4.82; // shower max parameter - first guess; ca=TMath::Log(1000./8.07)
  double tmax = 0.;       // position of electromagnetic shower max in cm

  const double x0 = 1.31; // radiation lenght (cm)

  if (e > 0.1) {
    tmax = TMath::Log(e) + ca;
    if (key == 0) {
      tmax += 0.5;
    } else {
      tmax -= 0.5;
    }
    tmax *= x0; // convert to cm
  }

  return tmax;
}

///
/// Converts Theta (Radians) to Eta (Radians)
//______________________________________________________________________________
template <class InputType>
float ClusterFactory<InputType>::etaToTheta(float arg) const
{
  return (2. * TMath::ATan(TMath::Exp(-arg)));
}

///
/// Converts Eta (Radians) to Theta (Radians)
//______________________________________________________________________________
template <class InputType>
float ClusterFactory<InputType>::thetaToEta(float arg) const
{
  return (-1 * TMath::Log(TMath::Tan(0.5 * arg)));
}

template <class InputType>
ClusterFactory<InputType>::ClusterIterator::ClusterIterator(const ClusterFactory& factory, int clusterIndex, bool forward) : mClusterFactory(factory),
                                                                                                                             mCurrentCluster(),
                                                                                                                             mClusterID(clusterIndex),
                                                                                                                             mForward(forward)
{
  mCurrentCluster = mClusterFactory.buildCluster(mClusterID);
}

template <class InputType>
bool ClusterFactory<InputType>::ClusterIterator::operator==(const ClusterFactory<InputType>::ClusterIterator& rhs) const
{
  return &mClusterFactory == &rhs.mClusterFactory && mClusterID == rhs.mClusterID && mForward == rhs.mForward;
}

template <class InputType>
typename ClusterFactory<InputType>::ClusterIterator& ClusterFactory<InputType>::ClusterIterator::operator++()
{
  if (mForward) {
    mClusterID++;
  } else {
    mClusterID--;
  }
  mCurrentCluster = mClusterFactory.buildCluster(mClusterID);
  return *this;
}

template <class InputType>
typename ClusterFactory<InputType>::ClusterIterator ClusterFactory<InputType>::ClusterIterator::operator++(int)
{
  auto tmp = *this;
  ++(*this);
  return tmp;
}

template <class InputType>
typename ClusterFactory<InputType>::ClusterIterator& ClusterFactory<InputType>::ClusterIterator::operator--()
{
  if (mForward) {
    mClusterID--;
  } else {
    mClusterID++;
  }
  mCurrentCluster = mClusterFactory.buildCluster(mClusterID);
  return *this;
}

template <class InputType>
typename ClusterFactory<InputType>::ClusterIterator ClusterFactory<InputType>::ClusterIterator::operator--(int)
{
  auto tmp = *this;
  --(*this);
  return tmp;
}

template class o2::emcal::ClusterFactory<o2::emcal::Cell>;
template class o2::emcal::ClusterFactory<o2::emcal::Digit>;