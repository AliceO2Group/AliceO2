// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

#include <cmath>
#include "ITSReconstruction/CA/vertexer/ClusterLines.h"

namespace o2
{
namespace ITS
{
namespace CA
{

Line::Line() : originPoint{}, weightMatrix{ std::array<float, 6>{ 1., 0., 0., 1., 0., 1. } }, cosinesDirector{}
{
  // Nothing to do
}

Line::Line(std::array<float, 3> firstPoint, std::array<float, 3> secondPoint)
  : originPoint{ firstPoint }, weightMatrix{ std::array<float, 6>{ 1., 0., 0., 1., 0., 1. } } // dummy, ATM
{
  for (int index{ 0 }; index < 3; ++index)
    cosinesDirector[index] = secondPoint[index] - firstPoint[index];
  float inverseNorm{ 1.f / std::sqrt(cosinesDirector[0] * cosinesDirector[0] + cosinesDirector[1] * cosinesDirector[1] +
                                     cosinesDirector[2] * cosinesDirector[2]) };
  for (int index{ 0 }; index < 3; ++index)
    cosinesDirector[index] *= inverseNorm;
}

bool Line::areParallel(const Line& firstLine, const Line& secondLine, const float precision)
{
  float crossProdX{ firstLine.cosinesDirector[1] * secondLine.cosinesDirector[2] -
                    firstLine.cosinesDirector[2] * secondLine.cosinesDirector[1] };
  float module{ std::abs(firstLine.cosinesDirector[1] * secondLine.cosinesDirector[2]) +
                std::abs(firstLine.cosinesDirector[2] * secondLine.cosinesDirector[1]) };
  if (std::abs(crossProdX) > precision * module)
    return false;

  float crossProdY{ -firstLine.cosinesDirector[0] * secondLine.cosinesDirector[2] +
                    firstLine.cosinesDirector[2] * secondLine.cosinesDirector[0] };
  module = std::abs(firstLine.cosinesDirector[0] * secondLine.cosinesDirector[2]) +
           std::abs(firstLine.cosinesDirector[2] * secondLine.cosinesDirector[0]);
  if (std::abs(crossProdY) > precision * module)
    return false;

  float crossProdZ = firstLine.cosinesDirector[0] * secondLine.cosinesDirector[1] -
                     firstLine.cosinesDirector[1] * secondLine.cosinesDirector[0];
  module = std::abs(firstLine.cosinesDirector[0] * secondLine.cosinesDirector[1]) +
           std::abs(firstLine.cosinesDirector[1] * secondLine.cosinesDirector[0]);
  if (std::abs(crossProdZ) > precision * module)
    return false;

  return true;
}

float Line::getDCA(const Line& firstLine, const Line& secondLine, const float precision)
{
  if (areParallel(firstLine, secondLine)) {
    float squaredOriginsDistance{ 0 }, projectionOnFirstVector{ 0 }, normFirstVector{ 0 };
    for (int i{ 0 }; i < 3; ++i) {
      squaredOriginsDistance +=
        (firstLine.originPoint[i] - secondLine.originPoint[i]) * (firstLine.originPoint[i] - secondLine.originPoint[i]);
      projectionOnFirstVector += (firstLine.originPoint[i] - secondLine.originPoint[i]) * firstLine.cosinesDirector[i];
      normFirstVector += firstLine.cosinesDirector[i] * firstLine.cosinesDirector[i];
    }

    if (std::abs(normFirstVector) > precision) {
      projectionOnFirstVector /= normFirstVector;
      return std::sqrt(squaredOriginsDistance - projectionOnFirstVector * projectionOnFirstVector);
    } else {
      return -1;
    }
  } else {
    std::array<float, 3> perpendicularVector{};
    perpendicularVector[0] = firstLine.cosinesDirector[1] * secondLine.cosinesDirector[2] -
                             firstLine.cosinesDirector[2] * secondLine.cosinesDirector[1];
    perpendicularVector[1] = -firstLine.cosinesDirector[0] * secondLine.cosinesDirector[2] +
                             firstLine.cosinesDirector[2] * secondLine.cosinesDirector[0];
    perpendicularVector[2] = firstLine.cosinesDirector[0] * secondLine.cosinesDirector[1] -
                             firstLine.cosinesDirector[1] * secondLine.cosinesDirector[0];

    float normFirstVector{ 0 }, distance{ 0 };
    for (int i{ 0 }; i < 3; ++i) {
      normFirstVector += perpendicularVector[i] * perpendicularVector[i];
      distance += (firstLine.originPoint[i] - secondLine.originPoint[i]) * perpendicularVector[i];
    }
    if (std::abs(normFirstVector) > precision) {
      return std::abs(distance / std::sqrt(normFirstVector));
    } else
      return -1;
  }
}

float Line::getDistanceFromPoint(const Line& line, const std::array<float, 3> point)
{
  Line tmpLine{};
  tmpLine.originPoint = point;
  for (int index{ 0 }; index < 3; ++index)
    tmpLine.cosinesDirector[index] = line.cosinesDirector[index];
  float inverseNorm{ 1.f / std::sqrt(tmpLine.cosinesDirector[0] * tmpLine.cosinesDirector[0] +
                                     tmpLine.cosinesDirector[1] * tmpLine.cosinesDirector[1] +
                                     tmpLine.cosinesDirector[2] * tmpLine.cosinesDirector[2]) };
  for (int index{ 0 }; index < 3; ++index)
    tmpLine.cosinesDirector[index] *= inverseNorm;
  return getDCA(line, tmpLine);
}

ClusterLines::ClusterLines(const int firstLabel, const Line& firstLine, const int secondLabel, const Line& secondLine,
                           const bool weight)
{
  mLabels.push_back(firstLabel);
  mLabels.push_back(secondLabel);

  std::array<float, 3> covarianceFirst{ 1., 1., 1. };
  std::array<float, 3> covarianceSecond{ 1., 1., 1. };

  for (int i{ 0 }; i < 6; ++i)
    mWeightMatrix[i] = firstLine.weightMatrix[i] + secondLine.weightMatrix[i];

  float determinantFirst =
    firstLine.cosinesDirector[2] * firstLine.cosinesDirector[2] * covarianceFirst[0] * covarianceFirst[1] +
    firstLine.cosinesDirector[1] * firstLine.cosinesDirector[1] * covarianceFirst[0] * covarianceFirst[2] +
    firstLine.cosinesDirector[0] * firstLine.cosinesDirector[0] * covarianceFirst[1] * covarianceFirst[2];
  float determinantSecond =
    secondLine.cosinesDirector[2] * secondLine.cosinesDirector[2] * covarianceSecond[0] * covarianceSecond[1] +
    secondLine.cosinesDirector[1] * secondLine.cosinesDirector[1] * covarianceSecond[0] * covarianceSecond[2] +
    secondLine.cosinesDirector[0] * secondLine.cosinesDirector[0] * covarianceSecond[1] * covarianceSecond[2];

  mAMatrix[0] = (firstLine.cosinesDirector[2] * firstLine.cosinesDirector[2] * covarianceFirst[1] +
                 firstLine.cosinesDirector[1] * firstLine.cosinesDirector[1] * covarianceFirst[2]) /
                  determinantFirst +
                (secondLine.cosinesDirector[2] * secondLine.cosinesDirector[2] * covarianceSecond[1] +
                 secondLine.cosinesDirector[1] * secondLine.cosinesDirector[1] * covarianceSecond[2]) /
                  determinantSecond;

  mAMatrix[1] = -firstLine.cosinesDirector[0] * firstLine.cosinesDirector[1] * covarianceFirst[2] / determinantFirst -
                secondLine.cosinesDirector[0] * secondLine.cosinesDirector[1] * covarianceSecond[2] / determinantSecond;

  mAMatrix[2] = -firstLine.cosinesDirector[0] * firstLine.cosinesDirector[2] * covarianceFirst[1] / determinantFirst -
                secondLine.cosinesDirector[0] * secondLine.cosinesDirector[2] * covarianceSecond[1] / determinantSecond;

  mAMatrix[3] = (firstLine.cosinesDirector[2] * firstLine.cosinesDirector[2] * covarianceFirst[0] +
                 firstLine.cosinesDirector[0] * firstLine.cosinesDirector[0] * covarianceFirst[2]) /
                  determinantFirst +
                (secondLine.cosinesDirector[2] * secondLine.cosinesDirector[2] * covarianceSecond[0] +
                 secondLine.cosinesDirector[0] * secondLine.cosinesDirector[0] * covarianceSecond[2]) /
                  determinantSecond;

  mAMatrix[4] = -firstLine.cosinesDirector[1] * firstLine.cosinesDirector[2] * covarianceFirst[0] / determinantFirst -
                secondLine.cosinesDirector[1] * secondLine.cosinesDirector[2] * covarianceSecond[0] / determinantSecond;

  mAMatrix[5] = (firstLine.cosinesDirector[1] * firstLine.cosinesDirector[1] * covarianceFirst[0] +
                 firstLine.cosinesDirector[0] * firstLine.cosinesDirector[0] * covarianceFirst[1]) /
                  determinantFirst +
                (secondLine.cosinesDirector[1] * secondLine.cosinesDirector[1] * covarianceSecond[0] +
                 secondLine.cosinesDirector[0] * secondLine.cosinesDirector[0] * covarianceSecond[1]) /
                  determinantSecond;

  mBMatrix[0] = (firstLine.cosinesDirector[1] * covarianceFirst[2] *
                   (-firstLine.cosinesDirector[1] * firstLine.originPoint[0] +
                    firstLine.cosinesDirector[0] * firstLine.originPoint[1]) +
                 firstLine.cosinesDirector[2] * covarianceFirst[1] *
                   (-firstLine.cosinesDirector[2] * firstLine.originPoint[0] +
                    firstLine.cosinesDirector[0] * firstLine.originPoint[2])) /
                determinantFirst;

  mBMatrix[0] += (secondLine.cosinesDirector[1] * covarianceSecond[2] *
                    (-secondLine.cosinesDirector[1] * secondLine.originPoint[0] +
                     secondLine.cosinesDirector[0] * secondLine.originPoint[1]) +
                  secondLine.cosinesDirector[2] * covarianceSecond[1] *
                    (-secondLine.cosinesDirector[2] * secondLine.originPoint[0] +
                     secondLine.cosinesDirector[0] * secondLine.originPoint[2])) /
                 determinantSecond;

  mBMatrix[1] = (firstLine.cosinesDirector[0] * covarianceFirst[2] *
                   (-firstLine.cosinesDirector[0] * firstLine.originPoint[1] +
                    firstLine.cosinesDirector[1] * firstLine.originPoint[0]) +
                 firstLine.cosinesDirector[2] * covarianceFirst[0] *
                   (-firstLine.cosinesDirector[2] * firstLine.originPoint[1] +
                    firstLine.cosinesDirector[1] * firstLine.originPoint[2])) /
                determinantFirst;

  mBMatrix[1] += (secondLine.cosinesDirector[0] * covarianceSecond[2] *
                    (-secondLine.cosinesDirector[0] * secondLine.originPoint[1] +
                     secondLine.cosinesDirector[1] * secondLine.originPoint[0]) +
                  secondLine.cosinesDirector[2] * covarianceSecond[0] *
                    (-secondLine.cosinesDirector[2] * secondLine.originPoint[1] +
                     secondLine.cosinesDirector[1] * secondLine.originPoint[2])) /
                 determinantSecond;

  mBMatrix[2] = (firstLine.cosinesDirector[0] * covarianceFirst[1] *
                   (-firstLine.cosinesDirector[0] * firstLine.originPoint[2] +
                    firstLine.cosinesDirector[2] * firstLine.originPoint[0]) +
                 firstLine.cosinesDirector[1] * covarianceFirst[0] *
                   (-firstLine.cosinesDirector[1] * firstLine.originPoint[2] +
                    firstLine.cosinesDirector[2] * firstLine.originPoint[1])) /
                determinantFirst;

  mBMatrix[2] += (secondLine.cosinesDirector[0] * covarianceSecond[1] *
                    (-secondLine.cosinesDirector[0] * secondLine.originPoint[2] +
                     secondLine.cosinesDirector[2] * secondLine.originPoint[0]) +
                  secondLine.cosinesDirector[1] * covarianceSecond[0] *
                    (-secondLine.cosinesDirector[1] * secondLine.originPoint[2] +
                     secondLine.cosinesDirector[2] * secondLine.originPoint[1])) /
                 determinantSecond;

  computeClusterCentroid();
}

void ClusterLines::add(const int lineLabel, const Line& line, const bool weight)
{
  mLabels.push_back(lineLabel);
  std::array<float, 3> covariance{ 1., 1., 1. };

  for (int i{ 0 }; i < 6; ++i)
    mWeightMatrix[i] += line.weightMatrix[i];
  // if(weight) line->GetSigma2P0(covariance);

  float determinant{ line.cosinesDirector[2] * line.cosinesDirector[2] * covariance[0] * covariance[1] +
                     line.cosinesDirector[1] * line.cosinesDirector[1] * covariance[0] * covariance[2] +
                     line.cosinesDirector[0] * line.cosinesDirector[0] * covariance[1] * covariance[2] };

  mAMatrix[0] += (line.cosinesDirector[2] * line.cosinesDirector[2] * covariance[1] +
                  line.cosinesDirector[1] * line.cosinesDirector[1] * covariance[2]) /
                 determinant;
  mAMatrix[1] += -line.cosinesDirector[0] * line.cosinesDirector[1] * covariance[2] / determinant;
  mAMatrix[2] += -line.cosinesDirector[0] * line.cosinesDirector[2] * covariance[1] / determinant;
  mAMatrix[3] += (line.cosinesDirector[2] * line.cosinesDirector[2] * covariance[0] +
                  line.cosinesDirector[0] * line.cosinesDirector[0] * covariance[2]) /
                 determinant;
  mAMatrix[4] += -line.cosinesDirector[1] * line.cosinesDirector[2] * covariance[0] / determinant;
  mAMatrix[5] += (line.cosinesDirector[1] * line.cosinesDirector[1] * covariance[0] +
                  line.cosinesDirector[0] * line.cosinesDirector[0] * covariance[1]) /
                 determinant;

  mBMatrix[0] += (line.cosinesDirector[1] * covariance[2] *
                    (-line.cosinesDirector[1] * line.originPoint[0] + line.cosinesDirector[0] * line.originPoint[1]) +
                  line.cosinesDirector[2] * covariance[1] *
                    (-line.cosinesDirector[2] * line.originPoint[0] + line.cosinesDirector[0] * line.originPoint[2])) /
                 determinant;
  mBMatrix[1] += (line.cosinesDirector[0] * covariance[2] *
                    (-line.cosinesDirector[0] * line.originPoint[1] + line.cosinesDirector[1] * line.originPoint[0]) +
                  line.cosinesDirector[2] * covariance[0] *
                    (-line.cosinesDirector[2] * line.originPoint[1] + line.cosinesDirector[1] * line.originPoint[2])) /
                 determinant;
  mBMatrix[2] += (line.cosinesDirector[0] * covariance[1] *
                    (-line.cosinesDirector[0] * line.originPoint[2] + line.cosinesDirector[2] * line.originPoint[0]) +
                  line.cosinesDirector[1] * covariance[0] *
                    (-line.cosinesDirector[1] * line.originPoint[2] + line.cosinesDirector[2] * line.originPoint[1])) /
                 determinant;

  computeClusterCentroid();
}

void ClusterLines::computeClusterCentroid()
{

  float determinant{ mAMatrix[0] * (mAMatrix[3] * mAMatrix[5] - mAMatrix[4] * mAMatrix[4]) -
                     mAMatrix[1] * (mAMatrix[1] * mAMatrix[5] - mAMatrix[4] * mAMatrix[2]) +
                     mAMatrix[2] * (mAMatrix[1] * mAMatrix[4] - mAMatrix[2] * mAMatrix[3]) };

  if (determinant == 0) {
    return;
  }

  mVertex[0] = -(mBMatrix[0] * (mAMatrix[3] * mAMatrix[5] - mAMatrix[4] * mAMatrix[4]) -
                 mAMatrix[1] * (mBMatrix[1] * mAMatrix[5] - mAMatrix[4] * mBMatrix[2]) +
                 mAMatrix[2] * (mBMatrix[1] * mAMatrix[4] - mBMatrix[2] * mAMatrix[3])) /
               determinant;
  mVertex[1] = -(mAMatrix[0] * (mBMatrix[1] * mAMatrix[5] - mBMatrix[2] * mAMatrix[4]) -
                 mBMatrix[0] * (mAMatrix[1] * mAMatrix[5] - mAMatrix[4] * mAMatrix[2]) +
                 mAMatrix[2] * (mAMatrix[1] * mBMatrix[2] - mAMatrix[2] * mBMatrix[1])) /
               determinant;
  mVertex[2] = -(mAMatrix[0] * (mAMatrix[3] * mBMatrix[2] - mBMatrix[1] * mAMatrix[4]) -
                 mAMatrix[1] * (mAMatrix[1] * mBMatrix[2] - mBMatrix[1] * mAMatrix[2]) +
                 mBMatrix[0] * (mAMatrix[1] * mAMatrix[4] - mAMatrix[2] * mAMatrix[3])) /
               determinant;
}

} // namespace CA
} // namespace ITS
} // namespace o2
