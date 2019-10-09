// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

#include "ITStrackingCUDA/ClusterLinesGPU.h"

namespace o2
{
namespace its
{
namespace GPU
{

GPUd() ClusterLinesGPU::ClusterLinesGPU(const Line& firstLine, const Line& secondLine)
{
  float covarianceFirst[3];
  float covarianceSecond[3];

  for (int i{0}; i < 3; ++i) {
    covarianceFirst[i] = 1.f;
    covarianceSecond[i] = 1.f;
  }

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

  mBMatrix[0] =
    (firstLine.cosinesDirector[1] * covarianceFirst[2] * (-firstLine.cosinesDirector[1] * firstLine.originPoint[0] + firstLine.cosinesDirector[0] * firstLine.originPoint[1]) +
     firstLine.cosinesDirector[2] * covarianceFirst[1] * (-firstLine.cosinesDirector[2] * firstLine.originPoint[0] + firstLine.cosinesDirector[0] * firstLine.originPoint[2])) /
    determinantFirst;

  mBMatrix[0] +=
    (secondLine.cosinesDirector[1] * covarianceSecond[2] * (-secondLine.cosinesDirector[1] * secondLine.originPoint[0] + secondLine.cosinesDirector[0] * secondLine.originPoint[1]) +
     secondLine.cosinesDirector[2] * covarianceSecond[1] *
       (-secondLine.cosinesDirector[2] * secondLine.originPoint[0] +
        secondLine.cosinesDirector[0] * secondLine.originPoint[2])) /
    determinantSecond;

  mBMatrix[1] =
    (firstLine.cosinesDirector[0] * covarianceFirst[2] * (-firstLine.cosinesDirector[0] * firstLine.originPoint[1] + firstLine.cosinesDirector[1] * firstLine.originPoint[0]) +
     firstLine.cosinesDirector[2] * covarianceFirst[0] * (-firstLine.cosinesDirector[2] * firstLine.originPoint[1] + firstLine.cosinesDirector[1] * firstLine.originPoint[2])) /
    determinantFirst;

  mBMatrix[1] +=
    (secondLine.cosinesDirector[0] * covarianceSecond[2] * (-secondLine.cosinesDirector[0] * secondLine.originPoint[1] + secondLine.cosinesDirector[1] * secondLine.originPoint[0]) +
     secondLine.cosinesDirector[2] * covarianceSecond[0] *
       (-secondLine.cosinesDirector[2] * secondLine.originPoint[1] +
        secondLine.cosinesDirector[1] * secondLine.originPoint[2])) /
    determinantSecond;

  mBMatrix[2] =
    (firstLine.cosinesDirector[0] * covarianceFirst[1] * (-firstLine.cosinesDirector[0] * firstLine.originPoint[2] + firstLine.cosinesDirector[2] * firstLine.originPoint[0]) +
     firstLine.cosinesDirector[1] * covarianceFirst[0] * (-firstLine.cosinesDirector[1] * firstLine.originPoint[2] + firstLine.cosinesDirector[2] * firstLine.originPoint[1])) /
    determinantFirst;

  mBMatrix[2] +=
    (secondLine.cosinesDirector[0] * covarianceSecond[1] * (-secondLine.cosinesDirector[0] * secondLine.originPoint[2] + secondLine.cosinesDirector[2] * secondLine.originPoint[0]) +
     secondLine.cosinesDirector[1] * covarianceSecond[0] *
       (-secondLine.cosinesDirector[1] * secondLine.originPoint[2] +
        secondLine.cosinesDirector[2] * secondLine.originPoint[1])) /
    determinantSecond;

  computeClusterCentroid();
}

GPUd() ClusterLinesGPU::ClusterLinesGPU(const int& firstLabel, const Line& firstLine, const int& secondLabel, const Line& secondLine,
                                        const unsigned char weight) : mNContributors{2}
{
  mLabels[0] = firstLabel;
  mLabels[1] = secondLabel;

  float covarianceFirst[3];
  float covarianceSecond[3];

  for (int i{0}; i < 3; ++i) {
    covarianceFirst[i] = 1.f;
    covarianceSecond[i] = 1.f;
  }

  for (int i{0}; i < 6; ++i)
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

  mBMatrix[0] =
    (firstLine.cosinesDirector[1] * covarianceFirst[2] * (-firstLine.cosinesDirector[1] * firstLine.originPoint[0] + firstLine.cosinesDirector[0] * firstLine.originPoint[1]) +
     firstLine.cosinesDirector[2] * covarianceFirst[1] * (-firstLine.cosinesDirector[2] * firstLine.originPoint[0] + firstLine.cosinesDirector[0] * firstLine.originPoint[2])) /
    determinantFirst;

  mBMatrix[0] +=
    (secondLine.cosinesDirector[1] * covarianceSecond[2] * (-secondLine.cosinesDirector[1] * secondLine.originPoint[0] + secondLine.cosinesDirector[0] * secondLine.originPoint[1]) +
     secondLine.cosinesDirector[2] * covarianceSecond[1] *
       (-secondLine.cosinesDirector[2] * secondLine.originPoint[0] +
        secondLine.cosinesDirector[0] * secondLine.originPoint[2])) /
    determinantSecond;

  mBMatrix[1] =
    (firstLine.cosinesDirector[0] * covarianceFirst[2] * (-firstLine.cosinesDirector[0] * firstLine.originPoint[1] + firstLine.cosinesDirector[1] * firstLine.originPoint[0]) +
     firstLine.cosinesDirector[2] * covarianceFirst[0] * (-firstLine.cosinesDirector[2] * firstLine.originPoint[1] + firstLine.cosinesDirector[1] * firstLine.originPoint[2])) /
    determinantFirst;

  mBMatrix[1] +=
    (secondLine.cosinesDirector[0] * covarianceSecond[2] * (-secondLine.cosinesDirector[0] * secondLine.originPoint[1] + secondLine.cosinesDirector[1] * secondLine.originPoint[0]) +
     secondLine.cosinesDirector[2] * covarianceSecond[0] *
       (-secondLine.cosinesDirector[2] * secondLine.originPoint[1] +
        secondLine.cosinesDirector[1] * secondLine.originPoint[2])) /
    determinantSecond;

  mBMatrix[2] =
    (firstLine.cosinesDirector[0] * covarianceFirst[1] * (-firstLine.cosinesDirector[0] * firstLine.originPoint[2] + firstLine.cosinesDirector[2] * firstLine.originPoint[0]) +
     firstLine.cosinesDirector[1] * covarianceFirst[0] * (-firstLine.cosinesDirector[1] * firstLine.originPoint[2] + firstLine.cosinesDirector[2] * firstLine.originPoint[1])) /
    determinantFirst;

  mBMatrix[2] +=
    (secondLine.cosinesDirector[0] * covarianceSecond[1] * (-secondLine.cosinesDirector[0] * secondLine.originPoint[2] + secondLine.cosinesDirector[2] * secondLine.originPoint[0]) +
     secondLine.cosinesDirector[1] * covarianceSecond[0] *
       (-secondLine.cosinesDirector[1] * secondLine.originPoint[2] +
        secondLine.cosinesDirector[2] * secondLine.originPoint[1])) /
    determinantSecond;

  computeClusterCentroid();

  Line::getDCAComponents(firstLine, mVertex, mRMS2);
  float tmpRMS2Line2[6];
  Line::getDCAComponents(secondLine, mVertex, tmpRMS2Line2);

  for (int ii{0}; ii < 6; ++ii) {
    mRMS2[ii] += (tmpRMS2Line2[ii] - mRMS2[ii]) / mNContributors;
  }
}

GPUd() void ClusterLinesGPU::add(const int& lineLabel, const Line& line, const unsigned char weight)
{
  mNContributors++;
  float covariance[3];
  for (int ii{0}; ii < 3; ++ii) {
    covariance[ii] = 1.f;
  }

  for (int i{0}; i < 6; ++i)
    mWeightMatrix[i] += line.weightMatrix[i];
  // if(weight) line->GetSigma2P0(covariance);

  float determinant{line.cosinesDirector[2] * line.cosinesDirector[2] * covariance[0] * covariance[1] +
                    line.cosinesDirector[1] * line.cosinesDirector[1] * covariance[0] * covariance[2] +
                    line.cosinesDirector[0] * line.cosinesDirector[0] * covariance[1] * covariance[2]};

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

  mAvgDistance2 += (Line::getDistanceFromPoint(line, mVertex) * Line::getDistanceFromPoint(line, mVertex) - mAvgDistance2) / mNContributors;
}

GPUd() void ClusterLinesGPU::computeClusterCentroid()
{

  float determinant{mAMatrix[0] * (mAMatrix[3] * mAMatrix[5] - mAMatrix[4] * mAMatrix[4]) -
                    mAMatrix[1] * (mAMatrix[1] * mAMatrix[5] - mAMatrix[4] * mAMatrix[2]) +
                    mAMatrix[2] * (mAMatrix[1] * mAMatrix[4] - mAMatrix[2] * mAMatrix[3])};

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
} // namespace GPU
} // namespace its
} // namespace o2