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