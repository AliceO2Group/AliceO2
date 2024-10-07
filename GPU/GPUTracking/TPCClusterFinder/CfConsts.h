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

/// \file CfConsts.h
/// \author Felix Weiglhofer

#ifndef O2_GPU_CF_CONSTS_H
#define O2_GPU_CF_CONSTS_H

#include "clusterFinderDefs.h"

namespace GPUCA_NAMESPACE
{
namespace gpu
{
namespace cfconsts
{

GPUconstexpr() tpccf::Delta2 InnerNeighbors[8] =
  {
    {-1, -1},

    {-1, 0},

    {-1, 1},

    {0, -1},

    {0, 1},
    {1, -1},
    {1, 0},
    {1, 1}};

GPUconstexpr() bool InnerTestEq[8] =
  {
    true, true, true, true,
    false, false, false, false};

GPUconstexpr() tpccf::Delta2 OuterNeighbors[16] =
  {
    {-2, -1},
    {-2, -2},
    {-1, -2},

    {-2, 0},

    {-2, 1},
    {-2, 2},
    {-1, 2},

    {0, -2},

    {0, 2},

    {2, -1},
    {2, -2},
    {1, -2},

    {2, 0},

    {2, 1},
    {2, 2},
    {1, 2}};

GPUconstexpr() uint8_t OuterToInner[16] =
  {
    0, 0, 0,

    1,

    2, 2, 2,

    3,

    4,

    5, 5, 5,

    6,

    7, 7, 7};

// outer to inner mapping change for the peak counting step,
// as the other position is the position of the peak
GPUconstexpr() uint8_t OuterToInnerInv[16] =
  {
    1,
    0,
    3,
    1,
    1,
    2,
    4,
    3,
    4,
    6,
    5,
    3,
    6,
    6,
    7,
    4};

#define NOISE_SUPPRESSION_NEIGHBOR_NUM 34

GPUconstexpr() tpccf::Delta2 NoiseSuppressionNeighbors[NOISE_SUPPRESSION_NEIGHBOR_NUM] =
  {
    {-2, -3},
    {-2, -2},
    {-2, -1},
    {-2, 0},
    {-2, 1},
    {-2, 2},
    {-2, 3},

    {-1, -3},
    {-1, -2},
    {-1, -1},
    {-1, 0},
    {-1, 1},
    {-1, 2},
    {-1, 3},

    {0, -3},
    {0, -2},
    {0, -1},

    {0, 1},
    {0, 2},
    {0, 3},

    {1, -3},
    {1, -2},
    {1, -1},
    {1, 0},
    {1, 1},
    {1, 2},
    {1, 3},

    {2, -3},
    {2, -2},
    {2, -1},
    {2, 0},
    {2, 1},
    {2, 2},
    {2, 3}};

GPUconstexpr() uint32_t NoiseSuppressionMinima[NOISE_SUPPRESSION_NEIGHBOR_NUM] =
  {
    (1 << 8) | (1 << 9),
    (1 << 9),
    (1 << 9),
    (1 << 10),
    (1 << 11),
    (1 << 11),
    (1 << 11) | (1 << 12),
    (1 << 8) | (1 << 9),
    (1 << 9),
    0,
    0,
    0,
    (1 << 11),
    (1 << 11) | (1 << 12),
    (1 << 15) | (1 << 16),
    (1 << 16),
    0,
    0,
    (1 << 17),
    (1 << 17) | (1 << 18),
    (1 << 21) | (1 << 22),
    (1 << 22),
    0,
    0,
    0,
    (1 << 24),
    (1 << 24) | (1 << 25),
    (1 << 21) | (1 << 22),
    (1 << 22),
    (1 << 22),
    (1 << 23),
    (1 << 24),
    (1 << 24),
    (1 << 24) | (1 << 25)};

} // namespace cfconsts
} // namespace gpu
} // namespace GPUCA_NAMESPACE

#endif
