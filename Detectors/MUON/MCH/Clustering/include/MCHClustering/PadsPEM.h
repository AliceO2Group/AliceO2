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

/// \file PadPEM.h
/// \brief Pads representation and transformation
///
/// \author Gilles Grasseau, Subatech

#ifndef ALICEO2_MCH_PADSPEM_H_
#define ALICEO2_MCH_PADSPEM_H_

#include "MCHClustering/dataStructure.h"
namespace o2
{
namespace mch
{
struct Pads {
  enum padMode {
    xydxdyMode = 0x0,  ///< x, y, dx, dy pad coordinates
    xyInfSupMode = 0x1 ///< xInf, xSup, yInf, ySup pad coordinates
  };
  // Representation mode  (see padMode)
  int mode;
  // Mode xydxdy
  double* x;
  double* y;
  double* dx;
  double* dy;
  // Mode xyInfSupMode
  double *xInf, *xSup;
  double *yInf, *ySup;
  Mask_t* cath;
  Mask_t* saturate;
  double* q;
  int nPads;
  int chamberId;

  Pads(int N, int chId, int mode = xydxdyMode);
  Pads(const Pads& pads, int mode_);
  Pads(const Pads& pads1, const Pads& pads2, int mode);
  Pads(const double* x_, const double* y_, const double* dx_, const double* dy_, const double* q_, const Mask_t* saturate, int chId, int nPads_);
  void removePad(int index);
  ~Pads();
  void allocate();
  void setToZero();
  void display(const char* str);
  void release();
};

} // namespace mch
} // namespace o2

#endif // ALICEO2_MCH_PADSPEM_H_
