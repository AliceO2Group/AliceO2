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

/// \file PadPEM.cxx
/// \brief Pads representation and transformation
///
/// \author Gilles Grasseau, Subatech

#include <stdexcept>
#include <cstring>

#include "MCHClustering/PadsPEM.h"

namespace o2
{
namespace mch
{
Pads::Pads(int N, int chId, int mode_)
{
  nPads = N;
  mode = mode_;
  chamberId = chId;
  allocate();
}

Pads::Pads(const Pads& pads, int mode_)
{
  nPads = pads.nPads;
  mode = mode_;
  chamberId = pads.chamberId;
  allocate();
  if (mode == pads.mode) {
    if (mode == xydxdyMode) {
      memcpy(x, pads.x, sizeof(double) * nPads);
      memcpy(y, pads.y, sizeof(double) * nPads);
      memcpy(dx, pads.dx, sizeof(double) * nPads);
      memcpy(dy, pads.dy, sizeof(double) * nPads);
      memcpy(q, pads.q, sizeof(double) * nPads);
    } else {
      memcpy(xInf, pads.xInf, sizeof(double) * nPads);
      memcpy(yInf, pads.yInf, sizeof(double) * nPads);
      memcpy(xSup, pads.xSup, sizeof(double) * nPads);
      memcpy(ySup, pads.ySup, sizeof(double) * nPads);
      memcpy(q, pads.q, sizeof(double) * nPads);
    }
  } else if (mode == xydxdyMode) {
    //  xyInfSupMode ->  xydxdyMode
    for (int i = 0; i < nPads; i++) {
      dx[i] = 0.5 * (pads.xSup[i] - pads.xInf[i]);
      dy[i] = 0.5 * (pads.ySup[i] - pads.yInf[i]);
      x[i] = pads.xInf[i] + dx[i];
      y[i] = pads.yInf[i] + dy[i];
    }
    memcpy(q, pads.q, sizeof(double) * nPads);
  } else {
    // xydxdyMode -> xyInfSupMode
    for (int i = 0; i < nPads; i++) {
      xInf[i] = pads.x[i] - pads.dx[i];
      xSup[i] = pads.x[i] + pads.dx[i];
      yInf[i] = pads.y[i] - pads.dy[i];
      ySup[i] = pads.y[i] + pads.dy[i];
    }
    memcpy(q, pads.q, sizeof(double) * nPads);
  }
  memcpy(saturate, pads.saturate, sizeof(Mask_t) * nPads);
}

Pads::Pads(const double* x_, const double* y_, const double* dx_, const double* dy_, const double* q_, const Mask_t* saturate_, int chId, int nPads_)
{
  mode = xydxdyMode;
  nPads = nPads_;
  chamberId = chId;
  allocate();
  // Copy pads
  memcpy(x, x_, sizeof(double) * nPads);
  memcpy(y, y_, sizeof(double) * nPads);
  memcpy(dx, dx_, sizeof(double) * nPads);
  memcpy(dy, dy_, sizeof(double) * nPads);
  memcpy(q, q_, sizeof(double) * nPads);
  if (saturate_ != nullptr) {
    memcpy(saturate, saturate_, sizeof(Mask_t) * nPads);
  }
  // ??? To remove memcpy ( pads->saturate, pads1.saturate, sizeof(double)*N1 );
}

Pads::Pads(const Pads& pads1, const Pads& pads2, int mode_)
{
  int N1 = pads1.nPads;
  int N2 = pads2.nPads;
  nPads = N1 + N2;
  chamberId = pads1.chamberId;
  mode = mode_;
  allocate();
  if (mode == xydxdyMode) {
    // Copy pads1
    memcpy(x, pads1.x, sizeof(double) * N1);
    memcpy(y, pads1.y, sizeof(double) * N1);
    memcpy(dx, pads1.dx, sizeof(double) * N1);
    memcpy(dy, pads1.dy, sizeof(double) * N1);
    memcpy(q, pads1.q, sizeof(double) * N1);
    memcpy(saturate, pads1.saturate, sizeof(double) * N1);
    // Copy pads2
    memcpy(&x[N1], pads2.x, sizeof(double) * N2);
    memcpy(&y[N1], pads2.y, sizeof(double) * N2);
    memcpy(&dx[N1], pads2.dx, sizeof(double) * N2);
    memcpy(&dy[N1], pads2.dy, sizeof(double) * N2);
    memcpy(&q[N1], pads2.q, sizeof(double) * N2);
    memcpy(&saturate[N1], pads2.saturate, sizeof(double) * N2);
  } else
    for (int i = 0; i < N1; i++) {
      xInf[i] = pads1.x[i] - pads1.dx[i];
      xSup[i] = pads1.x[i] + pads1.dx[i];
      yInf[i] = pads1.y[i] - pads1.dy[i];
      ySup[i] = pads1.y[i] + pads1.dy[i];
      q[i] = pads1.q[i];
      saturate[i] = pads1.saturate[i];
    }
  for (int i = 0; i < N2; i++) {
    xInf[i + N1] = pads2.x[i] - pads2.dx[i];
    xSup[i + N1] = pads2.x[i] + pads2.dx[i];
    yInf[i + N1] = pads2.y[i] - pads2.dy[i];
    ySup[i + N1] = pads2.y[i] + pads2.dy[i];
    q[i + N1] = pads2.q[i];
    saturate[i + N1] = pads2.saturate[i];
  }
}

void Pads::removePad(int index)
{

  if ((index < 0) || (index >= nPads))
    return;
  int nItems = nPads - index;
  if (index == nPads - 1) {
    nPads = nPads - 1;
    return;
  }
  if (mode == xydxdyMode) {
    vectorCopy(&x[index + 1], nItems, &x[index]);
    vectorCopy(&y[index + 1], nItems, &y[index]);
    vectorCopy(&dx[index + 1], nItems, &dx[index]);
    vectorCopy(&dy[index + 1], nItems, &dy[index]);
  } else {
    vectorCopy(&xInf[index + 1], nItems, &xInf[index]);
    vectorCopy(&yInf[index + 1], nItems, &yInf[index]);
    vectorCopy(&xSup[index + 1], nItems, &xSup[index]);
    vectorCopy(&ySup[index + 1], nItems, &ySup[index]);
  }
  vectorCopy(&q[index + 1], nItems, &q[index]);
  vectorCopyShort(&saturate[index + 1], nItems, &saturate[index]);

  nPads = nPads - 1;
}

void Pads::allocate()
{
  // Note: Must be deallocated/releases if required
  x = nullptr;
  y = nullptr;
  dx = nullptr;
  dy = nullptr;
  xInf = nullptr;
  xSup = nullptr;
  yInf = nullptr;
  ySup = nullptr;
  saturate = nullptr;
  q = nullptr;
  int N = nPads;
  if (mode == xydxdyMode) {
    x = new double[N];
    y = new double[N];
    dx = new double[N];
    dy = new double[N];
  } else {
    xInf = new double[N];
    xSup = new double[N];
    yInf = new double[N];
    ySup = new double[N];
  }
  saturate = new Mask_t[N];
  cath = new Mask_t[N];
  q = new double[N];
}

void Pads::setToZero()
{
  if (mode == xydxdyMode) {
    for (int i = 0; i < nPads; i++) {
      x[i] = 0.0;
      y[i] = 0.0;
      dx[i] = 0.0;
      dy[i] = 0.0;
      q[i] = 0.0;
    }
  } else {
    for (int i = 0; i < nPads; i++) {
      xInf[i] = 0.0;
      ySup[i] = 0.0;
      yInf[i] = 0.0;
      yInf[i] = 0.0;
      q[i] = 0.0;
    }
  }
}

void Pads::release()
{
  if (mode == xydxdyMode) {
    if (x != nullptr) {
      delete[] x;
      x = nullptr;
    }
    if (y != nullptr) {
      delete[] y;
      y = nullptr;
    }
    if (dx != nullptr) {
      delete[] dx;
      dx = nullptr;
    }
    if (dy != nullptr) {
      delete[] dy;
      dy = nullptr;
    }
  } else {
    if (xInf != nullptr) {
      delete[] xInf;
      xInf = nullptr;
    }
    if (xSup != nullptr) {
      delete[] xSup;
      xSup = nullptr;
    }
    if (yInf != nullptr) {
      delete[] yInf;
      yInf = nullptr;
    }
    if (ySup != nullptr) {
      delete[] ySup;
      ySup = nullptr;
    }
  }
  if (q != nullptr) {
    delete[] q;
    q = nullptr;
  }
  if (cath != nullptr) {
    delete[] cath;
    cath = nullptr;
  }
  if (saturate != nullptr) {
    delete[] saturate;
    saturate = nullptr;
  }
  nPads = 0;
}

void Pads::display(const char* str)
{
  printf("%s\n", str);
  printf("  nPads=%d, mode=%d, chId=%d \n", nPads, mode, chamberId);
  if (mode == xydxdyMode) {
    vectorPrint("  x", x, nPads);
    vectorPrint("  y", y, nPads);
    vectorPrint("  dx", dx, nPads);
    vectorPrint("  dy", dy, nPads);
  } else {
    vectorPrint("  xInf", xInf, nPads);
    vectorPrint("  xSup", xSup, nPads);
    vectorPrint("  yInf", yInf, nPads);
    vectorPrint("  ySup", ySup, nPads);
  }
  vectorPrint("  q", q, nPads);
}

Pads::~Pads()
{
  release();
}

} // namespace mch
} // namespace o2
