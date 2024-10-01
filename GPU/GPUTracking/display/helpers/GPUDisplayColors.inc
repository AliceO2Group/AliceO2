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

/// \file GPUDisplayColors.cxx
/// \author David Rohr

#include "GPUDisplay.h"

using namespace GPUCA_NAMESPACE::gpu;

inline void GPUDisplay::ActivateColor()
{
  mBackend->ActivateColor(mDrawColor);
}

inline void GPUDisplay::SetColorClusters()
{
  if (mCfgL.colorCollisions) {
    return;
  }
  if (mCfgL.invertColors) {
    mDrawColor = {0, 0.3, 0.7, 1.f};
  } else {
    mDrawColor = {0, 0.7, 1.0, 1.f};
  }
  ActivateColor();
}
inline void GPUDisplay::SetColorTRD()
{
  if (mCfgL.colorCollisions) {
    return;
  }
  if (mCfgL.invertColors) {
    mDrawColor = {0.7, 0.3, 0, 1.f};
  } else {
    mDrawColor = {1.0, 0.7, 0, 1.f};
  }
  ActivateColor();
}
inline void GPUDisplay::SetColorITS()
{
  if (mCfgL.colorCollisions) {
    return;
  }
  if (mCfgL.invertColors) {
    mDrawColor = {1.00, 0.1, 0.1, 1.f};
  } else {
    mDrawColor = {1.00, 0.3, 0.3, 1.f};
  }
  ActivateColor();
}
inline void GPUDisplay::SetColorTOF()
{
  if (mCfgL.colorCollisions) {
    return;
  }
  if (mCfgL.invertColors) {
    mDrawColor = {0.1, 1.0, 0.1, 1.f};
  } else {
    mDrawColor = {0.5, 1.0, 0.5, 1.f};
  }
  ActivateColor();
}
inline void GPUDisplay::SetColorInitLinks()
{
  if (mCfgL.invertColors) {
    mDrawColor = {0.42, 0.4, 0.1, 1.f};
  } else {
    mDrawColor = {0.42, 0.4, 0.1, 1.f};
  }
  ActivateColor();
}
inline void GPUDisplay::SetColorLinks()
{
  if (mCfgL.invertColors) {
    mDrawColor = {0.6, 0.1, 0.1, 1.f};
  } else {
    mDrawColor = {0.8, 0.2, 0.2, 1.f};
  }
  ActivateColor();
}
inline void GPUDisplay::SetColorSeeds()
{
  if (mCfgL.invertColors) {
    mDrawColor = {0.6, 0.0, 0.65, 1.f};
  } else {
    mDrawColor = {0.8, 0.1, 0.85, 1.f};
  }
  ActivateColor();
}
inline void GPUDisplay::SetColorTracklets()
{
  if (mCfgL.invertColors) {
    mDrawColor = {0, 0, 0, 1.f};
  } else {
    mDrawColor = {1, 1, 1, 1.f};
  }
  ActivateColor();
}
inline void GPUDisplay::SetColorTracks()
{
  if (mCfgL.invertColors) {
    mDrawColor = {0.6, 0, 0.1, 1.f};
  } else {
    mDrawColor = {0.8, 1., 0.15, 1.f};
  }
  ActivateColor();
}
inline void GPUDisplay::SetColorGlobalTracks()
{
  if (mCfgL.invertColors) {
    mDrawColor = {0.8, 0.2, 0, 1.f};
  } else {
    mDrawColor = {1.0, 0.4, 0, 1.f};
  }
  ActivateColor();
}
inline void GPUDisplay::SetColorFinal()
{
  if (mCfgL.colorCollisions) {
    return;
  }
  if (mCfgL.invertColors) {
    mDrawColor = {0, 0.6, 0.1, 1.f};
  } else {
    mDrawColor = {0, 0.7, 0.2, 1.f};
  }
  ActivateColor();
}
inline void GPUDisplay::SetColorGrid()
{
  if (mCfgL.invertColors) {
    mDrawColor = {0.5, 0.5, 0.0, 1.f};
  } else {
    mDrawColor = {0.7, 0.7, 0.0, 1.f};
  }
  ActivateColor();
}
inline void GPUDisplay::SetColorGridTRD()
{
  if (mCfgL.invertColors) {
    mDrawColor = {0.5, 0.5, 0.5, 1.f};
  } else {
    mDrawColor = {0.7, 0.7, 0.5, 1.f};
  }
  ActivateColor();
}
inline void GPUDisplay::SetColorMarked()
{
  if (mCfgL.invertColors) {
    mDrawColor = {0.8, 0, 0, 1.f};
  } else {
    mDrawColor = {1.0, 0.0, 0.0, 1.f};
  }
  ActivateColor();
}
inline void GPUDisplay::SetCollisionColor(int col)
{
  int red = (col * 2) % 5;
  int blue = (2 + col * 3) % 7;
  int green = (4 + col * 5) % 6;
  if (mCfgL.invertColors && red == 4 && blue == 5 && green == 6) {
    red = 0;
  }
  if (!mCfgL.invertColors && red == 0 && blue == 0 && green == 0) {
    red = 4;
  }
  mDrawColor = {red / 4.f, green / 5.f, blue / 6.f, 1.f};
  ActivateColor();
}
