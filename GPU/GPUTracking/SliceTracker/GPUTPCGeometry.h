// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

/// \file GPUTPCGeometry.h
/// \author David Rohr, Jens Wiechula, Sergey Gorbunov

#ifndef GPUTPCGEOMETRY_H
#define GPUTPCGEOMETRY_H

#include "GPUCommonDef.h"

namespace GPUCA_NAMESPACE
{
namespace gpu
{
// Copy of TPC constants from AliRoot:TPCGeometry / O2:TPC/Base/Mapper
// Should be unified, but cannot take the contants from the official headers for now, since we want it to be constexpr
class GPUTPCGeometry
{
#if defined(__OPENCL__) && !defined(__OPENCLCPP__)
  GPUTPCGeometry(); // Fake constructor declaration for OpenCL due to static members, does not exist!
#endif
#ifdef GPUCA_TPC_GEOMETRY_O2
  const float mX[GPUCA_ROW_COUNT] GPUCA_CPP11_INIT(= { 85.225f, 85.975f, 86.725f, 87.475f, 88.225f, 88.975f, 89.725f, 90.475f, 91.225f, 91.975f, 92.725f, 93.475f, 94.225f, 94.975f, 95.725f, 96.475f, 97.225f, 97.975f, 98.725f, 99.475f, 100.225f, 100.975f,
                                                       101.725f, 102.475f, 103.225f, 103.975f, 104.725f, 105.475f, 106.225f, 106.975f, 107.725f, 108.475f, 109.225f, 109.975f, 110.725f, 111.475f, 112.225f, 112.975f, 113.725f, 114.475f, 115.225f, 115.975f, 116.725f, 117.475f,
                                                       118.225f, 118.975f, 119.725f, 120.475f, 121.225f, 121.975f, 122.725f, 123.475f, 124.225f, 124.975f, 125.725f, 126.475f, 127.225f, 127.975f, 128.725f, 129.475f, 130.225f, 130.975f, 131.725f, 135.2f, 136.2f, 137.2f,
                                                       138.2f, 139.2f, 140.2f, 141.2f, 142.2f, 143.2f, 144.2f, 145.2f, 146.2f, 147.2f, 148.2f, 149.2f, 150.2f, 151.2f, 152.2f, 153.2f, 154.2f, 155.2f, 156.2f, 157.2f, 158.2f, 159.2f,
                                                       160.2f, 161.2f, 162.2f, 163.2f, 164.2f, 165.2f, 166.2f, 167.2f, 168.2f, 171.4f, 172.6f, 173.8f, 175.f, 176.2f, 177.4f, 178.6f, 179.8f, 181.f, 182.2f, 183.4f, 184.6f, 185.8f,
                                                       187.f, 188.2f, 189.4f, 190.6f, 191.8f, 193.f, 194.2f, 195.4f, 196.6f, 197.8f, 199.f, 200.2f, 201.4f, 202.6f, 203.8f, 205.f, 206.2f, 209.65f, 211.15f, 212.65f, 214.15f, 215.65f,
                                                       217.15f, 218.65f, 220.15f, 221.65f, 223.15f, 224.65f, 226.15f, 227.65f, 229.15f, 230.65f, 232.15f, 233.65f, 235.15f, 236.65f, 238.15f, 239.65f, 241.15f, 242.65f, 244.15f, 245.65f });

  const unsigned char mNPads[GPUCA_ROW_COUNT] GPUCA_CPP11_INIT(= { 66, 66, 66, 68, 68, 68, 70, 70, 70, 72, 72, 72, 74, 74, 74, 74, 76, 76, 76, 76, 78, 78, 78, 80, 80, 80, 82, 82, 82, 84, 84, 84, 86, 86, 86, 88, 88, 88,
                                                                   90, 90, 90, 90, 92, 92, 92, 94, 94, 94, 92, 92, 92, 94, 94, 94, 96, 96, 96, 98, 98, 98, 100, 100, 100, 76, 76, 76, 76, 78, 78, 78, 80, 80, 80, 80, 82, 82,
                                                                   82, 84, 84, 84, 84, 86, 86, 86, 88, 88, 88, 90, 90, 90, 90, 92, 92, 92, 94, 94, 94, 94, 96, 96, 96, 98, 98, 98, 100, 100, 102, 102, 102, 104, 104, 104, 106, 110,
                                                                   110, 112, 112, 112, 114, 114, 114, 116, 116, 116, 118, 118, 118, 118, 118, 120, 120, 122, 122, 124, 124, 124, 126, 126, 128, 128, 128, 130, 130, 132, 132, 132, 134, 134, 136, 136, 138, 138 });

  const unsigned char mRegion[GPUCA_ROW_COUNT] GPUCA_CPP11_INIT(= { 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4,
                                                                    4, 4, 4, 4, 4, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9 });

  const float mPadHeight[10] GPUCA_CPP11_INIT(= { .75f, .75f, .75f, .75f, 1.f, 1.f, 1.2f, 1.2f, 1.5f, 1.5f });
  const float mPadWidth[10] GPUCA_CPP11_INIT(= { .416f, .420f, .420f, .436f, .6f, .6f, .608f, .588f, .604f, .607f });
  GPUd() int GetRegion(int row) const { return mRegion[row]; }

#if !defined(__OPENCL__) || defined(__OPENCLCPP__)
  static CONSTEXPR float FACTOR_T2Z GPUCA_CPP11_INIT(= 250.f / 512.f);
#endif

 public:
  GPUd() int GetROC(int row) const { return row < 97 ? (row < 63 ? 0 : 1) : (row < 127 ? 2 : 3); }
  GPUd() int EndIROC() const { return 63; }
  GPUd() int EndOROC1() const { return 97; }
  GPUd() int EndOROC2() const { return 127; }
#else
  const float mX[GPUCA_ROW_COUNT] GPUCA_CPP11_INIT(= { 85.195f, 85.945f, 86.695f, 87.445f, 88.195f, 88.945f, 89.695f, 90.445f, 91.195f, 91.945f, 92.695f, 93.445f, 94.195f, 94.945f, 95.695f, 96.445f, 97.195f, 97.945f, 98.695f, 99.445f, 100.195f, 100.945f, 101.695f,
                                                       102.445f, 103.195f, 103.945f, 104.695f, 105.445f, 106.195f, 106.945f, 107.695f, 108.445f, 109.195f, 109.945f, 110.695f, 111.445f, 112.195f, 112.945f, 113.695f, 114.445f, 115.195f, 115.945f, 116.695f, 117.445f, 118.195f, 118.945f,
                                                       119.695f, 120.445f, 121.195f, 121.945f, 122.695f, 123.445f, 124.195f, 124.945f, 125.695f, 126.445f, 127.195f, 127.945f, 128.695f, 129.445f, 130.195f, 130.945f, 131.695f, 135.180f, 136.180f, 137.180f, 138.180f, 139.180f, 140.180f,
                                                       141.180f, 142.180f, 143.180f, 144.180f, 145.180f, 146.180f, 147.180f, 148.180f, 149.180f, 150.180f, 151.180f, 152.180f, 153.180f, 154.180f, 155.180f, 156.180f, 157.180f, 158.180f, 159.180f, 160.180f, 161.180f, 162.180f, 163.180f,
                                                       164.180f, 165.180f, 166.180f, 167.180f, 168.180f, 169.180f, 170.180f, 171.180f, 172.180f, 173.180f, 174.180f, 175.180f, 176.180f, 177.180f, 178.180f, 179.180f, 180.180f, 181.180f, 182.180f, 183.180f, 184.180f, 185.180f, 186.180f,
                                                       187.180f, 188.180f, 189.180f, 190.180f, 191.180f, 192.180f, 193.180f, 194.180f, 195.180f, 196.180f, 197.180f, 198.180f, 199.430f, 200.930f, 202.430f, 203.930f, 205.430f, 206.930f, 208.430f, 209.930f, 211.430f, 212.930f, 214.430f,
                                                       215.930f, 217.430f, 218.930f, 220.430f, 221.930f, 223.430f, 224.930f, 226.430f, 227.930f, 229.430f, 230.930f, 232.430f, 233.930f, 235.430f, 236.930f, 238.430f, 239.930f, 241.430f, 242.930f, 244.430f, 245.930f });

  const unsigned char mNPads[GPUCA_ROW_COUNT] GPUCA_CPP11_INIT(= { 68, 68, 68, 68, 70, 70, 70, 72, 72, 72, 74, 74, 74, 76, 76, 76, 78, 78, 78, 80, 80, 80, 82, 82, 82, 84, 84, 84, 86, 86, 86, 88, 88, 88, 90, 90, 90, 92, 92, 92,
                                                                   94, 94, 94, 96, 96, 96, 98, 98, 98, 100, 100, 100, 102, 102, 102, 104, 104, 104, 106, 106, 106, 108, 108, 74, 76, 76, 76, 76, 78, 78, 78, 80, 80, 80, 80, 82, 82, 82, 84, 84,
                                                                   84, 86, 86, 86, 86, 88, 88, 88, 90, 90, 90, 90, 92, 92, 92, 94, 94, 94, 96, 96, 96, 96, 98, 98, 98, 100, 100, 100, 100, 102, 102, 102, 104, 104, 104, 106, 106, 106, 106, 108,
                                                                   108, 108, 110, 110, 110, 110, 112, 112, 114, 114, 114, 116, 116, 118, 118, 120, 120, 122, 122, 122, 124, 124, 126, 126, 128, 128, 130, 130, 130, 132, 132, 134, 134, 136, 136, 138, 138, 138, 140 });

  GPUd() int GetRegion(int row) const { return (row < 63 ? 0 : row < 63 + 64 ? 1 : 2); }
  const float mPadHeight[3] GPUCA_CPP11_INIT(= { .75f, 1.f, 1.5f });
  const float mPadWidth[3] GPUCA_CPP11_INIT(= { .4f, .6f, .6f });

#if !defined(__OPENCL__) || defined(__OPENCLCPP__)
  static CONSTEXPR float FACTOR_T2Z GPUCA_CPP11_INIT(= 250.f / 1024.f);
#endif

 public:
  GPUd() int GetROC(int row) const { return GetRegion(row); }
  GPUd() int EndIROC() const { return 63; }
  GPUd() int EndOROC1() const { return 63 + 64; }
  GPUd() int EndOROC2() const { return GPUCA_ROW_COUNT; }
#endif
 private:
#if !defined(__OPENCL__) || defined(__OPENCLCPP__)
  static CONSTEXPR float FACTOR_Z2T GPUCA_CPP11_INIT(= 1.f / FACTOR_T2Z);
#endif
 public:
  GPUd() float Row2X(int row) const { return (mX[row]); }
  GPUd() float PadHeight(int row) const { return (mPadHeight[GetRegion(row)]); }
  GPUd() float PadWidth(int row) const { return (mPadWidth[GetRegion(row)]); }
  GPUd() unsigned char NPads(int row) const { return mNPads[row]; }

#if !defined(__OPENCL__) || defined(__OPENCLCPP__)
  GPUd() float LinearPad2Y(int slice, int row, float pad) const
  {
    const float u = (pad - 0.5 * mNPads[row]) * PadWidth(row);
    return (slice >= GPUCA_NSLICES / 2) ? -u : u;
  }

  GPUd() static float LinearTime2Z(int slice, float time)
  {
    const float v = 250.f - time * FACTOR_T2Z;
    return (slice >= GPUCA_NSLICES / 2) ? -v : v;
  }

  GPUd() float LinearY2Pad(int slice, int row, float y) const
  {
    const float u = (slice >= GPUCA_NSLICES / 2) ? -y : y;
    return u / PadWidth(row) + 0.5 * mNPads[row];
  }

  GPUd() static float LinearZ2Time(int slice, float z)
  {
    const float v = (slice >= GPUCA_NSLICES / 2) ? -z : z;
    return (250.f - v) * FACTOR_Z2T;
  }
#endif
};
} // namespace gpu
} // namespace GPUCA_NAMESPACE
#endif
