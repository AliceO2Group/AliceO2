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

/// \file SpatialPhotonResponse.h
/// \brief Visualizing spatial photon response in ZDC neutron and proton calorimeters

#ifndef DETECTORS_ZDC_SIMULATION_INCLUDE_ZDCSIMULATION_SPATIALPHOTONRESPONSE_H_
#define DETECTORS_ZDC_SIMULATION_INCLUDE_ZDCSIMULATION_SPATIALPHOTONRESPONSE_H_

#include <array>
#include <vector>

namespace o2
{
namespace zdc
{

static constexpr int ZDCCHANNELSPERTOWER = 1;

/// Class representing the spatial photon response in a ZDC
/// calorimeter
class SpatialPhotonResponse
{
 public:
  // Nx number of pixels in x direction
  // Ny number of pixels in y direction
  SpatialPhotonResponse(int Nx, int Ny, double lowerx, double lowery, double lengthx, double lengthy);

  SpatialPhotonResponse() = default;

  void addPhoton(double x, double y, int nphotons);
  // Adds photon to the image (as addPhoton does) but it takes pixel coordinates as arguments.
  void addPhotonByPixel(int xpixel, int ypixel, int nphotons);
  // void exportToPNG() const;
  void printToScreen() const;
  void reset();

  double getCellLx() const { return mLxOfCell; }
  double getCellLy() const { return mLyOfCell; }

  double getNx() const { return mNx; }
  double getNy() const { return mNy; }
  double getLowerX() const { return mLowerX; }
  double getLowerY() const { return mLowerY; }

  bool hasSignal() const { return mPhotonSum > 0; }
  int getPhotonSum() const { return mPhotonSum; }
  void setDetectorID(int det)
  {
    if (mDetectorID != -1 && det != mDetectorID) {
      printErrMsg("trying to change detector ID");
    }
    mDetectorID = det;
  }
  int getDetectorID() const { return mDetectorID; }
  void setHitTime(float t) { mTime = t; }
  float getHitTime() const { return mTime; }

  std::array<int, 5> getPhotonsPerChannel() const;

  std::vector<std::vector<int>> const& getImageData() const { return mImageData; }

 private:
  void printErrMsg(const char* mgs) const;

  double mLxOfCell = 1.; // x length of cell corresponding to one pixel
  double mLyOfCell = 1.; // y length of cell corresponding to one pixel

  double mInvLxOfCell = 1.; // x length of cell corresponding to one pixel
  double mInvLyOfCell = 1.; // y length of cell corresponding to one pixel

  double mLowerX = 0.; // lowerX coordinate of image (used to convert to pixels in addPhoton)
  double mLowerY = 0.; // lowerY coordinate of image (used to convert to pixels in addPhoton)

  int mNx = 1;        // number of "towers" in x direction
  int mNy = 1;        // number of "towers" in y direction
  int mPhotonSum = 0; // accumulated photon number (for quick filtering)

  int mDetectorID = -1; // the detectorID (ZNA, ZNC, ZPA, ZPC) to which this image corresponds (if available)

  float mTime = -1.; // the time (time difference between creation of response and the collission time); negative is unitialized

  std::vector<std::vector<int>> mImageData;
};

} // namespace zdc
} // namespace o2

#endif /* DETECTORS_ZDC_SIMULATION_INCLUDE_ZDCSIMULATION_SPATIALPHOTONRESPONSE_H_ */
