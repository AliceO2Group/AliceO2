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

/// \file ITS3Services.h
/// \brief Definition of the ITS3Services class
/// \author Fabrizio Grosa <fgrosa@cern.ch>

#ifndef ALICEO2_ITS3_ITS3SERVICES_H
#define ALICEO2_ITS3_ITS3SERVICES_H

class TGeoVolume;

namespace o2
{
namespace its3
{
/// This class defines the Geometry for the ITS3 services using TGeo.
class ITS3Services : public TObject
{
 public:
  // Default constructor
  ITS3Services() = default;

  /// Copy constructor
  ITS3Services(const ITS3Services&) = default;
  /// Assignment operator
  ITS3Services& operator=(const ITS3Services&) = default;

  /// Default destructor
  ~ITS3Services() override;

  /// Creates the CYSS Assembly (i.e. the supporting half cylinder and cone)
  TGeoVolume* createCYSSAssembly();

 private:
  /// Creates the CYSS cylinder of the Inner Barrel
  TGeoVolume* createCYSSCylinder();

  /// Creates the CYSS cone of the Inner Barrel
  TGeoVolume* createCYSSCone();

  /// Creates the CYSS Flange on Side A of the Inner Barrel
  TGeoVolume* createCYSSFlangeA();

  /// Creates the CYSS Flange on Side C of the Inner Barrel
  TGeoVolume* createCYSSFlangeC();

  /// Creates the hollows in the CYSS Flange on Side A of the Inner Barrel
  /// \param zlen the thickness of the ring where the hollows are
  TString createHollowsCYSSFlangeA(double zlen);

  /// Given two intersecting lines defined by the points (x0,y0), (x1,y1) and
  /// (x1,y1), (x2,y2) {intersecting at (x1,y1)} the point (x,y) a distance
  /// c away is returned such that two lines a distance c away from the
  /// lines defined above intersect at (x,y).
  /// \param x0 X point on the first intersecting sets of lines
  /// \param y0 Y point on the first intersecting sets of lines
  /// \param x1 X point on the first/second intersecting sets of lines
  /// \param y1 Y point on the first/second intersecting sets of lines
  /// \param x2 X point on the second intersecting sets of lines
  /// \param y2 Y point on the second intersecting sets of lines
  /// \param c  Distance the two sets of lines are from each other
  /// \param x  X point for the intersecting sets of parellel lines
  /// \param y  Y point for the intersecting sets of parellel lines
  void insidePoint(double x0, double y0, double x1, double y1, double x2,
                   double y2, double c, double& x, double& y) const;

  /// Given the two points (x0,y0) and (x1,y1) and the location x, returns
  /// the value y corresponding to that point x on the line defined by the
  /// two points. Returns the value y corresponding to the point x on the line defined by
  /// the two points (x0,y0) and (x1,y1).
  /// \param x0 The first x value defining the line
  /// \param y0 The first y value defining the line
  /// \param x1 The second x value defining the line
  /// \param y1 The second y value defining the line
  /// \param x The x value for which the y value is wanted.
  double yFrom2Points(double x0, double y0, double x1, double y1, double x) const;

  ClassDefOverride(ITS3Services, 0); // ITS3 services
};
} // namespace its3
} // namespace o2

#endif
