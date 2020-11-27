// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.
/// get the local-to-global transformation for a given detection element

#ifndef O2_MCH_GEOMETRY_TRANSFORMER_TRANSFORMATIONS_H
#define O2_MCH_GEOMETRY_TRANSFORMER_TRANSFORMATIONS_H

#include "MathUtils/Cartesian.h"
#include <array>
#include <tuple>
#include <gsl/span>
#include <functional>

class TGeoManager;

namespace o2::mch::geo
{

/** A geometry transformation creator must be able to 
 * create a transformation for any valid MCH detection element id.
 *
 * That transformation is used to convert coordinates from local 
 * (x,y in detection element plane) to global (x,y,z in Alice Coordinate 
 * System) and vice-versa
 * 
 * @param detElemId must be a valid detection element
 * @param geo a reference to a GeoManager that must contain MCH volumes
 *
 * @throw if detElemId is not valid
 */
using TransformationCreator = std::function<o2::math_utils::Transform3D(int detElemId)>;

/** Tranformation creator using TGeoManager.
 *
 * @param geo a reference to a GeoManager that must contain MCH volumes
 */
TransformationCreator transformationFromTGeoManager(const TGeoManager& geo);

TransformationCreator transformationFromJSON(std::istream& in);

/** The list of detection element identifiers for MCH 
 */
extern std::array<int, 156> allDeIds;

/** Convert the 3 Tait–Bryan angles (yaw,pitch,roll) into the 9 matrix 
  * elements of a rotation matrix.
  *
  * @param yaw rotation around z-axis, in radian
  * @param pitch rotation around y'-axis (new y-axis resulting from yaw 
  * rotation), in radian
  * @param roll rotation around x''-axis (new x-axis resulting from yaw 
  * and pitch rotations), in radian
  */
std::array<double, 9> angles2matrix(double yaw, double pitch, double roll);

/** Convert the 9 matrix elements of a rotation matrix
  * into 3 Tait-Bryan angles (yaw,pitch,roll).
  *
  * @param rot a 9-elements vector
  * @returns a tuple of the 3 angles <yaw,pitch,roll>. The angles are 
  * expressed in radian
  */
std::tuple<double, double, double> matrix2angles(gsl::span<double> rot);
} // namespace o2::mch::geo

#endif
