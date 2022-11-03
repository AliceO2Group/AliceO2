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

#include "MCHGeometryTransformer/Transformations.h"
#include "MCHGeometryTransformer/VolumePaths.h"
#include "MCHConstants/DetectionElements.h"
#include <TGeoManager.h>
#include <TGeoMatrix.h>
#include <array>
#include <fmt/format.h>
#include <rapidjson/document.h>
#include <rapidjson/istreamwrapper.h>
#include <string>
#include <vector>

namespace o2::mch::geo
{

std::array<int, 156> allDeIds = o2::mch::constants::deIdsForAllMCH;

TransformationCreator transformationFromTGeoManager(const TGeoManager& geo)
{
  return [&geo](int detElemId) -> o2::math_utils::Transform3D {
    if (std::find(begin(allDeIds), end(allDeIds), detElemId) == end(allDeIds)) {
      throw std::runtime_error("Wrong detection element Id");
    }

    std::string volPathName = geo.GetTopVolume()->GetName();

    int nCh = detElemId / 100;

    if (nCh <= 4 && geo.GetVolume("YOUT1")) {
      volPathName += "/YOUT1_1/";
    } else if ((nCh == 5 || nCh == 6) && geo.GetVolume("DDIP")) {
      volPathName += "/DDIP_1/";
    } else if (nCh >= 7 && geo.GetVolume("YOUT2")) {
      volPathName += "/YOUT2_1/";
    } else {
      volPathName += "/";
    }

    volPathName += volumePathName(detElemId);

    TGeoNavigator* navig = gGeoManager->GetCurrentNavigator();

    if (!navig->cd(volPathName.c_str())) {
      throw std::runtime_error("could not get to volPathName=" + volPathName);
    }

    return o2::math_utils::Transform3D{*(navig->GetCurrentMatrix())};
  };
}

TransformationCreator transformationFromJSON(std::istream& in)
{
  rapidjson::IStreamWrapper isw(in);

  rapidjson::Document d;
  d.ParseStream(isw);

  rapidjson::Value& alignables = d["alignables"];
  assert(alignables.IsArray());

  std::map<int, std::tuple<double, double, double>> angles;
  std::map<int, std::tuple<double, double, double>> translations;

  // loop over json document and extract Tait-Bryan angles (yaw,pitch,roll)
  // as well as translation vector (tx,ty,tz)
  // for each detection element

  constexpr double deg2rad = 3.14159265358979323846 / 180.0;

  for (auto& al : alignables.GetArray()) {
    auto itr = al.FindMember("deid");
    if (itr != al.MemberEnd()) {
      int deid = itr->value.GetInt();
      auto t = al["transform"].GetObject();
      angles[deid] = {
        deg2rad * t["yaw"].GetDouble(),
        deg2rad * t["pitch"].GetDouble(),
        deg2rad * t["roll"].GetDouble()};
      translations[deid] = {
        t["tx"].GetDouble(),
        t["ty"].GetDouble(),
        t["tz"].GetDouble()};
    }
  }

  return [angles, translations](int detElemId) -> o2::math_utils::Transform3D {
    if (std::find(begin(allDeIds), end(allDeIds), detElemId) == end(allDeIds)) {
      throw std::runtime_error("Wrong detection element Id");
    }
    auto [yaw, pitch, roll] = angles.at(detElemId);
    auto [tx, ty, tz] = translations.at(detElemId);
    double tr[3] = {tx, ty, tz};
    // get the angles, convert them to a matrix and build a Transform3D
    // from it
    auto rot = o2::mch::geo::angles2matrix(yaw, pitch, roll);
    TGeoHMatrix m;
    m.SetRotation(&rot[0]);
    m.SetTranslation(tr);
    return o2::math_utils::Transform3D(m);
  };
} // namespace o2::mch::geo

std::array<double, 9> angles2matrix(double yaw, double pitch, double roll)
{
  std::array<double, 9> rot;

  double sinpsi = std::sin(roll);
  double cospsi = std::cos(roll);
  double sinthe = std::sin(pitch);
  double costhe = std::cos(pitch);
  double sinphi = std::sin(yaw);
  double cosphi = std::cos(yaw);
  rot[0] = costhe * cosphi;
  rot[1] = -costhe * sinphi;
  rot[2] = sinthe;
  rot[3] = sinpsi * sinthe * cosphi + cospsi * sinphi;
  rot[4] = -sinpsi * sinthe * sinphi + cospsi * cosphi;
  rot[5] = -costhe * sinpsi;
  rot[6] = -cospsi * sinthe * cosphi + sinpsi * sinphi;
  rot[7] = cospsi * sinthe * sinphi + sinpsi * cosphi;
  rot[8] = costhe * cospsi;
  return rot;
}

std::tuple<double, double, double> matrix2angles(gsl::span<double> rot)
{
  double roll = std::atan2(-rot[5], rot[8]);
  double pitch = std::asin(rot[2]);
  double yaw = std::atan2(-rot[1], rot[0]);
  return std::make_tuple(yaw,
                         pitch,
                         roll);
}

} // namespace o2::mch::geo
