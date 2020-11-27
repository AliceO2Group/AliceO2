// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

#include "MCHGeometryCreator/Geometry.h"
#include "MCHGeometryTransformer/VolumePaths.h"
#include "Station1Geometry.h"
#include "Station2Geometry.h"
#include "Station345Geometry.h"
#include "Materials.h"
#include <iostream>
#include <TGeoPhysicalNode.h>
#include <fmt/format.h>
#include "TGeoVolume.h"
#include "TGeoManager.h"
#include "Framework/Logger.h"

namespace impl
{
void addAlignableVolumesHalfChamber(TGeoManager& geom, int hc, std::string& parent)
{
  //
  // Add alignable volumes for a half chamber and its daughters
  //
  std::vector<std::vector<int>> DEofHC{{100, 103},
                                       {101, 102},
                                       {200, 203},
                                       {201, 202},
                                       {300, 303},
                                       {301, 302},
                                       {400, 403},
                                       {401, 402},
                                       {500, 501, 502, 503, 504, 514, 515, 516, 517},
                                       {505, 506, 507, 508, 509, 510, 511, 512, 513},
                                       {600, 601, 602, 603, 604, 614, 615, 616, 617},
                                       {605, 606, 607, 608, 609, 610, 611, 612, 613},
                                       {700, 701, 702, 703, 704, 705, 706, 720, 721, 722, 723, 724, 725},
                                       {707, 708, 709, 710, 711, 712, 713, 714, 715, 716, 717, 718, 719},
                                       {800, 801, 802, 803, 804, 805, 806, 820, 821, 822, 823, 824, 825},
                                       {807, 808, 809, 810, 811, 812, 813, 814, 815, 816, 817, 818, 819},
                                       {900, 901, 902, 903, 904, 905, 906, 920, 921, 922, 923, 924, 925},
                                       {907, 908, 909, 910, 911, 912, 913, 914, 915, 916, 917, 918, 919},
                                       {1000, 1001, 1002, 1003, 1004, 1005, 1006, 1020, 1021, 1022, 1023, 1024, 1025},
                                       {1007, 1008, 1009, 1010, 1011, 1012, 1013, 1014, 1015, 1016, 1017, 1018, 1019}};

  for (int i = 0; i < DEofHC[hc].size(); i++) {
    std::string volPathName = o2::mch::geo::volumePathName(DEofHC[hc][i]);

    TString path = Form("%s%s", parent.c_str(), volPathName.c_str());
    TString sname = Form("MCH/HC%d/DE%d", hc, DEofHC[hc][i]);

    LOG(DEBUG) << "Add " << sname << " <-> " << path;

    if (!geom.SetAlignableEntry(sname.Data(), path.Data())) {
      LOG(FATAL) << "Unable to set alignable entry ! " << sname << " : " << path;
    }
  }

  return;
}

} // namespace impl

namespace o2::mch::geo
{

void createGeometry(TGeoManager& geom, TGeoVolume& topVolume)
{
  createMaterials();

  auto volYOUT1 = geom.GetVolume("YOUT1");

  createStation1Geometry((volYOUT1) ? *volYOUT1 : topVolume);
  createStation2Geometry((volYOUT1) ? *volYOUT1 : topVolume);

  createStation345Geometry(topVolume);
}

std::vector<TGeoVolume*> getSensitiveVolumes()
{
  auto st1 = getStation1SensitiveVolumes();
  auto st2 = getStation2SensitiveVolumes();
  auto st345 = getStation345SensitiveVolumes();

  auto vol = st1;
  vol.insert(vol.end(), st2.begin(), st2.end());
  vol.insert(vol.end(), st345.begin(), st345.end());

  return vol;
}

void addAlignableVolumes(TGeoManager& geom)
{
  if (!geom.IsClosed()) {
    geom.CloseGeometry();
  }

  LOG(INFO) << "Add MCH alignable volumes";

  for (int hc = 0; hc < 20; hc++) {
    int nCh = hc / 2 + 1;

    std::string volPathName = geom.GetTopVolume()->GetName();

    if (nCh <= 4 && geom.GetVolume("YOUT1")) {
      volPathName += "/YOUT1_1/";
    } else if ((nCh == 5 || nCh == 6) && geom.GetVolume("DDIP")) {
      volPathName += "/DDIP_1/";
    } else if (nCh >= 7 && geom.GetVolume("YOUT2")) {
      volPathName += "/YOUT2_1/";
    } else {
      volPathName += "/";
    }

    std::string path = fmt::format("{0}SC{1}{2}{3}_{4}", volPathName.c_str(), nCh < 10 ? "0" : "", nCh, hc % 2 ? "O" : "I", hc);
    std::string sname = fmt::format("MCH/HC{}", hc);

    LOG(DEBUG) << sname << " <-> " << path;

    auto ae = geom.SetAlignableEntry(sname.c_str(), path.c_str());
    if (!ae) {
      LOG(FATAL) << "Unable to set alignable entry ! " << sname << " : " << path;
    }

    Int_t lastUID = 0;

    impl::addAlignableVolumesHalfChamber(geom, hc, volPathName);
  }

  return;
}

} // namespace o2::mch::geo
