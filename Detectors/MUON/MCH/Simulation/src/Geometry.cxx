// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

#include "MCHSimulation/Geometry.h"

#include "Station1Geometry.h"
#include "Station2Geometry.h"
#include "Station345Geometry.h"
#include "Materials.h"
#include <iostream>
#include "TGeoVolume.h"
#include "TGeoManager.h"

namespace o2
{
namespace mch
{
void createGeometry(TGeoVolume& topVolume)
{
  createMaterials();
  createStation1Geometry(topVolume);
  createStation2Geometry(topVolume);
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

namespace impl
{
// return the path of the mother volume of detElemId, relative to MCH geometry
// (i.e. excluding general top node)
std::string getVolumePathName(int detElemId)
{
  std::string vp{"incorrect detElemId"};

  if (detElemId == 100) {
    return "SC01I_0/Quadrant (chamber 1)_100";
  }
  if (detElemId == 103) {
    return "SC01I_0/Quadrant (chamber 1)_103";
  }
  if (detElemId == 101) {
    return "SC01O_1/Quadrant (chamber 1)_101";
  }
  if (detElemId == 102) {
    return "SC01O_1/Quadrant (chamber 1)_102";
  }
  if (detElemId == 200) {
    return "SC02I_2/Quadrant (chamber 2)_200";
  }
  if (detElemId == 203) {
    return "SC02I_2/Quadrant (chamber 2)_203";
  }
  if (detElemId == 201) {
    return "SC02O_3/Quadrant (chamber 2)_201";
  }
  if (detElemId == 202) {
    return "SC02O_3/Quadrant (chamber 2)_202";
  }
  if (detElemId == 300) {
    return "SC03I_4/Station 2 quadrant_300";
  }
  if (detElemId == 303) {
    return "SC03I_4/Station 2 quadrant_303";
  }
  if (detElemId == 301) {
    return "SC03O_5/Station 2 quadrant_301";
  }
  if (detElemId == 302) {
    return "SC03O_5/Station 2 quadrant_302";
  }
  if (detElemId == 400) {
    return "SC04I_6/Station 2 quadrant_400";
  }
  if (detElemId == 403) {
    return "SC04I_6/Station 2 quadrant_403";
  }
  if (detElemId == 401) {
    return "SC04O_7/Station 2 quadrant_401";
  }
  if (detElemId == 402) {
    return "SC04O_7/Station 2 quadrant_402";
  }
  if (detElemId == 500) {
    return "SC05I_8/122000SR1_500";
  }
  if (detElemId == 501) {
    return "SC05I_8/112200SR2_501";
  }
  if (detElemId == 502) {
    return "SC05I_8/122200S_502";
  }
  if (detElemId == 503) {
    return "SC05I_8/222000N_503";
  }
  if (detElemId == 504) {
    return "SC05I_8/220000N_504";
  }
  if (detElemId == 514) {
    return "SC05I_8/220000N_514";
  }
  if (detElemId == 515) {
    return "SC05I_8/222000N_515";
  }
  if (detElemId == 516) {
    return "SC05I_8/122200S_516";
  }
  if (detElemId == 517) {
    return "SC05I_8/112200SR2_517";
  }
  if (detElemId == 505) {
    return "SC05O_9/220000N_505";
  }
  if (detElemId == 506) {
    return "SC05O_9/222000N_506";
  }
  if (detElemId == 507) {
    return "SC05O_9/122200S_507";
  }
  if (detElemId == 508) {
    return "SC05O_9/112200SR2_508";
  }
  if (detElemId == 509) {
    return "SC05O_9/122000SR1_509";
  }
  if (detElemId == 510) {
    return "SC05O_9/112200SR2_510";
  }
  if (detElemId == 511) {
    return "SC05O_9/122200S_511";
  }
  if (detElemId == 512) {
    return "SC05O_9/222000N_512";
  }
  if (detElemId == 513) {
    return "SC05O_9/220000N_513";
  }
  if (detElemId == 600) {
    return "SC06I_10/122000NR1_600";
  }
  if (detElemId == 601) {
    return "SC06I_10/112200NR2_601";
  }
  if (detElemId == 602) {
    return "SC06I_10/122200N_602";
  }
  if (detElemId == 603) {
    return "SC06I_10/222000N_603";
  }
  if (detElemId == 604) {
    return "SC06I_10/220000N_604";
  }
  if (detElemId == 614) {
    return "SC06I_10/220000N_614";
  }
  if (detElemId == 615) {
    return "SC06I_10/222000N_615";
  }
  if (detElemId == 616) {
    return "SC06I_10/122200N_616";
  }
  if (detElemId == 617) {
    return "SC06I_10/112200NR2_617";
  }
  if (detElemId == 605) {
    return "SC06O_11/220000N_605";
  }
  if (detElemId == 606) {
    return "SC06O_11/222000N_606";
  }
  if (detElemId == 607) {
    return "SC06O_11/122200N_607";
  }
  if (detElemId == 608) {
    return "SC06O_11/112200NR2_608";
  }
  if (detElemId == 609) {
    return "SC06O_11/122000NR1_609";
  }
  if (detElemId == 610) {
    return "SC06O_11/112200NR2_610";
  }
  if (detElemId == 611) {
    return "SC06O_11/122200N_611";
  }
  if (detElemId == 612) {
    return "SC06O_11/222000N_612";
  }
  if (detElemId == 613) {
    return "SC06O_11/220000N_613";
  }
  if (detElemId == 700) {
    return "SC07I_12/122330N_700";
  }
  if (detElemId == 701) {
    return "SC07I_12/112233NR3_701";
  }
  if (detElemId == 702) {
    return "SC07I_12/112230N_702";
  }
  if (detElemId == 703) {
    return "SC07I_12/222330N_703";
  }
  if (detElemId == 704) {
    return "SC07I_12/223300N_704";
  }
  if (detElemId == 705) {
    return "SC07I_12/333000N_705";
  }
  if (detElemId == 706) {
    return "SC07I_12/330000N_706";
  }
  if (detElemId == 720) {
    return "SC07I_12/330000N_720";
  }
  if (detElemId == 721) {
    return "SC07I_12/333000N_721";
  }
  if (detElemId == 722) {
    return "SC07I_12/223300N_722";
  }
  if (detElemId == 723) {
    return "SC07I_12/222330N_723";
  }
  if (detElemId == 724) {
    return "SC07I_12/112230N_724";
  }
  if (detElemId == 725) {
    return "SC07I_12/112233NR3_725";
  }
  if (detElemId == 707) {
    return "SC07O_13/330000N_707";
  }
  if (detElemId == 708) {
    return "SC07O_13/333000N_708";
  }
  if (detElemId == 709) {
    return "SC07O_13/223300N_709";
  }
  if (detElemId == 710) {
    return "SC07O_13/222330N_710";
  }
  if (detElemId == 711) {
    return "SC07O_13/112230N_711";
  }
  if (detElemId == 712) {
    return "SC07O_13/112233NR3_712";
  }
  if (detElemId == 713) {
    return "SC07O_13/122330N_713";
  }
  if (detElemId == 714) {
    return "SC07O_13/112233NR3_714";
  }
  if (detElemId == 715) {
    return "SC07O_13/112230N_715";
  }
  if (detElemId == 716) {
    return "SC07O_13/222330N_716";
  }
  if (detElemId == 717) {
    return "SC07O_13/223300N_717";
  }
  if (detElemId == 718) {
    return "SC07O_13/333000N_718";
  }
  if (detElemId == 719) {
    return "SC07O_13/330000N_719";
  }
  if (detElemId == 800) {
    return "SC08I_14/122330N_800";
  }
  if (detElemId == 801) {
    return "SC08I_14/112233NR3_801";
  }
  if (detElemId == 802) {
    return "SC08I_14/112230N_802";
  }
  if (detElemId == 803) {
    return "SC08I_14/222330N_803";
  }
  if (detElemId == 804) {
    return "SC08I_14/223300N_804";
  }
  if (detElemId == 805) {
    return "SC08I_14/333000N_805";
  }
  if (detElemId == 806) {
    return "SC08I_14/330000N_806";
  }
  if (detElemId == 820) {
    return "SC08I_14/330000N_820";
  }
  if (detElemId == 821) {
    return "SC08I_14/333000N_821";
  }
  if (detElemId == 822) {
    return "SC08I_14/223300N_822";
  }
  if (detElemId == 823) {
    return "SC08I_14/222330N_823";
  }
  if (detElemId == 824) {
    return "SC08I_14/112230N_824";
  }
  if (detElemId == 825) {
    return "SC08I_14/112233NR3_825";
  }
  if (detElemId == 807) {
    return "SC08O_15/330000N_807";
  }
  if (detElemId == 808) {
    return "SC08O_15/333000N_808";
  }
  if (detElemId == 809) {
    return "SC08O_15/223300N_809";
  }
  if (detElemId == 810) {
    return "SC08O_15/222330N_810";
  }
  if (detElemId == 811) {
    return "SC08O_15/112230N_811";
  }
  if (detElemId == 812) {
    return "SC08O_15/112233NR3_812";
  }
  if (detElemId == 813) {
    return "SC08O_15/122330N_813";
  }
  if (detElemId == 814) {
    return "SC08O_15/112233NR3_814";
  }
  if (detElemId == 815) {
    return "SC08O_15/112230N_815";
  }
  if (detElemId == 816) {
    return "SC08O_15/222330N_816";
  }
  if (detElemId == 817) {
    return "SC08O_15/223300N_817";
  }
  if (detElemId == 818) {
    return "SC08O_15/333000N_818";
  }
  if (detElemId == 819) {
    return "SC08O_15/330000N_819";
  }
  if (detElemId == 900) {
    return "SC09I_16/122330N_900";
  }
  if (detElemId == 901) {
    return "SC09I_16/112233NR3_901";
  }
  if (detElemId == 902) {
    return "SC09I_16/112233N_902";
  }
  if (detElemId == 903) {
    return "SC09I_16/222333N_903";
  }
  if (detElemId == 904) {
    return "SC09I_16/223330N_904";
  }
  if (detElemId == 905) {
    return "SC09I_16/333300N_905";
  }
  if (detElemId == 906) {
    return "SC09I_16/333000N_906";
  }
  if (detElemId == 920) {
    return "SC09I_16/333000N_920";
  }
  if (detElemId == 921) {
    return "SC09I_16/333300N_921";
  }
  if (detElemId == 922) {
    return "SC09I_16/223330N_922";
  }
  if (detElemId == 923) {
    return "SC09I_16/222333N_923";
  }
  if (detElemId == 924) {
    return "SC09I_16/112233N_924";
  }
  if (detElemId == 925) {
    return "SC09I_16/112233NR3_925";
  }
  if (detElemId == 907) {
    return "SC09O_17/333000N_907";
  }
  if (detElemId == 908) {
    return "SC09O_17/333300N_908";
  }
  if (detElemId == 909) {
    return "SC09O_17/223330N_909";
  }
  if (detElemId == 910) {
    return "SC09O_17/222333N_910";
  }
  if (detElemId == 911) {
    return "SC09O_17/112233N_911";
  }
  if (detElemId == 912) {
    return "SC09O_17/112233NR3_912";
  }
  if (detElemId == 913) {
    return "SC09O_17/122330N_913";
  }
  if (detElemId == 914) {
    return "SC09O_17/112233NR3_914";
  }
  if (detElemId == 915) {
    return "SC09O_17/112233N_915";
  }
  if (detElemId == 916) {
    return "SC09O_17/222333N_916";
  }
  if (detElemId == 917) {
    return "SC09O_17/223330N_917";
  }
  if (detElemId == 918) {
    return "SC09O_17/333300N_918";
  }
  if (detElemId == 919) {
    return "SC09O_17/333000N_919";
  }
  if (detElemId == 1000) {
    return "SC10I_18/122330N_1000";
  }
  if (detElemId == 1001) {
    return "SC10I_18/112233NR3_1001";
  }
  if (detElemId == 1002) {
    return "SC10I_18/112233N_1002";
  }
  if (detElemId == 1003) {
    return "SC10I_18/222333N_1003";
  }
  if (detElemId == 1004) {
    return "SC10I_18/223330N_1004";
  }
  if (detElemId == 1005) {
    return "SC10I_18/333300N_1005";
  }
  if (detElemId == 1006) {
    return "SC10I_18/333000N_1006";
  }
  if (detElemId == 1020) {
    return "SC10I_18/333000N_1020";
  }
  if (detElemId == 1021) {
    return "SC10I_18/333300N_1021";
  }
  if (detElemId == 1022) {
    return "SC10I_18/223330N_1022";
  }
  if (detElemId == 1023) {
    return "SC10I_18/222333N_1023";
  }
  if (detElemId == 1024) {
    return "SC10I_18/112233N_1024";
  }
  if (detElemId == 1025) {
    return "SC10I_18/112233NR3_1025";
  }
  if (detElemId == 1007) {
    return "SC10O_19/333000N_1007";
  }
  if (detElemId == 1008) {
    return "SC10O_19/333300N_1008";
  }
  if (detElemId == 1009) {
    return "SC10O_19/223330N_1009";
  }
  if (detElemId == 1010) {
    return "SC10O_19/222333N_1010";
  }
  if (detElemId == 1011) {
    return "SC10O_19/112233N_1011";
  }
  if (detElemId == 1012) {
    return "SC10O_19/112233NR3_1012";
  }
  if (detElemId == 1013) {
    return "SC10O_19/122330N_1013";
  }
  if (detElemId == 1014) {
    return "SC10O_19/112233NR3_1014";
  }
  if (detElemId == 1015) {
    return "SC10O_19/112233N_1015";
  }
  if (detElemId == 1016) {
    return "SC10O_19/222333N_1016";
  }
  if (detElemId == 1017) {
    return "SC10O_19/223330N_1017";
  }
  if (detElemId == 1018) {
    return "SC10O_19/333300N_1018";
  }
  if (detElemId == 1019) {
    return "SC10O_19/333000N_1019";
  }

  return vp;
}

} // namespace impl

o2::Transform3D getTransformation(int detElemId, const TGeoManager& geo)
{
  std::string volPathName = geo.GetTopVolume()->GetName();

  volPathName += "/";
  volPathName += impl::getVolumePathName(detElemId);

  TGeoNavigator* navig = gGeoManager->GetCurrentNavigator();

  if (!navig->cd(volPathName.c_str())) {
    throw std::runtime_error("could not get to volPathName=" + volPathName);
  }

  return o2::Transform3D{*(navig->GetCurrentMatrix())};
}

} // namespace mch
} // namespace o2
