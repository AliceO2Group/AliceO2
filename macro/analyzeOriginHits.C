// Brief macro to get origins of tracks leaving at least one hit in a detector.
// All histograms are summarised in a ROOT file to be processed further.

#if !defined(__CLING__) || defined(__ROOTCLING__)

#include <vector>
#include <unordered_map>
#include <iostream>
#include <fstream>
#include <numeric>
#include <string>
#include <type_traits>

#include "TFile.h"
#include "TTree.h"
#include "TH1F.h"
#include "TGeoManager.h"
#include "TGeoNode.h"
#include "TVector3.h"

#include "SimulationDataFormat/MCTrack.h"
#include "DetectorsCommonDataFormats/DetID.h"

#endif

const std::string UNKNOWN = "NULL";

class VolModMap
{
 public:
  VolModMap(const std::string& volMapFile)
  {
    std::ifstream ifs;
    ifs.open(volMapFile.c_str());
    if (ifs.is_open()) {
      std::string line;
      while (std::getline(ifs, line)) {
        std::istringstream ss(line);
        std::string token;
        int counter = 0;
        std::string keyvalue[2] = {UNKNOWN, UNKNOWN};
        while (counter < 2 && std::getline(ss, token, ':')) {
          if (!token.empty()) {
            keyvalue[counter] = token;
            counter++;
          }
        }
        insertVolMod({keyvalue[0], keyvalue[1]});
      }
      ifs.close();
    } else {
      std::cerr << "Cannot read vol->mod map file " << volMapFile << "\n";
      exit(1);
    }
  }

  const std::string& getModNameFromVol(const std::string& volName)
  {
    auto it = volModMap.find(volName);
    if (it == volModMap.end()) {
      return UNKNOWN;
    }
    return it->second;
  }

  int getModIdFromVol(const std::string& volName)
  {
    auto modName = getModNameFromVol(volName);
    if (modName.compare(UNKNOWN) == 0) {
      return -1;
    }
    return modToIdMap[modName];
  }

  const std::vector<std::string>& getIdToModMap() const
  {
    return idToModMap;
  }

 private:
  void insertVolMod(const std::pair<std::string, std::string>& keyvalues)
  {
    volModMap.insert(keyvalues);
    auto it = std::find(idToModMap.begin(), idToModMap.end(), keyvalues.second);
    if (it == idToModMap.end()) {
      modToIdMap[keyvalues.second] = idToModMap.size();
      idToModMap.push_back(keyvalues.second);
    }
  }
  std::unordered_map<std::string, std::string> volModMap;
  std::unordered_map<std::string, unsigned int> modToIdMap;
  std::vector<std::string> idToModMap;
};

const char* getVolNameFromGeom(const TVector3& pos, TGeoManager* geoManager = gGeoManager)
{
  auto node = geoManager->FindNode(pos.X(), pos.Y(), pos.Z());
  if (!node) {
    return UNKNOWN.c_str();
  }
  return node->GetVolume()->GetName();
}

template <std::size_t DIMS, typename T, typename = typename std::enable_if<(DIMS > 0)>::type>
class Grid
{
  typedef unsigned int coord;

 public:
  template <typename... P>
  Grid(T init, P&&... points) : mDimLengths{std::forward<coord>(points)...}
  {
    mGrid.resize(std::accumulate(mDimLengths.begin(), mDimLengths.end(), 1, std::multiplies<coord>()), init);
    if (mGrid.empty()) {
      std::cerr << "At least one dimension of the grid has length 0\n";
      exit(1);
    }
  }

  template <typename... P,
            typename = typename std::enable_if<sizeof...(P) == DIMS>::type>
  T& getPoint(P&&... points)
  {
    int absPoint = findAbsPoint(points...);
    return mGrid[absPoint];
  }

  const std::vector<T>& getPoints() const
  {
    return mGrid;
  }

 private:
  template <typename... P>
  int findAbsPoint(P&&... points)
  {
    std::array<coord, sizeof...(P)> l = {std::forward<coord>(points)...};
    auto absPoint = std::inner_product(l.begin() + 1, l.end(), mDimLengths.begin() + 1, l[0]);
    assert(absPoint < mGrid.size());
    return absPoint;
  }

 private:
  std::vector<T> mGrid;
  std::array<coord, DIMS> mDimLengths;
};

template <class... P, typename T>
Grid(T init, P&&... points)->Grid<sizeof...(P), T>;

void analyzeOriginHits(const char* filename = "o2sim.root",
                       const std::string& volMapFile = "MCStepLoggerVolMap.dat",
                       const std::string& geomFile = "O2geometry.root",
                       bool ignorePrimaries = false)
{
  TFile rf(filename, "OPEN");
  auto reftree = (TTree*)rf.Get("o2sim");
  const char* brname = "MCTrack";
  auto br = reftree->GetBranch(brname);
  if (!br) {
    std::cerr << "Unknown branch " << brname << "\n";
    return;
  }
  std::vector<o2::MCTrack>* tracks = nullptr;
  br->SetAddress(&tracks);

  TGeoManager::Import(geomFile.c_str());

  VolModMap volModMap(volMapFile);
  auto originNames = volModMap.getIdToModMap();

  auto nDet = o2::detectors::DetID::getNDetectors();
  std::vector<std::string> detectorNames(nDet);

  TH1* init = nullptr;
  Grid grid(init, static_cast<unsigned int>(originNames.size()), static_cast<unsigned int>(nDet));

  for (int i = 0; i < nDet; i++) {
    detectorNames[i] = o2::detectors::DetID::getName(i);
    for (int j = 0; j < static_cast<int>(originNames.size()); j++) {
      auto& histo = grid.getPoint(i, j);
      std::string name = originNames[j] + "_to_" + detectorNames[i];
      std::string title = originNames[j] + " #rightarrow " + detectorNames[i];
      histo = new TH1F(name.c_str(), title.c_str(), 10, 0, -1);
      histo->GetXaxis()->SetTitle("log10(E_{origin} / GeV)");
      histo->GetYaxis()->SetTitle("# Entries");
    }
  }

  for (int entry = 0; entry < br->GetEntries(); entry++) {
    br->GetEntry(entry);
    for (auto& track : *tracks) {
      auto hitmask = track.getHitMask();
      if (hitmask == 0) {
        continue;
      }
      if (hitmask == 0 || (ignorePrimaries && track.getMotherTrackId() < 0)) {
        continue;
      }
      TVector3 startVtx;
      track.GetStartVertex(startVtx);
      auto originModId = volModMap.getModIdFromVol(getVolNameFromGeom(startVtx));
      auto value = log10(track.GetEnergy());
      for (int i = 0; i < nDet; i++) {
        if ((hitmask & (1 << i)) > 0) {
          grid.getPoint(i, originModId)->Fill(value);
        }
      }
    }
  }

  TFile outFile("originHits.root", "RECREATE");
  outFile.cd();
  for (auto& point : grid.getPoints()) {
    point->Write();
  }
  outFile.Close();
}
