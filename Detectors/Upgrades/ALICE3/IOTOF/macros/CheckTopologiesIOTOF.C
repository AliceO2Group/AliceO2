#include <iostream>
#include <vector>
#include <string>
#include <unordered_map>
#include <algorithm>
#include <bitset>

#include "TFile.h"
#include "TH2F.h"
#include "TGraphErrors.h"

#include "Framework/Logger.h"
#include "IOTOFBase/IOTOFBaseParam.h"
#include "IOTOFReconstruction/TopologyClassifier.h"

using namespace o2::iotof;

void CheckTopologiesIOTOF(const char* topoFileName = "TF3ClusterTopologies.root",
                          std::string chipCfgStr = "",
                          const char* outFileName = "CheckTopologies.root")
{

  o2::conf::ConfigurableParam::updateFromString(chipCfgStr);
  const auto& chipInfo = o2::iotof::ChipSpecificsParam::Instance();

  // Cluster topologies dictionary
  TFile* clsTopoFile = TFile::Open(topoFileName, "READ");
  auto* clsTopoMapPtr = clsTopoFile->Get<std::unordered_map<uint32_t, o2::iotof::TopologyInfo>>("TF3ClusterTopologies");
  if (clsTopoMapPtr) {
    std::cout << "Loaded " << clsTopoMapPtr->size() << " entries from " << topoFileName << std::endl;
  } else {
    std::cerr << "Failed to load TF3ClusterTopologies from " << topoFileName << std::endl;
  }

  // Construct map directly from the vector pairs
  std::unordered_map<uint32_t, o2::iotof::TopologyInfo> topoMap(clsTopoMapPtr->begin(), clsTopoMapPtr->end());
  std::cout << "\nTopologies summary:" << std::endl;
  TopologyClassifier topoClassifier(std::move(topoMap));
  topoClassifier.print();
  std::cout << std::endl;
  clsTopoFile->Close();

  // Sorted topology map by spanRow, spanCol, and then by bitmask for better organization in the output file
  auto topologyMap = topoClassifier.getTopologyMap();
  std::vector<std::pair<uint32_t, TopologyInfo>> sortedTopoMap(topologyMap.begin(), topologyMap.end());
  std::sort(sortedTopoMap.begin(), sortedTopoMap.end(), [](const auto& a, const auto& b) {
    int topoA = a.second.mTopology;
    int topoB = b.second.mTopology;
    uint8_t spanRowA = (a.first >> 24) & 0xFF;
    uint8_t spanColA = (a.first >> 16) & 0xFF;
    uint8_t spanRowB = (b.first >> 24) & 0xFF;
    uint8_t spanColB = (b.first >> 16) & 0xFF;
    int nPixelsA = a.second.mNPixels;
    int nPixelsB = b.second.mNPixels;
    int frequencyA = a.second.mFrequency;
    int frequencyB = b.second.mFrequency;
    if (topoA != topoB)
      return topoA < topoB;
    if (frequencyA != frequencyB)
      return frequencyA > frequencyB;
    if (spanRowA != spanRowB)
      return spanRowA < spanRowB;
    if (spanColA != spanColB)
      return spanColA < spanColB;
    if (nPixelsA != nPixelsB)
      return nPixelsA < nPixelsB;
    return a.first < b.first; // Finally sort by bitmask if spans are equal
  });

  // Print the sorted topology map
  for (const auto& entry : sortedTopoMap) {
    const uint32_t key = entry.first;
    const TopologyInfo& topoInfo = entry.second;

    uint8_t spanRow = (key >> 24) & 0xFF;
    uint8_t spanCol = (key >> 16) & 0xFF;
    uint16_t bitmask = key & 0xFFFF;

    LOG(info) << "Key: " << key
              << ", SpanRow: " << static_cast<int>(spanRow)
              << ", SpanCol: " << static_cast<int>(spanCol)
              << ", Bitmask: " << std::bitset<16>(bitmask);
    topoInfo.print();
    LOG(info) << "";
  }

  // Topology names
  const std::array<std::string, kNTopologies> topologyNames = {
    "kSingleDigit", "kLineOnRow", "kLineOnCol", "kSquare", "kRectangle", "kDiagonal",
    "kLowerTriangleLeft", "kLowerTriangleRight", "kUpperTriangleLeft", "kUpperTriangleRight",
    "kSnake", "kSnakeRefl", "kSnakeRot90", "kSnakeRot90Refl", "kHuge", "kOther"};

  // Create output ROOT file
  auto* outFile = TFile::Open(outFileName, "RECREATE");
  TH1F* hTopoSummaryDictionary = new TH1F("hTopoSummaryDictionary", "Cluster Topology Count Summary;;Counts", kNTopologies, 0, kNTopologies);
  for (const auto& topoName : topologyNames) {
    hTopoSummaryDictionary->GetXaxis()->SetBinLabel(&topoName - &topologyNames[0] + 1, topoName.c_str());
  }
  for (const auto& [topoKey, topology] : topoClassifier.getTopologyMap()) {
    hTopoSummaryDictionary->Fill(topology.mTopology, topology.mFrequency);
    hTopoSummaryDictionary->SetBinError(topology.mTopology + 1, 0);
  }
  hTopoSummaryDictionary->Write();
  // Create directory structures for all categories
  for (const auto& topoName : topologyNames) {
    outFile->mkdir(topoName.c_str());
  }

  for (int iMapEntry = 0; iMapEntry < sortedTopoMap.size(); ++iMapEntry) {
    const auto& [topoKey, topology] = sortedTopoMap[iMapEntry];
    std::string topoName = topologyNames[topology.mTopology];
    int spanRow = topology.mSizeX;
    int spanCol = topology.mSizeZ;
    uint16_t bitmask = topology.mPattern;
    int frequency = topology.mFrequency;

    float minRowCoord = -1.5 * chipInfo.PitchRow;
    float maxRowCoord = chipInfo.PitchRow * (spanRow + 0.5);
    float minColCoord = -1.5 * chipInfo.PitchCol;
    float maxColCoord = chipInfo.PitchCol * (spanCol + 0.5);
    TH2F* hTopoDisplay = new TH2F(Form("spanRow_%i_spanCol_%i_key_%i_all", spanRow, spanCol, topoKey), Form("Cluster Topology %s;Row;Column", topoName.c_str()),
                                  spanRow + 2, minRowCoord, maxRowCoord, spanCol + 2, minColCoord, maxColCoord);

    // One-point TGraph for COG
    TGraphErrors* gTopoCOG = new TGraphErrors(1);
    gTopoCOG->SetName(Form("spanRow_%i_spanCol_%i_key_%i_COG", spanRow, spanCol, topoKey));
    gTopoCOG->SetTitle(Form("Cluster Topology %s COG", topoName.c_str()));
    gTopoCOG->SetPoint(0, topology.mXMean, topology.mZMean);
    gTopoCOG->SetPointError(0, std::sqrt(topology.mXSigma2), std::sqrt(topology.mZSigma2));
    gTopoCOG->SetMarkerStyle(20);
    gTopoCOG->SetMarkerColor(kBlue);

    // Loop over the bits of bitmask and fill the histogram
    for (int row = 0; row < spanRow; ++row) {
      for (int col = 0; col < spanCol; ++col) {
        int bitIndex = row * spanCol + col;
        if (bitmask & (1 << bitIndex)) {
          hTopoDisplay->SetBinContent(row + 2, col + 2, frequency);
        }
      }
    }
    outFile->cd(topoName.c_str());
    hTopoDisplay->Write();
    gTopoCOG->Write();
    delete hTopoDisplay;
    delete gTopoCOG;
  }

  outFile->Close();
  LOG(info) << "Successfully wrote topology displays to " << outFileName;
}
