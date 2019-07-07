/// \file CheckDigits.C
/// \brief Simple macro to check ITSU digits

#if !defined(__CLING__) || defined(__ROOTCLING__)
#include <climits>

#include <TFile.h>
#include <TTree.h>
#include <TH2F.h>
#include <TNtuple.h>
#include <TCanvas.h>

#include "ITSBase/GeometryTGeo.h"
#include "ITSMFTBase/SegmentationAlpide.h"
#include "ITSMFTBase/Digit.h"
#include "ITSMFTSimulation/ClusterShape.h"
#include "SimulationDataFormat/MCTruthContainer.h"
#include "SimulationDataFormat/MCCompLabel.h"
#endif

using o2::itsmft::Digit;
using o2::itsmft::SegmentationAlpide;
using namespace o2::its;

//////////////////////////////////////////
//////////////////////////////////////////
template <typename A, typename B>
std::pair<B, A> flip_pair(const std::pair<A, B>& p)
{
  return std::pair<B, A>(p.second, p.first);
}

template <typename A, typename B>
std::multimap<B, A> flip_map(const std::map<A, B>& src)
{
  std::multimap<B, A> dst;
  std::transform(src.begin(), src.end(), std::inserter(dst, dst.begin()),
                 flip_pair<A, B>);
  return dst;
}
//////////////////////////////////////////
//////////////////////////////////////////

//////////////////////////////////////////
//////////////////////////////////////////
class Pixel
{
 public:
  Pixel()
  {
    m_row = 0;
    m_col = 0;
  }
  Pixel(UInt_t row, UInt_t col)
  {
    m_row = row;
    m_col = col;
  }
  UInt_t GetRow() const { return m_row; };
  UInt_t GetCol() const { return m_col; };
  bool operator==(const Pixel& rhs) const
  {
    if (m_row == rhs.m_row && m_col == rhs.m_col) {
      return true;
    }
    return false;
  }

 private:
  UInt_t m_row;
  UInt_t m_col;
};
//////////////////////////////////////////
//////////////////////////////////////////

//////////////////////////////////////////
//////////////////////////////////////////
class Cluster
{
 public:
  Cluster()
  {
    m_pixels.clear();
  }
  void AddPixel(UInt_t l, Pixel p)
  {
    bool found = false;
    if (m_pixels.find(l) == m_pixels.end())
      m_pixels[l] = std::vector<Pixel>();
    for (size_t i = 0; i < m_pixels[l].size(); ++i) {
      if (m_pixels[l][i] == p)
        found = true;
    }
    if (!found) {
      m_pixels[l].push_back(p);
    } else {
      std::cout << "l: " << l << " - d:"
                << "(" << p.GetRow() << ", " << p.GetCol() << ") - already in cluster!" << std::endl;
    }
  }
  UInt_t GetNClusters(UInt_t l)
  {
    if (m_pixels.find(l) == m_pixels.end())
      return 0;
    return m_pixels[l].size();
  }
  Pixel GetPixel(UInt_t l, UInt_t digit)
  {
    if (m_pixels.find(l) == m_pixels.end())
      return Pixel();
    return m_pixels[l][digit];
  }
  const vector<Pixel>& GetPixels(UInt_t l)
  {
    return m_pixels[l];
  }

  static o2::itsmft::ClusterShape* PixelsToClusterShape(std::vector<Pixel> v)
  {
    if (v.size() == 0)
      return new o2::itsmft::ClusterShape();

    // To remove the multiple hits problem
    Pixel p0 = v[0];
    for (auto p = v.begin() + 1; p != v.end(); ++p) {
      if (fabs(p0.GetRow() - p->GetRow()) > v.size() || fabs(p0.GetCol() - p->GetCol()) > v.size()) {
        v.erase(p);
        p--;
      }
    }

    vector<UInt_t> r;
    vector<UInt_t> c;
    UInt_t min_r = UINT_MAX, min_c = UINT_MAX;
    for (auto& p : v) {
      if (p.GetRow() < min_r)
        min_r = p.GetRow();
      if (p.GetCol() < min_c)
        min_c = p.GetCol();
      if (find(begin(r), end(r), p.GetRow()) == end(r))
        r.push_back(p.GetRow());
      if (find(begin(c), end(c), p.GetCol()) == end(c))
        c.push_back(p.GetCol());
    }

    UInt_t rows = r.size();
    UInt_t cols = c.size();
    std::vector<UInt_t> pindex;
    for (auto& p : v) {
      Int_t rd = p.GetRow() - min_r;
      Int_t cd = p.GetCol() - min_c;
      Int_t index = rd * cols + cd;
      pindex.push_back(index);
    }

    return new o2::itsmft::ClusterShape(rows, cols, pindex);
  }

 private:
  map<UInt_t, vector<Pixel>> m_pixels;
};
//////////////////////////////////////////
//////////////////////////////////////////

//////////////////////////////////////////
//////////////////////////////////////////
void AnalyzeClusters(Int_t nev, const map<UInt_t, Cluster>& clusters, TH1F* freqDist, TH1F* cSizeDist)
{
  cout << ">>> Event " << nev << endl;
  Float_t x, z;
  Long64_t shapeId;
  map<Long64_t, Int_t> csFrequency;
  for (auto const& it : clusters) {
    Cluster cls = it.second;
    //cout << "id: " << it.first << " Ndigit: ";
    for (auto i = 0; i < 7; ++i) {
      //cout << cls.GetNClusters(i);
      vector<Pixel> pv = cls.GetPixels(i);
      if (pv.size() == 0)
        continue;
      o2::itsmft::ClusterShape* cs = Cluster::PixelsToClusterShape(pv);
      shapeId = cs->GetShapeID();
      cSizeDist->Fill(cs->GetNFiredPixels());
      cout << endl
           << shapeId << ":" << endl
           << *cs << endl
           << endl;

      if (csFrequency.find(shapeId) == csFrequency.end())
        csFrequency[shapeId] = 0;
      csFrequency[shapeId] += 1;

      // if (cls.GetNClusters(i) > 0) {
      //   cout << " [";
      //   for (auto j = 0; j < cls.GetNClusters(i); ++j) {
      //     Pixel p = cls.GetPixel(i, j);
      //     cout << "(" << p.GetRow() << "," << p.GetCol() << ")";
      //     //SegmentationAlpide::detectorToLocal(p.GetRow(),p.GetCol(),x,z);
      //     //cout << "(" << x << "," << z << ")";
      //     if (j < cls.GetNClusters(i)-1) cout << "  ";
      //   }
      //   cout << "]";
      // }

      //if (i < 6) cout << ", ";
      //else cout << endl;
    }
  }
  cout << ">>>>>>>>>>>>>>>>>" << endl
       << endl;

  int i = 0;
  std::multimap<Int_t, Long64_t> fmap = flip_map(csFrequency);
  for (auto e = fmap.rbegin(); e != fmap.rend(); ++e) {
    freqDist->Fill(i, e->first);
    string s = to_string(e->second);
    freqDist->GetXaxis()->SetBinLabel(i + 1, s.c_str());
    i++;
  }
  freqDist->GetXaxis()->SetRangeUser(0, fmap.size());
}
//////////////////////////////////////////
//////////////////////////////////////////

void CheckClusterShape(std::string digifile = "o2digi_its.root", std::string inputGeom = "O2geometry.root")
{
  // Geometry
  o2::base::GeometryManager::loadGeometry(inputGeom, "FAIRGeom");
  auto* gman = o2::its::GeometryTGeo::Instance();

  // Digits
  TFile* file1 = TFile::Open(digifile.data());
  TTree* digTree = (TTree*)gFile->Get("o2sim");
  std::vector<o2::itsmft::Digit>* digArr = nullptr;
  digTree->SetBranchAddress("ITSDigit", &digArr);
  o2::dataformats::MCTruthContainer<o2::MCCompLabel>* labels = nullptr;
  digTree->SetBranchAddress("ITSDigitMCTruth", &labels);

  TH1F* freqDist = new TH1F("freqDist", "", 300, 0, 300);
  //freqDist->GetXaxis()->SetTitle("Shape ID");
  //freqDist->GetYaxis()->SetTitle("Frequency");

  TH1F* cSizeDist = new TH1F("cSizeDist", "", 19, 1, 20);
  cSizeDist->GetXaxis()->SetTitle("Cluster Size");

  Int_t nev = digTree->GetEntries();
  int iev = -1;
  while (++iev < nev) {
    map<UInt_t, Cluster> clusters;
    digTree->GetEvent(iev);
    Int_t nd = digArr->size();
    while (nd--) {
      const Digit& d = (*digArr)[nd];
      const auto& labs = labels->getLabels(nd);
      Int_t ix = d.getRow(), iz = d.getColumn();

      Int_t chipID = d.getChipIndex();
      UInt_t layer = gman->getLayer(chipID);

      Pixel pixels(ix, iz);
      if (clusters.find(labs[0]) == clusters.end()) {
        Cluster c;
        clusters[labs[0]] = c;
      }
      clusters[labs[0]].AddPixel(layer, pixels);
    }
    AnalyzeClusters(nev, clusters, freqDist, cSizeDist);
  }

  freqDist->GetXaxis()->SetLabelSize(0.02);
  freqDist->Draw("HIST");

  new TCanvas;
  cSizeDist->Draw("HIST");
}
