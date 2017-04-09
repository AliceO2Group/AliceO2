#include "TCanvas.h"
#include "TString.h"
#include "TH1.h"
#include "TH2.h"

#include "TPCBase/CalDet.h"
#include "TPCBase/CalArray.h"
#include "TPCBase/Painter.h"

using namespace o2::TPC;

template <class T>
void Painter::Draw(CalDet<T> calDet)
{
  using DetType = CalDet<T>;
  using CalType = CalArray<T>;

  auto name = calDet.getName().c_str();
  auto c = new TCanvas(Form("c_%s", name));
  c->Divide(2,2);

  auto hAside1D = new TH1F(Form("h_Aside_%s", name), Form("%s (A-Side)", name),
                         300, -100, 100); //TODO: modify ranges

  auto hCside1D = new TH1F(Form("h_Cside_%s", name), Form("%s (C-Side)", name),
                         300, -100, 100); //TODO: modify ranges

  auto hAside2D = new TH2F(Form("h_Aside_%s", name), Form("%s (A-Side)", name),
                         300, -300, 300, 300, -300, 300);

  auto hCside2D = new TH2F(Form("h_Cside_%s", name), Form("%s (C-Side)", name),
                         300, -300, 300, 300, -300, 300);


  for (auto cal : calDet.getData()) {

    int calPadSubsetNumber = cal.getPadSubsetNumber();
    int row = -1;
    int pad = -1;
    switch (cal.getPadSubset()) {
      case CalType::calPadSubset::ROC: {
        break;
      }
      case CalType::PadSubset::Partition: {
        break;
      }
      case CalType::PadSubset::Region: {
        break;
      }
    }
  }

}

template <class T>
void Painter::Draw(CalArray<T>)
{
}
