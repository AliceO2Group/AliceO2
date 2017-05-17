#ifndef ALICEO2_TPC_CONVERTRAWCLUSTERS_C_
#define ALICEO2_TPC_CONVERTRAWCLUSTERS_C_

/// \file   RawClusterFinder.h
/// \author Jens Wiechula, Jens.Wiechula@ikf.uni-frankfurt.de

#include <vector>
#include <memory>

#include "Rtypes.h"
#include "TClonesArray.h"
#include "TFile.h"

#include "TPCBase/Defs.h"
#include "TPCBase/CalDet.h"
#include "TPCBase/CRU.h"
#include "TPCCalibration/CalibRawBase.h"
#include "TPCSimulation/HwClusterer.h"
#include "TPCSimulation/BoxClusterer.h"
#include "TPCSimulation/ClusterContainer.h"
#include "TPCSimulation/DigitMC.h"

namespace o2
{
namespace TPC
{


/// \brief Raw cluster conversion
///
/// This class is used to produce pad wise pedestal and noise calibration data
///
/// origin: TPC
/// \author Jens Wiechula, Jens.Wiechula@ikf.uni-frankfurt.de

class RawClusterFinder : public CalibRawBase
{
  public:
    enum class ClustererType : char {
      Box,  ///< use box clusterer
      HW    ///< use HW clusterer
    };

    using vectorType = std::vector<float>;

    /// default constructor
    RawClusterFinder(PadSubset padSubset = PadSubset::ROC) : CalibRawBase(padSubset), mClustererType(ClustererType::HW), mPedestals(nullptr), mArrDigits("o2::TPC::DigitMC") {;}

    /// default destructor
    virtual ~RawClusterFinder() = default;

    /// set clusterer type
    void setClustererType(ClustererType clustererType) { mClustererType = clustererType; }

    /// not used
    Int_t UpdateROC(const Int_t sector, const Int_t row, const Int_t pad,
                    const Int_t timeBin, const Float_t signal) final { return 0;}

    /// 
    Int_t UpdateCRU(const CRU& cru, const Int_t row, const Int_t pad,
                    const Int_t timeBin, const Float_t signal) final;

    void setPedestals(CalPad* pedestals) { mPedestals = pedestals; }
    static void ProcessEvents(TString fileInfo, TString pedestalFile, TString outputFileName="clusters.root", Int_t maxEvents=-1, ClustererType clustererType=ClustererType::HW);

    TClonesArray& getDigitArray() { return mArrDigits; }

    /// Dummy end event
    virtual void EndEvent() final {};

  private:
    ClustererType     mClustererType;
    CalPad       *mPedestals;
    TClonesArray mArrDigits;

    /// dummy reset
    void ResetEvent() final { mArrDigits.Clear(); }
};

Int_t RawClusterFinder::UpdateCRU(const CRU& cru, const Int_t row, const Int_t pad,
                                     const Int_t timeBin, const Float_t signal)
{
  float corrSignal = signal;

  // ===| get pedestal |========================================================
  if (mPedestals) {
    corrSignal -= mPedestals->getValue(cru, row, pad);
  }

  // ===| add new digit |=======================================================
  const size_t nDig = mArrDigits.GetEntries();
  new (mArrDigits[nDig]) DigitMC(cru, signal, row, pad, timeBin);

  return 1;
}

void RawClusterFinder::ProcessEvents(TString fileInfo, TString pedestalFile, TString outputFileName, Int_t maxEvents, ClustererType clustererType)
{

  // ===| create raw converter |================================================
  RawClusterFinder converter;
  converter.setupContainers(fileInfo);

  // ===| load pedestals |======================================================
  TFile f(pedestalFile);
  CalDet<float> *pedestal = nullptr;
  if (f.IsOpen() && !f.IsZombie()) {
    f.GetObject("Pedestals", pedestal);
    printf("pedestal: %.2f\n", pedestal->getValue(CRU(0), 0, 0));
  }

  // ===| output file and container |===========================================
  TClonesArray arrCluster("o2::TPC::Cluster");
  TFile fout(outputFileName,"recreate");
  TTree t("cbmsim","cbmsim");
  t.Branch("TPCClusterHW", &arrCluster);

  // ===| cluster finder |======================================================
  // HW cluster finder
  std::unique_ptr<Clusterer> cl;
  if (clustererType == ClustererType::HW) {
    HwClusterer *hwCl = new HwClusterer;
    hwCl->setContinuousReadout(false);
    hwCl->setPedestalObject(pedestal);
    cl = std::unique_ptr<Clusterer>(hwCl);
  }
  else if (clustererType == ClustererType::Box) {
    BoxClusterer *boxCl = new BoxClusterer;
    boxCl->setPedestals(pedestal);
    cl = std::unique_ptr<Clusterer>(boxCl);
  }
  else {
    return;
  }
    
  cl->Init();

  // Box cluster finder
  
  // ===| loop over all data |==================================================
  int events = 0;
  bool data = true;
  while ((converter.ProcessEvent() == CalibRawBase::ProcessStatus::Ok) && (maxEvents>0)?events<maxEvents:1) {

    TClonesArray &arr = converter.getDigitArray();
    printf("========| Event %4zu |========\n", converter.getNumberOfProcessedEvents());
    printf("Converted digits: %d %f\n", arr.GetEntriesFast(), ((DigitMC*)arr.At(0))->getChargeFloat());
    //for (Int_t i=0; i<10; ++i) {
      //printf("%.2f ", ((DigitMC*)arr.At(i))->getChargeFloat());
    //}
    //printf("\n");


    ClusterContainer* clCont = cl->Process(&arr);

    clCont->FillOutputContainer(&arrCluster);
    t.Fill();

    printf("Found clusters: %d\n", arrCluster.GetEntriesFast());
    arrCluster.Clear();
    ++events;
  }


  fout.Write();
  fout.Close();
}

} // namespace TPC

} // namespace o2
#endif
