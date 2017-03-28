/// \file Digitizer.cxx
/// \author Andi Mathis, andreas.mathis@ph.tum.de

#include "TPCSimulation/Digitizer.h"
#include "TPCSimulation/Constants.h"
#include "TPCSimulation/ElectronTransport.h"
#include "TPCSimulation/GEMAmplification.h"
#include "TPCSimulation/PadResponse.h"
#include "TPCSimulation/Point.h"
#include "TPCSimulation/SAMPAProcessing.h"

#include "TPCBase/Mapper.h"

#include "FairLogger.h"

ClassImp(AliceO2::TPC::Digitizer)

using namespace AliceO2::TPC;

bool AliceO2::TPC::Digitizer::mDebugFlagPRF = false;

Digitizer::Digitizer()
  : TObject()
  , mDigitContainer(nullptr)
  , mDebugTreePRF(nullptr)
{}

Digitizer::~Digitizer()
{
  delete mDigitContainer;
}

void Digitizer::init()
{
  /// @todo get rid of new? check with Mohammad
  mDigitContainer = new DigitContainer();

//  mDebugTreePRF = std::unique_ptr<TTree> (new TTree("PRFdebug", "PRFdebug"));
//  mDebugTreePRF->Branch("GEMresponse", &GEMresponse, "CRU:timeBin:row:pad:nElectrons");
}

DigitContainer *Digitizer::Process(TClonesArray *points)
{
  mDigitContainer->reset();
  const Mapper& mapper = Mapper::instance();

  /// @todo static_thread for thread savety?
  static GEMAmplification gemAmplification;
  static ElectronTransport electronTransport;
  static PadResponse padResponse;

  static std::array<float, mNShapedPoints> signalArray;

  for(auto pointObject : *points) {
    Point *inputpoint = static_cast<Point *>(pointObject);

    const GlobalPosition3D posEle(inputpoint->GetX(), inputpoint->GetY(), inputpoint->GetZ());
    int nPrimaryElectrons = static_cast<int>(inputpoint->GetEnergyLoss()/WION);

    int MCEventID = inputpoint->GetEventID();
    int MCTrackID = inputpoint->GetTrackID();

    /// Loop over electrons
    /// @todo can be vectorized?
    /// @todo split transport and signal formation in two separate loops?
    for(int iEle=0; iEle < nPrimaryElectrons; ++iEle) {

      /// Drift and Diffusion
      const GlobalPosition3D posEleDiff = electronTransport.getElectronDrift(posEle);

      /// @todo Time management in continuous mode (adding the time of the event?)
      const float driftTime = getTime(posEleDiff.getZ());

      /// Attachment
      if(electronTransport.isElectronAttachment(driftTime)) continue;

      /// Remove electrons that end up outside the active volume
      /// @todo should go to mapper?
      if(fabs(posEleDiff.getZ()) > TPCLENGTH) continue;

      const DigitPos digiPadPos = mapper.findDigitPosFromGlobalPosition(posEleDiff);
      if(!digiPadPos.isValid()) continue;

      const int nElectronsGEM = gemAmplification.getStackAmplification();

      /// Loop over all individual pads with signal due to pad response function
      /// Currently the PRF is not applied yet due to some problems with the mapper
      /// which results in most of the cases in a normalized pad response = 0
      /// @todo Problems of the mapper to be fixed
      /// @todo Mapper should provide a functionality which finds the adjacent pads of a given pad
      // for(int ipad = -2; ipad<3; ++ipad) {
      //   for(int irow = -2; irow<3; ++irow) {
      //     PadPos padPos(digiPadPos.getPadPos().getRow() + irow, digiPadPos.getPadPos().getPad() + ipad);
      //     DigitPos digiPos(digiPadPos.getCRU(), padPos);

      DigitPos digiPos = digiPadPos;
      if (!digiPos.isValid()) continue;
      // const float normalizedPadResponse = padResponse.getPadResponse(posEleDiff, digiPos);

      const float normalizedPadResponse = 1.f;
      if (normalizedPadResponse <= 0) continue;
      const int pad = digiPos.getPadPos().getPad();
      const int row = digiPos.getPadPos().getRow();

      if(mDebugFlagPRF) {
        /// @todo Write out the debug output
        GEMresponse.CRU = digiPos.getCRU().number();
        GEMresponse.time = driftTime;
        GEMresponse.row = row;
        GEMresponse.pad = pad;
        GEMresponse.nElectrons = nElectronsGEM * normalizedPadResponse;
        //mDebugTreePRF->Fill();
      }

      const float ADCsignal = SAMPAProcessing::getADCvalue(nElectronsGEM * normalizedPadResponse);
      SAMPAProcessing::getShapedSignal(ADCsignal, driftTime, signalArray);

      for(float i=0; i<mNShapedPoints; ++i) {
        float time = driftTime + i * ZBINWIDTH;
        mDigitContainer->addDigit(MCEventID, MCTrackID, digiPos.getCRU().number(), getTimeBinFromTime(time), row, pad, signalArray[i]);
      }

      // }
      // }
      /// end of loop over prf
    }
    /// end of loop over electrons
  }
  /// end of loop over points

  return mDigitContainer;
}
