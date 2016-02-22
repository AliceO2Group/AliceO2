#include <TSystem.h>
#include <FairMQLogger.h>
#include <FairMQTransportFactoryZMQ.h>
#include <chrono>
#include <thread>

#include "HistogramViewer.h"
#include "HistogramTMessage.h"

using namespace std;

HistogramViewer::HistogramViewer(std::string viewerId, int numIoThreads)
{
    this->SetTransport(new FairMQTransportFactoryZMQ);
    this->SetProperty(HistogramViewer::Id, viewerId);
    this->SetProperty(HistogramViewer::NumIoThreads, numIoThreads);   
}

void HistogramViewer::CustomCleanup(void *data, void *hint)
{
    delete (string*)hint;
}

void HistogramViewer::Run()
{
    mHistogramCanvas = new TCanvas("mHistogramCanvas", "Gauss histogram", 100, 10, 1200, 800);

    while (GetCurrentState() == RUNNING) {
        this_thread::sleep_for(chrono::milliseconds(1000));

        TH1F* receivedHistogram = receiveHistogramFromMerger();
        
        if (receivedHistogram != nullptr) {
            updateCanvas(receivedHistogram);
            updateHistogramCanvas(receivedHistogram); 
            sendReplyToMerger(new string("VIEWER_OK"));
        }
    }

    delete mHistogramCanvas;
}

void HistogramViewer::updateCanvas(TH1F* receivedHistogra)
{
    if (mNamesOfHistogramsToDraw.find(receivedHistogra->GetTitle()) == mNamesOfHistogramsToDraw.end()) {
 
        mNamesOfHistogramsToDraw.insert(receivedHistogra->GetTitle());
    
        mHistogramCanvas->Clear();
        mHistogramCanvas->Divide(mNamesOfHistogramsToDraw.size(), 1);
        mHistogramCanvas->cd(mNamesOfHistogramsToDraw.size());
        mHistogramCanvas->Update();
    }
    else {
        unsigned padId = mNamesOfHistogramsToDraw.size();
        for (auto const & name : mNamesOfHistogramsToDraw) {
            if (receivedHistogra->GetTitle() == name) {
                break;
            }
            padId--;
        }
        mHistogramCanvas->cd(padId);      
    }
}

void HistogramViewer::sendReplyToMerger(string* message)
{
    LOG(INFO) << "Sending reply to merger";
    FairMQMessage* reply = fTransportFactory->CreateMessage(const_cast<char*>(message->c_str()), 
                                                            message->length(), 
                                                            CustomCleanup, 
                                                            message);
    fChannels["data"].at(0).Send(reply);
}

TH1F* HistogramViewer::receiveHistogramFromMerger()
{
    TH1F* receivedHistogram;
    unique_ptr<FairMQMessage> request(move(receiveMessageFromMerger()));

    if (request->GetSize() != 0) {
        HistogramTMessage tm(request->GetData(), request->GetSize());
        receivedHistogram = static_cast<TH1F*>(tm.ReadObject(tm.GetClass()));
    }
    else {
        LOG(ERROR) << "Received empty request from merger";
        receivedHistogram = nullptr;
    }
    
    return receivedHistogram;
}

unique_ptr<FairMQMessage> HistogramViewer::receiveMessageFromMerger()
{
    unique_ptr<FairMQMessage> request(fTransportFactory->CreateMessage());
    fChannels["data"].at(0).Receive(&(*request));
    LOG(INFO) << "Received histogram from merger";
    return request;
}

void HistogramViewer::updateHistogramCanvas(TH1F* receivedHistogram) 
{
    receivedHistogram->Draw();
    mHistogramCanvas->Modified();
    mHistogramCanvas->Update();
    gSystem->ProcessEvents();
}

void HistogramViewer::executeRunLoop()
{
    ChangeState("INIT_DEVICE");
    WaitForEndOfState("INIT_DEVICE");

    ChangeState("INIT_TASK");
    WaitForEndOfState("INIT_TASK");

    ChangeState("RUN");
    InteractiveStateLoop();
}

void HistogramViewer::establishChannel(string type, string method, string address, string channelName)
{
    FairMQChannel requestChannel(type, method, address);
    requestChannel.UpdateSndBufSize(1000);
    requestChannel.UpdateRcvBufSize(1000);
    requestChannel.UpdateRateLogging(1);
    fChannels[channelName].push_back(requestChannel);
}
