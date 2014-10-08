/**
 * runMerger.cxx
 *
 * @since 2012-12-06
 * @author D. Klein, A. Rybalchenko, M.Al-Turany
 */

#include <iostream>
#include <csignal>

#include "FairMQLogger.h"
#include "O2Merger.h"

#ifdef NANOMSG
  #include "FairMQTransportFactoryNN.h"
#else
  #include "FairMQTransportFactoryZMQ.h"
#endif

using std::cout;
using std::cin;
using std::endl;
using std::stringstream;


O2Merger merger;

static void s_signal_handler (int signal)
{
  cout << endl << "Caught signal " << signal << endl;

  merger.ChangeState(O2Merger::STOP);
  merger.ChangeState(O2Merger::END);

  cout << "Shutdown complete. Bye!" << endl;
  exit(1);
}

static void s_catch_signals (void)
{
  struct sigaction action;
  action.sa_handler = s_signal_handler;
  action.sa_flags = 0;
  sigemptyset(&action.sa_mask);
  sigaction(SIGINT, &action, NULL);
  sigaction(SIGTERM, &action, NULL);
}

int main(int argc, char** argv)
{
  if ( argc < 16 || (argc - 8) % 4 != 0 ) {
    cout << "Usage: merger \tID numIoTreads numInputs\n"
              << "\t\tinputSocketType inputRcvBufSize inputMethod inputAddress\n"
              << "\t\tinputSocketType inputRcvBufSize inputMethod inputAddress\n"
              << "\t\t...\n"
              << "\t\toutputSocketType outputSndBufSize outputMethod outputAddress\n"
              << argc << " arguments provided" << endl;
    return 1;
  }

  s_catch_signals();

  LOG(INFO) << "PID: " << getpid();

#ifdef NANOMSG
  FairMQTransportFactory* transportFactory = new FairMQTransportFactoryNN();
#else
  FairMQTransportFactory* transportFactory = new FairMQTransportFactoryZMQ();
#endif

  merger.SetTransport(transportFactory);

  int i = 1;

  merger.SetProperty(O2Merger::Id, argv[i]);
  ++i;

  int numIoThreads;
  stringstream(argv[i]) >> numIoThreads;
  merger.SetProperty(O2Merger::NumIoThreads, numIoThreads);
  ++i;

  int numInputs;
  stringstream(argv[i]) >> numInputs;
  merger.SetProperty(O2Merger::NumInputs, numInputs);
  ++i;

  merger.SetProperty(O2Merger::NumOutputs, 1);

  merger.ChangeState(O2Merger::INIT);

  for (int iInput = 0; iInput < numInputs; iInput++ ) {
    merger.SetProperty(O2Merger::InputSocketType, argv[i], iInput);
    ++i;
    int inputRcvBufSize;
    stringstream(argv[i]) >> inputRcvBufSize;
    merger.SetProperty(O2Merger::InputRcvBufSize, inputRcvBufSize, iInput);
    ++i;
    merger.SetProperty(O2Merger::InputMethod, argv[i], iInput);
    ++i;
    merger.SetProperty(O2Merger::InputAddress, argv[i], iInput);
    ++i;
  }

  merger.SetProperty(O2Merger::OutputSocketType, argv[i], 0);
  ++i;
  int outputSndBufSize;
  stringstream(argv[i]) >> outputSndBufSize;
  merger.SetProperty(O2Merger::OutputSndBufSize, outputSndBufSize, 0);
  ++i;
  merger.SetProperty(O2Merger::OutputMethod, argv[i], 0);
  ++i;
  merger.SetProperty(O2Merger::OutputAddress, argv[i], 0);
  ++i;

  merger.ChangeState(O2Merger::SETOUTPUT);
  merger.ChangeState(O2Merger::SETINPUT);
  merger.ChangeState(O2Merger::RUN);

  // wait until the running thread has finished processing.
  boost::unique_lock<boost::mutex> lock(merger.fRunningMutex);
  while (!merger.fRunningFinished)
  {
      merger.fRunningCondition.wait(lock);
  }

  merger.ChangeState(O2Merger::STOP);
  merger.ChangeState(O2Merger::END);

  return 0;
}

