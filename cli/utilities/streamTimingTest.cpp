#include "args.hxx"

#include "../common.h"

#include "limesuiteng/RFStream.h"
#include "limesuiteng/SDRDescriptor.h"
#include "limesuiteng/StreamConfig.h"
#include "limesuiteng/StreamMeta.h"

#include <atomic>
#include <chrono>
#include <csignal>
#include <cinttypes>
#include <cmath>
#include <iostream>
#include <limits>
#include <memory>
#include <thread>
#include <vector>

using namespace std::literals;

namespace {

std::atomic_bool gStopRequested{ false };

void SignalHandler(int)
{
    gStopRequested.store(true, std::memory_order_relaxed);
}

lime::DataFormat ParseLinkFormat(const std::string& value)
{
    if (value == "I16")
        return lime::DataFormat::I16;
    if (value == "I12")
        return lime::DataFormat::I12;
    throw std::invalid_argument("Invalid link format: "s + value);
}

struct TxLoadStats {
    uint64_t calls{ 0 };
    uint64_t samples{ 0 };
    uint64_t shortWrites{ 0 };
    uint64_t zeroWrites{ 0 };
};

class TxLoadWorker
{
  public:
    TxLoadWorker(lime::RFStream* stream, uint32_t blockSamples)
        : stream(stream)
        , pattern(blockSamples)
        , buffers(1, nullptr)
    {
        constexpr float amplitude = 2048.0f;
        constexpr lime::complex16_t symbols[4] = {
            lime::complex16_t(0, amplitude),
            lime::complex16_t(amplitude, 0),
            lime::complex16_t(0, -amplitude),
            lime::complex16_t(-amplitude, 0),
        };
        for (size_t i = 0; i < pattern.size(); ++i)
            pattern[i] = symbols[i % 4];
        buffers[0] = pattern.data();
    }

    void Start()
    {
        worker = std::thread(&TxLoadWorker::Run, this);
    }

    void Stop()
    {
        stopRequested.store(true, std::memory_order_relaxed);
        if (worker.joinable())
            worker.join();
    }

    TxLoadStats Snapshot() const
    {
        TxLoadStats snapshot;
        snapshot.calls = calls.load(std::memory_order_relaxed);
        snapshot.samples = samples.load(std::memory_order_relaxed);
        snapshot.shortWrites = shortWrites.load(std::memory_order_relaxed);
        snapshot.zeroWrites = zeroWrites.load(std::memory_order_relaxed);
        return snapshot;
    }

  private:
    void Run()
    {
        lime::StreamTxMeta meta{};
        meta.hasTimestamp = false;
        meta.flags = 0;
        while (!stopRequested.load(std::memory_order_relaxed) && !gStopRequested.load(std::memory_order_relaxed))
        {
            const uint32_t sent = stream->Transmit(buffers.data(), pattern.size(), &meta);
            calls.fetch_add(1, std::memory_order_relaxed);
            samples.fetch_add(sent, std::memory_order_relaxed);
            if (sent == 0)
            {
                zeroWrites.fetch_add(1, std::memory_order_relaxed);
                std::this_thread::yield();
                continue;
            }
            if (sent != pattern.size())
                shortWrites.fetch_add(1, std::memory_order_relaxed);
        }
    }

    lime::RFStream* stream;
    std::vector<lime::complex16_t> pattern;
    std::vector<const lime::complex16_t*> buffers;
    std::atomic_bool stopRequested{ false };
    std::thread worker;
    std::atomic<uint64_t> calls{ 0 };
    std::atomic<uint64_t> samples{ 0 };
    std::atomic<uint64_t> shortWrites{ 0 };
    std::atomic<uint64_t> zeroWrites{ 0 };
};

struct ContinuitySummary {
    uint64_t receiveCalls{ 0 };
    uint64_t samplesRead{ 0 };
    uint64_t shortReads{ 0 };
    uint64_t zeroReads{ 0 };
    uint64_t missingTimestamps{ 0 };
    uint64_t continuityErrors{ 0 };
    uint64_t overrunEvents{ 0 };
    uint64_t lossEvents{ 0 };
    int64_t minGap{ 0 };
    int64_t maxGap{ 0 };
};

void PrintHumanSummary(double elapsedSeconds,
    const ContinuitySummary& summary,
    const lime::StreamStats& rxStats,
    const lime::StreamStats& txStats,
    const TxLoadStats& txLoadStats,
    bool txEnabled)
{
    std::cout << "t=" << elapsedSeconds << "s"
              << " calls=" << summary.receiveCalls
              << " samples=" << summary.samplesRead
              << " short=" << summary.shortReads
              << " zero=" << summary.zeroReads
              << " gaps=" << summary.continuityErrors
              << " rx_ov=" << summary.overrunEvents
              << " rx_loss=" << summary.lossEvents
              << " rx_fifo=" << rxStats.FIFO.usedCount << "/" << rxStats.FIFO.totalCount;
    if (txEnabled)
    {
        std::cout << " tx_calls=" << txLoadStats.calls
                  << " tx_samples=" << txLoadStats.samples
                  << " tx_short=" << txLoadStats.shortWrites
                  << " tx_zero=" << txLoadStats.zeroWrites
                  << " tx_ur=" << txStats.underrun
                  << " tx_fifo=" << txStats.FIFO.usedCount << "/" << txStats.FIFO.totalCount;
    }
    std::cout << '\n';
}

} // namespace

int main(int argc, char** argv)
{
    args::ArgumentParser parser("streamTimingTest - continuous RX timestamp continuity tester", "");
    args::HelpFlag help(parser, "help", "Display this help menu", { 'h', "help" });

    args::ValueFlag<std::string> deviceFlag(parser, "name", "Specifies which device to use", { 'd', "device" }, "");
    args::ValueFlag<int> chipFlag(parser, "index", "RFSoC chip index", { 'c', "chip" }, 0);
    args::ValueFlag<int> channelFlag(parser, "index", "API channel index", { 'C', "channel" }, 0);
    args::ValueFlag<double> timeFlag(parser, "seconds", "Run time in seconds", { 't', "time" }, 10.0);
    args::ValueFlag<uint32_t> blockSizeFlag(parser, "samples", "Samples per Receive() call", { 's', "samples" }, 4096);
    args::ValueFlag<double> reportPeriodFlag(parser, "seconds", "Summary print period", { 'r', "report" }, 1.0);
    args::ValueFlag<int> maxAnomaliesFlag(parser, "count", "Maximum immediate anomaly lines to print in human-readable mode (-1 for unlimited)", { "maxAnomalies" }, 32);
    args::ValueFlag<std::string> linkFormatFlag(parser, "I16|I12", "Transport link format", { "linkFormat" }, "I12");
    args::ValueFlag<int> rxSamplesInPacketFlag(parser, "samples", "Rx samples per packet override", { "rxSamplesInPacket" }, 0);
    args::ValueFlag<int> txSamplesInPacketFlag(parser, "samples", "Tx samples per packet override", { "txSamplesInPacket" }, 0);
    args::ValueFlag<int> rxPacketsInBatchFlag(parser, "packets", "Rx packets per batch override", { "rxPacketsInBatch" }, 0);
    args::ValueFlag<int> txPacketsInBatchFlag(parser, "packets", "Tx packets per batch override", { "txPacketsInBatch" }, 0);
    args::Flag txLoadFlag(parser, "txload", "Continuously transmit a simple pattern while checking RX continuity", { "txload" });
    args::Flag csvFlag(parser, "csv", "Print one CSV row per Receive() call", { "csv" });
    args::Flag strictFlag(parser, "strict", "Exit with failure if any continuity or stream anomaly is detected", { "strict" });
    args::ValueFlag<std::string> logFlag(parser, "level", "Log verbosity: info, warning, error, verbose, debug", { 'l', "log" }, "error");

    try
    {
        parser.ParseCLI(argc, argv);
    } catch (const args::Help&)
    {
        std::cout << parser;
        return EXIT_SUCCESS;
    } catch (const args::ParseError& e)
    {
        std::cerr << e.what() << '\n' << parser;
        return EXIT_FAILURE;
    }

    const std::string deviceName = args::get(deviceFlag);
    const int chipIndex = args::get(chipFlag);
    const int channelIndex = args::get(channelFlag);
    const double runtimeSeconds = args::get(timeFlag);
    const uint32_t blockSamples = args::get(blockSizeFlag);
    const double reportPeriodSeconds = args::get(reportPeriodFlag);
    const bool useTxLoad = txLoadFlag;
    const bool csvMode = csvFlag;
    const bool strictMode = strictFlag;
    const int maxAnomalies = args::get(maxAnomaliesFlag);

    if (blockSamples == 0)
    {
        std::cerr << "samples must be greater than 0\n";
        return EXIT_FAILURE;
    }
    if (runtimeSeconds <= 0)
    {
        std::cerr << "time must be greater than 0\n";
        return EXIT_FAILURE;
    }

    lime::cli::logVerbosity = lime::cli::strToLogLevel(args::get(logFlag));
    lime::registerLogHandler(lime::cli::LogCallback);

    std::unique_ptr<lime::SDRDevice, void (*)(lime::SDRDevice*)> device(
        lime::cli::ConnectToFilteredOrDefaultDevice(deviceName), lime::DeviceRegistry::freeDevice);
    if (!device)
        return EXIT_FAILURE;

    device->SetMessageLogCallback(lime::cli::LogCallback);

    if (chipIndex < 0 || static_cast<size_t>(chipIndex) >= device->GetDescriptor().rfSOC.size())
    {
        std::cerr << "Invalid chip index " << chipIndex << '\n';
        return EXIT_FAILURE;
    }

    const int channelCount = device->GetDescriptor().rfSOC[chipIndex].channelCount;
    if (channelIndex < 0 || channelIndex >= channelCount)
    {
        std::cerr << "Invalid channel index " << channelIndex << " for chip " << chipIndex << '\n';
        return EXIT_FAILURE;
    }

    double sampleRate = device->GetSampleRate(chipIndex, lime::TRXDir::Rx, channelIndex);
    if (sampleRate <= 0)
    {
        sampleRate = 1e6;
        std::cerr << "Sample-rate readback unavailable, assuming 1000000 Hz for timestamp tick rate\n";
    }

    lime::StreamConfig streamCfg;
    streamCfg.channels.at(lime::TRXDir::Rx).push_back(channelIndex);
    if (useTxLoad)
        streamCfg.channels.at(lime::TRXDir::Tx).push_back(channelIndex);
    streamCfg.format = lime::DataFormat::I16;
    try
    {
        streamCfg.linkFormat = ParseLinkFormat(args::get(linkFormatFlag));
    } catch (const std::invalid_argument& e)
    {
        std::cerr << e.what() << '\n';
        return EXIT_FAILURE;
    }
    streamCfg.hintSampleRate = sampleRate;
    streamCfg.timestampType = lime::TimestampType::SAMPLE_TICKS;
    streamCfg.extraConfig.rx.samplesInPacket = args::get(rxSamplesInPacketFlag);
    streamCfg.extraConfig.tx.samplesInPacket = args::get(txSamplesInPacketFlag);
    streamCfg.extraConfig.rx.packetsInBatch = args::get(rxPacketsInBatchFlag);
    streamCfg.extraConfig.tx.packetsInBatch = args::get(txPacketsInBatchFlag);

    auto stream = device->StreamCreate(streamCfg, chipIndex);
    if (!stream)
    {
        std::cerr << "Failed to create stream\n";
        return EXIT_FAILURE;
    }

    std::signal(SIGINT, SignalHandler);

    std::vector<lime::complex16_t> rxData(blockSamples);
    std::vector<lime::complex16_t*> rxBuffers(1, rxData.data());

    lime::StreamStats rxStats{};
    lime::StreamStats txStats{};
    TxLoadStats txLoadStats{};
    ContinuitySummary summary{};

    uint32_t prevRxOverrun = 0;
    uint32_t prevRxLoss = 0;
    bool havePreviousTimestamp = false;
    uint64_t previousTimestamp = 0;
    uint32_t previousSamplesRead = 0;

    const auto startWall = std::chrono::steady_clock::now();
    auto lastReport = startWall;
    auto previousCallEnd = startWall;
    bool anomalyOutputSuppressed = false;
    int printedAnomalies = 0;

    if (csvMode)
    {
        std::cout << "call,samples_read,rx_timestamp,expected_timestamp,timestamp_gap,call_us,idle_us,hw_timestamp,"
                     "rx_overrun,rx_loss,rx_fifo_used,rx_fifo_total,tx_underrun,tx_fifo_used,tx_fifo_total\n";
    }

    stream->Start();

    TxLoadWorker txLoadWorker(stream.get(), blockSamples);
    if (useTxLoad)
        txLoadWorker.Start();

    while (!gStopRequested.load(std::memory_order_relaxed))
    {
        const auto now = std::chrono::steady_clock::now();
        const auto elapsed = std::chrono::duration<double>(now - startWall).count();
        if (elapsed >= runtimeSeconds)
            break;

        lime::StreamRxMeta rxMeta{};
        const auto callStart = std::chrono::steady_clock::now();
        const uint32_t samplesRead = stream->Receive(rxBuffers.data(), blockSamples, &rxMeta);
        const auto callEnd = std::chrono::steady_clock::now();
        const auto callUs = std::chrono::duration_cast<std::chrono::microseconds>(callEnd - callStart).count();
        const auto idleUs = std::chrono::duration_cast<std::chrono::microseconds>(callStart - previousCallEnd).count();
        previousCallEnd = callEnd;

        stream->StreamStatus(&rxStats, useTxLoad ? &txStats : nullptr);
        if (useTxLoad)
            txLoadStats = txLoadWorker.Snapshot();

        ++summary.receiveCalls;
        summary.samplesRead += samplesRead;
        if (samplesRead == 0)
            ++summary.zeroReads;
        else if (samplesRead != blockSamples)
            ++summary.shortReads;

        if (!rxMeta.hasTimestamp)
            ++summary.missingTimestamps;

        uint64_t rxTimestamp = rxMeta.hasTimestamp ? rxMeta.timestamp.GetTicks() : 0;
        uint64_t expectedTimestamp = 0;
        int64_t timestampGap = 0;
        bool continuityError = false;

        if (havePreviousTimestamp && rxMeta.hasTimestamp && samplesRead > 0)
        {
            expectedTimestamp = previousTimestamp + previousSamplesRead;
            timestampGap = static_cast<int64_t>(rxTimestamp) - static_cast<int64_t>(expectedTimestamp);
            if (timestampGap != 0)
            {
                continuityError = true;
                ++summary.continuityErrors;
                summary.minGap = std::min(summary.minGap, timestampGap);
                summary.maxGap = std::max(summary.maxGap, timestampGap);
            }
        }

        const uint32_t overrunDelta = rxStats.overrun - prevRxOverrun;
        const uint32_t lossDelta = rxStats.loss - prevRxLoss;
        summary.overrunEvents += overrunDelta;
        summary.lossEvents += lossDelta;
        prevRxOverrun = rxStats.overrun;
        prevRxLoss = rxStats.loss;

        if (csvMode)
        {
            std::cout << summary.receiveCalls << ','
                      << samplesRead << ','
                      << rxTimestamp << ','
                      << expectedTimestamp << ','
                      << timestampGap << ','
                      << callUs << ','
                      << idleUs << ','
                      << stream->GetHardwareTimestamp() << ','
                      << rxStats.overrun << ','
                      << rxStats.loss << ','
                      << rxStats.FIFO.usedCount << ','
                      << rxStats.FIFO.totalCount << ','
                      << txStats.underrun << ','
                      << txStats.FIFO.usedCount << ','
                      << txStats.FIFO.totalCount << '\n';
        }
        else if (continuityError || samplesRead == 0 || samplesRead != blockSamples || overrunDelta != 0 || lossDelta != 0)
        {
            if (maxAnomalies < 0 || printedAnomalies < maxAnomalies)
            {
                ++printedAnomalies;
                std::cout << "anomaly"
                          << " call=" << summary.receiveCalls
                          << " samples=" << samplesRead
                          << " ts=" << rxTimestamp
                          << " expected=" << expectedTimestamp
                          << " gap=" << timestampGap
                          << " call_us=" << callUs
                          << " idle_us=" << idleUs
                          << " hwts=" << stream->GetHardwareTimestamp()
                          << " rx_ov_delta=" << overrunDelta
                          << " rx_loss_delta=" << lossDelta;
                if (useTxLoad)
                {
                    std::cout << " tx_short=" << txLoadStats.shortWrites
                              << " tx_zero=" << txLoadStats.zeroWrites
                              << " tx_ur=" << txStats.underrun;
                }
                std::cout << '\n';
            }
            else if (!anomalyOutputSuppressed)
            {
                anomalyOutputSuppressed = true;
                std::cout << "additional anomalies suppressed; use --csv for every Receive() record\n";
            }
        }

        if (rxMeta.hasTimestamp && samplesRead > 0)
        {
            previousTimestamp = rxTimestamp;
            previousSamplesRead = samplesRead;
            havePreviousTimestamp = true;
        }

        const auto reportElapsed = std::chrono::duration<double>(callEnd - lastReport).count();
        if (reportElapsed >= reportPeriodSeconds)
        {
            lastReport = callEnd;
            PrintHumanSummary(std::chrono::duration<double>(callEnd - startWall).count(), summary, rxStats, txStats, txLoadStats, useTxLoad);
        }
    }

    if (useTxLoad)
        txLoadWorker.Stop();
    stream->Stop();

    PrintHumanSummary(
        std::chrono::duration<double>(std::chrono::steady_clock::now() - startWall).count(), summary, rxStats, txStats, txLoadStats, useTxLoad);

    std::cout << "final"
              << " calls=" << summary.receiveCalls
              << " samples=" << summary.samplesRead
              << " short=" << summary.shortReads
              << " zero=" << summary.zeroReads
              << " missing_ts=" << summary.missingTimestamps
              << " continuity_errors=" << summary.continuityErrors
              << " min_gap=" << summary.minGap
              << " max_gap=" << summary.maxGap
              << " overrun_events=" << summary.overrunEvents
              << " loss_events=" << summary.lossEvents;
    if (useTxLoad)
    {
        std::cout << " tx_calls=" << txLoadStats.calls
                  << " tx_samples=" << txLoadStats.samples
                  << " tx_short=" << txLoadStats.shortWrites
                  << " tx_zero=" << txLoadStats.zeroWrites
                  << " tx_underrun=" << txStats.underrun;
    }
    std::cout << '\n';

    const bool failed =
        summary.continuityErrors != 0 || summary.zeroReads != 0 || summary.shortReads != 0 || summary.overrunEvents != 0 ||
        summary.lossEvents != 0 || summary.missingTimestamps != 0 || (useTxLoad && (txLoadStats.zeroWrites != 0 || txStats.underrun != 0));

    if (strictMode && failed)
        return EXIT_FAILURE;

    return EXIT_SUCCESS;
}
