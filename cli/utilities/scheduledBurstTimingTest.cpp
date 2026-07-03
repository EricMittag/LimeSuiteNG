#include "args.hxx"

#include "../common.h"
#include "../src/DSP/math/math.h"

#include "limesuiteng/RFStream.h"
#include "limesuiteng/SDRDevice.h"
#include "limesuiteng/SDRDescriptor.h"
#include "limesuiteng/StreamConfig.h"
#include "limesuiteng/StreamMeta.h"

#include <chrono>
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <iostream>
#include <limits>
#include <memory>
#include <string>
#include <string_view>
#include <vector>

#define _USE_MATH_DEFINES
#include <math.h>

using namespace std::literals;

namespace {

lime::LogLevel logVerbosity = lime::LogLevel::Error;

lime::LogLevel ParseLogLevel(std::string_view value)
{
    if (value == "debug"sv)
        return lime::LogLevel::Debug;
    if (value == "verbose"sv)
        return lime::LogLevel::Verbose;
    if (value == "info"sv)
        return lime::LogLevel::Info;
    if (value == "warning"sv)
        return lime::LogLevel::Warning;
    return lime::LogLevel::Error;
}

void LogCallback(lime::LogLevel level, const std::string& message)
{
    if (level > logVerbosity)
        return;
    std::cerr << message << '\n';
}

lime::complex32f_t Chirp(double startFreq, double stopFreq, double amplitude, double chirpDuration, double time)
{
    const double phase = startFreq * time + (stopFreq - startFreq) * time * time / (2 * chirpDuration);
    return lime::complex32f_t(amplitude * std::cos(phase), amplitude * std::sin(phase));
}

std::vector<lime::complex32f_t> GenerateChirp(double duration, double sampleRate, double startFreq, double stopFreq)
{
    std::vector<lime::complex32f_t> values;
    const double step = 1.0 / sampleRate;
    const double angularStart = startFreq * 2 * M_PI * sampleRate / 2;
    const double angularStop = stopFreq * 2 * M_PI * sampleRate / 2;
    constexpr double amplitude = 0.9;
    for (double time = 0; time <= duration; time += step)
        values.push_back(Chirp(angularStart, angularStop, amplitude, duration, time));
    return values;
}

std::vector<lime::complex32f_t> BuildChirp(std::size_t chirpLength, double sampleRate)
{
    const double chirpTime = static_cast<double>(chirpLength) / sampleRate;
    std::vector<lime::complex32f_t> chirp = GenerateChirp(chirpTime, sampleRate, 0.005, 0.04);
    chirp.resize(chirpLength);

    double scale = 0.0;
    for (std::size_t i = 0; i < chirp.size(); ++i)
    {
        chirp[i] = lime::complex32f_t(chirp[i].real() * scale, chirp[i].imag() * scale);
        if (i < chirp.size() / 2)
            scale += 2.0 / chirp.size();
        else
            scale -= 2.0 / chirp.size();
    }

    if (!chirp.empty())
    {
        chirp.front() = lime::complex32f_t(0.8f, -0.8f);
        if (chirp.size() > 1)
            chirp[chirp.size() - 2] = lime::complex32f_t(0.8f, -0.8f);
        chirp.back() = lime::complex32f_t(-0.8f, 0.8f);
    }

    return chirp;
}

template<class T> std::size_t GetMaxElementIndex(const std::vector<T>& values)
{
    std::size_t maxIndex = 0;
    for (std::size_t i = 1; i < values.size(); ++i)
    {
        if (values[i] > values[maxIndex])
            maxIndex = i;
    }
    return maxIndex;
}

struct ReceiveSummary {
    uint32_t samplesRead{ 0 };
    uint64_t firstTimestamp{ 0 };
    bool hasTimestamp{ false };
    bool timedOut{ false };
    uint32_t zeroReads{ 0 };
    uint32_t shortReads{ 0 };
    uint32_t missingTimestamps{ 0 };
    uint32_t continuityErrors{ 0 };
};

ReceiveSummary ReceiveExact(lime::RFStream* stream,
    std::vector<lime::complex32f_t>& storage,
    uint32_t targetSamples,
    uint32_t blockSamples)
{
    ReceiveSummary summary;
    storage.resize(targetSamples);
    lime::complex32f_t* buffer = storage.data();

    uint64_t expectedTimestamp = 0;
    bool expectTimestamp = false;
    uint32_t consecutiveZeroReads = 0;

    while (summary.samplesRead < targetSamples)
    {
        lime::StreamRxMeta meta{};
        const uint32_t request = std::min(blockSamples, targetSamples - summary.samplesRead);
        lime::complex32f_t* destination = buffer + summary.samplesRead;
        const uint32_t got = stream->Receive(&destination, request, &meta);
        if (got == 0)
        {
            ++summary.zeroReads;
            ++consecutiveZeroReads;
            if (consecutiveZeroReads >= 3)
            {
                summary.timedOut = true;
                break;
            }
            continue;
        }
        consecutiveZeroReads = 0;

        if (got != request)
            ++summary.shortReads;

        if (!meta.hasTimestamp)
            ++summary.missingTimestamps;
        else
        {
            const uint64_t timestamp = meta.timestamp.GetTicks();
            if (!summary.hasTimestamp)
            {
                summary.firstTimestamp = timestamp;
                summary.hasTimestamp = true;
            }
            if (expectTimestamp && timestamp != expectedTimestamp)
                ++summary.continuityErrors;
            expectedTimestamp = timestamp + got;
            expectTimestamp = true;
        }

        summary.samplesRead += got;
    }

    return summary;
}

struct BurstResult {
    uint32_t iteration{ 0 };
    uint64_t primeStartTimestamp{ 0 };
    uint32_t primeSamplesRead{ 0 };
    uint64_t txTimestamp{ 0 };
    uint32_t txSamplesSent{ 0 };
    uint64_t hwTimestampBeforeTx{ 0 };
    uint64_t hwTimestampAfterTx{ 0 };
    ReceiveSummary capture;
    int64_t expectedIndex{ 0 };
    int64_t foundIndex{ -1 };
    int64_t sampleDelta{ 0 };
    double timeDeltaUs{ 0.0 };
    double correlationPeak{ 0.0 };
    bool correlationWindowClipped{ false };
    lime::StreamStats rxStats{};
    lime::StreamStats txStats{};
};

BurstResult RunBurst(lime::SDRDevice* device,
    const lime::StreamConfig& streamCfg,
    int chipIndex,
    const std::vector<lime::complex32f_t>& chirp,
    uint32_t iteration,
    uint32_t leadSamples,
    uint32_t captureTailSamples,
    uint32_t captureBlockSamples,
    uint32_t searchRadiusSamples,
    double sampleRate)
{
    BurstResult result;
    result.iteration = iteration;

    std::unique_ptr<lime::RFStream> stream = device->StreamCreate(streamCfg, chipIndex);
    if (!stream)
        return result;

    if (stream->Start() != lime::OpStatus::Success)
        return result;

    result.txTimestamp = leadSamples;
    result.hwTimestampBeforeTx = stream->GetHardwareTimestamp();

    std::vector<const lime::complex32f_t*> txBuffers(1, chirp.data());
    lime::StreamTxMeta txMeta{};
    txMeta.hasTimestamp = true;
    txMeta.timestamp = lime::Timespec(static_cast<int64_t>(result.txTimestamp));
    txMeta.flags = lime::StreamTxMeta::StartOfBurst | lime::StreamTxMeta::EndOfBurst;
    result.txSamplesSent = stream->Transmit(txBuffers.data(), chirp.size(), &txMeta);
    result.hwTimestampAfterTx = stream->GetHardwareTimestamp();

    std::vector<lime::complex32f_t> captureBuffer;
    const uint32_t captureSamples = leadSamples + chirp.size() + captureTailSamples;
    result.capture = ReceiveExact(stream.get(), captureBuffer, captureSamples, captureBlockSamples);
    stream->StreamStatus(&result.rxStats, &result.txStats);
    stream->Stop();
    stream->Teardown();

    if (!result.capture.hasTimestamp || result.capture.samplesRead == 0)
        return result;

    result.expectedIndex = static_cast<int64_t>(result.txTimestamp) - static_cast<int64_t>(result.capture.firstTimestamp);
    const int64_t expectedStart = std::max<int64_t>(0, result.expectedIndex - static_cast<int64_t>(searchRadiusSamples));
    const int64_t expectedStop =
        std::min<int64_t>(result.capture.samplesRead, result.expectedIndex + static_cast<int64_t>(chirp.size() + searchRadiusSamples));

    if (expectedStart != result.expectedIndex - static_cast<int64_t>(searchRadiusSamples)
        || expectedStop != result.expectedIndex + static_cast<int64_t>(chirp.size() + searchRadiusSamples))
        result.correlationWindowClipped = true;

    if (expectedStop <= expectedStart)
        return result;

    std::vector<lime::complex32f_t> searchWindow(expectedStop - expectedStart);
    std::copy_n(captureBuffer.data() + expectedStart, searchWindow.size(), searchWindow.data());

    const std::vector<double> correlation = lime::CrossCorrelation(searchWindow, chirp);
    if (!correlation.empty())
    {
        const std::size_t peakIndex = GetMaxElementIndex(correlation);
        result.foundIndex = expectedStart + static_cast<int64_t>(peakIndex);
        result.sampleDelta = result.foundIndex - result.expectedIndex;
        result.timeDeltaUs = 1e6 * static_cast<double>(result.sampleDelta) / sampleRate;
        result.correlationPeak = correlation[peakIndex];
    }
    return result;
}

void PrintCsvHeader()
{
    std::cout << "iteration,prime_start,prime_samples,tx_timestamp,hw_before_tx,hw_after_tx,tx_sent,"
                 "capture_start,capture_samples,expected_index,found_index,sample_delta,time_delta_us,correlation,"
                 "capture_short,capture_zero,capture_missing_ts,capture_gap,rx_overrun,rx_loss,tx_underrun,tx_late,"
                 "rx_fifo_used,rx_fifo_total,tx_fifo_used,tx_fifo_total,window_clipped\n";
}

void PrintResultCsv(const BurstResult& result)
{
    std::cout << result.iteration << ','
              << result.primeStartTimestamp << ','
              << result.primeSamplesRead << ','
              << result.txTimestamp << ','
              << result.hwTimestampBeforeTx << ','
              << result.hwTimestampAfterTx << ','
              << result.txSamplesSent << ','
              << result.capture.firstTimestamp << ','
              << result.capture.samplesRead << ','
              << result.expectedIndex << ','
              << result.foundIndex << ','
              << result.sampleDelta << ','
              << result.timeDeltaUs << ','
              << result.correlationPeak << ','
              << result.capture.shortReads << ','
              << result.capture.zeroReads << ','
              << result.capture.missingTimestamps << ','
              << result.capture.continuityErrors << ','
              << result.rxStats.overrun << ','
              << result.rxStats.loss << ','
              << result.txStats.underrun << ','
              << result.txStats.late << ','
              << result.rxStats.FIFO.usedCount << ','
              << result.rxStats.FIFO.totalCount << ','
              << result.txStats.FIFO.usedCount << ','
              << result.txStats.FIFO.totalCount << ','
              << (result.correlationWindowClipped ? 1 : 0) << '\n';
}

void PrintResultHuman(const BurstResult& result)
{
    std::cout << "burst=" << result.iteration
              << " tx=" << result.txTimestamp
              << " capture=" << result.capture.firstTimestamp
              << " expected=" << result.expectedIndex
              << " found=" << result.foundIndex
              << " diff=" << result.sampleDelta
              << " (" << result.timeDeltaUs << " us)"
              << " corr=" << result.correlationPeak
              << " tx_sent=" << result.txSamplesSent
              << " rx_ov=" << result.rxStats.overrun
              << " rx_loss=" << result.rxStats.loss
              << " tx_ur=" << result.txStats.underrun
              << " tx_late=" << result.txStats.late
              << " rx_fifo=" << result.rxStats.FIFO.usedCount << '/' << result.rxStats.FIFO.totalCount
              << " tx_fifo=" << result.txStats.FIFO.usedCount << '/' << result.txStats.FIFO.totalCount;

    if (result.capture.shortReads || result.capture.zeroReads || result.capture.missingTimestamps || result.capture.continuityErrors
        || result.capture.timedOut)
    {
        std::cout << " capture_anom="
                  << "short:" << result.capture.shortReads
                  << " zero:" << result.capture.zeroReads
                  << " missing_ts:" << result.capture.missingTimestamps
                  << " gaps:" << result.capture.continuityErrors
                  << " timeout:" << result.capture.timedOut;
    }
    if (result.correlationWindowClipped)
        std::cout << " clipped_search=1";
    std::cout << '\n';
}

} // namespace

int main(int argc, char** argv)
{
    args::ArgumentParser parser("scheduledBurstTimingTest - repeated timestamped chirp timing tester", "");
    args::HelpFlag help(parser, "help", "Display this help menu", { 'h', "help" });

    args::ValueFlag<std::string> deviceFlag(parser, "name", "Specifies which device to use", { 'd', "device" }, "");
    args::ValueFlag<int> chipFlag(parser, "index", "RFSoC chip index", { 'c', "chip" }, 0);
    args::ValueFlag<int> channelFlag(parser, "index", "API channel index", { 'C', "channel" }, 0);
    args::ValueFlag<uint32_t> countFlag(parser, "count", "Number of chirp bursts to test", { 'n', "count" }, 32);
    args::ValueFlag<uint32_t> chirpLengthFlag(parser, "samples", "Chirp length in samples", { "chirpLength" }, 1024 * 16 + 137);
    args::ValueFlag<uint32_t> primeSamplesFlag(
        parser, "samples", "Compatibility option; ignored in fresh-stream burst mode", { "prime" }, 4096);
    args::ValueFlag<uint32_t> leadSamplesFlag(parser, "samples", "Samples between the prime read and the scheduled Tx timestamp (0 = sampleRate/1000)", { "lead" }, 0);
    args::ValueFlag<uint32_t> tailSamplesFlag(parser, "samples", "Rx samples to capture after the chirp (0 = 2.2 * sampleRate / 1000)", { "tail" }, 0);
    args::ValueFlag<uint32_t> captureBlockFlag(parser, "samples", "Samples per Receive() call while filling the capture window", { "captureBlock" }, 4096);
    args::ValueFlag<uint32_t> searchRadiusFlag(parser, "samples", "Correlation search radius around the expected chirp position", { "search" }, 8192);
    args::ValueFlag<int64_t> txOnOffsetFlag(parser, "samples", "TxOn offset", { "txon" }, 0);
    args::ValueFlag<int64_t> txOffOffsetFlag(parser, "samples", "TxOff offset", { "txoff" }, 0);
    args::Flag csvFlag(parser, "csv", "Print one CSV row per burst", { "csv" });
    args::ValueFlag<std::string> logFlag(parser, "level", "Log verbosity: info, warning, error, verbose, debug", { 'l', "log" }, "error");

    try
    {
        parser.ParseCLI(argc, argv);
    } catch (const args::Help&)
    {
        std::cout << parser;
        return EXIT_SUCCESS;
    } catch (const args::ParseError& error)
    {
        std::cerr << error.what() << '\n' << parser;
        return EXIT_FAILURE;
    }

    const std::string deviceName = args::get(deviceFlag);
    const int chipIndex = args::get(chipFlag);
    const int channelIndex = args::get(channelFlag);
    const uint32_t burstCount = args::get(countFlag);
    const uint32_t chirpLength = args::get(chirpLengthFlag);
    const uint32_t captureBlockSamples = args::get(captureBlockFlag);
    const uint32_t configuredSearchRadius = args::get(searchRadiusFlag);
    const bool csvMode = csvFlag;

    if (burstCount == 0 || chirpLength == 0 || captureBlockSamples == 0)
    {
        std::cerr << "count, chirpLength, and captureBlock must be greater than 0\n";
        return EXIT_FAILURE;
    }

    logVerbosity = ParseLogLevel(args::get(logFlag));
    lime::cli::logVerbosity = logVerbosity;
    lime::registerLogHandler(LogCallback);

    std::unique_ptr<lime::SDRDevice, void (*)(lime::SDRDevice*)> device(
        lime::cli::ConnectToFilteredOrDefaultDevice(deviceName), lime::DeviceRegistry::freeDevice);
    if (!device)
        return EXIT_FAILURE;

    device->SetMessageLogCallback(LogCallback);

    if (chipIndex < 0 || static_cast<std::size_t>(chipIndex) >= device->GetDescriptor().rfSOC.size())
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

    const uint32_t leadSamples =
        args::get(leadSamplesFlag) != 0 ? args::get(leadSamplesFlag) : static_cast<uint32_t>(sampleRate / 1000.0);
    const uint32_t tailSamples =
        args::get(tailSamplesFlag) != 0 ? args::get(tailSamplesFlag) : static_cast<uint32_t>(2.2 * sampleRate / 1000.0);
    const uint32_t searchRadiusSamples = std::max<uint32_t>(configuredSearchRadius, chirpLength / 4);

    std::vector<lime::complex32f_t> chirp = BuildChirp(chirpLength, sampleRate);

    lime::StreamConfig streamCfg;
    streamCfg.channels.at(lime::TRXDir::Rx).push_back(channelIndex);
    streamCfg.channels.at(lime::TRXDir::Tx).push_back(channelIndex);
    streamCfg.format = lime::DataFormat::F32;
    streamCfg.linkFormat = lime::DataFormat::I12;
    streamCfg.hintSampleRate = sampleRate;
    streamCfg.timestampType = lime::TimestampType::SAMPLE_TICKS;
    streamCfg.extraConfig.txonoffset = args::get(txOnOffsetFlag);
    streamCfg.extraConfig.txoffoffset = args::get(txOffOffsetFlag);

    if (csvMode)
        PrintCsvHeader();
    else
    {
        std::cout << "sample_rate=" << sampleRate
                  << " chirp=" << chirpLength
                  << " prime=0"
                  << " lead=" << leadSamples
                  << " tail=" << tailSamples
                  << " search=" << searchRadiusSamples
                  << " txon=" << streamCfg.extraConfig.txonoffset
                  << " txoff=" << streamCfg.extraConfig.txoffoffset << '\n';
    }

    int exitCode = EXIT_SUCCESS;
    for (uint32_t iteration = 0; iteration < burstCount; ++iteration)
    {
        const BurstResult result = RunBurst(
            device.get(), streamCfg, chipIndex, chirp, iteration, leadSamples, tailSamples, captureBlockSamples, searchRadiusSamples, sampleRate);

        if (csvMode)
            PrintResultCsv(result);
        else
            PrintResultHuman(result);

        if (result.txSamplesSent != chirp.size() || !result.capture.hasTimestamp || result.capture.timedOut
            || result.capture.continuityErrors != 0
            || result.rxStats.overrun != 0 || result.txStats.underrun != 0 || result.txStats.late != 0)
            exitCode = EXIT_FAILURE;
    }

    return exitCode;
}
