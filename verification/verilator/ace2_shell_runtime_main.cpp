#include <openssl/sha.h>

#include <algorithm>
#include <array>
#include <cfenv>
#include <cerrno>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <map>
#include <optional>
#include <sstream>
#include <stdexcept>
#include <string>
#include <type_traits>
#include <unordered_map>
#include <utility>
#include <vector>

#include <fcntl.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <sys/types.h>
#include <unistd.h>

#include "Vace2_shell_runtime_harness.h"
#include "verilated.h"

namespace fs = std::filesystem;

namespace {

constexpr std::uint32_t kPackageVersion = 1;
constexpr std::uint32_t kJournalVersion = 1;
constexpr std::uint64_t kRopeTableBase = 0x0000000400000000ULL;
constexpr std::uint64_t kRmsnormBase = 0x0000000300000000ULL;
constexpr std::uint64_t kRmsnormGainBytes = 1792;
constexpr std::uint64_t kRmsnormScaleOffset = kRmsnormGainBytes + 8;
constexpr std::uint32_t kVocab = 151936;
constexpr std::uint32_t kHidden = 896;
constexpr std::uint32_t kLmTile = 32;
constexpr std::uint32_t kLastLmTile = kVocab / kLmTile - 1;
constexpr std::uint32_t kCsrControl = 0x18;
constexpr std::uint64_t kDefaultTimeoutCycles = 100000000ULL;

const std::array<const char*, 21> kOperatorNames = {
    "input_rmsnorm",
    "q_proj",
    "k_proj",
    "v_proj",
    "rope_q",
    "rope_k",
    "kv_write",
    "attention_score",
    "softmax",
    "attention_value",
    "attention_compose",
    "o_proj",
    "attention_residual_add",
    "post_attention_rmsnorm",
    "mlp_gate_proj",
    "mlp_up_proj",
    "silu_gate",
    "mlp_down_proj",
    "mlp_residual_add",
    "final_rmsnorm",
    "lm_head_tile",
};

struct BoundaryError : public std::runtime_error {
    std::string category;
    std::optional<std::uint64_t> address;

    BoundaryError(std::string category_in, std::string detail)
        : std::runtime_error(std::move(detail)), category(std::move(category_in)) {}

    BoundaryError(std::string category_in, std::string detail, std::uint64_t address_in)
        : std::runtime_error(std::move(detail)),
          category(std::move(category_in)),
          address(address_in) {}
};

std::string errno_string(const std::string& prefix) {
    return prefix + ": " + std::strerror(errno);
}

void write_all(int fd, const std::uint8_t* data, std::size_t size) {
    while (size != 0) {
        const ssize_t written = ::write(fd, data, size);
        if (written < 0) {
            if (errno == EINTR) {
                continue;
            }
            throw std::runtime_error(errno_string("write failed"));
        }
        data += written;
        size -= static_cast<std::size_t>(written);
    }
}

void write_all(int fd, const std::string& text) {
    write_all(fd, reinterpret_cast<const std::uint8_t*>(text.data()), text.size());
}

void fsync_directory(const fs::path& path) {
    const fs::path directory = path.parent_path().empty() ? fs::path(".") : path.parent_path();
    const int fd = ::open(directory.c_str(), O_RDONLY | O_DIRECTORY);
    if (fd < 0) {
        throw std::runtime_error(errno_string("open directory for fsync failed"));
    }
    if (::fsync(fd) != 0) {
        const std::string message = errno_string("directory fsync failed");
        ::close(fd);
        throw std::runtime_error(message);
    }
    ::close(fd);
}

void write_atomic(const fs::path& path, const std::string& text) {
    fs::create_directories(path.parent_path());
    const fs::path temporary = path.string() + ".tmp";
    const int fd = ::open(temporary.c_str(), O_WRONLY | O_CREAT | O_TRUNC, 0644);
    if (fd < 0) {
        throw std::runtime_error(errno_string("open temporary file failed"));
    }
    try {
        write_all(fd, text);
        if (::fsync(fd) != 0) {
            throw std::runtime_error(errno_string("file fsync failed"));
        }
        if (::close(fd) != 0) {
            throw std::runtime_error(errno_string("close temporary file failed"));
        }
    } catch (...) {
        ::close(fd);
        ::unlink(temporary.c_str());
        throw;
    }
    if (::rename(temporary.c_str(), path.c_str()) != 0) {
        ::unlink(temporary.c_str());
        throw std::runtime_error(errno_string("atomic rename failed"));
    }
    fsync_directory(path);
}

std::string json_escape(const std::string& value) {
    std::ostringstream out;
    for (const unsigned char ch : value) {
        switch (ch) {
            case '\\': out << "\\\\"; break;
            case '"': out << "\\\""; break;
            case '\n': out << "\\n"; break;
            case '\r': out << "\\r"; break;
            case '\t': out << "\\t"; break;
            default:
                if (ch < 0x20) {
                    out << "\\u" << std::hex << std::setw(4) << std::setfill('0')
                        << static_cast<unsigned>(ch) << std::dec;
                } else {
                    out << ch;
                }
        }
    }
    return out.str();
}

std::string hex_digest(const std::array<std::uint8_t, SHA256_DIGEST_LENGTH>& digest) {
    std::ostringstream out;
    out << std::hex << std::setfill('0');
    for (const std::uint8_t byte : digest) {
        out << std::setw(2) << static_cast<unsigned>(byte);
    }
    return out.str();
}

std::array<std::uint8_t, SHA256_DIGEST_LENGTH> sha256_bytes(
    const std::uint8_t* data,
    std::size_t size
) {
    std::array<std::uint8_t, SHA256_DIGEST_LENGTH> digest{};
    SHA256(data, size, digest.data());
    return digest;
}

template <typename T>
T read_le(const std::vector<std::uint8_t>& data, std::size_t& offset) {
    static_assert(std::is_integral<T>::value, "read_le requires an integer type");
    if (offset + sizeof(T) > data.size()) {
        throw std::runtime_error("truncated little-endian value");
    }
    using U = typename std::make_unsigned<T>::type;
    U value = 0;
    for (std::size_t index = 0; index < sizeof(T); ++index) {
        value |= static_cast<U>(data[offset + index]) << (index * 8);
    }
    offset += sizeof(T);
    return static_cast<T>(value);
}

template <typename T>
void append_le(std::vector<std::uint8_t>& data, T raw_value) {
    static_assert(std::is_integral<T>::value, "append_le requires an integer type");
    using U = typename std::make_unsigned<T>::type;
    const U value = static_cast<U>(raw_value);
    for (std::size_t index = 0; index < sizeof(T); ++index) {
        data.push_back(static_cast<std::uint8_t>((value >> (index * 8)) & 0xffU));
    }
}

std::vector<std::uint8_t> read_file(const fs::path& path) {
    std::ifstream stream(path, std::ios::binary);
    if (!stream) {
        throw std::runtime_error("cannot open file: " + path.string());
    }
    stream.seekg(0, std::ios::end);
    const auto end = stream.tellg();
    if (end < 0) {
        throw std::runtime_error("cannot size file: " + path.string());
    }
    std::vector<std::uint8_t> data(static_cast<std::size_t>(end));
    stream.seekg(0, std::ios::beg);
    if (!data.empty()) {
        stream.read(reinterpret_cast<char*>(data.data()), data.size());
    }
    if (!stream) {
        throw std::runtime_error("cannot read file: " + path.string());
    }
    return data;
}

struct Command {
    std::uint32_t ordinal = 0;
    std::uint16_t token_step = 0;
    std::uint8_t layer_id = 0;
    std::uint8_t operator_id = 0;
    std::uint8_t opcode = 0;
    std::uint8_t flags = 0;
    std::uint16_t m = 0;
    std::uint16_t n = 0;
    std::uint16_t k = 0;
    std::uint16_t sequence_position = 0;
    std::uint16_t completion_tag = 0;
    std::int16_t query_head = -1;
    std::int16_t context_token = -1;
    std::int32_t vocab_tile = -1;
    std::uint64_t src0 = 0;
    std::uint64_t src1 = 0;
    std::uint64_t dst = 0;
    std::uint64_t scale = 0;
    std::uint64_t scratch = 0;

    const char* operator_name() const {
        if (operator_id >= kOperatorNames.size()) {
            return "invalid_operator";
        }
        return kOperatorNames[operator_id];
    }
};

struct Package {
    std::uint32_t seed_token = 0;
    std::uint32_t embedding_rows = 0;
    std::uint32_t embedding_cols = 0;
    std::uint64_t embedding_offset = 0;
    std::array<std::uint8_t, 32> schedule_sha{};
    std::array<std::uint8_t, 32> image_sha{};
    std::array<std::uint8_t, 32> model_sha{};
    std::array<std::uint8_t, 256> rope_records{};
    std::vector<Command> commands;
};

Package load_package(const fs::path& path) {
    const std::vector<std::uint8_t> data = read_file(path);
    std::size_t offset = 0;
    if (data.size() < 8 || std::memcmp(data.data(), "ACE2RT1", 7) != 0) {
        throw std::runtime_error("runtime package magic differs");
    }
    offset += 8;
    const std::uint32_t version = read_le<std::uint32_t>(data, offset);
    const std::uint32_t record_bytes = read_le<std::uint32_t>(data, offset);
    const std::uint32_t command_count = read_le<std::uint32_t>(data, offset);
    Package package;
    package.seed_token = read_le<std::uint32_t>(data, offset);
    package.embedding_rows = read_le<std::uint32_t>(data, offset);
    package.embedding_cols = read_le<std::uint32_t>(data, offset);
    package.embedding_offset = read_le<std::uint64_t>(data, offset);
    if (version != kPackageVersion || record_bytes != 68) {
        throw std::runtime_error("runtime package version or record width differs");
    }
    for (auto* digest : {&package.schedule_sha, &package.image_sha, &package.model_sha}) {
        if (offset + digest->size() > data.size()) {
            throw std::runtime_error("runtime package digest is truncated");
        }
        std::memcpy(digest->data(), data.data() + offset, digest->size());
        offset += digest->size();
    }
    if (offset + package.rope_records.size() > data.size()) {
        throw std::runtime_error("runtime package RoPE records are truncated");
    }
    std::memcpy(package.rope_records.data(), data.data() + offset,
                package.rope_records.size());
    offset += package.rope_records.size();
    package.commands.reserve(command_count);
    for (std::uint32_t index = 0; index < command_count; ++index) {
        const std::size_t record_start = offset;
        Command command;
        command.ordinal = read_le<std::uint32_t>(data, offset);
        command.token_step = read_le<std::uint16_t>(data, offset);
        command.layer_id = read_le<std::uint8_t>(data, offset);
        command.operator_id = read_le<std::uint8_t>(data, offset);
        command.opcode = read_le<std::uint8_t>(data, offset);
        command.flags = read_le<std::uint8_t>(data, offset);
        command.m = read_le<std::uint16_t>(data, offset);
        command.n = read_le<std::uint16_t>(data, offset);
        command.k = read_le<std::uint16_t>(data, offset);
        command.sequence_position = read_le<std::uint16_t>(data, offset);
        command.completion_tag = read_le<std::uint16_t>(data, offset);
        command.query_head = read_le<std::int16_t>(data, offset);
        command.context_token = read_le<std::int16_t>(data, offset);
        command.vocab_tile = read_le<std::int32_t>(data, offset);
        command.src0 = read_le<std::uint64_t>(data, offset);
        command.src1 = read_le<std::uint64_t>(data, offset);
        command.dst = read_le<std::uint64_t>(data, offset);
        command.scale = read_le<std::uint64_t>(data, offset);
        command.scratch = read_le<std::uint64_t>(data, offset);
        if (offset - record_start != record_bytes || command.ordinal != index ||
            command.operator_id >= kOperatorNames.size()) {
            throw std::runtime_error("runtime command package is not ordinally canonical");
        }
        package.commands.push_back(command);
    }
    if (offset != data.size()) {
        throw std::runtime_error("runtime package has trailing bytes");
    }
    if (package.commands.size() != 13914 || package.embedding_rows != kVocab ||
        package.embedding_cols != kHidden) {
        throw std::runtime_error("runtime package geometry differs from the accepted contract");
    }
    return package;
}

struct Beat {
    std::array<std::uint8_t, 16> data{};
    std::uint16_t valid = 0;
};

struct ImageRegion {
    std::uint64_t base;
    std::uint64_t bytes;
    std::uint64_t file_offset;
};

class Memory {
  public:
    explicit Memory(const fs::path& image_path) {
        image_fd_ = ::open(image_path.c_str(), O_RDONLY);
        if (image_fd_ < 0) {
            throw std::runtime_error(errno_string("open sealed image failed"));
        }
        struct stat metadata {};
        if (::fstat(image_fd_, &metadata) != 0) {
            throw std::runtime_error(errno_string("stat sealed image failed"));
        }
        image_bytes_ = static_cast<std::size_t>(metadata.st_size);
        if (image_bytes_ != 254421520ULL) {
            throw std::runtime_error("sealed image byte count differs");
        }
        void* mapped = ::mmap(
            nullptr, image_bytes_, PROT_READ, MAP_PRIVATE, image_fd_, 0
        );
        if (mapped == MAP_FAILED) {
            throw std::runtime_error(errno_string("mmap sealed image failed"));
        }
        image_ = static_cast<const std::uint8_t*>(mapped);
    }

    Memory(const Memory&) = delete;
    Memory& operator=(const Memory&) = delete;

    ~Memory() {
        if (image_ != nullptr) {
            ::munmap(const_cast<std::uint8_t*>(image_), image_bytes_);
        }
        if (image_fd_ >= 0) {
            ::close(image_fd_);
        }
    }

    bool in_image(std::uint64_t address, std::size_t bytes, std::size_t& file_offset) const {
        static const std::array<ImageRegion, 4> regions = {{
            {0x0000000100000000ULL, 246980608ULL, 0ULL},
            {0x0000000200000000ULL, 7297024ULL, 246980608ULL},
            {0x0000000300000000ULL, 88592ULL, 254277632ULL},
            {0x0000000310000000ULL, 55296ULL, 254366224ULL},
        }};
        for (const auto& region : regions) {
            if (address >= region.base &&
                address - region.base <= region.bytes &&
                bytes <= region.bytes - (address - region.base)) {
                file_offset = static_cast<std::size_t>(
                    region.file_offset + address - region.base
                );
                return true;
            }
        }
        return false;
    }

    std::array<std::uint8_t, 16> read_beat(std::uint64_t address) const {
        if ((address & 15ULL) != 0) {
            throw BoundaryError("memory_alignment", "unaligned 16-byte read", address);
        }
        std::size_t file_offset = 0;
        std::array<std::uint8_t, 16> result{};
        if (in_image(address, result.size(), file_offset)) {
            std::memcpy(result.data(), image_ + file_offset, result.size());
            return result;
        }
        const auto found = mutable_.find(address);
        if (found == mutable_.end() || found->second.valid != 0xffffU) {
            throw BoundaryError(
                "missing_memory",
                "read reached an uninitialized runtime address",
                address
            );
        }
        return found->second.data;
    }

    std::vector<std::uint8_t> read_range(std::uint64_t address, std::size_t bytes) const {
        std::vector<std::uint8_t> result(bytes);
        for (std::size_t index = 0; index < bytes; ++index) {
            const std::uint64_t beat_address = (address + index) & ~15ULL;
            const auto beat = read_beat(beat_address);
            result[index] = beat[(address + index) & 15ULL];
        }
        return result;
    }

    void write_beat(
        std::uint64_t address,
        const std::array<std::uint8_t, 16>& data,
        std::uint16_t strobe
    ) {
        if ((address & 15ULL) != 0) {
            throw BoundaryError("memory_alignment", "unaligned 16-byte write", address);
        }
        std::size_t ignored = 0;
        if (in_image(address, 16, ignored)) {
            throw BoundaryError("memory_protection", "write targeted sealed image data", address);
        }
        Beat& beat = mutable_[address];
        for (std::size_t lane = 0; lane < 16; ++lane) {
            if ((strobe >> lane) & 1U) {
                beat.data[lane] = data[lane];
                beat.valid |= static_cast<std::uint16_t>(1U << lane);
            }
        }
    }

    void preload(std::uint64_t address, const std::uint8_t* data, std::size_t bytes) {
        for (std::size_t index = 0; index < bytes; ++index) {
            const std::uint64_t absolute = address + index;
            const std::uint64_t beat_address = absolute & ~15ULL;
            Beat& beat = mutable_[beat_address];
            const unsigned lane = static_cast<unsigned>(absolute & 15ULL);
            beat.data[lane] = data[index];
            beat.valid |= static_cast<std::uint16_t>(1U << lane);
        }
    }

  private:
    int image_fd_ = -1;
    const std::uint8_t* image_ = nullptr;
    std::size_t image_bytes_ = 0;
    std::unordered_map<std::uint64_t, Beat> mutable_;
};

class EmbeddingSource {
  public:
    EmbeddingSource(const fs::path& path, std::uint64_t data_offset)
        : data_offset_(data_offset) {
        fd_ = ::open(path.c_str(), O_RDONLY);
        if (fd_ < 0) {
            throw std::runtime_error(errno_string("open raw safetensors failed"));
        }
    }

    EmbeddingSource(const EmbeddingSource&) = delete;
    EmbeddingSource& operator=(const EmbeddingSource&) = delete;

    ~EmbeddingSource() {
        if (fd_ >= 0) {
            ::close(fd_);
        }
    }

    std::vector<std::uint8_t> quantized_row(std::uint32_t token, float scale) const {
        if (token >= kVocab || !std::isfinite(scale) || scale <= 0.0f) {
            throw BoundaryError("embedding_preload", "invalid embedding token or scale");
        }
        std::array<std::uint8_t, kHidden * 2> raw{};
        const off_t offset = static_cast<off_t>(
            data_offset_ + static_cast<std::uint64_t>(token) * raw.size()
        );
        std::size_t consumed = 0;
        while (consumed < raw.size()) {
            const ssize_t count = ::pread(
                fd_, raw.data() + consumed, raw.size() - consumed,
                offset + static_cast<off_t>(consumed)
            );
            if (count < 0) {
                if (errno == EINTR) {
                    continue;
                }
                throw std::runtime_error(errno_string("read embedding row failed"));
            }
            if (count == 0) {
                throw std::runtime_error("embedding row is truncated");
            }
            consumed += static_cast<std::size_t>(count);
        }
        std::vector<std::uint8_t> output(kHidden);
        for (std::size_t lane = 0; lane < kHidden; ++lane) {
            const std::uint16_t bf16 = static_cast<std::uint16_t>(raw[lane * 2]) |
                (static_cast<std::uint16_t>(raw[lane * 2 + 1]) << 8);
            const std::uint32_t fp32_bits = static_cast<std::uint32_t>(bf16) << 16;
            float value = 0.0f;
            std::memcpy(&value, &fp32_bits, sizeof(value));
            float rounded = std::nearbyintf(value / scale);
            rounded = std::max(-128.0f, std::min(127.0f, rounded));
            output[lane] = static_cast<std::uint8_t>(
                static_cast<std::int8_t>(static_cast<int>(rounded))
            );
        }
        return output;
    }

  private:
    int fd_ = -1;
    std::uint64_t data_offset_ = 0;
};

struct WriteRecord {
    std::uint64_t address = 0;
    std::uint16_t strobe = 0;
    std::array<std::uint8_t, 16> data{};
};

struct CompletedRecord {
    std::uint32_t ordinal = 0;
    std::uint64_t cycles = 0;
    std::uint32_t read_beats = 0;
    std::uint32_t write_beats = 0;
    std::uint16_t done_tag = 0;
    bool done_error = false;
    bool saturation = false;
    std::int32_t argmax_token = -1;
    std::int32_t argmax_logit = -129;
    std::int32_t generated0 = -1;
    std::int32_t generated1 = -1;
    std::array<std::uint8_t, 32> source_sha{};
    std::array<std::uint8_t, 32> destination_sha{};
    std::vector<WriteRecord> writes;
};

std::vector<std::uint8_t> serialize_completed(const CompletedRecord& record) {
    std::vector<std::uint8_t> body;
    append_le(body, record.ordinal);
    append_le(body, record.cycles);
    append_le(body, record.read_beats);
    append_le(body, record.write_beats);
    append_le(body, record.done_tag);
    append_le(body, static_cast<std::uint8_t>(record.done_error));
    append_le(body, static_cast<std::uint8_t>(record.saturation));
    append_le(body, record.argmax_token);
    append_le(body, record.argmax_logit);
    append_le(body, record.generated0);
    append_le(body, record.generated1);
    body.insert(body.end(), record.source_sha.begin(), record.source_sha.end());
    body.insert(body.end(), record.destination_sha.begin(), record.destination_sha.end());
    append_le(body, static_cast<std::uint32_t>(record.writes.size()));
    for (const auto& write : record.writes) {
        append_le(body, write.address);
        append_le(body, write.strobe);
        body.insert(body.end(), write.data.begin(), write.data.end());
    }
    return body;
}

CompletedRecord parse_completed(const std::vector<std::uint8_t>& body) {
    std::size_t offset = 0;
    CompletedRecord record;
    record.ordinal = read_le<std::uint32_t>(body, offset);
    record.cycles = read_le<std::uint64_t>(body, offset);
    record.read_beats = read_le<std::uint32_t>(body, offset);
    record.write_beats = read_le<std::uint32_t>(body, offset);
    record.done_tag = read_le<std::uint16_t>(body, offset);
    record.done_error = read_le<std::uint8_t>(body, offset) != 0;
    record.saturation = read_le<std::uint8_t>(body, offset) != 0;
    record.argmax_token = read_le<std::int32_t>(body, offset);
    record.argmax_logit = read_le<std::int32_t>(body, offset);
    record.generated0 = read_le<std::int32_t>(body, offset);
    record.generated1 = read_le<std::int32_t>(body, offset);
    for (auto* digest : {&record.source_sha, &record.destination_sha}) {
        if (offset + digest->size() > body.size()) {
            throw std::runtime_error("journal digest is truncated");
        }
        std::memcpy(digest->data(), body.data() + offset, digest->size());
        offset += digest->size();
    }
    const std::uint32_t write_count = read_le<std::uint32_t>(body, offset);
    record.writes.reserve(write_count);
    for (std::uint32_t index = 0; index < write_count; ++index) {
        WriteRecord write;
        write.address = read_le<std::uint64_t>(body, offset);
        write.strobe = read_le<std::uint16_t>(body, offset);
        if (offset + write.data.size() > body.size()) {
            throw std::runtime_error("journal write data is truncated");
        }
        std::memcpy(write.data.data(), body.data() + offset, write.data.size());
        offset += write.data.size();
        record.writes.push_back(write);
    }
    if (offset != body.size() || record.write_beats != record.writes.size()) {
        throw std::runtime_error("journal frame width differs");
    }
    return record;
}

class Journal {
  public:
    Journal(const fs::path& path, const Package& package, bool resume)
        : path_(path), package_(package) {
        if (resume) {
            records_ = load_existing();
        } else {
            if (fs::exists(path_)) {
                throw std::runtime_error("fresh runtime journal already exists");
            }
            create_header();
        }
        fd_ = ::open(path_.c_str(), O_WRONLY | O_APPEND);
        if (fd_ < 0) {
            throw std::runtime_error(errno_string("open runtime journal append failed"));
        }
    }

    Journal(const Journal&) = delete;
    Journal& operator=(const Journal&) = delete;

    ~Journal() {
        if (fd_ >= 0) {
            ::close(fd_);
        }
    }

    const std::vector<CompletedRecord>& records() const { return records_; }

    void append(const CompletedRecord& record) {
        const std::vector<std::uint8_t> body = serialize_completed(record);
        const auto digest = sha256_bytes(body.data(), body.size());
        std::vector<std::uint8_t> frame;
        append_le(frame, static_cast<std::uint32_t>(body.size()));
        frame.insert(frame.end(), body.begin(), body.end());
        frame.insert(frame.end(), digest.begin(), digest.end());
        append_le(frame, static_cast<std::uint32_t>(body.size()));
        write_all(fd_, frame.data(), frame.size());
        if (::fsync(fd_) != 0) {
            throw std::runtime_error(errno_string("runtime journal fsync failed"));
        }
        records_.push_back(record);
    }

  private:
    static constexpr std::size_t kHeaderBytes = 8 + 4 + 32 * 3;

    void create_header() {
        fs::create_directories(path_.parent_path());
        std::vector<std::uint8_t> header;
        const std::array<std::uint8_t, 8> magic = {'A','C','E','2','J','1',0,0};
        header.insert(header.end(), magic.begin(), magic.end());
        append_le(header, kJournalVersion);
        header.insert(header.end(), package_.schedule_sha.begin(), package_.schedule_sha.end());
        header.insert(header.end(), package_.image_sha.begin(), package_.image_sha.end());
        header.insert(header.end(), package_.model_sha.begin(), package_.model_sha.end());
        const int fd = ::open(path_.c_str(), O_WRONLY | O_CREAT | O_EXCL, 0644);
        if (fd < 0) {
            throw std::runtime_error(errno_string("create runtime journal failed"));
        }
        write_all(fd, header.data(), header.size());
        if (::fsync(fd) != 0) {
            const std::string message = errno_string("runtime journal header fsync failed");
            ::close(fd);
            throw std::runtime_error(message);
        }
        ::close(fd);
        fsync_directory(path_);
    }

    std::vector<CompletedRecord> load_existing() {
        std::vector<std::uint8_t> data = read_file(path_);
        if (data.size() < kHeaderBytes || std::memcmp(data.data(), "ACE2J1", 6) != 0) {
            throw std::runtime_error("runtime journal header differs");
        }
        std::size_t offset = 8;
        if (read_le<std::uint32_t>(data, offset) != kJournalVersion) {
            throw std::runtime_error("runtime journal version differs");
        }
        for (const auto* expected : {&package_.schedule_sha, &package_.image_sha, &package_.model_sha}) {
            if (offset + expected->size() > data.size() ||
                std::memcmp(data.data() + offset, expected->data(), expected->size()) != 0) {
                throw std::runtime_error("runtime journal provenance differs");
            }
            offset += expected->size();
        }
        std::vector<CompletedRecord> records;
        std::size_t valid_bytes = offset;
        while (offset < data.size()) {
            if (data.size() - offset < 4) {
                break;
            }
            const std::size_t frame_start = offset;
            const std::uint32_t body_size = read_le<std::uint32_t>(data, offset);
            if (data.size() - offset < static_cast<std::size_t>(body_size) + 32 + 4) {
                break;
            }
            std::vector<std::uint8_t> body(
                data.begin() + static_cast<std::ptrdiff_t>(offset),
                data.begin() + static_cast<std::ptrdiff_t>(offset + body_size)
            );
            offset += body_size;
            const auto digest = sha256_bytes(body.data(), body.size());
            if (std::memcmp(data.data() + offset, digest.data(), digest.size()) != 0) {
                throw std::runtime_error("runtime journal frame digest differs");
            }
            offset += digest.size();
            const std::uint32_t trailing_size = read_le<std::uint32_t>(data, offset);
            if (trailing_size != body_size) {
                throw std::runtime_error("runtime journal frame trailer differs");
            }
            CompletedRecord record = parse_completed(body);
            if (record.ordinal != records.size()) {
                throw std::runtime_error("runtime journal ordinal sequence differs");
            }
            records.push_back(std::move(record));
            valid_bytes = offset;
            if (offset <= frame_start) {
                throw std::runtime_error("runtime journal parser made no progress");
            }
        }
        if (valid_bytes != data.size()) {
            if (::truncate(path_.c_str(), static_cast<off_t>(valid_bytes)) != 0) {
                throw std::runtime_error(errno_string("truncate incomplete journal tail failed"));
            }
            fsync_directory(path_);
        }
        return records;
    }

    fs::path path_;
    const Package& package_;
    int fd_ = -1;
    std::vector<CompletedRecord> records_;
};

std::array<std::uint8_t, 16> get_wide128(const WData* words) {
    std::array<std::uint8_t, 16> bytes{};
    for (std::size_t word = 0; word < 4; ++word) {
        const std::uint32_t value = words[word];
        for (std::size_t lane = 0; lane < 4; ++lane) {
            bytes[word * 4 + lane] = static_cast<std::uint8_t>(
                (value >> (lane * 8)) & 0xffU
            );
        }
    }
    return bytes;
}

void set_wide128(WData* words, const std::array<std::uint8_t, 16>& bytes) {
    for (std::size_t word = 0; word < 4; ++word) {
        std::uint32_t value = 0;
        for (std::size_t lane = 0; lane < 4; ++lane) {
            value |= static_cast<std::uint32_t>(bytes[word * 4 + lane]) << (lane * 8);
        }
        words[word] = value;
    }
}

void sha_update_u64(SHA256_CTX& context, std::uint64_t value) {
    std::array<std::uint8_t, 8> bytes{};
    for (std::size_t index = 0; index < bytes.size(); ++index) {
        bytes[index] = static_cast<std::uint8_t>((value >> (index * 8)) & 0xffU);
    }
    SHA256_Update(&context, bytes.data(), bytes.size());
}

void sha_update_u16(SHA256_CTX& context, std::uint16_t value) {
    const std::array<std::uint8_t, 2> bytes = {
        static_cast<std::uint8_t>(value & 0xffU),
        static_cast<std::uint8_t>((value >> 8) & 0xffU),
    };
    SHA256_Update(&context, bytes.data(), bytes.size());
}

struct CommandResult {
    std::uint64_t cycles = 0;
    std::uint32_t read_beats = 0;
    std::uint16_t done_tag = 0;
    bool done_error = false;
    bool saturation = false;
    std::array<std::uint8_t, 32> source_sha{};
    std::array<std::uint8_t, 32> destination_sha{};
    std::vector<WriteRecord> writes;
};

class Simulator {
  public:
    Simulator(Memory& memory, std::uint64_t timeout_cycles)
        : memory_(memory), timeout_cycles_(timeout_cycles) {
        clear_inputs();
        top_.rst_ni = 0;
        for (int cycle = 0; cycle < 5; ++cycle) {
            step();
        }
        top_.rst_ni = 1;
        enable_shell();
    }

    std::uint64_t cycles() const { return cycles_; }

    CommandResult execute(const Command& command) {
        if (read_response_.has_value() || write_request_.has_value() ||
            write_response_.has_value()) {
            throw BoundaryError("memory_protocol", "command started with outstanding memory traffic");
        }
        SHA256_CTX source_context;
        SHA256_CTX destination_context;
        SHA256_Init(&source_context);
        SHA256_Init(&destination_context);
        active_source_ = &source_context;
        active_destination_ = &destination_context;
        active_writes_.clear();
        active_read_beats_ = 0;

        drive_command(command);
        top_.cmd_valid_i = 1;
        top_.cmd_done_ready_i = 1;
        const std::uint64_t start_cycle = cycles_;
        bool accepted = false;
        while (true) {
            const Event event = step();
            if (event.cmd_fire) {
                if (accepted) {
                    throw BoundaryError("descriptor_interface", "command accepted more than once");
                }
                accepted = true;
                top_.cmd_valid_i = 0;
            }
            if (event.done_fire) {
                if (!accepted) {
                    throw BoundaryError("descriptor_interface", "completion preceded command acceptance");
                }
                CommandResult result;
                result.cycles = cycles_ - start_cycle;
                result.read_beats = active_read_beats_;
                result.done_tag = event.done_tag;
                result.done_error = event.done_error;
                result.saturation = event.saturation;
                result.writes = active_writes_;
                SHA256_Final(result.source_sha.data(), &source_context);
                SHA256_Final(result.destination_sha.data(), &destination_context);
                active_source_ = nullptr;
                active_destination_ = nullptr;
                top_.cmd_done_ready_i = 0;
                if (read_response_.has_value() || write_request_.has_value() ||
                    write_response_.has_value()) {
                    throw BoundaryError(
                        "memory_protocol",
                        "completion retired with outstanding memory traffic"
                    );
                }
                validate_result(command, result);
                return result;
            }
            if (cycles_ - start_cycle > timeout_cycles_) {
                top_.cmd_valid_i = 0;
                top_.cmd_done_ready_i = 0;
                active_source_ = nullptr;
                active_destination_ = nullptr;
                throw BoundaryError("timeout", "command exceeded the host timeout policy");
            }
        }
    }

  private:
    struct ReadResponse {
        std::array<std::uint8_t, 16> data{};
        std::uint8_t tag = 0;
    };

    struct WriteRequest {
        std::uint64_t address = 0;
        std::uint8_t tag = 0;
    };

    struct WriteResponse {
        std::uint8_t tag = 0;
    };

    struct Event {
        bool cmd_fire = false;
        bool done_fire = false;
        std::uint16_t done_tag = 0;
        bool done_error = false;
        bool saturation = false;
    };

    void clear_inputs() {
        top_.clk_i = 0;
        top_.rst_ni = 0;
        top_.csr_valid_i = 0;
        top_.csr_write_i = 0;
        top_.csr_addr_i = 0;
        top_.csr_wdata_i = 0;
        top_.csr_wstrb_i = 0;
        top_.csr_rready_i = 0;
        top_.cmd_valid_i = 0;
        top_.cmd_opcode_i = 0;
        top_.cmd_flags_i = 0;
        top_.cmd_layer_id_i = 0;
        top_.cmd_m_i = 0;
        top_.cmd_n_i = 0;
        top_.cmd_k_i = 0;
        top_.cmd_sequence_position_i = 0;
        top_.cmd_completion_tag_i = 0;
        top_.cmd_src0_addr_i = 0;
        top_.cmd_src1_addr_i = 0;
        top_.cmd_dst_addr_i = 0;
        top_.cmd_scale_addr_i = 0;
        top_.cmd_scratch_addr_i = 0;
        top_.mem_req_ready_i = 0;
        top_.mem_wready_i = 0;
        top_.mem_rvalid_i = 0;
        set_wide128(top_.mem_rdata_i, {});
        top_.mem_rtag_i = 0;
        top_.mem_rerror_i = 0;
        top_.mem_bvalid_i = 0;
        top_.mem_btag_i = 0;
        top_.mem_berror_i = 0;
        top_.cmd_done_ready_i = 0;
    }

    void enable_shell() {
        top_.csr_valid_i = 1;
        top_.csr_write_i = 1;
        top_.csr_addr_i = kCsrControl;
        top_.csr_wdata_i = 1;
        top_.csr_wstrb_i = 0xff;
        const std::uint64_t start = cycles_;
        while (true) {
            top_.clk_i = 0;
            drive_memory_inputs();
            top_.eval();
            const bool fire = top_.csr_valid_i && top_.csr_ready_o;
            const Event ignored = capture_and_rise();
            (void)ignored;
            if (fire) {
                break;
            }
            if (cycles_ - start > 1024) {
                throw BoundaryError("csr_interface", "shell enable CSR timed out");
            }
        }
        top_.csr_valid_i = 0;
        top_.csr_write_i = 0;
        top_.csr_wstrb_i = 0;
    }

    void drive_command(const Command& command) {
        top_.cmd_opcode_i = command.opcode;
        top_.cmd_flags_i = command.flags;
        top_.cmd_layer_id_i = command.layer_id;
        top_.cmd_m_i = command.m;
        top_.cmd_n_i = command.n;
        top_.cmd_k_i = command.k;
        top_.cmd_sequence_position_i = command.sequence_position;
        top_.cmd_completion_tag_i = command.completion_tag;
        top_.cmd_src0_addr_i = command.src0;
        top_.cmd_src1_addr_i = command.src1;
        top_.cmd_dst_addr_i = command.dst;
        top_.cmd_scale_addr_i = command.scale;
        top_.cmd_scratch_addr_i = command.scratch;
    }

    void drive_memory_inputs() {
        top_.mem_req_ready_i = (cycles_ % 17ULL) != 3ULL;
        top_.mem_wready_i = (cycles_ % 19ULL) != 5ULL;
        top_.mem_rvalid_i = read_response_.has_value();
        if (read_response_.has_value()) {
            set_wide128(top_.mem_rdata_i, read_response_->data);
            top_.mem_rtag_i = read_response_->tag;
        } else {
            set_wide128(top_.mem_rdata_i, {});
            top_.mem_rtag_i = 0;
        }
        top_.mem_rerror_i = 0;
        top_.mem_bvalid_i = write_response_.has_value();
        top_.mem_btag_i = write_response_.has_value() ? write_response_->tag : 0;
        top_.mem_berror_i = 0;
    }

    Event capture_and_rise() {
        Event event;
        const bool req_fire = top_.mem_req_valid_o && top_.mem_req_ready_i;
        const bool write_data_fire = top_.mem_wvalid_o && top_.mem_wready_i;
        const bool read_response_fire = top_.mem_rvalid_i && top_.mem_rready_o;
        const bool write_response_fire = top_.mem_bvalid_i && top_.mem_bready_o;
        event.cmd_fire = top_.cmd_valid_i && top_.cmd_ready_o;
        event.done_fire = top_.cmd_done_valid_o && top_.cmd_done_ready_i;
        event.done_tag = top_.cmd_done_tag_o;
        event.done_error = top_.cmd_done_error_o;
        event.saturation = top_.cmd_done_saturation_seen_o;

        const bool req_write = top_.mem_req_write_o;
        const std::uint64_t req_address = top_.mem_req_addr_o;
        const std::uint16_t req_len = top_.mem_req_len_o;
        const std::uint8_t req_tag = top_.mem_req_tag_o;
        const std::uint8_t write_tag = top_.mem_wtag_o;
        const std::uint16_t write_strobe = top_.mem_wstrb_o;
        const auto write_data = get_wide128(top_.mem_wdata_o);

        top_.clk_i = 1;
        top_.eval();

        if (read_response_fire) {
            read_response_.reset();
        }
        if (write_response_fire) {
            write_response_.reset();
        }
        if (req_fire) {
            if (req_len != 1) {
                throw BoundaryError("memory_protocol", "shell emitted mem_req_len other than one");
            }
            if (req_write) {
                if (write_request_.has_value()) {
                    throw BoundaryError("memory_protocol", "overlapping write requests", req_address);
                }
                write_request_ = WriteRequest{req_address, req_tag};
            } else {
                if (read_response_.has_value()) {
                    throw BoundaryError("memory_protocol", "overlapping read requests", req_address);
                }
                const auto data = memory_.read_beat(req_address);
                read_response_ = ReadResponse{data, req_tag};
                if (active_source_ != nullptr) {
                    sha_update_u64(*active_source_, req_address);
                    SHA256_Update(active_source_, data.data(), data.size());
                    ++active_read_beats_;
                }
            }
        }
        if (write_data_fire) {
            if (!write_request_.has_value()) {
                throw BoundaryError("memory_protocol", "write data lacked a write request");
            }
            if (write_request_->tag != write_tag) {
                throw BoundaryError("memory_protocol", "write request/data tags differ");
            }
            memory_.write_beat(write_request_->address, write_data, write_strobe);
            WriteRecord write{write_request_->address, write_strobe, write_data};
            active_writes_.push_back(write);
            if (active_destination_ != nullptr) {
                sha_update_u64(*active_destination_, write.address);
                sha_update_u16(*active_destination_, write.strobe);
                SHA256_Update(active_destination_, write.data.data(), write.data.size());
            }
            if (write_response_.has_value()) {
                throw BoundaryError("memory_protocol", "overlapping write responses");
            }
            write_response_ = WriteResponse{write_tag};
            write_request_.reset();
        }
        if (event.done_fire && (read_response_.has_value() || write_request_.has_value() ||
                                write_response_.has_value())) {
            throw BoundaryError("memory_protocol", "completion overlapped an unretired response");
        }
        top_.clk_i = 0;
        top_.eval();
        ++cycles_;
        return event;
    }

    Event step() {
        top_.clk_i = 0;
        drive_memory_inputs();
        top_.eval();
        return capture_and_rise();
    }

    static std::uint32_t expected_writes(const Command& command) {
        switch (command.opcode) {
            case 1: return command.n / 16;
            case 2: return command.n / 16;
            case 3: return command.n / 16;
            case 4: return 1;
            case 5: return 1;
            case 6: return command.k / 16;
            case 7: return command.n / 16;
            case 8: return command.n / 16;
            case 9: return command.flags == 6 ? command.k / 16 : 0;
            case 10: return command.n / 16 * 2 + 1;
            default:
                throw BoundaryError("descriptor_interface", "runtime saw an unknown opcode");
        }
    }

    static void validate_result(const Command& command, const CommandResult& result) {
        if (result.done_tag != command.completion_tag) {
            throw BoundaryError("completion_tag", "completion tag differs from the descriptor");
        }
        if (result.done_error) {
            throw BoundaryError(
                "command_error",
                "shell completed the descriptor with cmd_done_error asserted"
            );
        }
        const std::uint32_t expected = expected_writes(command);
        if (result.writes.size() != expected) {
            std::ostringstream detail;
            detail << "write count differs: expected=" << expected
                   << " actual=" << result.writes.size();
            throw BoundaryError("dropped_writes", detail.str());
        }
    }

    Memory& memory_;
    std::uint64_t timeout_cycles_;
    Vace2_shell_runtime_harness top_;
    std::uint64_t cycles_ = 0;
    std::optional<ReadResponse> read_response_;
    std::optional<WriteRequest> write_request_;
    std::optional<WriteResponse> write_response_;
    SHA256_CTX* active_source_ = nullptr;
    SHA256_CTX* active_destination_ = nullptr;
    std::vector<WriteRecord> active_writes_;
    std::uint32_t active_read_beats_ = 0;
};

struct RuntimeState {
    std::int32_t argmax_token = -1;
    std::int32_t argmax_logit = -129;
    std::int32_t generated0 = -1;
    std::int32_t generated1 = -1;
    std::uint32_t embedding_token = 0;
    std::string embedding_sha;
    std::uint64_t resume_warmup_cycles = 0;
};

std::string preload_embedding(
    Memory& memory,
    const EmbeddingSource& embeddings,
    std::uint64_t destination,
    std::uint32_t token,
    float scale
) {
    const auto row = embeddings.quantized_row(token, scale);
    memory.preload(destination, row.data(), row.size());
    return hex_digest(sha256_bytes(row.data(), row.size()));
}

void preload_rope_command(
    Memory& memory,
    const Package& package,
    const Command& command
) {
    if (std::string(command.operator_name()) != "rope_q" &&
        std::string(command.operator_name()) != "rope_k") {
        return;
    }
    if (command.sequence_position > 1 || (command.n != 896 && command.n != 128)) {
        throw BoundaryError("descriptor_interface", "runtime RoPE preload geometry changed");
    }
    const std::uint8_t* record = package.rope_records.data() +
        static_cast<std::size_t>(command.sequence_position) * 128;
    const std::uint64_t position_stride = command.n == 128 ? 512ULL : 3584ULL;
    const std::uint64_t position_base = command.src1 +
        static_cast<std::uint64_t>(command.sequence_position) * position_stride;
    for (std::uint32_t beat = 0; beat < command.n / 16; ++beat) {
        std::array<std::uint8_t, 64> expanded{};
        const std::size_t pair_offset = static_cast<std::size_t>(beat & 1U) * 32;
        std::memcpy(expanded.data(), record + pair_offset, 32);
        std::memcpy(expanded.data() + 32, record + 64 + pair_offset, 32);
        memory.preload(position_base + static_cast<std::uint64_t>(beat) * 64,
                       expanded.data(), expanded.size());
    }
}

void update_argmax(const Command& command, const Memory& memory, RuntimeState& state) {
    if (std::string(command.operator_name()) != "lm_head_tile") {
        return;
    }
    if (command.vocab_tile < 0) {
        throw BoundaryError("token_selection", "lm-head command lacks a vocab tile");
    }
    const auto logits = memory.read_range(command.dst, kLmTile);
    for (std::uint32_t lane = 0; lane < kLmTile; ++lane) {
        const std::int32_t token = command.vocab_tile * kLmTile + lane;
        const std::int32_t logit = static_cast<std::int8_t>(logits[lane]);
        if (state.argmax_token < 0 || logit > state.argmax_logit ||
            (logit == state.argmax_logit && token < state.argmax_token)) {
            state.argmax_token = token;
            state.argmax_logit = logit;
        }
    }
    if (static_cast<std::uint32_t>(command.vocab_tile) == kLastLmTile) {
        if (command.token_step == 0) {
            state.generated0 = state.argmax_token;
        } else if (command.token_step == 1) {
            state.generated1 = state.argmax_token;
        }
    }
}

std::uint32_t first_token1_ordinal(const Package& package) {
    for (const Command& command : package.commands) {
        if (command.token_step == 1) {
            return command.ordinal;
        }
    }
    throw std::runtime_error("accepted schedule lacks token step one");
}

void replay_records(
    const Package& package,
    const std::vector<CompletedRecord>& records,
    Memory& memory,
    const EmbeddingSource& embeddings,
    std::uint64_t embedding_destination,
    float embedding_scale,
    RuntimeState& state
) {
    const std::uint32_t token1_start = first_token1_ordinal(package);
    for (const CompletedRecord& record : records) {
        if (record.ordinal == token1_start) {
            if (state.generated0 < 0) {
                throw std::runtime_error("journal lacks first generated token before feedback");
            }
            state.embedding_token = static_cast<std::uint32_t>(state.generated0);
            state.embedding_sha = preload_embedding(
                memory, embeddings, embedding_destination, state.embedding_token,
                embedding_scale
            );
            state.argmax_token = -1;
            state.argmax_logit = -129;
        }
        for (const auto& write : record.writes) {
            memory.write_beat(write.address, write.data, write.strobe);
        }
        state.argmax_token = record.argmax_token;
        state.argmax_logit = record.argmax_logit;
        state.generated0 = record.generated0;
        state.generated1 = record.generated1;
    }
}

std::string command_json_line(
    const CompletedRecord& record,
    const Command& command
) {
    std::ostringstream out;
    out << "{\"ordinal\":" << record.ordinal
        << ",\"token_step\":" << command.token_step
        << ",\"layer_id\":" << static_cast<unsigned>(command.layer_id)
        << ",\"operator\":\"" << command.operator_name() << "\""
        << ",\"cycles\":" << record.cycles
        << ",\"read_beats\":" << record.read_beats
        << ",\"write_beats\":" << record.write_beats
        << ",\"source_sha256\":\"" << hex_digest(record.source_sha) << "\""
        << ",\"destination_sha256\":\"" << hex_digest(record.destination_sha) << "\""
        << ",\"completion_tag\":" << record.done_tag
        << ",\"completion_error\":" << (record.done_error ? "true" : "false")
        << ",\"saturation_seen\":" << (record.saturation ? "true" : "false")
        << ",\"argmax_token\":" << record.argmax_token
        << ",\"argmax_logit\":" << record.argmax_logit
        << ",\"generated_token_after\":";
    if (record.generated1 >= 0) {
        out << record.generated1;
    } else if (record.generated0 >= 0) {
        out << record.generated0;
    } else {
        out << "null";
    }
    out << "}\n";
    return out.str();
}

void rebuild_jsonl(
    const fs::path& path,
    const Package& package,
    const std::vector<CompletedRecord>& records
) {
    std::string content;
    for (const auto& record : records) {
        content += command_json_line(record, package.commands.at(record.ordinal));
    }
    write_atomic(path, content);
}

void append_jsonl(const fs::path& path, const std::string& line) {
    const int fd = ::open(path.c_str(), O_WRONLY | O_CREAT | O_APPEND, 0644);
    if (fd < 0) {
        throw std::runtime_error(errno_string("open command JSONL failed"));
    }
    write_all(fd, line);
    if (::fsync(fd) != 0) {
        const std::string message = errno_string("command JSONL fsync failed");
        ::close(fd);
        throw std::runtime_error(message);
    }
    ::close(fd);
}

std::string progress_json(
    const Package& package,
    const RuntimeState& state,
    std::uint32_t next_ordinal,
    std::uint64_t cycles,
    const fs::path& journal,
    const std::string& status,
    unsigned resume_count
) {
    std::ostringstream out;
    out << "{\n"
        << "  \"schema_version\": 1,\n"
        << "  \"status\": \"" << json_escape(status) << "\",\n"
        << "  \"total_commands\": " << package.commands.size() << ",\n"
        << "  \"next_ordinal\": " << next_ordinal << ",\n"
        << "  \"last_committed_ordinal\": ";
    if (next_ordinal == 0) {
        out << "null";
    } else {
        out << (next_ordinal - 1);
    }
    out << ",\n"
        << "  \"simulator_cycles\": " << cycles << ",\n"
        << "  \"schedule_sha256\": \"" << hex_digest(package.schedule_sha) << "\",\n"
        << "  \"image_sha256\": \"" << hex_digest(package.image_sha) << "\",\n"
        << "  \"model_safetensors_sha256\": \"" << hex_digest(package.model_sha) << "\",\n"
        << "  \"embedding_token\": " << state.embedding_token << ",\n"
        << "  \"embedding_s8_sha256\": \"" << state.embedding_sha << "\",\n"
        << "  \"argmax\": {\"token_id\": " << state.argmax_token
        << ", \"logit_s8\": " << state.argmax_logit << "},\n"
        << "  \"generated_token_ids\": [";
    bool comma = false;
    if (state.generated0 >= 0) {
        out << state.generated0;
        comma = true;
    }
    if (state.generated1 >= 0) {
        if (comma) out << ", ";
        out << state.generated1;
    }
    out << "],\n"
        << "  \"journal_bytes\": " << (fs::exists(journal) ? fs::file_size(journal) : 0)
        << ",\n"
        << "  \"resume_count\": " << resume_count << ",\n"
        << "  \"resume_warmup_cycles\": " << state.resume_warmup_cycles << ",\n"
        << "  \"first_failure\": null\n"
        << "}\n";
    return out.str();
}

std::string summary_json(
    const Package& package,
    const RuntimeState& state,
    const std::vector<CompletedRecord>& records,
    std::uint64_t cycles,
    const std::string& status
) {
    struct Aggregate {
        std::uint64_t commands = 0;
        std::uint64_t cycles = 0;
        std::uint64_t max_cycles = 0;
        std::uint64_t reads = 0;
        std::uint64_t writes = 0;
    };
    std::map<std::string, Aggregate> aggregates;
    for (const auto& record : records) {
        const Command& command = package.commands.at(record.ordinal);
        Aggregate& aggregate = aggregates[command.operator_name()];
        ++aggregate.commands;
        aggregate.cycles += record.cycles;
        aggregate.max_cycles = std::max(aggregate.max_cycles, record.cycles);
        aggregate.reads += record.read_beats;
        aggregate.writes += record.write_beats;
    }
    std::ostringstream out;
    out << "{\n"
        << "  \"schema_version\": 1,\n"
        << "  \"status\": \"" << json_escape(status) << "\",\n"
        << "  \"commands_completed\": " << records.size() << ",\n"
        << "  \"commands_total\": " << package.commands.size() << ",\n"
        << "  \"simulator_cycles\": " << cycles << ",\n"
        << "  \"generated_token_ids\": [";
    bool comma = false;
    if (state.generated0 >= 0) {
        out << state.generated0;
        comma = true;
    }
    if (state.generated1 >= 0) {
        if (comma) out << ", ";
        out << state.generated1;
    }
    out << "],\n"
        << "  \"resume_warmup_cycles\": " << state.resume_warmup_cycles << ",\n"
        << "  \"operator_totals\": {\n";
    std::size_t index = 0;
    for (const auto& item : aggregates) {
        out << "    \"" << json_escape(item.first) << "\": {"
            << "\"commands\":" << item.second.commands
            << ",\"cycles\":" << item.second.cycles
            << ",\"max_cycles\":" << item.second.max_cycles
            << ",\"read_beats\":" << item.second.reads
            << ",\"write_beats\":" << item.second.writes << "}";
        out << (++index == aggregates.size() ? "\n" : ",\n");
    }
    out << "  },\n"
        << "  \"first_failure\": null\n"
        << "}\n";
    return out.str();
}

std::string failure_json(
    const Package& package,
    const RuntimeState& state,
    const Command& command,
    const BoundaryError& error,
    std::uint64_t cycles
) {
    std::ostringstream out;
    out << "{\n"
        << "  \"schema_version\": 1,\n"
        << "  \"status\": \"STOPPED_AT_FIRST_GENUINE_BOUNDARY\",\n"
        << "  \"ordinal\": " << command.ordinal << ",\n"
        << "  \"token_step\": " << command.token_step << ",\n"
        << "  \"layer_id\": " << static_cast<unsigned>(command.layer_id) << ",\n"
        << "  \"operator\": \"" << command.operator_name() << "\",\n"
        << "  \"category\": \"" << json_escape(error.category) << "\",\n"
        << "  \"detail\": \"" << json_escape(error.what()) << "\",\n"
        << "  \"address\": ";
    if (error.address.has_value()) {
        out << *error.address;
    } else {
        out << "null";
    }
    out << ",\n"
        << "  \"simulator_cycles\": " << cycles << ",\n"
        << "  \"schedule_sha256\": \"" << hex_digest(package.schedule_sha) << "\",\n"
        << "  \"image_sha256\": \"" << hex_digest(package.image_sha) << "\",\n"
        << "  \"generated_token_ids\": [";
    if (state.generated0 >= 0) {
        out << state.generated0;
        if (state.generated1 >= 0) out << ", " << state.generated1;
    }
    out << "]\n"
        << "}\n";
    return out.str();
}

std::vector<Command> resume_warmup_commands(
    const Package& package,
    std::uint32_t next_ordinal
) {
    if (next_ordinal >= package.commands.size()) {
        return {};
    }
    const Command& target = package.commands[next_ordinal];
    if (std::string(target.operator_name()) != "attention_compose" || target.flags == 0) {
        return {};
    }
    std::uint32_t start = next_ordinal;
    while (start != 0) {
        --start;
        const Command& candidate = package.commands[start];
        if (std::string(candidate.operator_name()) == "attention_compose" &&
            candidate.token_step == target.token_step &&
            candidate.layer_id == target.layer_id &&
            candidate.query_head == target.query_head &&
            candidate.flags == 0) {
            return std::vector<Command>(
                package.commands.begin() + start,
                package.commands.begin() + next_ordinal
            );
        }
    }
    throw BoundaryError("state_propagation", "cannot find attention-compose replay anchor");
}

struct Arguments {
    fs::path package;
    fs::path image;
    fs::path model;
    fs::path output;
    std::uint32_t stop_after = 0;
    std::uint64_t timeout_cycles = kDefaultTimeoutCycles;
    bool resume = false;
};

Arguments parse_arguments(int argc, char** argv) {
    Arguments arguments;
    for (int index = 1; index < argc; ++index) {
        const std::string option = argv[index];
        auto require_value = [&]() -> std::string {
            if (++index >= argc) {
                throw std::runtime_error("missing value after " + option);
            }
            return argv[index];
        };
        if (option == "--package") {
            arguments.package = require_value();
        } else if (option == "--image") {
            arguments.image = require_value();
        } else if (option == "--model") {
            arguments.model = require_value();
        } else if (option == "--output") {
            arguments.output = require_value();
        } else if (option == "--stop-after") {
            arguments.stop_after = static_cast<std::uint32_t>(std::stoul(require_value()));
        } else if (option == "--timeout-cycles") {
            arguments.timeout_cycles = std::stoull(require_value());
        } else if (option == "--resume") {
            arguments.resume = true;
        } else {
            throw std::runtime_error("unknown runtime option: " + option);
        }
    }
    if (arguments.package.empty() || arguments.image.empty() || arguments.model.empty() ||
        arguments.output.empty()) {
        throw std::runtime_error("--package, --image, --model, and --output are required");
    }
    return arguments;
}

}  // namespace

int main(int argc, char** argv) {
    Verilated::commandArgs(argc, argv);
    try {
        std::fesetround(FE_TONEAREST);
        const Arguments arguments = parse_arguments(argc, argv);
        const Package package = load_package(arguments.package);
        const std::uint32_t stop_after = arguments.stop_after == 0
            ? static_cast<std::uint32_t>(package.commands.size())
            : arguments.stop_after;
        if (stop_after > package.commands.size()) {
            throw std::runtime_error("--stop-after exceeds accepted command count");
        }
        fs::create_directories(arguments.output);
        const fs::path journal_path = arguments.output / "progress.journal";
        const fs::path jsonl_path = arguments.output / "commands.jsonl";
        const fs::path progress_path = arguments.output / "progress.json";
        const fs::path summary_path = arguments.output / "summary.json";
        const fs::path failure_path = arguments.output / "first_failure.json";
        if (!arguments.resume &&
            (fs::exists(jsonl_path) || fs::exists(progress_path) || fs::exists(summary_path) ||
             fs::exists(failure_path))) {
            throw std::runtime_error("fresh runtime output already contains progress artifacts");
        }

        Memory memory(arguments.image);
        EmbeddingSource embeddings(arguments.model, package.embedding_offset);
        const auto scale_raw = memory.read_range(kRmsnormBase + kRmsnormScaleOffset, 8);
        double scale_f64 = 0.0;
        std::memcpy(&scale_f64, scale_raw.data(), sizeof(scale_f64));
        const float embedding_scale = static_cast<float>(scale_f64);
        if (!std::isfinite(embedding_scale) || embedding_scale <= 0.0f) {
            throw std::runtime_error("layer-0 embedding scale is invalid");
        }
        const std::uint64_t embedding_destination = package.commands.front().src0;

        RuntimeState state;
        state.embedding_token = package.seed_token;
        state.embedding_sha = preload_embedding(
            memory, embeddings, embedding_destination, state.embedding_token,
            embedding_scale
        );

        Journal journal(journal_path, package, arguments.resume);
        replay_records(
            package, journal.records(), memory, embeddings, embedding_destination,
            embedding_scale, state
        );
        if (arguments.resume) {
            rebuild_jsonl(jsonl_path, package, journal.records());
        } else {
            write_atomic(jsonl_path, "");
        }
        if (journal.records().size() > stop_after) {
            throw std::runtime_error("journal progress exceeds requested stop ordinal");
        }

        Simulator simulator(memory, arguments.timeout_cycles);
        unsigned resume_count = arguments.resume ? 1U : 0U;
        const std::uint32_t next_ordinal = static_cast<std::uint32_t>(journal.records().size());
        if (arguments.resume) {
            const auto warmup = resume_warmup_commands(package, next_ordinal);
            const std::uint64_t start = simulator.cycles();
            for (const auto& command : warmup) {
                simulator.execute(command);
            }
            state.resume_warmup_cycles += simulator.cycles() - start;
        }
        write_atomic(
            progress_path,
            progress_json(
                package, state, next_ordinal, simulator.cycles(), journal_path,
                "RUNNING", resume_count
            )
        );

        const std::uint32_t token1_start = first_token1_ordinal(package);
        for (std::uint32_t ordinal = next_ordinal; ordinal < stop_after; ++ordinal) {
            const Command& command = package.commands[ordinal];
            if (ordinal == token1_start) {
                if (state.generated0 < 0) {
                    throw BoundaryError(
                        "token_selection",
                        "second-token feedback began before first-token argmax completed"
                    );
                }
                state.embedding_token = static_cast<std::uint32_t>(state.generated0);
                state.embedding_sha = preload_embedding(
                    memory, embeddings, embedding_destination, state.embedding_token,
                    embedding_scale
                );
                state.argmax_token = -1;
                state.argmax_logit = -129;
            }
            try {
                preload_rope_command(memory, package, command);
                const CommandResult result = simulator.execute(command);
                update_argmax(command, memory, state);
                CompletedRecord completed;
                completed.ordinal = ordinal;
                completed.cycles = result.cycles;
                completed.read_beats = result.read_beats;
                completed.write_beats = static_cast<std::uint32_t>(result.writes.size());
                completed.done_tag = result.done_tag;
                completed.done_error = result.done_error;
                completed.saturation = result.saturation;
                completed.argmax_token = state.argmax_token;
                completed.argmax_logit = state.argmax_logit;
                completed.generated0 = state.generated0;
                completed.generated1 = state.generated1;
                completed.source_sha = result.source_sha;
                completed.destination_sha = result.destination_sha;
                completed.writes = result.writes;
                journal.append(completed);
                append_jsonl(jsonl_path, command_json_line(completed, command));
                const std::string status = ordinal + 1 == package.commands.size()
                    ? "COMPLETE" : "RUNNING";
                write_atomic(
                    progress_path,
                    progress_json(
                        package, state, ordinal + 1, simulator.cycles(), journal_path,
                        status, resume_count
                    )
                );
                if (ordinal < 2 || (ordinal + 1) % 100 == 0 ||
                    ordinal + 1 == stop_after || ordinal + 1 == token1_start) {
                    std::cout << "ACE2_RUNTIME_PROGRESS next=" << (ordinal + 1)
                              << "/" << package.commands.size()
                              << " operator=" << command.operator_name()
                              << " command_cycles=" << result.cycles
                              << " simulator_cycles=" << simulator.cycles() << '\n';
                    std::cout.flush();
                }
            } catch (const BoundaryError& error) {
                write_atomic(
                    failure_path,
                    failure_json(package, state, command, error, simulator.cycles())
                );
                write_atomic(
                    summary_path,
                    summary_json(
                        package, state, journal.records(), simulator.cycles(),
                        "STOPPED_AT_FIRST_GENUINE_BOUNDARY"
                    )
                );
                write_atomic(
                    progress_path,
                    progress_json(
                        package, state, ordinal, simulator.cycles(), journal_path,
                        "STOPPED_AT_FIRST_GENUINE_BOUNDARY", resume_count
                    )
                );
                std::cerr << "ACE2_RUNTIME_BOUNDARY ordinal=" << ordinal
                          << " operator=" << command.operator_name()
                          << " category=" << error.category
                          << " detail=" << error.what() << '\n';
                return 2;
            }
        }

        const bool full_complete = stop_after == package.commands.size();
        const std::string final_status = full_complete ? "PASS" : "PREFIX_COMPLETE";
        write_atomic(
            summary_path,
            summary_json(package, state, journal.records(), simulator.cycles(), final_status)
        );
        write_atomic(
            progress_path,
            progress_json(
                package, state, stop_after, simulator.cycles(), journal_path,
                final_status, resume_count
            )
        );
        std::cout << "ACE2_RUNTIME_" << final_status
                  << " commands=" << journal.records().size()
                  << " simulator_cycles=" << simulator.cycles();
        if (state.generated0 >= 0) {
            std::cout << " token0=" << state.generated0;
        }
        if (state.generated1 >= 0) {
            std::cout << " token1=" << state.generated1;
        }
        std::cout << '\n';
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "ACE2_RUNTIME_SETUP_FAIL detail=" << error.what() << '\n';
        return 3;
    }
}
