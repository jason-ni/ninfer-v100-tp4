#pragma once

#include <cstddef>
#include <cstdint>

namespace ninfer {

enum class DType : std::uint8_t {
    BF16       = 0,
    FP32       = 1,
    I32        = 2,
    U8         = 3,
    I64        = 4,
    I8         = 5,
    FP16       = 6,
    FP8_E4M3FN = 7,
    // Packed iq4_nl nibble codes for KV storage: one code (index into a 16-entry asymmetric LUT)
    // per head-dim value, stored two per byte. Logical size is 1 byte per row of head_dim/2
    // elements; it is never an arithmetic operand dtype.
    I4NL       = 8,
};

std::size_t dtype_size(DType dtype);

} // namespace ninfer
