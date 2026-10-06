#pragma once

#include <cstdint>
#include <limits>
#include <stdexcept>

namespace ninfer::targets::qwen3_6::detail::NINFER_QWEN36_RUNTIME_NS {

/** Qwen3.6's target-local mapping from a stable request lane to its state roles.
 *
 * Every lane owns two slots: the current decode state and the rewrite-checkpoint image. When the
 * hybrid prefix cache is enabled, `snapshot_slots` additional slots hold cached prefix state images
 * (GDN recurrent + conv history), addressed by `snapshot_state_slot`. */
struct LinearStateSlots {
    [[nodiscard]] static std::int32_t state_slot_count(std::uint32_t max_concurrency,
                                                       std::uint32_t snapshot_slots = 0) {
        if (max_concurrency == 0 ||
            max_concurrency >
                static_cast<std::uint32_t>(std::numeric_limits<std::int32_t>::max() / 2)) {
            throw std::invalid_argument("Qwen3.6 Linear Attention concurrency is invalid");
        }
        const std::uint32_t lane_slots = 2U * max_concurrency;
        if (snapshot_slots >
            static_cast<std::uint32_t>(std::numeric_limits<std::int32_t>::max()) - lane_slots) {
            throw std::invalid_argument("Qwen3.6 Linear Attention snapshot slot count is invalid");
        }
        return static_cast<std::int32_t>(lane_slots + snapshot_slots);
    }

    [[nodiscard]] static std::int32_t current_state_slot(std::uint32_t lane,
                                                         std::uint32_t max_concurrency) {
        if (lane >= max_concurrency) {
            throw std::out_of_range("Qwen3.6 Linear Attention lane is out of range");
        }
        return static_cast<std::int32_t>(lane);
    }

    [[nodiscard]] static std::int32_t rewrite_checkpoint_state_slot(std::uint32_t lane,
                                                                    std::uint32_t max_concurrency) {
        return static_cast<std::int32_t>(max_concurrency) +
               current_state_slot(lane, max_concurrency);
    }

    [[nodiscard]] static std::int32_t snapshot_state_slot(std::uint32_t index, std::uint32_t max_concurrency,
                                                          std::uint32_t snapshot_slots) {
        if (index >= snapshot_slots) {
            throw std::out_of_range("Qwen3.6 Linear Attention snapshot slot index is out of range");
        }
        return static_cast<std::int32_t>(2U * max_concurrency + index);
    }
};

} // namespace ninfer::targets::qwen3_6::detail::NINFER_QWEN36_RUNTIME_NS
