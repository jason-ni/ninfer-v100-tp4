#pragma once

#include <ninfer/types.h>

#include <nlohmann/json.hpp>

#include <string>

namespace ninfer::serve {

// Prometheus text exposition (text/plain; version=0.0.4) of a resource snapshot. Device, KV-pool,
// and prefix-cache figures are gauges; the execution counters carry a `_total` suffix.
std::string metrics_to_prometheus(const ninfer::MetricsSnapshot& metrics);

// JSON object mirror of the same snapshot, one key per top-level group.
nlohmann::json metrics_to_json(const ninfer::MetricsSnapshot& metrics);

} // namespace ninfer::serve
