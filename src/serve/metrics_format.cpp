#include "serve/metrics_format.h"

#include <cstdint>
#include <string>

namespace ninfer::serve {
namespace {

void header(std::string& out, const char* name, const char* help, const char* type) {
    out += "# HELP ";
    out += name;
    out += ' ';
    out += help;
    out += "\n# TYPE ";
    out += name;
    out += ' ';
    out += type;
    out += '\n';
}

void sample(std::string& out, const char* name, const std::string& labels, std::uint64_t value) {
    out += name;
    out += labels;
    out += ' ';
    out += std::to_string(value);
    out += '\n';
}

std::string labels(std::initializer_list<std::pair<const char*, std::string>> pairs) {
    std::string out = "{";
    bool first      = true;
    for (const auto& [key, value] : pairs) {
        if (!first) { out += ','; }
        first = false;
        out += key;
        out += "=\"";
        out += value;
        out += '"';
    }
    out += '}';
    return out;
}

nlohmann::json kv_to_json(const ninfer::KvPoolMetrics& kv) {
    return nlohmann::json{{"capacity_pages", kv.capacity_pages},
                          {"mapped_pages", kv.mapped_pages},
                          {"cached_pages", kv.cached_pages},
                          {"free_pages", kv.free_pages},
                          {"page_bytes", kv.page_bytes}};
}

} // namespace

std::string metrics_to_prometheus(const ninfer::MetricsSnapshot& metrics) {
    std::string out;
    out.reserve(2048);

    header(out, "ninfer_device_memory_bytes",
           "Device memory per tensor-parallel rank (cudaMemGetInfo).", "gauge");
    for (const ninfer::DeviceMemoryMetrics& device : metrics.devices) {
        const std::string dev = std::to_string(device.device);
        sample(out, "ninfer_device_memory_bytes", labels({{"device", dev}, {"kind", "total"}}),
               device.total_bytes);
        sample(out, "ninfer_device_memory_bytes", labels({{"device", dev}, {"kind", "used"}}),
               device.used_bytes);
        sample(out, "ninfer_device_memory_bytes", labels({{"device", dev}, {"kind", "free"}}),
               device.free_bytes);
    }

    header(out, "ninfer_kv_pages", "Paged KV pool page-group occupancy by pool.", "gauge");
    header(out, "ninfer_kv_page_bytes", "Bytes per physical KV page group by pool.", "gauge");
    const std::pair<const char*, const ninfer::KvPoolMetrics*> pools[] = {
        {"main", &metrics.main_kv}, {"backend", &metrics.backend_kv}};
    for (const auto& [name, kv] : pools) {
        sample(out, "ninfer_kv_pages", labels({{"pool", name}, {"kind", "capacity"}}),
               kv->capacity_pages);
        sample(out, "ninfer_kv_pages", labels({{"pool", name}, {"kind", "mapped"}}),
               kv->mapped_pages);
        sample(out, "ninfer_kv_pages", labels({{"pool", name}, {"kind", "cached"}}),
               kv->cached_pages);
        sample(out, "ninfer_kv_pages", labels({{"pool", name}, {"kind", "free"}}), kv->free_pages);
        sample(out, "ninfer_kv_page_bytes", labels({{"pool", name}}), kv->page_bytes);
    }

    header(out, "ninfer_prefix_cache_enabled", "1 when the hybrid prefix cache is enabled.",
           "gauge");
    sample(out, "ninfer_prefix_cache_enabled", "", metrics.prefix_cache.enabled ? 1U : 0U);
    header(out, "ninfer_prefix_cache_device_snapshot_slots",
           "Device state-image slots held by the prefix cache.", "gauge");
    sample(out, "ninfer_prefix_cache_device_snapshot_slots", labels({{"kind", "total"}}),
           metrics.prefix_cache.device_snapshot_slots);
    sample(out, "ninfer_prefix_cache_device_snapshot_slots", labels({{"kind", "used"}}),
           metrics.prefix_cache.device_snapshot_slots_used);
    header(out, "ninfer_prefix_cache_host_enabled",
           "1 when the pinned host KV tier is allocated.", "gauge");
    sample(out, "ninfer_prefix_cache_host_enabled", "", metrics.prefix_cache.host.enabled ? 1U : 0U);
    header(out, "ninfer_prefix_cache_host_bytes",
           "Pinned host KV tier bytes allocated and occupied.", "gauge");
    sample(out, "ninfer_prefix_cache_host_bytes", labels({{"kind", "pinned"}}),
           metrics.prefix_cache.host.pinned_bytes);
    sample(out, "ninfer_prefix_cache_host_bytes", labels({{"kind", "used"}}),
           metrics.prefix_cache.host.used_bytes);
    header(out, "ninfer_prefix_cache_host_slabs", "Pinned host KV tier slab count.", "gauge");
    sample(out, "ninfer_prefix_cache_host_slabs", labels({{"kind", "block"}}),
           metrics.prefix_cache.host.block_slab_count);
    sample(out, "ninfer_prefix_cache_host_slabs", labels({{"kind", "snapshot"}}),
           metrics.prefix_cache.host.snapshot_slab_count);

    header(out, "ninfer_requests", "Requests by scheduler state.", "gauge");
    sample(out, "ninfer_requests", labels({{"state", "running"}}),
           metrics.scheduler.running_requests);
    sample(out, "ninfer_requests", labels({{"state", "prefilling"}}),
           metrics.scheduler.prefilling_requests);
    sample(out, "ninfer_requests", labels({{"state", "decode_ready"}}),
           metrics.scheduler.decode_ready_requests);
    sample(out, "ninfer_requests", labels({{"state", "waiting"}}),
           metrics.scheduler.waiting_requests);

    header(out, "ninfer_computed_prefill_tokens_total", "Prompt tokens evaluated by prefill.",
           "counter");
    sample(out, "ninfer_computed_prefill_tokens_total", "",
           metrics.scheduler.computed_prefill_tokens);
    header(out, "ninfer_committed_decode_tokens_total", "Tokens committed by decode rounds.",
           "counter");
    sample(out, "ninfer_committed_decode_tokens_total", "",
           metrics.scheduler.committed_decode_tokens);
    header(out, "ninfer_decode_rounds_total", "Decode batch executions.", "counter");
    sample(out, "ninfer_decode_rounds_total", "", metrics.scheduler.decode_rounds);
    header(out, "ninfer_decode_row_rounds_total", "Sum of decode batch sizes.", "counter");
    sample(out, "ninfer_decode_row_rounds_total", "", metrics.scheduler.decode_row_rounds);
    return out;
}

nlohmann::json metrics_to_json(const ninfer::MetricsSnapshot& metrics) {
    nlohmann::json devices = nlohmann::json::array();
    for (const ninfer::DeviceMemoryMetrics& device : metrics.devices) {
        devices.push_back(nlohmann::json{{"device", device.device},
                                         {"total_bytes", device.total_bytes},
                                         {"used_bytes", device.used_bytes},
                                         {"free_bytes", device.free_bytes}});
    }

    const ninfer::HostCacheMetrics& host = metrics.prefix_cache.host;
    return nlohmann::json{
        {"scheduler", nlohmann::json{{"running_requests", metrics.scheduler.running_requests},
                                     {"prefilling_requests", metrics.scheduler.prefilling_requests},
                                     {"decode_ready_requests", metrics.scheduler.decode_ready_requests},
                                     {"waiting_requests", metrics.scheduler.waiting_requests},
                                     {"computed_prefill_tokens", metrics.scheduler.computed_prefill_tokens},
                                     {"committed_decode_tokens", metrics.scheduler.committed_decode_tokens},
                                     {"decode_rounds", metrics.scheduler.decode_rounds},
                                     {"decode_row_rounds", metrics.scheduler.decode_row_rounds}}},
        {"devices", std::move(devices)},
        {"main_kv", kv_to_json(metrics.main_kv)},
        {"backend_kv", kv_to_json(metrics.backend_kv)},
        {"prefix_cache",
         nlohmann::json{{"enabled", metrics.prefix_cache.enabled},
                        {"device_snapshot_slots", metrics.prefix_cache.device_snapshot_slots},
                        {"device_snapshot_slots_used",
                         metrics.prefix_cache.device_snapshot_slots_used},
                        {"host",
                         nlohmann::json{{"enabled", host.enabled},
                                        {"pinned_bytes", host.pinned_bytes},
                                        {"used_bytes", host.used_bytes},
                                        {"block_slab_count", host.block_slab_count},
                                        {"snapshot_slab_count", host.snapshot_slab_count}}}}}};
}

} // namespace ninfer::serve
