#include "BatchDataLoader.h"

// Constructor
BatchDataLoader::BatchDataLoader(const std::vector<std::vector<double>>& features,
                                 const std::vector<double>& targets,
                                 size_t batch_size,
                                 bool shuffle) // shuffle parameter is present but not used for time-series default
    : all_features_(features),
      all_targets_(targets),
      batch_size_(batch_size),
      num_samples_(features.size()),
      current_sample_idx_(0)
      {
    if (features.size() != targets.size()) {
        throw std::invalid_argument("BatchDataLoader Error: Features and targets must have the same number of samples.");
    }
    if (batch_size == 0) {
        throw std::invalid_argument("BatchDataLoader Error: Batch size cannot be zero.");
    }
    if (num_samples_ > 0 && batch_size_ > num_samples_) {
    }

    (void)shuffle; // Suppress unused parameter warning if shuffle is not used
}

bool BatchDataLoader::next_batch(std::vector<std::vector<double>>& out_batch_features,
                                 std::vector<double>& out_batch_targets) {
    if (current_sample_idx_ >= num_samples_) {
        return false; // All samples processed for this epoch
    }

    out_batch_features.clear();
    out_batch_targets.clear();

    size_t end_idx = std::min(current_sample_idx_ + batch_size_, num_samples_);

    // Reserve space for efficiency
    out_batch_features.reserve(end_idx - current_sample_idx_);
    out_batch_targets.reserve(end_idx - current_sample_idx_);

    for (size_t i = current_sample_idx_; i < end_idx; ++i) {
        // For sequential access
        size_t actual_idx = i;

        out_batch_features.push_back(all_features_[actual_idx]);
        out_batch_targets.push_back(all_targets_[actual_idx]);
    }

    current_sample_idx_ = end_idx;
    return !out_batch_features.empty(); // Return true if any samples were added to the batch
}

void BatchDataLoader::reset(bool reshuffle_on_reset) {
    current_sample_idx_ = 0;
    (void)reshuffle_on_reset; // Suppress unused parameter warning
}

size_t BatchDataLoader::get_total_batches() const {
    if (num_samples_ == 0) return 0;
    return (num_samples_ + batch_size_ - 1) / batch_size_; // Ceiling division
}

size_t BatchDataLoader::get_num_samples() const {
    return num_samples_;
}

size_t BatchDataLoader::get_current_batch_actual_size(size_t batch_start_idx) const {
    if (batch_start_idx >= num_samples_) return 0;
    return std::min(batch_size_, num_samples_ - batch_start_idx);
}
