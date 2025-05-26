#ifndef BATCH_DATA_LOADER_HPP
#define BATCH_DATA_LOADER_HPP

#include <vector>
#include <string>
#include <stdexcept>
#include <numeric>
#include <algorithm>
#include <random>

class BatchDataLoader {
public:

    //Constructs
    BatchDataLoader(const std::vector<std::vector<double>>& features,
                    const std::vector<double>& targets,
                    size_t batch_size,
                    bool shuffle = false);


    //Retrieves the next batch
    bool next_batch(std::vector<std::vector<double>>& out_batch_features,
                    std::vector<double>& out_batch_targets);


    //Resets the loader to the beginning of the dataset, typically for a new epoch
    void reset(bool reshuffle_on_reset = false); // Parameter kept for signature consistency

    //Gets the total number of batches
    size_t get_total_batches() const;


    //Gets the total number of samples in the dataset.
    size_t get_num_samples() const;


    //Gets the actual size of the current batch being processed
    size_t get_current_batch_actual_size(size_t batch_start_idx) const;


private:
    const std::vector<std::vector<double>>& all_features_; // Using const reference to avoid copying large data
    const std::vector<double>& all_targets_;               // Using const reference
    size_t batch_size_;
    size_t num_samples_;
    size_t current_sample_idx_;
};

#endif
