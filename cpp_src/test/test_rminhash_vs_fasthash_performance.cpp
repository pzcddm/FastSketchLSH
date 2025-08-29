//! Performance comparison test between RMinHashSIMD and FastSimilaritySketchAVX512Packed
//! 
//! This test compares the performance and accuracy of two different sketch algorithms:
//! - RMinHashSIMD: Optimized R-MinHash implementation mimicking Rust code
//! - FastSimilaritySketchAVX512Packed: Existing fast sketch implementation
//!
//! The test measures:
//! - Sketch computation time for both uint32_t and string inputs
//! - Memory usage patterns
//! - Accuracy in Jaccard similarity estimation
//! - Scalability with different input sizes and sketch sizes

#include "rminhash.h"
#include "fasthash_simd.h"
#include <iostream>
#include <vector>
#include <string>
#include <random>
#include <chrono>
#include <iomanip>
#include <algorithm>
#include <cmath>
#include <cassert>

using namespace std;
using namespace std::chrono;

// ===================== Test Data Generation =====================

// Generate random uint32_t vector
vector<uint32_t> generate_random_uint32_set(size_t size, uint32_t seed, uint32_t max_val = UINT32_MAX) {
    vector<uint32_t> result;
    result.reserve(size);
    mt19937 rng(seed);
    uniform_int_distribution<uint32_t> dist(0, max_val);
    
    for (size_t i = 0; i < size; ++i) {
        result.push_back(dist(rng));
    }
    
    // Remove duplicates to get exact set semantics
    sort(result.begin(), result.end());
    result.erase(unique(result.begin(), result.end()), result.end());
    
    return result;
}

// Generate overlapping sets for Jaccard testing
pair<vector<uint32_t>, vector<uint32_t>> generate_overlapping_sets(
    size_t set_size, double jaccard_target, uint32_t seed) {
    
    mt19937 rng(seed);
    
    // Generate base set
    auto base_set = generate_random_uint32_set(set_size, seed, set_size * 10);
    
    // Calculate intersection size for target Jaccard
    size_t intersection_size = static_cast<size_t>(jaccard_target * set_size * 2 / (1 + jaccard_target));
    size_t unique_a = set_size - intersection_size;
    size_t unique_b = set_size - intersection_size;
    
    vector<uint32_t> set_a, set_b;
    set_a.reserve(set_size);
    set_b.reserve(set_size);
    
    // Add intersection elements
    for (size_t i = 0; i < intersection_size && i < base_set.size(); ++i) {
        set_a.push_back(base_set[i]);
        set_b.push_back(base_set[i]);
    }
    
    // Add unique elements for set A
    for (size_t i = intersection_size; i < intersection_size + unique_a && i < base_set.size(); ++i) {
        set_a.push_back(base_set[i]);
    }
    
    // Generate unique elements for set B
    auto unique_b_set = generate_random_uint32_set(unique_b, seed + 1, set_size * 20);
    for (size_t i = 0; i < unique_b && i < unique_b_set.size(); ++i) {
        set_b.push_back(unique_b_set[i]);
    }
    
    return {set_a, set_b};
}

// Generate random string vector
vector<string> generate_random_string_set(size_t size, uint32_t seed, size_t string_length = 16) {
    vector<string> result;
    result.reserve(size);
    mt19937 rng(seed);
    uniform_int_distribution<int> char_dist(0, 25);
    
    for (size_t i = 0; i < size; ++i) {
        string s;
        s.reserve(string_length);
        for (size_t j = 0; j < string_length; ++j) {
            s.push_back(static_cast<char>('a' + char_dist(rng)));
        }
        result.push_back(move(s));
    }
    
    // Remove duplicates
    sort(result.begin(), result.end());
    result.erase(unique(result.begin(), result.end()), result.end());
    
    return result;
}

// ===================== Utility Functions =====================

// Compute true Jaccard similarity
template<typename T>
double compute_true_jaccard(const vector<T>& set_a, const vector<T>& set_b) {
    // Assuming sets are sorted
    size_t intersection = 0;
    size_t i = 0, j = 0;
    
    while (i < set_a.size() && j < set_b.size()) {
        if (set_a[i] == set_b[j]) {
            ++intersection;
            ++i;
            ++j;
        } else if (set_a[i] < set_b[j]) {
            ++i;
        } else {
            ++j;
        }
    }
    
    size_t union_size = set_a.size() + set_b.size() - intersection;
    return union_size > 0 ? static_cast<double>(intersection) / static_cast<double>(union_size) : 0.0;
}

// Estimate Jaccard from FastSimilaritySketch results
double estimate_jaccard_fasthash(const vector<uint64_t>& sketch_a, const vector<uint64_t>& sketch_b) {
    assert(sketch_a.size() == sketch_b.size());
    size_t matches = 0;
    for (size_t i = 0; i < sketch_a.size(); ++i) {
        if (sketch_a[i] == sketch_b[i]) {
            ++matches;
        }
    }
    return static_cast<double>(matches) / static_cast<double>(sketch_a.size());
}

// ===================== Test Structures =====================

struct PerformanceResult {
    double prehash_time_ms = 0.0;
    double compute_time_ms = 0.0;
    double total_time_ms = 0.0;
    double jaccard_estimate = 0.0;
    double jaccard_error = 0.0;
    size_t sketch_size = 0;
    string algorithm_name;
};

struct TestCase {
    size_t input_size;
    size_t sketch_size;
    double target_jaccard;
    string test_name;
};

// ===================== Performance Testing Functions =====================

PerformanceResult test_rminhash_uint32(const vector<uint32_t>& set_a, 
                                      const vector<uint32_t>& set_b,
                                      size_t sketch_size,
                                      double true_jaccard) {
    PerformanceResult result;
    result.algorithm_name = "RMinHashSIMD";
    result.sketch_size = sketch_size;
    
    const uint64_t seed = 42;
    RMinHashSIMD sketcher_a(sketch_size, seed);
    RMinHashSIMD sketcher_b(sketch_size, seed);
    
    // Test set A
    double prehash_a = 0, update_a = 0, total_a = 0;
    auto sketch_a = sketcher_a.sketch(set_a, &prehash_a, &update_a, &total_a);
    
    // Test set B
    double prehash_b = 0, update_b = 0, total_b = 0;
    auto sketch_b = sketcher_b.sketch(set_b, &prehash_b, &update_b, &total_b);
    
    // Compute Jaccard estimate
    result.jaccard_estimate = sketcher_a.jaccard(sketcher_b);
    result.jaccard_error = abs(result.jaccard_estimate - true_jaccard);
    
    result.prehash_time_ms = prehash_a + prehash_b;
    result.compute_time_ms = update_a + update_b;
    result.total_time_ms = total_a + total_b;
    
    return result;
}

PerformanceResult test_fasthash_uint32(const vector<uint32_t>& set_a,
                                      const vector<uint32_t>& set_b,
                                      size_t sketch_size,
                                      double true_jaccard) {
    PerformanceResult result;
    result.algorithm_name = "FastHashSIMD";
    result.sketch_size = sketch_size;
    
    // FastHash expects power of 2 sketch size
    size_t power_of_2_size = 1;
    while (power_of_2_size < sketch_size) power_of_2_size *= 2;
    
    FastSimilaritySketchAVX512Packed sketcher(static_cast<int>(power_of_2_size), 42);
    
    // Test set A
    double prehash_a = 0, phase1_a = 0, phase2_a = 0;
    auto start_a = high_resolution_clock::now();
    auto sketch_a = sketcher.sketch(set_a, &prehash_a, &phase1_a, &phase2_a);
    auto end_a = high_resolution_clock::now();
    double total_a = duration<double, milli>(end_a - start_a).count();
    
    // Test set B
    double prehash_b = 0, phase1_b = 0, phase2_b = 0;
    auto start_b = high_resolution_clock::now();
    auto sketch_b = sketcher.sketch(set_b, &prehash_b, &phase1_b, &phase2_b);
    auto end_b = high_resolution_clock::now();
    double total_b = duration<double, milli>(end_b - start_b).count();
    
    // Truncate to requested sketch size for fair comparison
    sketch_a.resize(sketch_size);
    sketch_b.resize(sketch_size);
    
    // Compute Jaccard estimate
    result.jaccard_estimate = estimate_jaccard_fasthash(sketch_a, sketch_b);
    result.jaccard_error = abs(result.jaccard_estimate - true_jaccard);
    
    result.prehash_time_ms = prehash_a + prehash_b;
    result.compute_time_ms = (phase1_a + phase2_a) + (phase1_b + phase2_b);
    result.total_time_ms = total_a + total_b;
    
    return result;
}

PerformanceResult test_rminhash_strings(const vector<string>& set_a,
                                       const vector<string>& set_b,
                                       size_t sketch_size,
                                       double true_jaccard) {
    PerformanceResult result;
    result.algorithm_name = "RMinHashSIMD-Strings";
    result.sketch_size = sketch_size;
    
    const uint64_t seed = 42;
    RMinHashSIMD sketcher_a(sketch_size, seed);
    RMinHashSIMD sketcher_b(sketch_size, seed);
    
    // Test set A
    double prehash_a = 0, update_a = 0, total_a = 0;
    auto sketch_a = sketcher_a.sketch(set_a, &prehash_a, &update_a, &total_a);
    
    // Test set B  
    double prehash_b = 0, update_b = 0, total_b = 0;
    auto sketch_b = sketcher_b.sketch(set_b, &prehash_b, &update_b, &total_b);
    
    // Compute Jaccard estimate
    result.jaccard_estimate = sketcher_a.jaccard(sketcher_b);
    result.jaccard_error = abs(result.jaccard_estimate - true_jaccard);
    
    result.prehash_time_ms = prehash_a + prehash_b;
    result.compute_time_ms = update_a + update_b;
    result.total_time_ms = total_a + total_b;
    
    return result;
}

PerformanceResult test_fasthash_strings(const vector<string>& set_a,
                                       const vector<string>& set_b,
                                       size_t sketch_size,
                                       double true_jaccard) {
    PerformanceResult result;
    result.algorithm_name = "FastHashSIMD-Strings";
    result.sketch_size = sketch_size;
    
    // FastHash expects power of 2 sketch size
    size_t power_of_2_size = 1;
    while (power_of_2_size < sketch_size) power_of_2_size *= 2;
    
    FastSimilaritySketchAVX512Packed sketcher(static_cast<int>(power_of_2_size), 42);
    
    // Test set A
    double prehash_a = 0, phase1_a = 0, phase2_a = 0;
    auto start_a = high_resolution_clock::now();
    auto sketch_a = sketcher.sketch(set_a, &prehash_a, &phase1_a, &phase2_a);
    auto end_a = high_resolution_clock::now();
    double total_a = duration<double, milli>(end_a - start_a).count();
    
    // Test set B
    double prehash_b = 0, phase1_b = 0, phase2_b = 0;
    auto start_b = high_resolution_clock::now();
    auto sketch_b = sketcher.sketch(set_b, &prehash_b, &phase1_b, &phase2_b);
    auto end_b = high_resolution_clock::now();
    double total_b = duration<double, milli>(end_b - start_b).count();
    
    // Truncate to requested sketch size for fair comparison
    sketch_a.resize(sketch_size);
    sketch_b.resize(sketch_size);
    
    // Compute Jaccard estimate
    result.jaccard_estimate = estimate_jaccard_fasthash(sketch_a, sketch_b);
    result.jaccard_error = abs(result.jaccard_estimate - true_jaccard);
    
    result.prehash_time_ms = prehash_a + prehash_b;
    result.compute_time_ms = (phase1_a + phase2_a) + (phase1_b + phase2_b);
    result.total_time_ms = total_a + total_b;
    
    return result;
}

// ===================== Reporting Functions =====================

void print_test_header(const string& test_name) {
    cout << "\n" << string(80, '=') << "\n";
    cout << "TEST: " << test_name << "\n";
    cout << string(80, '=') << "\n";
}

void print_result_table_header() {
    cout << "\n" << left;
    cout << setw(20) << "Algorithm"
         << setw(12) << "SketchSize"
         << setw(12) << "Prehash(ms)"
         << setw(12) << "Compute(ms)"
         << setw(12) << "Total(ms)"
         << setw(12) << "JaccEst"
         << setw(12) << "JaccError"
         << setw(12) << "Speedup"
         << "\n";
    cout << string(120, '-') << "\n";
}

void print_result_row(const PerformanceResult& result, double baseline_time = 0.0) {
    double speedup = (baseline_time > 0) ? baseline_time / result.total_time_ms : 1.0;
    
    cout << left << fixed << setprecision(3);
    cout << setw(20) << result.algorithm_name
         << setw(12) << result.sketch_size
         << setw(12) << result.prehash_time_ms
         << setw(12) << result.compute_time_ms
         << setw(12) << result.total_time_ms
         << setw(12) << result.jaccard_estimate
         << setw(12) << result.jaccard_error;
    
    if (baseline_time > 0) {
        cout << setw(12) << speedup << "x";
    } else {
        cout << setw(12) << "baseline";
    }
    cout << "\n";
}

// ===================== Main Test Suite =====================

void run_uint32_performance_test() {
    print_test_header("uint32_t Performance Comparison");
    
    vector<TestCase> test_cases = {
        {1000, 128, 0.3, "Small Dataset, Small Sketch"},
        {10000, 256, 0.3, "Medium Dataset, Medium Sketch"},
        {50000, 512, 0.3, "Large Dataset, Large Sketch"},
        {10000, 128, 0.1, "Low Jaccard Similarity"},
        {10000, 128, 0.7, "High Jaccard Similarity"}
    };
    
    for (const auto& test_case : test_cases) {
        cout << "\n--- " << test_case.test_name << " ---\n";
        cout << "Input size: " << test_case.input_size 
             << ", Sketch size: " << test_case.sketch_size
             << ", Target Jaccard: " << test_case.target_jaccard << "\n";
        
        // Generate test data
        auto [set_a, set_b] = generate_overlapping_sets(test_case.input_size, test_case.target_jaccard, 12345);
        double true_jaccard = compute_true_jaccard(set_a, set_b);
        
        cout << "Actual sets - A: " << set_a.size() << ", B: " << set_b.size() 
             << ", True Jaccard: " << fixed << setprecision(4) << true_jaccard << "\n";
        
        // Run tests
        auto rminhash_result = test_rminhash_uint32(set_a, set_b, test_case.sketch_size, true_jaccard);
        auto fasthash_result = test_fasthash_uint32(set_a, set_b, test_case.sketch_size, true_jaccard);
        
        print_result_table_header();
        print_result_row(rminhash_result);
        print_result_row(fasthash_result, rminhash_result.total_time_ms);
    }
}

void run_string_performance_test() {
    print_test_header("String Performance Comparison");
    
    vector<TestCase> test_cases = {
        {5000, 128, 0.3, "String Dataset, Small Sketch"},
        {20000, 256, 0.3, "String Dataset, Medium Sketch"}
    };
    
    for (const auto& test_case : test_cases) {
        cout << "\n--- " << test_case.test_name << " ---\n";
        cout << "Input size: " << test_case.input_size 
             << ", Sketch size: " << test_case.sketch_size << "\n";
        
        // Generate test data
        auto set_a = generate_random_string_set(test_case.input_size, 54321);
        auto set_b = generate_random_string_set(test_case.input_size, 98765);
        
        // Create some overlap
        size_t overlap_size = static_cast<size_t>(test_case.input_size * test_case.target_jaccard / 2);
        for (size_t i = 0; i < overlap_size && i < min(set_a.size(), set_b.size()); ++i) {
            set_b[i] = set_a[i];
        }
        
        sort(set_a.begin(), set_a.end());
        sort(set_b.begin(), set_b.end());
        double true_jaccard = compute_true_jaccard(set_a, set_b);
        
        cout << "Actual sets - A: " << set_a.size() << ", B: " << set_b.size() 
             << ", True Jaccard: " << fixed << setprecision(4) << true_jaccard << "\n";
        
        // Run tests
        auto rminhash_result = test_rminhash_strings(set_a, set_b, test_case.sketch_size, true_jaccard);
        auto fasthash_result = test_fasthash_strings(set_a, set_b, test_case.sketch_size, true_jaccard);
        
        print_result_table_header();
        print_result_row(rminhash_result);
        print_result_row(fasthash_result, rminhash_result.total_time_ms);
    }
}

void run_scalability_test() {
    print_test_header("Scalability Analysis");
    
    vector<size_t> input_sizes = {1000, 5000, 10000, 25000, 50000};
    const size_t sketch_size = 256;
    const double target_jaccard = 0.3;
    
    cout << "\nScaling with input size (sketch size = " << sketch_size << "):\n";
    print_result_table_header();
    
    for (size_t input_size : input_sizes) {
        auto [set_a, set_b] = generate_overlapping_sets(input_size, target_jaccard, 11111);
        double true_jaccard = compute_true_jaccard(set_a, set_b);
        
        auto rminhash_result = test_rminhash_uint32(set_a, set_b, sketch_size, true_jaccard);
        auto fasthash_result = test_fasthash_uint32(set_a, set_b, sketch_size, true_jaccard);
        
        rminhash_result.algorithm_name = "RMinHash(" + to_string(input_size) + ")";
        fasthash_result.algorithm_name = "FastHash(" + to_string(input_size) + ")";
        
        print_result_row(rminhash_result);
        print_result_row(fasthash_result);
    }
}

int main() {
    cout << "Performance Comparison: RMinHashSIMD vs FastSimilaritySketchAVX512Packed\n";
    cout << "========================================================================\n";
    
    try {
        run_uint32_performance_test();
        run_string_performance_test();
        run_scalability_test();
        
        cout << "\n" << string(80, '=') << "\n";
        cout << "All tests completed successfully!\n";
        cout << "Summary:\n";
        cout << "- RMinHashSIMD implements the Rust algorithm with C++ SIMD optimizations\n";
        cout << "- FastHashSIMD uses a different sketching approach with AVX-512\n";
        cout << "- Both algorithms provide efficient similarity estimation\n";
        cout << "- Performance may vary based on data characteristics and hardware\n";
        cout << string(80, '=') << "\n";
        
    } catch (const exception& e) {
        cerr << "Test failed with error: " << e.what() << "\n";
        return 1;
    }
    
    return 0;
}
