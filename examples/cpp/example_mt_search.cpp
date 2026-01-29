#include "../../hnswlib/hnswlib.h"
#include <thread>
#include <fcntl.h>
#include <unistd.h>
#include <sys/mman.h>
#include <cstring>
#include <stdexcept>
#include <unordered_set>
#include <vector>


template<typename T>
class BinaryFile {
public:
    BinaryFile(const char* filepath, uint32_t max_rows = 0) 
        : fd_(-1), mapped_ptr_(nullptr), data_(nullptr), file_size_(0) {
        
        fd_ = open(filepath, O_RDONLY);
        if (fd_ == -1) {
            throw std::runtime_error(std::string("Error opening file: ") + filepath);
        }
        
        uint32_t shape[2];
        ssize_t bytesRead = read(fd_, shape, 8);
        if (bytesRead != 8) {
            close(fd_);
            throw std::runtime_error(std::string("Error reading shape from file: ") + filepath);
        }
        
        shape_[0] = (max_rows > 0 && max_rows < shape[0]) ? max_rows : shape[0];
        shape_[1] = shape[1];
        
        size_t data_size = shape_[0] * static_cast<size_t>(shape_[1]);
        size_t header_size = 8;
        file_size_ = data_size * sizeof(T) + header_size;
        
        mapped_ptr_ = (uint8_t*)mmap(nullptr, file_size_, PROT_READ, MAP_SHARED, fd_, 0);
        if (mapped_ptr_ == MAP_FAILED) {
            close(fd_);
            throw std::runtime_error(std::string("Error mmapping file: ") + filepath);
        }
        
        data_ = reinterpret_cast<T*>(mapped_ptr_ + header_size);
    }
    
    ~BinaryFile() {
        if (mapped_ptr_ != nullptr && mapped_ptr_ != MAP_FAILED) {
            munmap(mapped_ptr_, file_size_);
        }
        if (fd_ != -1) {
            close(fd_);
        }
    }
    
    BinaryFile(const BinaryFile&) = delete;
    BinaryFile& operator=(const BinaryFile&) = delete;
    
    T* data() { return data_; }
    const T* data() const { return data_; }
    
    uint32_t rows() const { return shape_[0]; }
    uint32_t cols() const { return shape_[1]; }
    
private:
    int fd_;
    uint8_t* mapped_ptr_;
    T* data_;
    size_t file_size_;
    uint32_t shape_[2];
};


// Multithreaded executor
// The helper function copied from python_bindings/bindings.cpp (and that itself is copied from nmslib)
// An alternative is using #pragme omp parallel for or any other C++ threading
template<class Function>
inline void ParallelFor(size_t start, size_t end, size_t numThreads, Function fn) {
    if (numThreads <= 0) {
        numThreads = std::thread::hardware_concurrency();
    }

    if (numThreads == 1) {
        for (size_t id = start; id < end; id++) {
            fn(id, 0);
        }
    } else {
        std::vector<std::thread> threads;
        std::atomic<size_t> current(start);

        // keep track of exceptions in threads
        // https://stackoverflow.com/a/32428427/1713196
        std::exception_ptr lastException = nullptr;
        std::mutex lastExceptMutex;

        for (size_t threadId = 0; threadId < numThreads; ++threadId) {
            threads.push_back(std::thread([&, threadId] {
                while (true) {
                    size_t id = current.fetch_add(1);

                    if (id >= end) {
                        break;
                    }

                    try {
                        fn(id, threadId);
                    } catch (...) {
                        std::unique_lock<std::mutex> lastExcepLock(lastExceptMutex);
                        lastException = std::current_exception();
                        /*
                         * This will work even when current is the largest value that
                         * size_t can fit, because fetch_add returns the previous value
                         * before the increment (what will result in overflow
                         * and produce 0 instead of current + 1).
                         */
                        current = end;
                        break;
                    }
                }
            }));
        }
        for (auto &thread : threads) {
            thread.join();
        }
        if (lastException) {
            std::rethrow_exception(lastException);
        }
    }
}


int main(int argc, char* argv[]) {
    // Parse command-line options
    const char* index_load_path = nullptr;
    const char* index_save_path = nullptr;
    std::vector<const char*> positional_args;
    
    for (int i = 1; i < argc; i++) {
        if (std::strcmp(argv[i], "-i") == 0 && i + 1 < argc) {
            index_load_path = argv[++i];
        } else if (std::strcmp(argv[i], "-o") == 0 && i + 1 < argc) {
            index_save_path = argv[++i];
        } else {
            positional_args.push_back(argv[i]);
        }
    }
    
    if (positional_args.size() != 3) {
        std::cerr << "Usage: " << argv[0] << " [-i index_file] [-o index_file] <dataset_file> <queries_file> <groundtruth_file>" << std::endl;
        return EXIT_FAILURE;
    }

    try {
        BinaryFile<float> dataset(positional_args[0], 100);
        BinaryFile<float> queries(positional_args[1]);
        BinaryFile<int> groundtruth(positional_args[2]);
        
        std::cout << "Dataset shape: [" << dataset.rows() << ", " << dataset.cols() << "]" << std::endl;
        std::cout << "Queries shape: [" << queries.rows() << ", " << queries.cols() << "]" << std::endl;
        std::cout << "Groundtruth shape: [" << groundtruth.rows() << ", " << groundtruth.cols() << "]" << std::endl;
        
        if (dataset.cols() != queries.cols()) {
            std::cerr << "Error: Dataset and queries dimensions don't match ("
                      << dataset.cols() << " vs " << queries.cols() << ")" << std::endl;
            return EXIT_FAILURE;
        }
        
        int dim = dataset.cols();
        int max_elements = dataset.rows();
        int num_queries = queries.rows();
        int M = 16;
        int ef_construction = 200;
        int num_threads = 20;
        
        // Initing or loading index
        hnswlib::L2Space space(dim);
        hnswlib::HierarchicalNSW<float>* alg_hnsw = nullptr;
        
        if (index_load_path) {
            std::cout << "Loading index from " << index_load_path << "..." << std::endl;
            alg_hnsw = new hnswlib::HierarchicalNSW<float>(&space, index_load_path, false);
            std::cout << "Index loaded with " << alg_hnsw->cur_element_count << " elements" << std::endl;
        } else {
            alg_hnsw = new hnswlib::HierarchicalNSW<float>(&space, max_elements, M, ef_construction);
            
            // Add data to index
            std::cout << "Building index with " << max_elements << " points..." << std::endl;
            ParallelFor(0, max_elements, num_threads, [&](size_t row, size_t threadId) {
                alg_hnsw->addPoint((void*)(dataset.data() + dim * row), row);
            });
            std::cout << "Index built successfully" << std::endl;
        }
        
        // Save index if requested
        if (index_save_path) {
            std::cout << "Saving index to " << index_save_path << "..." << std::endl;
            alg_hnsw->saveIndex(index_save_path);
            std::cout << "Index saved successfully" << std::endl;
        }
        
        // Query with the loaded queries
        int k = 10;
        std::cout << "Searching " << num_queries << " queries for " << k << " neighbors..." << std::endl;
        std::vector<std::vector<hnswlib::labeltype>> neighbors(num_queries, std::vector<hnswlib::labeltype>(k));
        ParallelFor(0, num_queries, num_threads, [&](size_t row, size_t threadId) {
            std::priority_queue<std::pair<float, hnswlib::labeltype>> result = alg_hnsw->searchKnn(queries.data() + dim * row, k);
            for (int j = k - 1; j >= 0; j--) {
                neighbors[row][j] = result.top().second;
                result.pop();
            }
        });
        
        std::cout << "Search completed. First 5 queries:" << std::endl;
        for (int i = 0; i < std::min(5, num_queries); i++) {
            std::cout << "Query " << i << " -> Neighbors: ";
            for (int j = 0; j < k; j++) {
                std::cout << neighbors[i][j];
                if (j < k - 1) std::cout << ", ";
            }
            std::cout << std::endl;
        }
        
        // Calculate recall
        int gt_cols = std::min(k, static_cast<int>(groundtruth.cols()));
        int total_matches = 0;
        for (int i = 0; i < num_queries; i++) {
            std::unordered_set<int> gt_set;
            for (int j = 0; j < gt_cols; j++) {
                gt_set.insert(groundtruth.data()[i * groundtruth.cols() + j]);
            }
            for (int j = 0; j < k; j++) {
                if (gt_set.count(neighbors[i][j])) {
                    total_matches++;
                }
            }
        }
        
        double recall = static_cast<double>(total_matches) / (num_queries * k);
        std::cout << "\nRecall@" << k << ": " << recall 
                  << " (" << total_matches << "/" << (num_queries * k) << ")" << std::endl;
        
        delete alg_hnsw;
    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << std::endl;
        return EXIT_FAILURE;
    }
    
    return 0;
}
