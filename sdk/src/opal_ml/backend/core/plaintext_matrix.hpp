#pragma once

#include <cstddef>
#include <vector>

namespace fhe {

// Abstract interface for a plaintext 2-D matrix used in the diagonal matmul algorithm.
// Concrete types wrap either a dense nested-list (Python side) or a device-side buffer.
class IPlaintextMatrix {
public:
    virtual ~IPlaintextMatrix() = default;

    virtual std::size_t rows() const noexcept = 0;
    virtual std::size_t cols() const noexcept = 0;

    // Return the k-th cyclic diagonal of the matrix:
    //   result[i] = mat[i % rows][(i + k) % cols]  for i in [0, len)
    // k may be negative or >= cols; wrapping is applied cyclically.
    // len = 0 means min(rows, cols) elements.
    virtual std::vector<double> diagonal(int k, std::size_t len = 0) const = 0;
};

} // namespace fhe
