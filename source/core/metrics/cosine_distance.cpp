/*
 * Copyright (C) 2025 Advanced Micro Devices, Inc. All rights reserved.
 *
 * Redistribution and use in source and binary forms, with or without modification,
 * are permitted provided that the following conditions are met:
 * 1. Redistributions of source code must retain the above copyright notice,
 *    this list of conditions and the following disclaimer.
 * 2. Redistributions in binary form must reproduce the above copyright notice,
 *    this list of conditions and the following disclaimer in the documentation
 *    and/or other materials provided with the distribution.
 * 3. Neither the name of the copyright holder nor the names of its contributors
 *    may be used to endorse or promote products derived from this software without
 *    specific prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND
 * ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
 * WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE DISCLAIMED.
 * IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT,
 * INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING,
 * BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA,
 * OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY,
 * WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
 * ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
 * POSSIBILITY OF SUCH DAMAGE.
 *
 */

#include "aoclda.h"
#include "aoclda_types.h"
#include "da_cblas.hh"
#include "da_error.hpp"
#include "pairwise_distances.hpp"

namespace ARCH {
namespace da_metrics {
namespace pairwise {

template <typename T>
void cosine_kernel(da_int m, da_int n, da_int k, const T *X, da_int ldx,
                 const T *Y, da_int ldy, T *D, da_int ldd, bool compute_distance, T *X_norms, T *Y_norms) {
    // Go through the rows of D
    for (da_int i = 0; i < m; i++) {
        // Go through the columns of X (also updating the columns of D)
        for (da_int j = 0; j < n; j++) {
            D[i * ldd + j] =
                da_blas::cblas_dot(k, X + i * ldx, 1, Y + j * ldy, 1);
        }

        if (compute_distance) {
            for (da_int j = 0; j < n; j++)
                D[i * ldd + j] = 1.0 - (D[i * ldd + j] / (X_norms[i] * Y_norms[j]));
        } else {
            for (da_int j = 0; j < n; j++)
                D[i * ldd + j] /= (X_norms[i] * Y_norms[j]);
        }
    }
}

template <typename T>
da_status cosine(da_order order, da_int m, da_int n, da_int k, const T *X, da_int ldx,
                 const T *Y, da_int ldy, T *D, da_int ldd, bool compute_distance) {
    da_status status = da_status_success;
    const T *Y_new = Y;

    // T normX, normY;
    // We want to compute the distance of X to itself
    // The sizes are copies so it's safe to update them
    if (Y == nullptr) {
        n = m;
        ldy = ldx;
        Y_new = X;
    }

    T *D_new = D;
    da_int ldd_new = ldd;
    const T *X_new = X;
    // Create temporary vectors X_row and Y_row
    std::vector<T> X_row, Y_row, D_row;
    if (order == column_major) {
        try {
            X_row.resize(m * k);
            Y_row.resize(n * k);
            D_row.resize(m * n);
        } catch (std::bad_alloc const &) {
            return da_status_memory_error;
        }
        // Transpose X and Y so that the data is stored in row major order
        da_blas::omatcopy('T', m, k, 1.0, X_new, ldx, X_row.data(), k);
        da_blas::omatcopy('T', n, k, 1.0, Y_new, ldy, Y_row.data(), k);
        D_new = D_row.data();
        X_new = X_row.data();
        Y_new = Y_row.data();
        ldx = k;
        ldy = k;
        ldd_new = n;
    }

    // Allocate norm arrays
    std::vector<T> X_norms, Y_norms;
    try {
        X_norms.resize(m);
        Y_norms.resize(n);
    } catch (std::bad_alloc const &) {
        return da_status_memory_error;
    }

    // Precalculate the norms, setting any 0s to 1 to avoid division by 0 later
    // Norm is only 0 for the 0 vector, and any vector dotted with the 0 vector is 0
    // So this doesn't interfere with accuracy
    // Should probably do norms[i] <= epsilon for the 0 check instead.
    for (da_int i = 0; i < m; ++i) {
        X_norms[i] = da_blas::cblas_nrm2(k, X_new + i * ldx, 1);

        if (X_norms[i] == 0.0) {
            X_norms[i] = 1.0;
        }
    }

    if (Y == nullptr) {
        for (da_int j = 0; j < n; ++j) {
            Y_norms[j] = X_norms[j];
        }
    } else {
        for (da_int j = 0; j < n; ++j) {
            Y_norms[j] = da_blas::cblas_nrm2(k, Y_new + j * ldy, 1);

            if (Y_norms[j] == 0.0) {
                Y_norms[j] = 1.0;
            }
        }
    }

    if (Y != nullptr) {
        // Block sizes
        da_int i_block_size = std::min(200, m);
        da_int j_block_size = std::min(200, n);
        #pragma omp parallel for shared(m, n, k, i_block_size, j_block_size, X_new, ldx, Y_new, ldy, D, ldd, compute_distance, X_norms, Y_norms) collapse(2)
        for (da_int jj = 0; jj < n; jj+=j_block_size) {
            for (da_int ii = 0; ii < m; ii+=i_block_size) {
                int jb = std::min(n-jj, j_block_size);
                int ib = std::min(m-ii, i_block_size);

                cosine_kernel(ib, jb, k,
                    X_new + ii*ldx, ldx,
                    Y_new + jj*ldy, ldy,
                    D_new + jj + ii*ldd_new, ldd_new,
                    compute_distance, X_norms.data(), Y_norms.data());
            }
        }
    } else {
        // Block sizes
        da_int block_size = std::min(2048, m);
        #pragma omp parallel for shared(m, k, block_size, X_new, ldx, D, ldd, compute_distance, X_norms) collapse(2)
        for (da_int jj = 0; jj < m; jj+=block_size) {
            for (da_int ii = 0; ii < m; ii+=block_size) {
                int jb = std::min(n-jj, block_size);
                int ib = std::min(m-ii, block_size);

                cosine_kernel(ib, jb, k,
                    X_new + ii*ldx, ldx,
                    X_new + jj*ldx, ldx,
                    D_new + jj + ii*ldd_new, ldd_new,
                    compute_distance, X_norms.data(), X_norms.data());
            }
        }
    }
    if (order == column_major) {
        // Transpose D to return data in column major order
        da_blas::omatcopy('T', n, m, 1.0, D_new, ldd_new, D, ldd);
    }

    return status;
}

} // namespace pairwise

namespace pairwise_distances {

template <typename T>
da_status cosine(da_order order, da_int m, da_int n, da_int k, const T *X, da_int ldx,
                 const T *Y, da_int ldy, T *D, da_int ldd) {
    return da_metrics::pairwise::cosine(order, m, n, k, X, ldx, Y, ldy, D, ldd, true);
}

template da_status cosine<float>(da_order order, da_int m, da_int n, da_int k,
                                 const float *X, da_int ldx, const float *Y, da_int ldy,
                                 float *D, da_int ldd);

template da_status cosine<double>(da_order order, da_int m, da_int n, da_int k,
                                  const double *X, da_int ldx, const double *Y,
                                  da_int ldy, double *D, da_int ldd);

} // namespace pairwise_distances
} // namespace da_metrics
} // namespace ARCH