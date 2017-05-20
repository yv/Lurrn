# -*- mode:cython -*-
cdef extern from "Python.h":
    ctypedef int Py_ssize_t
    int PyObject_AsCharBuffer(object obj, char **buffer, Py_ssize_t *buffer_len) except -1
    int PyObject_AsReadBuffer(object obj, char **buffer, Py_ssize_t *buffer_len) except -1
    int PyObject_CheckReadBuffer(object o)
    int PyObject_AsWriteBuffer(object obj, char **buffer, Py_ssize_t *buffer_len) except -1
    object PyBytes_FromStringAndSize(char *v, Py_ssize_t len)
    void *PyMem_Malloc(int)
    void PyMem_Free(void *p)

cdef extern from "string.h":
    ctypedef int size_t
    void *memcpy(void *dest, void *src, size_t n)

cdef extern from "math.h":
    double log(double x)
    double log1p(double x)
    double sqrt(double x)
    double fabs(double x)
    double pow(double x, double y)
    double M_LN2

cdef extern from "math_stuff.h":
    double inverse_erf(double x)

import numpy
cimport numpy

cimport cython
from libcpp.vector cimport vector
from libcpp.algorithm cimport sort

cdef extern from "cxx_gram.h":
    ctypedef unsigned int coordinate_t

    ctypedef int *const_int_ptr "const int *"
    struct c_CItemI1 "CountItem<1,int> ":
        int *addr
        int item
    ctypedef vector[c_CItemI1] c_VecI1 "std::vector<CountItem<1,int> >"
    ctypedef vector[c_CItemI1].iterator c_VecIterI1 "std::vector<CountItem<1,int> >::iterator"
    struct c_SmallerAddrI1 "smallerAddr<1,int> ":
        pass
    c_CItemI1 c_VecI1_get_pointer "get_pointer" (c_VecI1 *v, size_t k)

    void c_compactifyI1 "compactify"(c_VecI1 *v)
    int c_get_countI1 "get_count_v"(c_VecI1 *v,
                                    c_CItemI1)
    struct c_CItemI2 "CountItem<2,int> ":
        int *addr
        int item
    ctypedef vector[c_CItemI2] c_VecI2 "std::vector<CountItem<2,int> >"
    ctypedef vector[c_CItemI2].iterator c_VecIterI2 "std::vector<CountItem<2,int> >::iterator"
    struct c_SmallerAddrI2 "smallerAddr<2,int> ":
        pass
    c_CItemI2 c_VecI2_get_pointer "get_pointer" (c_VecI2 *v, size_t k)

    void c_compactifyI2 "compactify"(c_VecI2 *v)
    int c_get_countI2 "get_count_v"(c_VecI2 *v,
                                    c_CItemI2)
    struct c_CItemI3 "CountItem<3,int> ":
        int *addr
        int item
    ctypedef vector[c_CItemI3] c_VecI3 "std::vector<CountItem<3,int> >"
    ctypedef vector[c_CItemI3].iterator c_VecIterI3 "std::vector<CountItem<3,int> >::iterator"
    struct c_SmallerAddrI3 "smallerAddr<3,int> ":
        pass
    c_CItemI3 c_VecI3_get_pointer "get_pointer" (c_VecI3 *v, size_t k)

    void c_compactifyI3 "compactify"(c_VecI3 *v)
    int c_get_countI3 "get_count_v"(c_VecI3 *v,
                                    c_CItemI3)
    struct c_CSRMatrixI "CSRMatrix<int>":
        coordinate_t num_rows
        int *offsets
        coordinate_t *rightColumns
        int *values
        void write_binary(int fileno)
        c_CSRMatrixI *transpose()
        void compute_left_marginals(int *vec)
        void compute_right_marginals(int *vec)
        void compute_right_squared_marginals(int *vec)
    c_CSRMatrixI *new_CSRMatrixI "new CSRMatrix<int>" ()
    c_CSRMatrixI *vec2csrI "vec2csr"(c_VecI2 *v)
    c_CSRMatrixI *add_csrI "add_csr"(c_CSRMatrixI *a,c_CSRMatrixI *b)
    int csrFromBufferI "csrFromBuffer"(void *buf, c_CSRMatrixI *m)
    c_CSRMatrixI *new_csrI "new_csr<int>"(int numRows, int numNonZero)
    void print_csrI "print_csr<int>"(c_CSRMatrixI *v)

    ctypedef float *const_float_ptr "const float *"
    struct c_CItemF1 "CountItem<1,float> ":
        int *addr
        float item
    ctypedef vector[c_CItemF1] c_VecF1 "std::vector<CountItem<1,float> >"
    ctypedef vector[c_CItemF1].iterator c_VecIterF1 "std::vector<CountItem<1,float> >::iterator"
    struct c_SmallerAddrF1 "smallerAddr<1,float> ":
        pass
    c_CItemF1 c_VecF1_get_pointer "get_pointer" (c_VecF1 *v, size_t k)

    void c_compactifyF1 "compactify"(c_VecF1 *v)
    float c_get_countF1 "get_count_v"(c_VecF1 *v,
                                    c_CItemF1)
    struct c_CItemF2 "CountItem<2,float> ":
        int *addr
        float item
    ctypedef vector[c_CItemF2] c_VecF2 "std::vector<CountItem<2,float> >"
    ctypedef vector[c_CItemF2].iterator c_VecIterF2 "std::vector<CountItem<2,float> >::iterator"
    struct c_SmallerAddrF2 "smallerAddr<2,float> ":
        pass
    c_CItemF2 c_VecF2_get_pointer "get_pointer" (c_VecF2 *v, size_t k)

    void c_compactifyF2 "compactify"(c_VecF2 *v)
    float c_get_countF2 "get_count_v"(c_VecF2 *v,
                                    c_CItemF2)
    struct c_CItemF3 "CountItem<3,float> ":
        int *addr
        float item
    ctypedef vector[c_CItemF3] c_VecF3 "std::vector<CountItem<3,float> >"
    ctypedef vector[c_CItemF3].iterator c_VecIterF3 "std::vector<CountItem<3,float> >::iterator"
    struct c_SmallerAddrF3 "smallerAddr<3,float> ":
        pass
    c_CItemF3 c_VecF3_get_pointer "get_pointer" (c_VecF3 *v, size_t k)

    void c_compactifyF3 "compactify"(c_VecF3 *v)
    float c_get_countF3 "get_count_v"(c_VecF3 *v,
                                    c_CItemF3)
    struct c_CSRMatrixF "CSRMatrix<float>":
        coordinate_t num_rows
        int *offsets
        coordinate_t *rightColumns
        float *values
        void write_binary(int fileno)
        c_CSRMatrixF *transpose()
        void compute_left_marginals(float *vec)
        void compute_right_marginals(float *vec)
        void compute_right_squared_marginals(float *vec)
    c_CSRMatrixF *new_CSRMatrixF "new CSRMatrix<float>" ()
    c_CSRMatrixF *vec2csrF "vec2csr"(c_VecF2 *v)
    c_CSRMatrixF *add_csrF "add_csr"(c_CSRMatrixF *a,c_CSRMatrixF *b)
    int csrFromBufferF "csrFromBuffer"(void *buf, c_CSRMatrixF *m)
    c_CSRMatrixF *new_csrF "new_csr<float>"(int numRows, int numNonZero)
    void print_csrF "print_csr<float>"(c_CSRMatrixF *v)

    ctypedef double *const_double_ptr "const double *"
    struct c_CItemD1 "CountItem<1,double> ":
        int *addr
        double item
    ctypedef vector[c_CItemD1] c_VecD1 "std::vector<CountItem<1,double> >"
    ctypedef vector[c_CItemD1].iterator c_VecIterD1 "std::vector<CountItem<1,double> >::iterator"
    struct c_SmallerAddrD1 "smallerAddr<1,double> ":
        pass
    c_CItemD1 c_VecD1_get_pointer "get_pointer" (c_VecD1 *v, size_t k)

    void c_compactifyD1 "compactify"(c_VecD1 *v)
    double c_get_countD1 "get_count_v"(c_VecD1 *v,
                                    c_CItemD1)
    struct c_CItemD2 "CountItem<2,double> ":
        int *addr
        double item
    ctypedef vector[c_CItemD2] c_VecD2 "std::vector<CountItem<2,double> >"
    ctypedef vector[c_CItemD2].iterator c_VecIterD2 "std::vector<CountItem<2,double> >::iterator"
    struct c_SmallerAddrD2 "smallerAddr<2,double> ":
        pass
    c_CItemD2 c_VecD2_get_pointer "get_pointer" (c_VecD2 *v, size_t k)

    void c_compactifyD2 "compactify"(c_VecD2 *v)
    double c_get_countD2 "get_count_v"(c_VecD2 *v,
                                    c_CItemD2)
    struct c_CItemD3 "CountItem<3,double> ":
        int *addr
        double item
    ctypedef vector[c_CItemD3] c_VecD3 "std::vector<CountItem<3,double> >"
    ctypedef vector[c_CItemD3].iterator c_VecIterD3 "std::vector<CountItem<3,double> >::iterator"
    struct c_SmallerAddrD3 "smallerAddr<3,double> ":
        pass
    c_CItemD3 c_VecD3_get_pointer "get_pointer" (c_VecD3 *v, size_t k)

    void c_compactifyD3 "compactify"(c_VecD3 *v)
    double c_get_countD3 "get_count_v"(c_VecD3 *v,
                                    c_CItemD3)
    struct c_CSRMatrixD "CSRMatrix<double>":
        coordinate_t num_rows
        int *offsets
        coordinate_t *rightColumns
        double *values
        void write_binary(int fileno)
        c_CSRMatrixD *transpose()
        void compute_left_marginals(double *vec)
        void compute_right_marginals(double *vec)
        void compute_right_squared_marginals(double *vec)
    c_CSRMatrixD *new_CSRMatrixD "new CSRMatrix<double>" ()
    c_CSRMatrixD *vec2csrD "vec2csr"(c_VecD2 *v)
    c_CSRMatrixD *add_csrD "add_csr"(c_CSRMatrixD *a,c_CSRMatrixD *b)
    int csrFromBufferD "csrFromBuffer"(void *buf, c_CSRMatrixD *m)
    c_CSRMatrixD *new_csrD "new_csr<double>"(int numRows, int numNonZero)
    void print_csrD "print_csr<double>"(c_CSRMatrixD *v)

    ctypedef void *const_void_ptr "const void *"
    struct c_CItemV1 "CountItem<1,void> ":
        int *addr
    ctypedef vector[c_CItemV1] c_VecV1 "std::vector<CountItem<1,void> >"
    ctypedef vector[c_CItemV1].iterator c_VecIterV1 "std::vector<CountItem<1,void> >::iterator"
    struct c_SmallerAddrV1 "smallerAddr<1,void> ":
        pass
    c_CItemV1 c_VecV1_get_pointer "get_pointer" (c_VecV1 *v, size_t k)

    void c_compactifyV1 "compactify_set"(c_VecV1 *v)
    bint c_get_countV1 "get_count_set"(c_VecV1 *v,
                                    c_CItemV1)

    struct c_CItemV2 "CountItem<2,void> ":
        int *addr
    ctypedef vector[c_CItemV2] c_VecV2 "std::vector<CountItem<2,void> >"
    ctypedef vector[c_CItemV2].iterator c_VecIterV2 "std::vector<CountItem<2,void> >::iterator"
    struct c_SmallerAddrV2 "smallerAddr<2,void> ":
        pass
    c_CItemV2 c_VecV2_get_pointer "get_pointer" (c_VecV2 *v, size_t k)

    void c_compactifyV2 "compactify_set"(c_VecV2 *v)
    bint c_get_countV2 "get_count_set"(c_VecV2 *v,
                                    c_CItemV2)

    struct c_CItemV3 "CountItem<3,void> ":
        int *addr
    ctypedef vector[c_CItemV3] c_VecV3 "std::vector<CountItem<3,void> >"
    ctypedef vector[c_CItemV3].iterator c_VecIterV3 "std::vector<CountItem<3,void> >::iterator"
    struct c_SmallerAddrV3 "smallerAddr<3,void> ":
        pass
    c_CItemV3 c_VecV3_get_pointer "get_pointer" (c_VecV3 *v, size_t k)

    void c_compactifyV3 "compactify_set"(c_VecV3 *v)
    bint c_get_countV3 "get_count_set"(c_VecV3 *v,
                                    c_CItemV3)

    struct c_CSRMatrixV "CSRMatrix<void>":
        coordinate_t num_rows
        int *offsets
        coordinate_t *rightColumns
        void *values
        void write_binary(int fileno)
        c_CSRMatrixV *transpose()
        void compute_left_marginals(void *vec)
        void compute_right_marginals(void *vec)
        void compute_right_squared_marginals(void *vec)
    c_CSRMatrixV *new_CSRMatrixV "new CSRMatrix<void>" ()
    c_CSRMatrixV *vec2csrV "vec2csr"(c_VecV2 *v)
    c_CSRMatrixV *add_csrV "add_csr"(c_CSRMatrixV *a,c_CSRMatrixV *b)
    int csrFromBufferV "csrFromBuffer"(void *buf, c_CSRMatrixV *m)
    c_CSRMatrixV *new_csrV "new_csr<void>"(int numRows, int numNonZero)
    void print_csrV "print_csr<void>"(c_CSRMatrixV *v)

    void cxx_delete "delete" (void *p)
    void cxx_deleteA "delete []"(void *p)

cdef extern from *:
    ctypedef char* const_char_ptr "const char*"

cdef class CSRMatrixD

## I -> int
cdef class CSRMatrixI
cdef class SparseVectorI


cdef class VecI1:
    cdef c_VecI1 vec
    cdef bint is_compact
    cpdef int get_count(self, coordinate_t k0)
    cdef void compactify(self)
    cdef void ensure_compact(self)
    cdef void c_add(self, coordinate_t k0, int item)
    cpdef SparseVectorI to_sparse(self)
    cpdef int get_maxcol(self, int k=*)
    cpdef numpy.ndarray get_type_counts(self, int k=*)
    cpdef VecI1 remap(self, int k, numpy.ndarray filt)


cdef class VecI2:
    cdef c_VecI2 vec
    cdef bint is_compact
    cpdef int get_count(self, coordinate_t k0,coordinate_t k1)
    cdef void compactify(self)
    cdef void ensure_compact(self)
    cdef void c_add(self, coordinate_t k0,coordinate_t k1, int item)
    cpdef CSRMatrixI to_csr(self)
    cpdef int get_maxcol(self, int k=*)
    cpdef numpy.ndarray get_type_counts(self, int k=*)
    cpdef VecI2 remap(self, int k, numpy.ndarray filt)


cdef class VecI3:
    cdef c_VecI3 vec
    cdef bint is_compact
    cpdef int get_count(self, coordinate_t k0,coordinate_t k1,coordinate_t k2)
    cdef void compactify(self)
    cdef void ensure_compact(self)
    cdef void c_add(self, coordinate_t k0,coordinate_t k1,coordinate_t k2, int item)
    cpdef int get_maxcol(self, int k=*)
    cpdef numpy.ndarray get_type_counts(self, int k=*)
    cpdef VecI3 remap(self, int k, numpy.ndarray filt)
cdef class CSRMatrixI:
    cdef c_CSRMatrixI *mat
    cdef int cache_maxcol
    # buf contains a reference to a buffer that contains the
    # arrays in the case of a mmap'ed matrix
    cdef object buf
    cdef void set_matrix(self,c_CSRMatrixI *mat_new)
    cpdef int get_count(self, coordinate_t k, coordinate_t k2)
    cpdef int get_maxcol(self)
    cpdef CSRMatrixD transform_mi(self)
    cpdef CSRMatrixD transform_mi_discount(self)
    cpdef CSRMatrixD transform_ll(self)
    cpdef CSRMatrixD transform_l1(self)
    cpdef CSRMatrixI apply_threshold(self, numpy.ndarray thresholds)
    cpdef CSRMatrixD apply_scaling(self, numpy.ndarray thresholds)
    cpdef CSRMatrixI fromVectors(self, vectors)

cdef class SparseVectorI:
    cdef object buf
    cdef int *vals_ptr
    cdef coordinate_t *idx_ptr
    cdef unsigned int my_len
    cpdef int from_dense(self, int[:] dense)
    cpdef int dotSelf(self)
    cdef int _dotFull(self, const_int_ptr full_ptr)
    cdef int _dotFull_partial(self, const_int_ptr full_ptr, int my_len)
    cpdef int dotSparse(self, SparseVectorI other)
    cdef void _axpy(self, int *x_ptr, int a)
    cpdef int sqdist(self, SparseVectorI other)
    cpdef double jaccard(self, SparseVectorI other)
    cpdef double cosine(self, SparseVectorI other)
    cpdef double min_sum(self, SparseVectorI other)
    cpdef double jsd_unnorm(self, SparseVectorI other)
    cpdef double skew_unnorm(self, SparseVectorI other, double alpha)
    cpdef SparseVectorI min_vals(self, SparseVectorI other)
    cpdef double norm_l1(self)
    cpdef double norm_l2(self)
    cpdef double norm_lp(self, double p)
cdef class SparseVectorsI:
    cdef object vecs
    cdef vector[cython.int] weights
    cpdef SparseVectorI to_vec(self)
    cpdef add(self, SparseVectorI vec, int weight=*)
    cdef int _dotFull(self, const_int_ptr full_ptr)
## F -> float
cdef class CSRMatrixF
cdef class SparseVectorF


cdef class VecF1:
    cdef c_VecF1 vec
    cdef bint is_compact
    cpdef float get_count(self, coordinate_t k0)
    cdef void compactify(self)
    cdef void ensure_compact(self)
    cdef void c_add(self, coordinate_t k0, float item)
    cpdef SparseVectorF to_sparse(self)
    cpdef int get_maxcol(self, int k=*)
    cpdef numpy.ndarray get_type_counts(self, int k=*)
    cpdef VecF1 remap(self, int k, numpy.ndarray filt)


cdef class VecF2:
    cdef c_VecF2 vec
    cdef bint is_compact
    cpdef float get_count(self, coordinate_t k0,coordinate_t k1)
    cdef void compactify(self)
    cdef void ensure_compact(self)
    cdef void c_add(self, coordinate_t k0,coordinate_t k1, float item)
    cpdef CSRMatrixF to_csr(self)
    cpdef int get_maxcol(self, int k=*)
    cpdef numpy.ndarray get_type_counts(self, int k=*)
    cpdef VecF2 remap(self, int k, numpy.ndarray filt)


cdef class VecF3:
    cdef c_VecF3 vec
    cdef bint is_compact
    cpdef float get_count(self, coordinate_t k0,coordinate_t k1,coordinate_t k2)
    cdef void compactify(self)
    cdef void ensure_compact(self)
    cdef void c_add(self, coordinate_t k0,coordinate_t k1,coordinate_t k2, float item)
    cpdef int get_maxcol(self, int k=*)
    cpdef numpy.ndarray get_type_counts(self, int k=*)
    cpdef VecF3 remap(self, int k, numpy.ndarray filt)
cdef class CSRMatrixF:
    cdef c_CSRMatrixF *mat
    cdef int cache_maxcol
    # buf contains a reference to a buffer that contains the
    # arrays in the case of a mmap'ed matrix
    cdef object buf
    cdef void set_matrix(self,c_CSRMatrixF *mat_new)
    cpdef float get_count(self, coordinate_t k, coordinate_t k2)
    cpdef int get_maxcol(self)
    cpdef CSRMatrixD transform_mi(self)
    cpdef CSRMatrixD transform_mi_discount(self)
    cpdef CSRMatrixD transform_ll(self)
    cpdef CSRMatrixD transform_l1(self)
    cpdef CSRMatrixF apply_threshold(self, numpy.ndarray thresholds)
    cpdef CSRMatrixD apply_scaling(self, numpy.ndarray thresholds)
    cpdef CSRMatrixF fromVectors(self, vectors)

cdef class SparseVectorF:
    cdef object buf
    cdef float *vals_ptr
    cdef coordinate_t *idx_ptr
    cdef unsigned int my_len
    cpdef int from_dense(self, float[:] dense)
    cpdef float dotSelf(self)
    cdef float _dotFull(self, const_float_ptr full_ptr)
    cdef float _dotFull_partial(self, const_float_ptr full_ptr, int my_len)
    cpdef float dotSparse(self, SparseVectorF other)
    cdef void _axpy(self, float *x_ptr, float a)
    cpdef float sqdist(self, SparseVectorF other)
    cpdef double jaccard(self, SparseVectorF other)
    cpdef double cosine(self, SparseVectorF other)
    cpdef double min_sum(self, SparseVectorF other)
    cpdef double jsd_unnorm(self, SparseVectorF other)
    cpdef double skew_unnorm(self, SparseVectorF other, double alpha)
    cpdef SparseVectorF min_vals(self, SparseVectorF other)
    cpdef double norm_l1(self)
    cpdef double norm_l2(self)
    cpdef double norm_lp(self, double p)
cdef class SparseVectorsF:
    cdef object vecs
    cdef vector[cython.float] weights
    cpdef SparseVectorF to_vec(self)
    cpdef add(self, SparseVectorF vec, float weight=*)
    cdef float _dotFull(self, const_float_ptr full_ptr)
## D -> double
cdef class CSRMatrixD
cdef class SparseVectorD


cdef class VecD1:
    cdef c_VecD1 vec
    cdef bint is_compact
    cpdef double get_count(self, coordinate_t k0)
    cdef void compactify(self)
    cdef void ensure_compact(self)
    cdef void c_add(self, coordinate_t k0, double item)
    cpdef SparseVectorD to_sparse(self)
    cpdef int get_maxcol(self, int k=*)
    cpdef numpy.ndarray get_type_counts(self, int k=*)
    cpdef VecD1 remap(self, int k, numpy.ndarray filt)


cdef class VecD2:
    cdef c_VecD2 vec
    cdef bint is_compact
    cpdef double get_count(self, coordinate_t k0,coordinate_t k1)
    cdef void compactify(self)
    cdef void ensure_compact(self)
    cdef void c_add(self, coordinate_t k0,coordinate_t k1, double item)
    cpdef CSRMatrixD to_csr(self)
    cpdef int get_maxcol(self, int k=*)
    cpdef numpy.ndarray get_type_counts(self, int k=*)
    cpdef VecD2 remap(self, int k, numpy.ndarray filt)


cdef class VecD3:
    cdef c_VecD3 vec
    cdef bint is_compact
    cpdef double get_count(self, coordinate_t k0,coordinate_t k1,coordinate_t k2)
    cdef void compactify(self)
    cdef void ensure_compact(self)
    cdef void c_add(self, coordinate_t k0,coordinate_t k1,coordinate_t k2, double item)
    cpdef int get_maxcol(self, int k=*)
    cpdef numpy.ndarray get_type_counts(self, int k=*)
    cpdef VecD3 remap(self, int k, numpy.ndarray filt)
cdef class CSRMatrixD:
    cdef c_CSRMatrixD *mat
    cdef int cache_maxcol
    # buf contains a reference to a buffer that contains the
    # arrays in the case of a mmap'ed matrix
    cdef object buf
    cdef void set_matrix(self,c_CSRMatrixD *mat_new)
    cpdef double get_count(self, coordinate_t k, coordinate_t k2)
    cpdef int get_maxcol(self)
    cpdef CSRMatrixD transform_mi(self)
    cpdef CSRMatrixD transform_mi_discount(self)
    cpdef CSRMatrixD transform_ll(self)
    cpdef CSRMatrixD transform_l1(self)
    cpdef CSRMatrixD apply_threshold(self, numpy.ndarray thresholds)
    cpdef CSRMatrixD apply_scaling(self, numpy.ndarray thresholds)
    cpdef CSRMatrixD fromVectors(self, vectors)

cdef class SparseVectorD:
    cdef object buf
    cdef double *vals_ptr
    cdef coordinate_t *idx_ptr
    cdef unsigned int my_len
    cpdef int from_dense(self, double[:] dense)
    cpdef double dotSelf(self)
    cdef double _dotFull(self, const_double_ptr full_ptr)
    cdef double _dotFull_partial(self, const_double_ptr full_ptr, int my_len)
    cpdef double dotSparse(self, SparseVectorD other)
    cdef void _axpy(self, double *x_ptr, double a)
    cpdef double sqdist(self, SparseVectorD other)
    cpdef double jaccard(self, SparseVectorD other)
    cpdef double cosine(self, SparseVectorD other)
    cpdef double min_sum(self, SparseVectorD other)
    cpdef double jsd_unnorm(self, SparseVectorD other)
    cpdef double skew_unnorm(self, SparseVectorD other, double alpha)
    cpdef SparseVectorD min_vals(self, SparseVectorD other)
    cpdef double norm_l1(self)
    cpdef double norm_l2(self)
    cpdef double norm_lp(self, double p)
cdef class SparseVectorsD:
    cdef object vecs
    cdef vector[cython.double] weights
    cpdef SparseVectorD to_vec(self)
    cpdef add(self, SparseVectorD vec, double weight=*)
    cdef double _dotFull(self, const_double_ptr full_ptr)
## V -> void


cdef class VecV1:
    cdef c_VecV1 vec
    cdef bint is_compact
    cpdef bint get_count(self, coordinate_t k0)
    cdef void compactify(self)
    cdef void ensure_compact(self)
    cdef void c_add(self, coordinate_t k0)
    cpdef int get_maxcol(self, int k=*)
    cpdef numpy.ndarray get_type_counts(self, int k=*)
    cpdef VecV1 remap(self, int k, numpy.ndarray filt)


cdef class VecV2:
    cdef c_VecV2 vec
    cdef bint is_compact
    cpdef bint get_count(self, coordinate_t k0,coordinate_t k1)
    cdef void compactify(self)
    cdef void ensure_compact(self)
    cdef void c_add(self, coordinate_t k0,coordinate_t k1)
    cpdef int get_maxcol(self, int k=*)
    cpdef numpy.ndarray get_type_counts(self, int k=*)
    cpdef VecV2 remap(self, int k, numpy.ndarray filt)


cdef class VecV3:
    cdef c_VecV3 vec
    cdef bint is_compact
    cpdef bint get_count(self, coordinate_t k0,coordinate_t k1,coordinate_t k2)
    cdef void compactify(self)
    cdef void ensure_compact(self)
    cdef void c_add(self, coordinate_t k0,coordinate_t k1,coordinate_t k2)
    cpdef int get_maxcol(self, int k=*)
    cpdef numpy.ndarray get_type_counts(self, int k=*)
    cpdef VecV3 remap(self, int k, numpy.ndarray filt)
