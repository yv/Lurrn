# -*- mode:cython *-
# vi:ft=python
import mmap
import sys
import numpy

cdef unsigned int BIG_ITEMS=1000000
DEF CHUNK_SIZE = 256

cdef double logB(double n, double p, double k):
    cdef double lp, lq
    if p==0.0 or p==1.0:
        return 0.0
    else:
        lp=log(p)
        lq=log1p(-p)
    return k*lp+(n-k)*lq

## I -> int, int32

cdef class VecI2
cdef class CSRMatrixI_item_iter
cdef class SparseVectorI
cdef SparseVectorI emptyvec_I

cdef class CSRMatrixI:
    def __cinit__(self, vecs=None):
        self.mat=NULL
        self.cache_maxcol=-1
        if vecs is not None:
            self.fromVectors(vecs)
    cdef void set_matrix(self,c_CSRMatrixI *mat_new):
        self.mat=mat_new
        self.cache_maxcol=-1
    cpdef CSRMatrixI fromVectors(self, vectors):
        cdef int spaceNeeded=0
        cdef int i=0
        cdef int j=0
        cdef SparseVectorI vec
        cdef c_CSRMatrixI *c_result
        cdef CSRMatrixI result
        for vec in vectors:
            spaceNeeded+=vec.my_len
            i+=1
        c_result=new_csrI(i,spaceNeeded)
        i=0
        spaceNeeded=0
        for vec in vectors:
            c_result.offsets[i]=spaceNeeded
            for j from 0<=j<vec.my_len:
                 c_result.rightColumns[spaceNeeded]=vec.idx_ptr[j]
                 c_result.values[spaceNeeded]=vec.vals_ptr[j]
                 spaceNeeded+=1
            i+=1
        c_result.offsets[i]=spaceNeeded
        self.set_matrix(c_result)
        return self
    def __lshift__(CSRMatrixI self,int k):
        cdef CSRMatrixI result
        cdef c_CSRMatrixI *c_mat
        cdef c_CSRMatrixI *c_result
        cdef int i, nnz
        c_mat=self.mat
        nnz=c_mat.offsets[c_mat.num_rows]
        c_result=new_csrI(c_mat.num_rows, nnz)
        for i from 0<=i<=c_mat.num_rows:
            c_result.offsets[i]=c_mat.offsets[i]
        for i from 0<=i<nnz:
            c_result.rightColumns[i]=c_mat.rightColumns[i]+k
        for i from 0<=i<nnz:
            c_result.values[i]=c_mat.values[i]
        result=CSRMatrixI()
        result.set_matrix(c_result)
        return result
    def __ilshift__(CSRMatrixI self, int k):
        cdef c_CSRMatrixI *c_mat
        cdef int i, nnz
        c_mat=self.mat
        nnz=c_mat.offsets[c_mat.num_rows]
        for i from 0<=i<nnz:
            c_mat.rightColumns[i]+=k
        return self
    def print_csr(self):
        print_csrI(self.mat)
    def get_size(self):
        return self.mat.offsets[self.mat.num_rows]
    def __len__(self):
        return self.mat.num_rows
    def __getitem__(self,k):
        cdef int off1, off2
        cdef SparseVectorI res
        if not (k>=0 and k<self.mat.num_rows):
            raise IndexError
        off1=self.mat.offsets[k]
        off2=self.mat.offsets[k+1]
        if off1==off2:
            return emptyvec_I
        res=SparseVectorI()
        res.buf=self
        res.my_len=off2-off1
        res.vals_ptr=&self.mat.values[off1]
        res.idx_ptr=&self.mat.rightColumns[off1]
        return res
    def item_iter(self):
        if self.mat.num_rows>0:
            return CSRMatrixI_item_iter(self)
        else:
            return ()
    def to_scipy(self):
        cdef numpy.ndarray[numpy.int32_t, ndim=1] data
        cdef numpy.ndarray[numpy.int32_t, ndim=1] indices
        cdef numpy.ndarray[numpy.int32_t, ndim=1] indptr
        cdef int num_rows=self.mat.num_rows
        cdef int max_offset=self.mat.offsets[num_rows]
        cdef int i,j
        indptr=numpy.zeros(num_rows+1, numpy.int32)
        indices=numpy.zeros(max_offset, numpy.int32)
        data=numpy.zeros(max_offset, numpy.int32)
        for i from 0<=i<=num_rows:
            indptr[i]=self.mat.offsets[i]
        for i from 0<=i<max_offset:
            indices[i]=self.mat.rightColumns[i]
        for i from 0<=i<max_offset:
            data[i]=self.mat.values[i]
        return (data,indices,indptr)
    cpdef int get_maxcol(self):
        cdef int result=0
        cdef int col_max
        cdef coordinate_t i
        cdef int off1, off2
        if self.cache_maxcol>=0:
            return self.cache_maxcol
        for i from 0<=i<self.mat.num_rows:
           off1=self.mat.offsets[i]
           off2=self.mat.offsets[i+1]
           if off2>off1:
              col_max=self.mat.rightColumns[off2-1]
              if result<col_max:
                 result=col_max
        self.cache_maxcol=result
        return result
    cpdef CSRMatrixD transform_mi(self):
        cdef numpy.ndarray[numpy.int32_t, ndim=1] marginals_l
        cdef numpy.ndarray[numpy.int32_t, ndim=1] marginals_r
        cdef VecD2 tmp=VecD2()
        cdef coordinate_t k, i
        cdef int off1, off2
        cdef double v, log_v, log_N, log1, log2
        if self.mat.num_rows<100000:
            tmp.vec.reserve(self.mat.offsets[self.mat.num_rows])
        else:
            tmp.vec.reserve(100000)
        marginals_l=self.left_marginals()
        marginals_r=self.right_marginals()
        log_N=log(marginals_l.sum())
        for k from 0<=k<self.mat.num_rows:
            log1=log(marginals_l[k])-log_N
            off1=self.mat.offsets[k]
            off2=self.mat.offsets[k+1]
            for i from off1<=i<off2:
               v=self.mat.values[i]
               if v>0:
                   log2=log(marginals_r[self.mat.rightColumns[i]])
                   log_v=log(v)-log2-log1
                   if log_v>0:
                      tmp.add_count(k,self.mat.rightColumns[i],log_v)
        return tmp.to_csr()
    cpdef CSRMatrixD transform_mi_discount(self):
        cdef numpy.ndarray[numpy.int32_t, ndim=1] marginals_l
        cdef numpy.ndarray[numpy.int32_t, ndim=1] marginals_r
        cdef VecD2 tmp=VecD2()
        cdef coordinate_t k, i
        cdef int off1, off2
        cdef double marg_l, marg_r, log_corr1, log_corr2
        cdef double v, log_v, log_N, log1, log2
        if self.mat.num_rows<100000:
            tmp.vec.reserve(self.mat.offsets[self.mat.num_rows])
        else:
            tmp.vec.reserve(100000)
        marginals_l=self.left_marginals()
        marginals_r=self.right_marginals()
        log_N=log(marginals_l.sum())
        for k from 0<=k<self.mat.num_rows:
            marg_l=marginals_l[k]
            log1=log(marg_l)-log_N
            off1=self.mat.offsets[k]
            off2=self.mat.offsets[k+1]
            for i from off1<=i<off2:
               v=self.mat.values[i]
               if v>0:
                   marg_r=marginals_r[self.mat.rightColumns[i]]
                   log2=log(marg_r)
                   log_corr1=log1p(-1.0/(v+1.0))
                   if marg_l<marg_r:
                       log_corr2=log1p(-1.0/(marg_l+1.0))
                   else:
                       log_corr2=log1p(-1.0/(marg_r+1.0))
                   log_v=log(v)-log2-log1+log_corr1+log_corr2
                   if log_v>0:
                      tmp.add_count(k,self.mat.rightColumns[i],log_v)
        return tmp.to_csr()
    cpdef CSRMatrixD transform_ll(self):
        cdef numpy.ndarray[numpy.int32_t, ndim=1] marginals_l
        cdef numpy.ndarray[numpy.int32_t, ndim=1] marginals_r
        cdef VecD2 tmp=VecD2()
        cdef coordinate_t k, i
        cdef int off1, off2
        cdef double marg_l, marg_r, c, c1, c2
        cdef double p, p1, pn1, ll
        if self.mat.num_rows<100000:
            tmp.vec.reserve(self.mat.offsets[self.mat.num_rows])
        else:
            tmp.vec.reserve(100000)
        marginals_l=self.left_marginals()
        marginals_r=self.right_marginals()
        c=marginals_l.sum()
        for k from 0<=k<self.mat.num_rows:
            c1=marginals_l[k]
            off1=self.mat.offsets[k]
            off2=self.mat.offsets[k+1]
            for i from off1<=i<off2:
               c12=self.mat.values[i]
               if c12>0:
                   c2=marginals_r[self.mat.rightColumns[i]]
                   p=c2/c
                   p1=c12/c1
                   if p1>p:
                       pn1=(c2-c12)/(c-c1)
                       ll=(-logB(c1,p,c12)-logB(c-c1,p,c2-c12)
                           +logB(c1,p1,c12)+logB(c-c1,pn1,c2-c12))
                       tmp.add_count(k,self.mat.rightColumns[i],ll)
        return tmp.to_csr()
    cpdef CSRMatrixD transform_l1(self):
        cdef numpy.ndarray[numpy.int32_t, ndim=1] marginals_l
        cdef VecD2 tmp=VecD2()
        cdef coordinate_t k, i
        cdef int off1, off2
        cdef double v, marg_l
        if self.mat.num_rows<100000:
            tmp.vec.reserve(self.mat.offsets[self.mat.num_rows])
        else:
            tmp.vec.reserve(100000)
        marginals_l=self.left_marginals()
        for k from 0<=k<self.mat.num_rows:
            marg_l=marginals_l[k]
            off1=self.mat.offsets[k]
            off2=self.mat.offsets[k+1]
            for i from off1<=i<off2:
               v=self.mat.values[i]
               if v>0:
                   tmp.add_count(k,self.mat.rightColumns[i],v/marg_l)
        return tmp.to_csr()
    def thresholds_norm(self, perc):
        cdef numpy.ndarray[numpy.int32_t, ndim=1] means
        cdef numpy.ndarray[numpy.int32_t, ndim=1] squared_means
        cdef numpy.ndarray[numpy.int32_t, ndim=1] stddev
        cdef double zscore=inverse_erf(2.0*perc-1.0)
        cdef int count_targets
        cdef int count_all=self.get_maxcol()+1
        cdef double corr_factor = count_all / (count_all - 1.0)
        count_targets=len(self)
        means=self.right_marginals()
        means /= count_targets
        squared_means = self.right_squared_marginals()
        squared_means /= count_targets
        stddev=numpy.sqrt(corr_factor * (squared_means - means*means))
        return means + zscore * stddev
    def thresholds_quantile(self, perc):
        cdef numpy.ndarray[numpy.int32_t, ndim=1] results
        cdef int i, count_nz, offset, abs_offset
        cdef int count_targets
        cdef int count_all=self.get_maxcol()+1
        cdef CSRMatrixD nonzeros=self.transpose()
        count_targets=len(self)
        count_features=len(nonzeros)
        results=numpy.zeros(count_features,numpy.int32)
        abs_offset=count_features*(1.0-perc)+0.5
        if abs_offset==0:
            abs_offset=1
        for i from 0<=i<count_features:
            nz_vals=[v for k,v in nonzeros[i]]
            nz_vals.sort()
            count_nz=len(nz_vals)
            offset=count_nz-abs_offset
            if offset<0:
                offset=0
            results[i] = nz_vals[offset]
        return results
    def thresholds_nonzero(self, perc):
        cdef numpy.ndarray[numpy.int32_t, ndim=1] results
        cdef int i
        cdef int count_all=self.get_maxcol()+1
        cdef CSRMatrixD nonzeros=self.transpose()
        results=numpy.zeros(count_all,numpy.int32)
        for i from 0<=i<count_all:
            nz_vals=[v for k,v in nonzeros[i]]
            nz_vals.sort()
            results[i] = nz_vals[int(len(nz_vals)*perc)]
        return results
    def get_quantiles(self, qs):
        cdef coordinate_t k, nnz
        cdef numpy.ndarray[numpy.int32_t, ndim=1] allvals
        nnz = self.mat.offsets[self.mat.num_rows]
        allvals = numpy.empty(self.mat.offsets[self.mat.num_rows],
                              numpy.int32)
        for k from 0<=k<self.mat.offsets[self.mat.num_rows]:
            allvals[k] = self.mat.values[k]
        allvals.sort()
        result = []
        for q in qs:
            k = nnz*q
            if k >= nnz:
                k = nnz-1
            result.append(allvals[k])
        return result
    cpdef CSRMatrixI apply_threshold(self, numpy.ndarray thr):
        cdef VecD2 tmp=VecD2()
        cdef coordinate_t k, i, k2
        cdef int off1, off2
        cdef double v, marg_l
        cdef numpy.ndarray[numpy.int32_t, ndim=1] thresholds=thr
        if self.mat.num_rows<100000:
            tmp.vec.reserve(self.mat.offsets[self.mat.num_rows])
        else:
            tmp.vec.reserve(100000)
        for k from 0<=k<self.mat.num_rows:
            off1=self.mat.offsets[k]
            off2=self.mat.offsets[k+1]
            for i from off1<=i<off2:
               v=self.mat.values[i]
               k2=self.mat.rightColumns[i]
               if v>=thresholds[k2]:
                   tmp.add_count(k,k2,1.0)
        return tmp.to_csr()
    cpdef CSRMatrixD apply_scaling(self, numpy.ndarray thr):
        cdef VecD2 tmp=VecD2()
        cdef numpy.ndarray[numpy.int32_t, ndim=1] thresholds=thr
        cdef coordinate_t k, i, k2
        cdef int off1, off2
        cdef double v, marg_l
        if self.mat.num_rows<100000:
            tmp.vec.reserve(self.mat.offsets[self.mat.num_rows])
        else:
            tmp.vec.reserve(100000)
        for k from 0<=k<self.mat.num_rows:
            off1=self.mat.offsets[k]
            off2=self.mat.offsets[k+1]
            for i from off1<=i<off2:
               k2=self.mat.rightColumns[i]
               v=self.mat.values[i]/thresholds[k2]
               if v>=1.0:
                   tmp.add_count(k,k2,1.0)
               else:
                   tmp.add_count(k,k2,v)
        return tmp.to_csr()
    def scale_columns(self, factors):
        cdef numpy.ndarray[numpy.int32_t, ndim=1] factors_l = factors
        cdef coordinate_t k
        cdef int off1, off2
        for k from 0<=k<self.mat.num_rows:
            off1 = self.mat.offsets[k]
            off2 = self.mat.offsets[k+1]
            for i from off1<=i<off2:
                k2 = self.mat.rightColumns[i]
                self.mat.values[i] *= factors_l[k2]
    def scale_rows(self, factors):
        cdef coordinate_t k
        cdef int factor
        cdef int off1, off2
        for k from 0<=k<self.mat.num_rows:
            off1 = self.mat.offsets[k]
            off2 = self.mat.offsets[k+1]
            factor = factors[k]
            for i from off1<=i<off2:
                self.mat.values[i] *= factor
    def left_marginals(self):
        cdef numpy.ndarray[numpy.int32_t, ndim=1] marginals_l
        marginals_l=numpy.zeros(self.mat.num_rows,numpy.int32)
        self.mat.compute_left_marginals(<int *> marginals_l.data)
        return marginals_l
    def right_marginals(self):
        cdef numpy.ndarray[numpy.int32_t, ndim=1] marginals_r
        marginals_r=numpy.zeros(self.get_maxcol()+1,numpy.int32)
        self.mat.compute_right_marginals(<int *> marginals_r.data)
        return marginals_r
    def right_squared_marginals(self):
        cdef numpy.ndarray[numpy.int32_t, ndim=1] marginals_r
        marginals_r=numpy.zeros(self.get_maxcol()+1,numpy.int32)
        self.mat.compute_right_squared_marginals(<int *> marginals_r.data)
        return marginals_r
    cpdef int get_count(self,coordinate_t k, coordinate_t k2):
        if k<0 or k>=self.mat.num_rows:
            return 0
        assert k<self.mat.num_rows
        cdef int lo=self.mat.offsets[k]
        cdef int hi=self.mat.offsets[k+1]
        cdef int mi
        cdef coordinate_t a_mi
        if hi==lo: return 0
        while hi-lo>1:
            mi=(hi+lo)//2
            a_mi=self.mat.rightColumns[mi]
            if a_mi>k2:
                hi=mi
            elif a_mi==k2:
                return self.mat.values[mi]
            else:
                lo=mi+1
        if self.mat.rightColumns[lo]==k2:
            return self.mat.values[lo]
        return 0
    def write_binary(self,f):
        """writes the matrix in binary format"""
        cdef int fileno=f.fileno()
        self.mat.write_binary(fileno)
    def transpose(self):
        cdef c_CSRMatrixI *matC=self.mat.transpose()
        cdef CSRMatrixI mat=CSRMatrixI()
        mat.set_matrix(matC)
        return mat
    def __add__(self,other):
        cdef CSRMatrixI mat1
        cdef CSRMatrixI mat2
        cdef c_CSRMatrixI *matC
        mat1=self
        mat2=other
        matC=add_csrI(mat1.mat,mat2.mat)
        cdef CSRMatrixI mat=CSRMatrixI()
        mat.set_matrix(matC)
        return mat
    def __dealloc__(self):
        if self.mat!=NULL:
            if self.buf is not None:
                self.buf=None
            else:
                cxx_deleteA(self.mat.offsets)
                if (self.mat.rightColumns!=NULL):
                    cxx_deleteA(self.mat.rightColumns)
                    cxx_deleteA(self.mat.values)
            cxx_delete(self.mat)
    def __reduce__(self):
        # choose compatibility over performance.
        cdef VecI2 tmp=VecI2()
        tmp.vec.reserve(self.mat.offsets[self.mat.num_rows])
        tmp.extend(self.item_iter())
        return (CSRMatrixI,(),tmp)
    def __setstate__(self,VecI2 tmp):
        tmp.ensure_compact()
        self.set_matrix(vec2csrI(&tmp.vec))

cdef class CSRMatrixI_item_iter:
    cdef CSRMatrixI mat
    cdef c_CSRMatrixI *matC
    cdef coordinate_t k
    cdef int off
    def __init__(self, mat):
        self.mat=mat
        self.matC=self.mat.mat
        self.k=0
        self.off=0
    def __iter__(self):
        return self
    def __next__(self):
        while self.off==self.matC.offsets[self.k+1]:
            self.k+=1
            if self.k>=self.matC.num_rows:
                raise StopIteration
        res=(self.k,
             self.matC.rightColumns[self.off],
             self.matC.values[self.off])
        self.off+=1
        return res
## F -> float, float32

cdef class VecF2
cdef class CSRMatrixF_item_iter
cdef class SparseVectorF
cdef SparseVectorF emptyvec_F

cdef class CSRMatrixF:
    def __cinit__(self, vecs=None):
        self.mat=NULL
        self.cache_maxcol=-1
        if vecs is not None:
            self.fromVectors(vecs)
    cdef void set_matrix(self,c_CSRMatrixF *mat_new):
        self.mat=mat_new
        self.cache_maxcol=-1
    cpdef CSRMatrixF fromVectors(self, vectors):
        cdef int spaceNeeded=0
        cdef int i=0
        cdef int j=0
        cdef SparseVectorF vec
        cdef c_CSRMatrixF *c_result
        cdef CSRMatrixF result
        for vec in vectors:
            spaceNeeded+=vec.my_len
            i+=1
        c_result=new_csrF(i,spaceNeeded)
        i=0
        spaceNeeded=0
        for vec in vectors:
            c_result.offsets[i]=spaceNeeded
            for j from 0<=j<vec.my_len:
                 c_result.rightColumns[spaceNeeded]=vec.idx_ptr[j]
                 c_result.values[spaceNeeded]=vec.vals_ptr[j]
                 spaceNeeded+=1
            i+=1
        c_result.offsets[i]=spaceNeeded
        self.set_matrix(c_result)
        return self
    def __lshift__(CSRMatrixF self,int k):
        cdef CSRMatrixF result
        cdef c_CSRMatrixF *c_mat
        cdef c_CSRMatrixF *c_result
        cdef int i, nnz
        c_mat=self.mat
        nnz=c_mat.offsets[c_mat.num_rows]
        c_result=new_csrF(c_mat.num_rows, nnz)
        for i from 0<=i<=c_mat.num_rows:
            c_result.offsets[i]=c_mat.offsets[i]
        for i from 0<=i<nnz:
            c_result.rightColumns[i]=c_mat.rightColumns[i]+k
        for i from 0<=i<nnz:
            c_result.values[i]=c_mat.values[i]
        result=CSRMatrixF()
        result.set_matrix(c_result)
        return result
    def __ilshift__(CSRMatrixF self, int k):
        cdef c_CSRMatrixF *c_mat
        cdef int i, nnz
        c_mat=self.mat
        nnz=c_mat.offsets[c_mat.num_rows]
        for i from 0<=i<nnz:
            c_mat.rightColumns[i]+=k
        return self
    def print_csr(self):
        print_csrF(self.mat)
    def get_size(self):
        return self.mat.offsets[self.mat.num_rows]
    def __len__(self):
        return self.mat.num_rows
    def __getitem__(self,k):
        cdef int off1, off2
        cdef SparseVectorF res
        if not (k>=0 and k<self.mat.num_rows):
            raise IndexError
        off1=self.mat.offsets[k]
        off2=self.mat.offsets[k+1]
        if off1==off2:
            return emptyvec_F
        res=SparseVectorF()
        res.buf=self
        res.my_len=off2-off1
        res.vals_ptr=&self.mat.values[off1]
        res.idx_ptr=&self.mat.rightColumns[off1]
        return res
    def item_iter(self):
        if self.mat.num_rows>0:
            return CSRMatrixF_item_iter(self)
        else:
            return ()
    def to_scipy(self):
        cdef numpy.ndarray[numpy.float32_t, ndim=1] data
        cdef numpy.ndarray[numpy.int32_t, ndim=1] indices
        cdef numpy.ndarray[numpy.int32_t, ndim=1] indptr
        cdef int num_rows=self.mat.num_rows
        cdef int max_offset=self.mat.offsets[num_rows]
        cdef int i,j
        indptr=numpy.zeros(num_rows+1, numpy.int32)
        indices=numpy.zeros(max_offset, numpy.int32)
        data=numpy.zeros(max_offset, numpy.float32)
        for i from 0<=i<=num_rows:
            indptr[i]=self.mat.offsets[i]
        for i from 0<=i<max_offset:
            indices[i]=self.mat.rightColumns[i]
        for i from 0<=i<max_offset:
            data[i]=self.mat.values[i]
        return (data,indices,indptr)
    cpdef int get_maxcol(self):
        cdef int result=0
        cdef int col_max
        cdef coordinate_t i
        cdef int off1, off2
        if self.cache_maxcol>=0:
            return self.cache_maxcol
        for i from 0<=i<self.mat.num_rows:
           off1=self.mat.offsets[i]
           off2=self.mat.offsets[i+1]
           if off2>off1:
              col_max=self.mat.rightColumns[off2-1]
              if result<col_max:
                 result=col_max
        self.cache_maxcol=result
        return result
    cpdef CSRMatrixD transform_mi(self):
        cdef numpy.ndarray[numpy.float32_t, ndim=1] marginals_l
        cdef numpy.ndarray[numpy.float32_t, ndim=1] marginals_r
        cdef VecD2 tmp=VecD2()
        cdef coordinate_t k, i
        cdef int off1, off2
        cdef double v, log_v, log_N, log1, log2
        if self.mat.num_rows<100000:
            tmp.vec.reserve(self.mat.offsets[self.mat.num_rows])
        else:
            tmp.vec.reserve(100000)
        marginals_l=self.left_marginals()
        marginals_r=self.right_marginals()
        log_N=log(marginals_l.sum())
        for k from 0<=k<self.mat.num_rows:
            log1=log(marginals_l[k])-log_N
            off1=self.mat.offsets[k]
            off2=self.mat.offsets[k+1]
            for i from off1<=i<off2:
               v=self.mat.values[i]
               if v>0:
                   log2=log(marginals_r[self.mat.rightColumns[i]])
                   log_v=log(v)-log2-log1
                   if log_v>0:
                      tmp.add_count(k,self.mat.rightColumns[i],log_v)
        return tmp.to_csr()
    cpdef CSRMatrixD transform_mi_discount(self):
        cdef numpy.ndarray[numpy.float32_t, ndim=1] marginals_l
        cdef numpy.ndarray[numpy.float32_t, ndim=1] marginals_r
        cdef VecD2 tmp=VecD2()
        cdef coordinate_t k, i
        cdef int off1, off2
        cdef double marg_l, marg_r, log_corr1, log_corr2
        cdef double v, log_v, log_N, log1, log2
        if self.mat.num_rows<100000:
            tmp.vec.reserve(self.mat.offsets[self.mat.num_rows])
        else:
            tmp.vec.reserve(100000)
        marginals_l=self.left_marginals()
        marginals_r=self.right_marginals()
        log_N=log(marginals_l.sum())
        for k from 0<=k<self.mat.num_rows:
            marg_l=marginals_l[k]
            log1=log(marg_l)-log_N
            off1=self.mat.offsets[k]
            off2=self.mat.offsets[k+1]
            for i from off1<=i<off2:
               v=self.mat.values[i]
               if v>0:
                   marg_r=marginals_r[self.mat.rightColumns[i]]
                   log2=log(marg_r)
                   log_corr1=log1p(-1.0/(v+1.0))
                   if marg_l<marg_r:
                       log_corr2=log1p(-1.0/(marg_l+1.0))
                   else:
                       log_corr2=log1p(-1.0/(marg_r+1.0))
                   log_v=log(v)-log2-log1+log_corr1+log_corr2
                   if log_v>0:
                      tmp.add_count(k,self.mat.rightColumns[i],log_v)
        return tmp.to_csr()
    cpdef CSRMatrixD transform_ll(self):
        cdef numpy.ndarray[numpy.float32_t, ndim=1] marginals_l
        cdef numpy.ndarray[numpy.float32_t, ndim=1] marginals_r
        cdef VecD2 tmp=VecD2()
        cdef coordinate_t k, i
        cdef int off1, off2
        cdef double marg_l, marg_r, c, c1, c2
        cdef double p, p1, pn1, ll
        if self.mat.num_rows<100000:
            tmp.vec.reserve(self.mat.offsets[self.mat.num_rows])
        else:
            tmp.vec.reserve(100000)
        marginals_l=self.left_marginals()
        marginals_r=self.right_marginals()
        c=marginals_l.sum()
        for k from 0<=k<self.mat.num_rows:
            c1=marginals_l[k]
            off1=self.mat.offsets[k]
            off2=self.mat.offsets[k+1]
            for i from off1<=i<off2:
               c12=self.mat.values[i]
               if c12>0:
                   c2=marginals_r[self.mat.rightColumns[i]]
                   p=c2/c
                   p1=c12/c1
                   if p1>p:
                       pn1=(c2-c12)/(c-c1)
                       ll=(-logB(c1,p,c12)-logB(c-c1,p,c2-c12)
                           +logB(c1,p1,c12)+logB(c-c1,pn1,c2-c12))
                       tmp.add_count(k,self.mat.rightColumns[i],ll)
        return tmp.to_csr()
    cpdef CSRMatrixD transform_l1(self):
        cdef numpy.ndarray[numpy.float32_t, ndim=1] marginals_l
        cdef VecD2 tmp=VecD2()
        cdef coordinate_t k, i
        cdef int off1, off2
        cdef double v, marg_l
        if self.mat.num_rows<100000:
            tmp.vec.reserve(self.mat.offsets[self.mat.num_rows])
        else:
            tmp.vec.reserve(100000)
        marginals_l=self.left_marginals()
        for k from 0<=k<self.mat.num_rows:
            marg_l=marginals_l[k]
            off1=self.mat.offsets[k]
            off2=self.mat.offsets[k+1]
            for i from off1<=i<off2:
               v=self.mat.values[i]
               if v>0:
                   tmp.add_count(k,self.mat.rightColumns[i],v/marg_l)
        return tmp.to_csr()
    def thresholds_norm(self, perc):
        cdef numpy.ndarray[numpy.float32_t, ndim=1] means
        cdef numpy.ndarray[numpy.float32_t, ndim=1] squared_means
        cdef numpy.ndarray[numpy.float32_t, ndim=1] stddev
        cdef double zscore=inverse_erf(2.0*perc-1.0)
        cdef int count_targets
        cdef int count_all=self.get_maxcol()+1
        cdef double corr_factor = count_all / (count_all - 1.0)
        count_targets=len(self)
        means=self.right_marginals()
        means /= count_targets
        squared_means = self.right_squared_marginals()
        squared_means /= count_targets
        stddev=numpy.sqrt(corr_factor * (squared_means - means*means))
        return means + zscore * stddev
    def thresholds_quantile(self, perc):
        cdef numpy.ndarray[numpy.float32_t, ndim=1] results
        cdef int i, count_nz, offset, abs_offset
        cdef int count_targets
        cdef int count_all=self.get_maxcol()+1
        cdef CSRMatrixD nonzeros=self.transpose()
        count_targets=len(self)
        count_features=len(nonzeros)
        results=numpy.zeros(count_features,numpy.float32)
        abs_offset=count_features*(1.0-perc)+0.5
        if abs_offset==0:
            abs_offset=1
        for i from 0<=i<count_features:
            nz_vals=[v for k,v in nonzeros[i]]
            nz_vals.sort()
            count_nz=len(nz_vals)
            offset=count_nz-abs_offset
            if offset<0:
                offset=0
            results[i] = nz_vals[offset]
        return results
    def thresholds_nonzero(self, perc):
        cdef numpy.ndarray[numpy.float32_t, ndim=1] results
        cdef int i
        cdef int count_all=self.get_maxcol()+1
        cdef CSRMatrixD nonzeros=self.transpose()
        results=numpy.zeros(count_all,numpy.float32)
        for i from 0<=i<count_all:
            nz_vals=[v for k,v in nonzeros[i]]
            nz_vals.sort()
            results[i] = nz_vals[int(len(nz_vals)*perc)]
        return results
    def get_quantiles(self, qs):
        cdef coordinate_t k, nnz
        cdef numpy.ndarray[numpy.float32_t, ndim=1] allvals
        nnz = self.mat.offsets[self.mat.num_rows]
        allvals = numpy.empty(self.mat.offsets[self.mat.num_rows],
                              numpy.float32)
        for k from 0<=k<self.mat.offsets[self.mat.num_rows]:
            allvals[k] = self.mat.values[k]
        allvals.sort()
        result = []
        for q in qs:
            k = nnz*q
            if k >= nnz:
                k = nnz-1
            result.append(allvals[k])
        return result
    cpdef CSRMatrixF apply_threshold(self, numpy.ndarray thr):
        cdef VecD2 tmp=VecD2()
        cdef coordinate_t k, i, k2
        cdef int off1, off2
        cdef double v, marg_l
        cdef numpy.ndarray[numpy.float32_t, ndim=1] thresholds=thr
        if self.mat.num_rows<100000:
            tmp.vec.reserve(self.mat.offsets[self.mat.num_rows])
        else:
            tmp.vec.reserve(100000)
        for k from 0<=k<self.mat.num_rows:
            off1=self.mat.offsets[k]
            off2=self.mat.offsets[k+1]
            for i from off1<=i<off2:
               v=self.mat.values[i]
               k2=self.mat.rightColumns[i]
               if v>=thresholds[k2]:
                   tmp.add_count(k,k2,1.0)
        return tmp.to_csr()
    cpdef CSRMatrixD apply_scaling(self, numpy.ndarray thr):
        cdef VecD2 tmp=VecD2()
        cdef numpy.ndarray[numpy.float32_t, ndim=1] thresholds=thr
        cdef coordinate_t k, i, k2
        cdef int off1, off2
        cdef double v, marg_l
        if self.mat.num_rows<100000:
            tmp.vec.reserve(self.mat.offsets[self.mat.num_rows])
        else:
            tmp.vec.reserve(100000)
        for k from 0<=k<self.mat.num_rows:
            off1=self.mat.offsets[k]
            off2=self.mat.offsets[k+1]
            for i from off1<=i<off2:
               k2=self.mat.rightColumns[i]
               v=self.mat.values[i]/thresholds[k2]
               if v>=1.0:
                   tmp.add_count(k,k2,1.0)
               else:
                   tmp.add_count(k,k2,v)
        return tmp.to_csr()
    def scale_columns(self, factors):
        cdef numpy.ndarray[numpy.float32_t, ndim=1] factors_l = factors
        cdef coordinate_t k
        cdef int off1, off2
        for k from 0<=k<self.mat.num_rows:
            off1 = self.mat.offsets[k]
            off2 = self.mat.offsets[k+1]
            for i from off1<=i<off2:
                k2 = self.mat.rightColumns[i]
                self.mat.values[i] *= factors_l[k2]
    def scale_rows(self, factors):
        cdef coordinate_t k
        cdef float factor
        cdef int off1, off2
        for k from 0<=k<self.mat.num_rows:
            off1 = self.mat.offsets[k]
            off2 = self.mat.offsets[k+1]
            factor = factors[k]
            for i from off1<=i<off2:
                self.mat.values[i] *= factor
    def left_marginals(self):
        cdef numpy.ndarray[numpy.float32_t, ndim=1] marginals_l
        marginals_l=numpy.zeros(self.mat.num_rows,numpy.float32)
        self.mat.compute_left_marginals(<float *> marginals_l.data)
        return marginals_l
    def right_marginals(self):
        cdef numpy.ndarray[numpy.float32_t, ndim=1] marginals_r
        marginals_r=numpy.zeros(self.get_maxcol()+1,numpy.float32)
        self.mat.compute_right_marginals(<float *> marginals_r.data)
        return marginals_r
    def right_squared_marginals(self):
        cdef numpy.ndarray[numpy.float32_t, ndim=1] marginals_r
        marginals_r=numpy.zeros(self.get_maxcol()+1,numpy.float32)
        self.mat.compute_right_squared_marginals(<float *> marginals_r.data)
        return marginals_r
    cpdef float get_count(self,coordinate_t k, coordinate_t k2):
        if k<0 or k>=self.mat.num_rows:
            return 0
        assert k<self.mat.num_rows
        cdef int lo=self.mat.offsets[k]
        cdef int hi=self.mat.offsets[k+1]
        cdef int mi
        cdef coordinate_t a_mi
        if hi==lo: return 0
        while hi-lo>1:
            mi=(hi+lo)//2
            a_mi=self.mat.rightColumns[mi]
            if a_mi>k2:
                hi=mi
            elif a_mi==k2:
                return self.mat.values[mi]
            else:
                lo=mi+1
        if self.mat.rightColumns[lo]==k2:
            return self.mat.values[lo]
        return 0
    def write_binary(self,f):
        """writes the matrix in binary format"""
        cdef int fileno=f.fileno()
        self.mat.write_binary(fileno)
    def transpose(self):
        cdef c_CSRMatrixF *matC=self.mat.transpose()
        cdef CSRMatrixF mat=CSRMatrixF()
        mat.set_matrix(matC)
        return mat
    def __add__(self,other):
        cdef CSRMatrixF mat1
        cdef CSRMatrixF mat2
        cdef c_CSRMatrixF *matC
        mat1=self
        mat2=other
        matC=add_csrF(mat1.mat,mat2.mat)
        cdef CSRMatrixF mat=CSRMatrixF()
        mat.set_matrix(matC)
        return mat
    def __dealloc__(self):
        if self.mat!=NULL:
            if self.buf is not None:
                self.buf=None
            else:
                cxx_deleteA(self.mat.offsets)
                if (self.mat.rightColumns!=NULL):
                    cxx_deleteA(self.mat.rightColumns)
                    cxx_deleteA(self.mat.values)
            cxx_delete(self.mat)
    def __reduce__(self):
        # choose compatibility over performance.
        cdef VecF2 tmp=VecF2()
        tmp.vec.reserve(self.mat.offsets[self.mat.num_rows])
        tmp.extend(self.item_iter())
        return (CSRMatrixF,(),tmp)
    def __setstate__(self,VecF2 tmp):
        tmp.ensure_compact()
        self.set_matrix(vec2csrF(&tmp.vec))

cdef class CSRMatrixF_item_iter:
    cdef CSRMatrixF mat
    cdef c_CSRMatrixF *matC
    cdef coordinate_t k
    cdef int off
    def __init__(self, mat):
        self.mat=mat
        self.matC=self.mat.mat
        self.k=0
        self.off=0
    def __iter__(self):
        return self
    def __next__(self):
        while self.off==self.matC.offsets[self.k+1]:
            self.k+=1
            if self.k>=self.matC.num_rows:
                raise StopIteration
        res=(self.k,
             self.matC.rightColumns[self.off],
             self.matC.values[self.off])
        self.off+=1
        return res
## D -> double, float64

cdef class VecD2
cdef class CSRMatrixD_item_iter
cdef class SparseVectorD
cdef SparseVectorD emptyvec_D

cdef class CSRMatrixD:
    def __cinit__(self, vecs=None):
        self.mat=NULL
        self.cache_maxcol=-1
        if vecs is not None:
            self.fromVectors(vecs)
    cdef void set_matrix(self,c_CSRMatrixD *mat_new):
        self.mat=mat_new
        self.cache_maxcol=-1
    cpdef CSRMatrixD fromVectors(self, vectors):
        cdef int spaceNeeded=0
        cdef int i=0
        cdef int j=0
        cdef SparseVectorD vec
        cdef c_CSRMatrixD *c_result
        cdef CSRMatrixD result
        for vec in vectors:
            spaceNeeded+=vec.my_len
            i+=1
        c_result=new_csrD(i,spaceNeeded)
        i=0
        spaceNeeded=0
        for vec in vectors:
            c_result.offsets[i]=spaceNeeded
            for j from 0<=j<vec.my_len:
                 c_result.rightColumns[spaceNeeded]=vec.idx_ptr[j]
                 c_result.values[spaceNeeded]=vec.vals_ptr[j]
                 spaceNeeded+=1
            i+=1
        c_result.offsets[i]=spaceNeeded
        self.set_matrix(c_result)
        return self
    def __lshift__(CSRMatrixD self,int k):
        cdef CSRMatrixD result
        cdef c_CSRMatrixD *c_mat
        cdef c_CSRMatrixD *c_result
        cdef int i, nnz
        c_mat=self.mat
        nnz=c_mat.offsets[c_mat.num_rows]
        c_result=new_csrD(c_mat.num_rows, nnz)
        for i from 0<=i<=c_mat.num_rows:
            c_result.offsets[i]=c_mat.offsets[i]
        for i from 0<=i<nnz:
            c_result.rightColumns[i]=c_mat.rightColumns[i]+k
        for i from 0<=i<nnz:
            c_result.values[i]=c_mat.values[i]
        result=CSRMatrixD()
        result.set_matrix(c_result)
        return result
    def __ilshift__(CSRMatrixD self, int k):
        cdef c_CSRMatrixD *c_mat
        cdef int i, nnz
        c_mat=self.mat
        nnz=c_mat.offsets[c_mat.num_rows]
        for i from 0<=i<nnz:
            c_mat.rightColumns[i]+=k
        return self
    def print_csr(self):
        print_csrD(self.mat)
    def get_size(self):
        return self.mat.offsets[self.mat.num_rows]
    def __len__(self):
        return self.mat.num_rows
    def __getitem__(self,k):
        cdef int off1, off2
        cdef SparseVectorD res
        if not (k>=0 and k<self.mat.num_rows):
            raise IndexError
        off1=self.mat.offsets[k]
        off2=self.mat.offsets[k+1]
        if off1==off2:
            return emptyvec_D
        res=SparseVectorD()
        res.buf=self
        res.my_len=off2-off1
        res.vals_ptr=&self.mat.values[off1]
        res.idx_ptr=&self.mat.rightColumns[off1]
        return res
    def item_iter(self):
        if self.mat.num_rows>0:
            return CSRMatrixD_item_iter(self)
        else:
            return ()
    def to_scipy(self):
        cdef numpy.ndarray[numpy.float64_t, ndim=1] data
        cdef numpy.ndarray[numpy.int32_t, ndim=1] indices
        cdef numpy.ndarray[numpy.int32_t, ndim=1] indptr
        cdef int num_rows=self.mat.num_rows
        cdef int max_offset=self.mat.offsets[num_rows]
        cdef int i,j
        indptr=numpy.zeros(num_rows+1, numpy.int32)
        indices=numpy.zeros(max_offset, numpy.int32)
        data=numpy.zeros(max_offset, numpy.float64)
        for i from 0<=i<=num_rows:
            indptr[i]=self.mat.offsets[i]
        for i from 0<=i<max_offset:
            indices[i]=self.mat.rightColumns[i]
        for i from 0<=i<max_offset:
            data[i]=self.mat.values[i]
        return (data,indices,indptr)
    cpdef int get_maxcol(self):
        cdef int result=0
        cdef int col_max
        cdef coordinate_t i
        cdef int off1, off2
        if self.cache_maxcol>=0:
            return self.cache_maxcol
        for i from 0<=i<self.mat.num_rows:
           off1=self.mat.offsets[i]
           off2=self.mat.offsets[i+1]
           if off2>off1:
              col_max=self.mat.rightColumns[off2-1]
              if result<col_max:
                 result=col_max
        self.cache_maxcol=result
        return result
    cpdef CSRMatrixD transform_mi(self):
        cdef numpy.ndarray[numpy.float64_t, ndim=1] marginals_l
        cdef numpy.ndarray[numpy.float64_t, ndim=1] marginals_r
        cdef VecD2 tmp=VecD2()
        cdef coordinate_t k, i
        cdef int off1, off2
        cdef double v, log_v, log_N, log1, log2
        if self.mat.num_rows<100000:
            tmp.vec.reserve(self.mat.offsets[self.mat.num_rows])
        else:
            tmp.vec.reserve(100000)
        marginals_l=self.left_marginals()
        marginals_r=self.right_marginals()
        log_N=log(marginals_l.sum())
        for k from 0<=k<self.mat.num_rows:
            log1=log(marginals_l[k])-log_N
            off1=self.mat.offsets[k]
            off2=self.mat.offsets[k+1]
            for i from off1<=i<off2:
               v=self.mat.values[i]
               if v>0:
                   log2=log(marginals_r[self.mat.rightColumns[i]])
                   log_v=log(v)-log2-log1
                   if log_v>0:
                      tmp.add_count(k,self.mat.rightColumns[i],log_v)
        return tmp.to_csr()
    cpdef CSRMatrixD transform_mi_discount(self):
        cdef numpy.ndarray[numpy.float64_t, ndim=1] marginals_l
        cdef numpy.ndarray[numpy.float64_t, ndim=1] marginals_r
        cdef VecD2 tmp=VecD2()
        cdef coordinate_t k, i
        cdef int off1, off2
        cdef double marg_l, marg_r, log_corr1, log_corr2
        cdef double v, log_v, log_N, log1, log2
        if self.mat.num_rows<100000:
            tmp.vec.reserve(self.mat.offsets[self.mat.num_rows])
        else:
            tmp.vec.reserve(100000)
        marginals_l=self.left_marginals()
        marginals_r=self.right_marginals()
        log_N=log(marginals_l.sum())
        for k from 0<=k<self.mat.num_rows:
            marg_l=marginals_l[k]
            log1=log(marg_l)-log_N
            off1=self.mat.offsets[k]
            off2=self.mat.offsets[k+1]
            for i from off1<=i<off2:
               v=self.mat.values[i]
               if v>0:
                   marg_r=marginals_r[self.mat.rightColumns[i]]
                   log2=log(marg_r)
                   log_corr1=log1p(-1.0/(v+1.0))
                   if marg_l<marg_r:
                       log_corr2=log1p(-1.0/(marg_l+1.0))
                   else:
                       log_corr2=log1p(-1.0/(marg_r+1.0))
                   log_v=log(v)-log2-log1+log_corr1+log_corr2
                   if log_v>0:
                      tmp.add_count(k,self.mat.rightColumns[i],log_v)
        return tmp.to_csr()
    cpdef CSRMatrixD transform_ll(self):
        cdef numpy.ndarray[numpy.float64_t, ndim=1] marginals_l
        cdef numpy.ndarray[numpy.float64_t, ndim=1] marginals_r
        cdef VecD2 tmp=VecD2()
        cdef coordinate_t k, i
        cdef int off1, off2
        cdef double marg_l, marg_r, c, c1, c2
        cdef double p, p1, pn1, ll
        if self.mat.num_rows<100000:
            tmp.vec.reserve(self.mat.offsets[self.mat.num_rows])
        else:
            tmp.vec.reserve(100000)
        marginals_l=self.left_marginals()
        marginals_r=self.right_marginals()
        c=marginals_l.sum()
        for k from 0<=k<self.mat.num_rows:
            c1=marginals_l[k]
            off1=self.mat.offsets[k]
            off2=self.mat.offsets[k+1]
            for i from off1<=i<off2:
               c12=self.mat.values[i]
               if c12>0:
                   c2=marginals_r[self.mat.rightColumns[i]]
                   p=c2/c
                   p1=c12/c1
                   if p1>p:
                       pn1=(c2-c12)/(c-c1)
                       ll=(-logB(c1,p,c12)-logB(c-c1,p,c2-c12)
                           +logB(c1,p1,c12)+logB(c-c1,pn1,c2-c12))
                       tmp.add_count(k,self.mat.rightColumns[i],ll)
        return tmp.to_csr()
    cpdef CSRMatrixD transform_l1(self):
        cdef numpy.ndarray[numpy.float64_t, ndim=1] marginals_l
        cdef VecD2 tmp=VecD2()
        cdef coordinate_t k, i
        cdef int off1, off2
        cdef double v, marg_l
        if self.mat.num_rows<100000:
            tmp.vec.reserve(self.mat.offsets[self.mat.num_rows])
        else:
            tmp.vec.reserve(100000)
        marginals_l=self.left_marginals()
        for k from 0<=k<self.mat.num_rows:
            marg_l=marginals_l[k]
            off1=self.mat.offsets[k]
            off2=self.mat.offsets[k+1]
            for i from off1<=i<off2:
               v=self.mat.values[i]
               if v>0:
                   tmp.add_count(k,self.mat.rightColumns[i],v/marg_l)
        return tmp.to_csr()
    def thresholds_norm(self, perc):
        cdef numpy.ndarray[numpy.float64_t, ndim=1] means
        cdef numpy.ndarray[numpy.float64_t, ndim=1] squared_means
        cdef numpy.ndarray[numpy.float64_t, ndim=1] stddev
        cdef double zscore=inverse_erf(2.0*perc-1.0)
        cdef int count_targets
        cdef int count_all=self.get_maxcol()+1
        cdef double corr_factor = count_all / (count_all - 1.0)
        count_targets=len(self)
        means=self.right_marginals()
        means /= count_targets
        squared_means = self.right_squared_marginals()
        squared_means /= count_targets
        stddev=numpy.sqrt(corr_factor * (squared_means - means*means))
        return means + zscore * stddev
    def thresholds_quantile(self, perc):
        cdef numpy.ndarray[numpy.float64_t, ndim=1] results
        cdef int i, count_nz, offset, abs_offset
        cdef int count_targets
        cdef int count_all=self.get_maxcol()+1
        cdef CSRMatrixD nonzeros=self.transpose()
        count_targets=len(self)
        count_features=len(nonzeros)
        results=numpy.zeros(count_features,numpy.float64)
        abs_offset=count_features*(1.0-perc)+0.5
        if abs_offset==0:
            abs_offset=1
        for i from 0<=i<count_features:
            nz_vals=[v for k,v in nonzeros[i]]
            nz_vals.sort()
            count_nz=len(nz_vals)
            offset=count_nz-abs_offset
            if offset<0:
                offset=0
            results[i] = nz_vals[offset]
        return results
    def thresholds_nonzero(self, perc):
        cdef numpy.ndarray[numpy.float64_t, ndim=1] results
        cdef int i
        cdef int count_all=self.get_maxcol()+1
        cdef CSRMatrixD nonzeros=self.transpose()
        results=numpy.zeros(count_all,numpy.float64)
        for i from 0<=i<count_all:
            nz_vals=[v for k,v in nonzeros[i]]
            nz_vals.sort()
            results[i] = nz_vals[int(len(nz_vals)*perc)]
        return results
    def get_quantiles(self, qs):
        cdef coordinate_t k, nnz
        cdef numpy.ndarray[numpy.float64_t, ndim=1] allvals
        nnz = self.mat.offsets[self.mat.num_rows]
        allvals = numpy.empty(self.mat.offsets[self.mat.num_rows],
                              numpy.float64)
        for k from 0<=k<self.mat.offsets[self.mat.num_rows]:
            allvals[k] = self.mat.values[k]
        allvals.sort()
        result = []
        for q in qs:
            k = nnz*q
            if k >= nnz:
                k = nnz-1
            result.append(allvals[k])
        return result
    cpdef CSRMatrixD apply_threshold(self, numpy.ndarray thr):
        cdef VecD2 tmp=VecD2()
        cdef coordinate_t k, i, k2
        cdef int off1, off2
        cdef double v, marg_l
        cdef numpy.ndarray[numpy.float64_t, ndim=1] thresholds=thr
        if self.mat.num_rows<100000:
            tmp.vec.reserve(self.mat.offsets[self.mat.num_rows])
        else:
            tmp.vec.reserve(100000)
        for k from 0<=k<self.mat.num_rows:
            off1=self.mat.offsets[k]
            off2=self.mat.offsets[k+1]
            for i from off1<=i<off2:
               v=self.mat.values[i]
               k2=self.mat.rightColumns[i]
               if v>=thresholds[k2]:
                   tmp.add_count(k,k2,1.0)
        return tmp.to_csr()
    cpdef CSRMatrixD apply_scaling(self, numpy.ndarray thr):
        cdef VecD2 tmp=VecD2()
        cdef numpy.ndarray[numpy.float64_t, ndim=1] thresholds=thr
        cdef coordinate_t k, i, k2
        cdef int off1, off2
        cdef double v, marg_l
        if self.mat.num_rows<100000:
            tmp.vec.reserve(self.mat.offsets[self.mat.num_rows])
        else:
            tmp.vec.reserve(100000)
        for k from 0<=k<self.mat.num_rows:
            off1=self.mat.offsets[k]
            off2=self.mat.offsets[k+1]
            for i from off1<=i<off2:
               k2=self.mat.rightColumns[i]
               v=self.mat.values[i]/thresholds[k2]
               if v>=1.0:
                   tmp.add_count(k,k2,1.0)
               else:
                   tmp.add_count(k,k2,v)
        return tmp.to_csr()
    def scale_columns(self, factors):
        cdef numpy.ndarray[numpy.float64_t, ndim=1] factors_l = factors
        cdef coordinate_t k
        cdef int off1, off2
        for k from 0<=k<self.mat.num_rows:
            off1 = self.mat.offsets[k]
            off2 = self.mat.offsets[k+1]
            for i from off1<=i<off2:
                k2 = self.mat.rightColumns[i]
                self.mat.values[i] *= factors_l[k2]
    def scale_rows(self, factors):
        cdef coordinate_t k
        cdef double factor
        cdef int off1, off2
        for k from 0<=k<self.mat.num_rows:
            off1 = self.mat.offsets[k]
            off2 = self.mat.offsets[k+1]
            factor = factors[k]
            for i from off1<=i<off2:
                self.mat.values[i] *= factor
    def left_marginals(self):
        cdef numpy.ndarray[numpy.float64_t, ndim=1] marginals_l
        marginals_l=numpy.zeros(self.mat.num_rows,numpy.float64)
        self.mat.compute_left_marginals(<double *> marginals_l.data)
        return marginals_l
    def right_marginals(self):
        cdef numpy.ndarray[numpy.float64_t, ndim=1] marginals_r
        marginals_r=numpy.zeros(self.get_maxcol()+1,numpy.float64)
        self.mat.compute_right_marginals(<double *> marginals_r.data)
        return marginals_r
    def right_squared_marginals(self):
        cdef numpy.ndarray[numpy.float64_t, ndim=1] marginals_r
        marginals_r=numpy.zeros(self.get_maxcol()+1,numpy.float64)
        self.mat.compute_right_squared_marginals(<double *> marginals_r.data)
        return marginals_r
    cpdef double get_count(self,coordinate_t k, coordinate_t k2):
        if k<0 or k>=self.mat.num_rows:
            return 0
        assert k<self.mat.num_rows
        cdef int lo=self.mat.offsets[k]
        cdef int hi=self.mat.offsets[k+1]
        cdef int mi
        cdef coordinate_t a_mi
        if hi==lo: return 0
        while hi-lo>1:
            mi=(hi+lo)//2
            a_mi=self.mat.rightColumns[mi]
            if a_mi>k2:
                hi=mi
            elif a_mi==k2:
                return self.mat.values[mi]
            else:
                lo=mi+1
        if self.mat.rightColumns[lo]==k2:
            return self.mat.values[lo]
        return 0
    def write_binary(self,f):
        """writes the matrix in binary format"""
        cdef int fileno=f.fileno()
        self.mat.write_binary(fileno)
    def transpose(self):
        cdef c_CSRMatrixD *matC=self.mat.transpose()
        cdef CSRMatrixD mat=CSRMatrixD()
        mat.set_matrix(matC)
        return mat
    def __add__(self,other):
        cdef CSRMatrixD mat1
        cdef CSRMatrixD mat2
        cdef c_CSRMatrixD *matC
        mat1=self
        mat2=other
        matC=add_csrD(mat1.mat,mat2.mat)
        cdef CSRMatrixD mat=CSRMatrixD()
        mat.set_matrix(matC)
        return mat
    def __dealloc__(self):
        if self.mat!=NULL:
            if self.buf is not None:
                self.buf=None
            else:
                cxx_deleteA(self.mat.offsets)
                if (self.mat.rightColumns!=NULL):
                    cxx_deleteA(self.mat.rightColumns)
                    cxx_deleteA(self.mat.values)
            cxx_delete(self.mat)
    def __reduce__(self):
        # choose compatibility over performance.
        cdef VecD2 tmp=VecD2()
        tmp.vec.reserve(self.mat.offsets[self.mat.num_rows])
        tmp.extend(self.item_iter())
        return (CSRMatrixD,(),tmp)
    def __setstate__(self,VecD2 tmp):
        tmp.ensure_compact()
        self.set_matrix(vec2csrD(&tmp.vec))

cdef class CSRMatrixD_item_iter:
    cdef CSRMatrixD mat
    cdef c_CSRMatrixD *matC
    cdef coordinate_t k
    cdef int off
    def __init__(self, mat):
        self.mat=mat
        self.matC=self.mat.mat
        self.k=0
        self.off=0
    def __iter__(self):
        return self
    def __next__(self):
        while self.off==self.matC.offsets[self.k+1]:
            self.k+=1
            if self.k>=self.matC.num_rows:
                raise StopIteration
        res=(self.k,
             self.matC.rightColumns[self.off],
             self.matC.values[self.off])
        self.off+=1
        return res



def csrFromBuffer(object buf):
    """takes an object conforming to the Buffer protocol and
    creates a CSR matrix from that."""
    cdef const_char_ptr buffer
    cdef Py_ssize_t buffer_len
    PyObject_AsCharBuffer(buf,&buffer,&buffer_len)
    cdef int *tmp=<int *>buffer
    cdef c_CSRMatrixI *c_matI
    cdef CSRMatrixI matI
    cdef c_CSRMatrixF *c_matF
    cdef CSRMatrixF matF
    cdef c_CSRMatrixD *c_matD
    cdef CSRMatrixD matD
    if tmp[1]&0xff==c'I':
        c_matI=new_CSRMatrixI()
        result=csrFromBufferI(<void *>buffer,c_matI)
        if result==-1:
            raise ValueError
        matI=CSRMatrixI()
        matI.set_matrix(c_matI)
        matI.buf=buf
        return matI
    elif tmp[1]&0xff==c'F' and (tmp[1]>>24)==c'4':
        c_matF=new_CSRMatrixF()
        result=csrFromBufferF(<void *>buffer,c_matF)
        if result==-1:
            raise ValueError
        matF=CSRMatrixF()
        matF.set_matrix(c_matF)
        matF.buf=buf
        return matF
    elif tmp[1]&0xff==c'F' and (tmp[1]>>24)==c'8':
        c_matD=new_CSRMatrixD()
        result=csrFromBufferD(<void *>buffer,c_matD)
        if result==-1:
            raise ValueError
        matD=CSRMatrixD()
        matD.set_matrix(c_matD)
        matD.buf=buf
        return matD
    else:
        # hrm... this only works on 32bit machines
        # what's the Python equivalent of sizeof(int)???
        tcode=PyBytes_FromStringAndSize(buffer+4,4)
        raise ValueError("wrong tcode "+tcode)

def mmapCSR(f):
    ## py2.4 compat: determine length
    ## instead of passing length=0
    # go to end of file and note position
    f.seek(0,2)
    flen=f.tell()
    a=mmap.mmap(f.fileno(),flen,access=mmap.ACCESS_READ)
    return csrFromBuffer(a)

## I -> int


cdef class IVecI1_iter

cdef class VecI1:
    """mutable sparse int matrix based on item vector"""
    def __init__(self):
        self.is_compact=True
    def item_iter(self):
        self.ensure_compact()
        return IVecI1_iter(self)
    def get_size(self,compactify=True):
        if compactify:
            self.ensure_compact()
        return self.vec.size()
    def __len__(self):
        return self.get_size(True)
    def clear(self):
        self.vec.resize(0)
        self.is_compact=True
    cdef void compactify(self):
        cdef c_SmallerAddrI1 comp
        c_IVecI1_sort(self.vec.begin(),self.vec.end(),comp)
        c_compactifyI1(&self.vec)
        self.is_compact=True
    cdef void ensure_compact(self):
        if not self.is_compact:
            self.compactify()
    cpdef int get_count(self,coordinate_t k0):
        cdef c_CItemI1 ci
        ci.addr[0]=k0
        return c_get_countI1(&self.vec,ci)
    cdef void c_add(self, coordinate_t k0, int item):
        cdef c_CItemI1 c
        c.addr[0]=k0
        c.item=item
        self.vec.push_back(c)
        self.is_compact=False
    def add_count(self, coordinate_t k0, item=1):
        self.c_add(k0, item)
    def __add__(VecI1 self, VecI1 other):
        cdef c_CItemI1 c1, c2
        cdef unsigned int i1, i2
        cdef int k, delta
        cdef VecI1 result
        self.ensure_compact()
        other.ensure_compact()
        result=VecI1()
        result.vec.reserve(max(self.vec.size(),other.vec.size()))
        i1=0
        i2=0
        while i1<self.vec.size() and i2<other.vec.size():
            c1=self.vec.at(i1)
            c2=other.vec.at(i2)
            delta=0
            for k from 0<=k<1:
                if c1.addr[k]<c2.addr[k]:
                    delta=-1
                    break
                elif c1.addr[k]>c2.addr[k]:
                    delta=1
                    break
            if delta==0:
                c1.item+=c2.item
                result.vec.push_back(c1)
                i1+=1
                i2+=1
            elif delta==-1:
                result.vec.push_back(c1)
                i1+=1
            else:
                result.vec.push_back(c2)
                i2+=1
        while i1<self.vec.size():
            c1=self.vec.at(i1)
            result.vec.push_back(c1)
            i1+=1
        while i2<other.vec.size():
            c2=other.vec.at(i2)
            result.vec.push_back(c2)
            i2+=1
        return result

    cpdef SparseVectorI to_sparse(self):
        cdef c_CItemI1 c
        cdef SparseVectorI result
        cdef coordinate_t n,i
        cdef coordinate_t *idx_ptr
        cdef int *vals_ptr
        self.ensure_compact()
        n=self.vec.size()
        idx_ptr=<coordinate_t *>PyMem_Malloc(n*sizeof(coordinate_t))
        vals_ptr=<int *>PyMem_Malloc(n*sizeof(int))
        for i from 0<=i<n:
          c=self.vec.at(i)
          idx_ptr[i]=c.addr[0]
          vals_ptr[i]=c.item
        result=SparseVectorI(None)
        result.idx_ptr=idx_ptr
        result.vals_ptr=vals_ptr
        result.my_len=n
        return result

    def tofile(self, f):
        cdef long i, n, k, k_max
        n=self.get_size()
        f.write('I1 %d\n'%(n,))
        k=0
        for i from 0<=i<n/CHUNK_SIZE:
            s=PyBytes_FromStringAndSize(<char *>c_VecI1_get_pointer(&self.vec,k), CHUNK_SIZE*sizeof(c_CItemI1))
            f.write(s)
            k+=CHUNK_SIZE
        if k<n:
            s=PyBytes_FromStringAndSize(<char *>c_VecI1_get_pointer(&self.vec,k), (n-k)*sizeof(c_CItemI1))
            f.write(s)
    def fromfile(self, f):
        cdef long i, n
        line=f.readline().strip().split()
        assert line[0]=='I1'
        n=long(line[1])
        self.clear()
        self.vec.resize(n)
        k=0
        for i from 0<=i<n/CHUNK_SIZE:
            s=f.read(CHUNK_SIZE*sizeof(c_CItemI1))
            assert len(s)==CHUNK_SIZE*sizeof(c_CItemI1)
            memcpy(<void *>c_VecI1_get_pointer(&self.vec,k), <char *>s, CHUNK_SIZE*sizeof(c_CItemI1))
            k+=CHUNK_SIZE
        if k<n:
            s=f.read((n-k)*sizeof(c_CItemI1))
            assert len(s)==(n-k)*sizeof(c_CItemI1)
            memcpy(<void *>c_VecI1_get_pointer(&self.vec,k), <char *>s, (n-k)*sizeof(c_CItemI1))
    cpdef int get_maxcol(self, int k=0):
        cdef int i, n, new_val, col
        assert k>=0 and k<1
        n=self.vec.size()
        if n==0:
            return -1
        if k==0:
            return self.vec.at(n-1).addr[0]
        col=-1
        n=self.vec.size()
        for i from 0<=i<n:
            new_val=self.vec.at(i).addr[k]
            if new_val>col:
                col=new_val
        return col
    cpdef numpy.ndarray get_type_counts(self, int k=0):
        cdef int old_val, new_val, i, n, count
        cdef numpy.ndarray[numpy.int32_t, ndim=1] result
        assert k>=0 and k<1
        if self.vec.size()==0:
            return numpy.zeros(0, numpy.int32)
        self.ensure_compact()
        n_types=self.get_maxcol(k)+1
        result=numpy.zeros(n_types,numpy.int32)
        n=self.vec.size()
        for i from 0<=i<n:
            new_val=self.vec.at(i).addr[k]
            result[new_val]+=1
        return result
    cpdef VecI1 remap(self, int k, numpy.ndarray filt):
        cdef numpy.ndarray[numpy.int8_t, ndim=1] wanted=filt.astype('b')
        cdef numpy.ndarray[numpy.int_t, ndim=1] target=filt.cumsum()-1
        self.ensure_compact()
        cdef int n=self.vec.size()
        cdef int i
        cdef VecI1 result=VecI1()
        cdef c_CItemI1 c
        for i from 0<=i<n:
            c=self.vec.at(i)
            if wanted[c.addr[k]]:
                c.addr[k]=target[c.addr[k]]
                result.vec.push_back(c)
        return result
    def append(self,tup):
        k0,val=tup
        self.c_add(k0,val)
    def extend(self,tups):
        for k0,val in tups:
            self.c_add(k0,val)
    def __iter__(self):
        return self.item_iter()
    def __reduce__(self):
        return (VecI1,(),(),self.item_iter())
    def __setstate__(self,state):
        pass
    def __repr__(self):
        return 'VecI1(%s)'%(str(list(self)),)
    def __dealloc__(self):
        # need to call destructor explicitly
        self.vec.cxx_destructor()

cdef class IVecI1_iter:
    cdef VecI1 vec
    cdef c_VecI1 *vecC
    cdef unsigned int k
    def __init__(self,vec):
        self.vec=vec
        self.vecC=&self.vec.vec
        self.k=0
    def __iter__(self):
        return self
    def __next__(self):
        if self.k>=self.vecC.size():
            raise StopIteration
        cdef c_CItemI1 res=self.vecC.at(self.k)
        self.k+=1
        return (res.addr[0],
                res.item)

cdef class LargeVecI1:
    cdef public object compact
    cdef public VecI1 loose
    def __init__(self):
        self.compact=[]
        self.loose=VecI1()
        self.loose.vec.reserve(100000)
    def ensure_compact(self):
        if self.loose.get_size()>0:
            self.compact.append(self.loose)
        self.loose=VecI1()
        while len(self.compact)>=2:
            self.compact[-2:]=[self.compact[-2]+self.compact[-1]]
    def add_count(self, coordinate_t k0, item=1):
        self.loose.c_add(k0, item)
        if self.loose.get_size(False)>=100000:
            self.loose.ensure_compact()
            self.compact.append(self.loose)
            self.loose=VecI1()
            self.loose.vec.reserve(100000)
            while (len(self.compact)>=2 and
                   self.compact[-1].get_size()>=0.6*self.compact[-2].get_size()):
                self.compact[-2:]=[self.compact[-2]+self.compact[-1]]
    def get_compact(self):
        self.ensure_compact()
        if len(self.compact)>0:
            return self.compact[0]
        else:
            return VecI1()
    def get_size(self, compactify=True):
        cdef int n
        if compactify:
            self.ensure_compact()
            if self.compact:
                return self.compact[0].get_size()
            else:
                return 0
        else:
            n=sum([c.get_size(False) for c in self.compact])
            n+=self.loose.get_size(False)
            return n
    def __iter__(self):
        self.ensure_compact()
        try:
            return iter(self.compact[0])
        except IndexError:
            return iter([])

    def get_type_counts(self, k=0):
        return self.get_compact().get_type_counts(k)
    def remap(self, k, filt):
        return self.get_compact().remap(k,filt)


cdef class IVecI2_iter

cdef class VecI2:
    """mutable sparse int matrix based on item vector"""
    def __init__(self):
        self.is_compact=True
    def item_iter(self):
        self.ensure_compact()
        return IVecI2_iter(self)
    def get_size(self,compactify=True):
        if compactify:
            self.ensure_compact()
        return self.vec.size()
    def __len__(self):
        return self.get_size(True)
    def clear(self):
        self.vec.resize(0)
        self.is_compact=True
    cdef void compactify(self):
        cdef c_SmallerAddrI2 comp
        c_IVecI2_sort(self.vec.begin(),self.vec.end(),comp)
        c_compactifyI2(&self.vec)
        self.is_compact=True
    cdef void ensure_compact(self):
        if not self.is_compact:
            self.compactify()
    cpdef int get_count(self,coordinate_t k0,coordinate_t k1):
        cdef c_CItemI2 ci
        ci.addr[0]=k0
        ci.addr[1]=k1
        return c_get_countI2(&self.vec,ci)
    cdef void c_add(self, coordinate_t k0,coordinate_t k1, int item):
        cdef c_CItemI2 c
        c.addr[0]=k0
        c.addr[1]=k1
        c.item=item
        self.vec.push_back(c)
        self.is_compact=False
    def add_count(self, coordinate_t k0,coordinate_t k1, item=1):
        self.c_add(k0,k1, item)
    def __add__(VecI2 self, VecI2 other):
        cdef c_CItemI2 c1, c2
        cdef unsigned int i1, i2
        cdef int k, delta
        cdef VecI2 result
        self.ensure_compact()
        other.ensure_compact()
        result=VecI2()
        result.vec.reserve(max(self.vec.size(),other.vec.size()))
        i1=0
        i2=0
        while i1<self.vec.size() and i2<other.vec.size():
            c1=self.vec.at(i1)
            c2=other.vec.at(i2)
            delta=0
            for k from 0<=k<2:
                if c1.addr[k]<c2.addr[k]:
                    delta=-1
                    break
                elif c1.addr[k]>c2.addr[k]:
                    delta=1
                    break
            if delta==0:
                c1.item+=c2.item
                result.vec.push_back(c1)
                i1+=1
                i2+=1
            elif delta==-1:
                result.vec.push_back(c1)
                i1+=1
            else:
                result.vec.push_back(c2)
                i2+=1
        while i1<self.vec.size():
            c1=self.vec.at(i1)
            result.vec.push_back(c1)
            i1+=1
        while i2<other.vec.size():
            c2=other.vec.at(i2)
            result.vec.push_back(c2)
            i2+=1
        return result

    cpdef CSRMatrixI to_csr(self):
        self.ensure_compact()
        cdef c_CSRMatrixI *matC=vec2csrI(&self.vec)
        cdef CSRMatrixI mat=CSRMatrixI()
        mat.set_matrix(matC)
        return mat

    def tofile(self, f):
        cdef long i, n, k, k_max
        n=self.get_size()
        f.write('I2 %d\n'%(n,))
        k=0
        for i from 0<=i<n/CHUNK_SIZE:
            s=PyBytes_FromStringAndSize(<char *>c_VecI2_get_pointer(&self.vec,k), CHUNK_SIZE*sizeof(c_CItemI2))
            f.write(s)
            k+=CHUNK_SIZE
        if k<n:
            s=PyBytes_FromStringAndSize(<char *>c_VecI2_get_pointer(&self.vec,k), (n-k)*sizeof(c_CItemI2))
            f.write(s)
    def fromfile(self, f):
        cdef long i, n
        line=f.readline().strip().split()
        assert line[0]=='I2'
        n=long(line[1])
        self.clear()
        self.vec.resize(n)
        k=0
        for i from 0<=i<n/CHUNK_SIZE:
            s=f.read(CHUNK_SIZE*sizeof(c_CItemI2))
            assert len(s)==CHUNK_SIZE*sizeof(c_CItemI2)
            memcpy(<void *>c_VecI2_get_pointer(&self.vec,k), <char *>s, CHUNK_SIZE*sizeof(c_CItemI2))
            k+=CHUNK_SIZE
        if k<n:
            s=f.read((n-k)*sizeof(c_CItemI2))
            assert len(s)==(n-k)*sizeof(c_CItemI2)
            memcpy(<void *>c_VecI2_get_pointer(&self.vec,k), <char *>s, (n-k)*sizeof(c_CItemI2))
    cpdef int get_maxcol(self, int k=0):
        cdef int i, n, new_val, col
        assert k>=0 and k<2
        n=self.vec.size()
        if n==0:
            return -1
        if k==0:
            return self.vec.at(n-1).addr[0]
        col=-1
        n=self.vec.size()
        for i from 0<=i<n:
            new_val=self.vec.at(i).addr[k]
            if new_val>col:
                col=new_val
        return col
    cpdef numpy.ndarray get_type_counts(self, int k=0):
        cdef int old_val, new_val, i, n, count
        cdef numpy.ndarray[numpy.int32_t, ndim=1] result
        assert k>=0 and k<2
        if self.vec.size()==0:
            return numpy.zeros(0, numpy.int32)
        self.ensure_compact()
        n_types=self.get_maxcol(k)+1
        result=numpy.zeros(n_types,numpy.int32)
        n=self.vec.size()
        for i from 0<=i<n:
            new_val=self.vec.at(i).addr[k]
            result[new_val]+=1
        return result
    cpdef VecI2 remap(self, int k, numpy.ndarray filt):
        cdef numpy.ndarray[numpy.int8_t, ndim=1] wanted=filt.astype('b')
        cdef numpy.ndarray[numpy.int_t, ndim=1] target=filt.cumsum()-1
        self.ensure_compact()
        cdef int n=self.vec.size()
        cdef int i
        cdef VecI2 result=VecI2()
        cdef c_CItemI2 c
        for i from 0<=i<n:
            c=self.vec.at(i)
            if wanted[c.addr[k]]:
                c.addr[k]=target[c.addr[k]]
                result.vec.push_back(c)
        return result
    def append(self,tup):
        k0,k1,val=tup
        self.c_add(k0,k1,val)
    def extend(self,tups):
        for k0,k1,val in tups:
            self.c_add(k0,k1,val)
    def __iter__(self):
        return self.item_iter()
    def __reduce__(self):
        return (VecI2,(),(),self.item_iter())
    def __setstate__(self,state):
        pass
    def __repr__(self):
        return 'VecI2(%s)'%(str(list(self)),)
    def __dealloc__(self):
        # need to call destructor explicitly
        self.vec.cxx_destructor()

cdef class IVecI2_iter:
    cdef VecI2 vec
    cdef c_VecI2 *vecC
    cdef unsigned int k
    def __init__(self,vec):
        self.vec=vec
        self.vecC=&self.vec.vec
        self.k=0
    def __iter__(self):
        return self
    def __next__(self):
        if self.k>=self.vecC.size():
            raise StopIteration
        cdef c_CItemI2 res=self.vecC.at(self.k)
        self.k+=1
        return (res.addr[0],
                res.addr[1],
                res.item)

cdef class LargeVecI2:
    cdef public object compact
    cdef public VecI2 loose
    def __init__(self):
        self.compact=[]
        self.loose=VecI2()
        self.loose.vec.reserve(100000)
    def ensure_compact(self):
        if self.loose.get_size()>0:
            self.compact.append(self.loose)
        self.loose=VecI2()
        while len(self.compact)>=2:
            self.compact[-2:]=[self.compact[-2]+self.compact[-1]]
    def add_count(self, coordinate_t k0,coordinate_t k1, item=1):
        self.loose.c_add(k0,k1, item)
        if self.loose.get_size(False)>=100000:
            self.loose.ensure_compact()
            self.compact.append(self.loose)
            self.loose=VecI2()
            self.loose.vec.reserve(100000)
            while (len(self.compact)>=2 and
                   self.compact[-1].get_size()>=0.6*self.compact[-2].get_size()):
                self.compact[-2:]=[self.compact[-2]+self.compact[-1]]
    def get_compact(self):
        self.ensure_compact()
        if len(self.compact)>0:
            return self.compact[0]
        else:
            return VecI2()
    def get_size(self, compactify=True):
        cdef int n
        if compactify:
            self.ensure_compact()
            if self.compact:
                return self.compact[0].get_size()
            else:
                return 0
        else:
            n=sum([c.get_size(False) for c in self.compact])
            n+=self.loose.get_size(False)
            return n
    def __iter__(self):
        self.ensure_compact()
        try:
            return iter(self.compact[0])
        except IndexError:
            return iter([])

    cpdef CSRMatrixI to_csr(self):
        self.ensure_compact()
        return self.compact[0].to_csr()

    def get_type_counts(self, k=0):
        return self.get_compact().get_type_counts(k)
    def remap(self, k, filt):
        return self.get_compact().remap(k,filt)


cdef class IVecI3_iter

cdef class VecI3:
    """mutable sparse int matrix based on item vector"""
    def __init__(self):
        self.is_compact=True
    def item_iter(self):
        self.ensure_compact()
        return IVecI3_iter(self)
    def get_size(self,compactify=True):
        if compactify:
            self.ensure_compact()
        return self.vec.size()
    def __len__(self):
        return self.get_size(True)
    def clear(self):
        self.vec.resize(0)
        self.is_compact=True
    cdef void compactify(self):
        cdef c_SmallerAddrI3 comp
        c_IVecI3_sort(self.vec.begin(),self.vec.end(),comp)
        c_compactifyI3(&self.vec)
        self.is_compact=True
    cdef void ensure_compact(self):
        if not self.is_compact:
            self.compactify()
    cpdef int get_count(self,coordinate_t k0,coordinate_t k1,coordinate_t k2):
        cdef c_CItemI3 ci
        ci.addr[0]=k0
        ci.addr[1]=k1
        ci.addr[2]=k2
        return c_get_countI3(&self.vec,ci)
    cdef void c_add(self, coordinate_t k0,coordinate_t k1,coordinate_t k2, int item):
        cdef c_CItemI3 c
        c.addr[0]=k0
        c.addr[1]=k1
        c.addr[2]=k2
        c.item=item
        self.vec.push_back(c)
        self.is_compact=False
    def add_count(self, coordinate_t k0,coordinate_t k1,coordinate_t k2, item=1):
        self.c_add(k0,k1,k2, item)
    def __add__(VecI3 self, VecI3 other):
        cdef c_CItemI3 c1, c2
        cdef unsigned int i1, i2
        cdef int k, delta
        cdef VecI3 result
        self.ensure_compact()
        other.ensure_compact()
        result=VecI3()
        result.vec.reserve(max(self.vec.size(),other.vec.size()))
        i1=0
        i2=0
        while i1<self.vec.size() and i2<other.vec.size():
            c1=self.vec.at(i1)
            c2=other.vec.at(i2)
            delta=0
            for k from 0<=k<3:
                if c1.addr[k]<c2.addr[k]:
                    delta=-1
                    break
                elif c1.addr[k]>c2.addr[k]:
                    delta=1
                    break
            if delta==0:
                c1.item+=c2.item
                result.vec.push_back(c1)
                i1+=1
                i2+=1
            elif delta==-1:
                result.vec.push_back(c1)
                i1+=1
            else:
                result.vec.push_back(c2)
                i2+=1
        while i1<self.vec.size():
            c1=self.vec.at(i1)
            result.vec.push_back(c1)
            i1+=1
        while i2<other.vec.size():
            c2=other.vec.at(i2)
            result.vec.push_back(c2)
            i2+=1
        return result

    def tofile(self, f):
        cdef long i, n, k, k_max
        n=self.get_size()
        f.write('I3 %d\n'%(n,))
        k=0
        for i from 0<=i<n/CHUNK_SIZE:
            s=PyBytes_FromStringAndSize(<char *>c_VecI3_get_pointer(&self.vec,k), CHUNK_SIZE*sizeof(c_CItemI3))
            f.write(s)
            k+=CHUNK_SIZE
        if k<n:
            s=PyBytes_FromStringAndSize(<char *>c_VecI3_get_pointer(&self.vec,k), (n-k)*sizeof(c_CItemI3))
            f.write(s)
    def fromfile(self, f):
        cdef long i, n
        line=f.readline().strip().split()
        assert line[0]=='I3'
        n=long(line[1])
        self.clear()
        self.vec.resize(n)
        k=0
        for i from 0<=i<n/CHUNK_SIZE:
            s=f.read(CHUNK_SIZE*sizeof(c_CItemI3))
            assert len(s)==CHUNK_SIZE*sizeof(c_CItemI3)
            memcpy(<void *>c_VecI3_get_pointer(&self.vec,k), <char *>s, CHUNK_SIZE*sizeof(c_CItemI3))
            k+=CHUNK_SIZE
        if k<n:
            s=f.read((n-k)*sizeof(c_CItemI3))
            assert len(s)==(n-k)*sizeof(c_CItemI3)
            memcpy(<void *>c_VecI3_get_pointer(&self.vec,k), <char *>s, (n-k)*sizeof(c_CItemI3))
    cpdef int get_maxcol(self, int k=0):
        cdef int i, n, new_val, col
        assert k>=0 and k<3
        n=self.vec.size()
        if n==0:
            return -1
        if k==0:
            return self.vec.at(n-1).addr[0]
        col=-1
        n=self.vec.size()
        for i from 0<=i<n:
            new_val=self.vec.at(i).addr[k]
            if new_val>col:
                col=new_val
        return col
    cpdef numpy.ndarray get_type_counts(self, int k=0):
        cdef int old_val, new_val, i, n, count
        cdef numpy.ndarray[numpy.int32_t, ndim=1] result
        assert k>=0 and k<3
        if self.vec.size()==0:
            return numpy.zeros(0, numpy.int32)
        self.ensure_compact()
        n_types=self.get_maxcol(k)+1
        result=numpy.zeros(n_types,numpy.int32)
        n=self.vec.size()
        for i from 0<=i<n:
            new_val=self.vec.at(i).addr[k]
            result[new_val]+=1
        return result
    cpdef VecI3 remap(self, int k, numpy.ndarray filt):
        cdef numpy.ndarray[numpy.int8_t, ndim=1] wanted=filt.astype('b')
        cdef numpy.ndarray[numpy.int_t, ndim=1] target=filt.cumsum()-1
        self.ensure_compact()
        cdef int n=self.vec.size()
        cdef int i
        cdef VecI3 result=VecI3()
        cdef c_CItemI3 c
        for i from 0<=i<n:
            c=self.vec.at(i)
            if wanted[c.addr[k]]:
                c.addr[k]=target[c.addr[k]]
                result.vec.push_back(c)
        return result
    def append(self,tup):
        k0,k1,k2,val=tup
        self.c_add(k0,k1,k2,val)
    def extend(self,tups):
        for k0,k1,k2,val in tups:
            self.c_add(k0,k1,k2,val)
    def __iter__(self):
        return self.item_iter()
    def __reduce__(self):
        return (VecI3,(),(),self.item_iter())
    def __setstate__(self,state):
        pass
    def __repr__(self):
        return 'VecI3(%s)'%(str(list(self)),)
    def __dealloc__(self):
        # need to call destructor explicitly
        self.vec.cxx_destructor()

cdef class IVecI3_iter:
    cdef VecI3 vec
    cdef c_VecI3 *vecC
    cdef unsigned int k
    def __init__(self,vec):
        self.vec=vec
        self.vecC=&self.vec.vec
        self.k=0
    def __iter__(self):
        return self
    def __next__(self):
        if self.k>=self.vecC.size():
            raise StopIteration
        cdef c_CItemI3 res=self.vecC.at(self.k)
        self.k+=1
        return (res.addr[0],
                res.addr[1],
                res.addr[2],
                res.item)

cdef class LargeVecI3:
    cdef public object compact
    cdef public VecI3 loose
    def __init__(self):
        self.compact=[]
        self.loose=VecI3()
        self.loose.vec.reserve(100000)
    def ensure_compact(self):
        if self.loose.get_size()>0:
            self.compact.append(self.loose)
        self.loose=VecI3()
        while len(self.compact)>=2:
            self.compact[-2:]=[self.compact[-2]+self.compact[-1]]
    def add_count(self, coordinate_t k0,coordinate_t k1,coordinate_t k2, item=1):
        self.loose.c_add(k0,k1,k2, item)
        if self.loose.get_size(False)>=100000:
            self.loose.ensure_compact()
            self.compact.append(self.loose)
            self.loose=VecI3()
            self.loose.vec.reserve(100000)
            while (len(self.compact)>=2 and
                   self.compact[-1].get_size()>=0.6*self.compact[-2].get_size()):
                self.compact[-2:]=[self.compact[-2]+self.compact[-1]]
    def get_compact(self):
        self.ensure_compact()
        if len(self.compact)>0:
            return self.compact[0]
        else:
            return VecI3()
    def get_size(self, compactify=True):
        cdef int n
        if compactify:
            self.ensure_compact()
            if self.compact:
                return self.compact[0].get_size()
            else:
                return 0
        else:
            n=sum([c.get_size(False) for c in self.compact])
            n+=self.loose.get_size(False)
            return n
    def __iter__(self):
        self.ensure_compact()
        try:
            return iter(self.compact[0])
        except IndexError:
            return iter([])

    def get_type_counts(self, k=0):
        return self.get_compact().get_type_counts(k)
    def remap(self, k, filt):
        return self.get_compact().remap(k,filt)

cdef class SparseVectorI:
    def __init__(self, pairs=None):
        cdef coordinate_t i
        if pairs is not None:
            self.my_len=len(pairs)
            self.idx_ptr=<coordinate_t *>PyMem_Malloc(self.my_len*sizeof(coordinate_t))
            self.vals_ptr=<int *>PyMem_Malloc(self.my_len*sizeof(int))
            for i from 0<=i<self.my_len:
                x,y=pairs[i]
                self.idx_ptr[i]=x
                self.vals_ptr[i]=y
        else:
            self.my_len=0
            self.idx_ptr=NULL
            self.vals_ptr=NULL
    cpdef int from_dense(self, int[:] dense):
        cdef int i, k
        assert self.my_len == 0
        k = 0
        for i from 0<=i<dense.shape[0]:
            if dense[i]!=0.0:
                k += 1
        self.my_len = k
        self.idx_ptr=<coordinate_t *>PyMem_Malloc(self.my_len*sizeof(coordinate_t))
        self.vals_ptr=<int *>PyMem_Malloc(self.my_len*sizeof(int))
        k = 0
        for i from 0<=i<dense.shape[0]:
            if dense[i] != 0.0:
                self.idx_ptr[k] = i
                self.vals_ptr[k] = dense[i]
                k += 1
        return k
    cpdef int dotSelf(self):
        cdef int s=0
        cdef coordinate_t i
        for i from 0<=i<self.my_len:
            s+=self.vals_ptr[i]*self.vals_ptr[i]
        return s
    cdef int _dotFull(self, const_int_ptr full_ptr):
        cdef int s=0
        cdef coordinate_t i
        for i from 0<=i<self.my_len:
            s+=self.vals_ptr[i]*full_ptr[self.idx_ptr[i]]
        return s
    cdef int _dotFull_partial(self, const_int_ptr full_ptr, int my_len):
        cdef int s=0
        cdef coordinate_t i
        for i from 0<=i<my_len:
            s+=self.vals_ptr[i]*full_ptr[self.idx_ptr[i]]
        return s
    cpdef double jaccard(self, SparseVectorI other):
        cdef double s_max=0
        cdef double s_min=0
        cdef int val1, val2
        cdef coordinate_t i,j, idx_i, idx_j
        i=0
        j=0
        while i<self.my_len and j<other.my_len:
            idx_i=self.idx_ptr[i]
            idx_j=other.idx_ptr[j]
            if idx_i<idx_j:
                s_max+=self.vals_ptr[i]
                i+=1
            elif idx_i>idx_j:
                s_max+=other.vals_ptr[j]
                j+=1
            else:
                val1=self.vals_ptr[i]
                val2=other.vals_ptr[j]
                if val1>val2:
                    s_max+=val1
                    s_min+=val2
                else:
                    s_max+=val2
                    s_min+=val1
                i+=1
                j+=1
        if i<self.my_len:
            while i<self.my_len:
                s_max+=self.vals_ptr[i]
                i+=1
        else:
            while j<other.my_len:
                s_max+=other.vals_ptr[j]
                j+=1
        if s_max==0:
            return 0.0
        else:
            return s_min/s_max
    cpdef int dotSparse(self, SparseVectorI other):
        cdef int product=0
        cdef int val1, val2
        cdef coordinate_t i, j, idx_i, idx_j
        i=0
        j=0
        while i<self.my_len and j<other.my_len:
           idx_i=self.idx_ptr[i]
           idx_j=other.idx_ptr[j]
           if idx_i<idx_j:
              i+=1
           elif idx_i>idx_j:
              j+=1
           else:
              val1=self.vals_ptr[i]
              val2=other.vals_ptr[j]
              product+=val1*val2
              i+=1
              j+=1
        return product
    def count_intersection(self, SparseVectorI other):
        cdef coordinate_t i, j, idx_i, idx_j
        cdef int n_ab=0
        i=0
        j=0
        while i<self.my_len and j<other.my_len:
           idx_i=self.idx_ptr[i]
           idx_j=other.idx_ptr[j]
           if idx_i<idx_j:
              i+=1
           elif idx_i>idx_j:
              j+=1
           else:
              i+=1
              j+=1
              n_ab+=1
        return n_ab
    cpdef double min_sum(self, SparseVectorI other):
        cdef double product=0.0
        cdef int val1, val2
        cdef coordinate_t i, j, idx_i, idx_j
        i=0
        j=0
        while i<self.my_len and j<other.my_len:
           idx_i=self.idx_ptr[i]
           idx_j=other.idx_ptr[j]
           if idx_i<idx_j:
              i+=1
           elif idx_i>idx_j:
              j+=1
           else:
              val1=self.vals_ptr[i]
              val2=other.vals_ptr[j]
              if val1<val2:
                  product+=val1
              else:
                  product+=val2
              i+=1
              j+=1
        return product
    cpdef double cosine(self, SparseVectorI other):
        cdef double sqsum_self=0.0
        cdef double sqsum_other=0.0
        cdef double product=0.0
        cdef int val1, val2
        cdef coordinate_t i, j, idx_i, idx_j
        i=0
        j=0
        while i<self.my_len and j<other.my_len:
           idx_i=self.idx_ptr[i]
           idx_j=other.idx_ptr[j]
           if idx_i<idx_j:
              val1=self.vals_ptr[i]
              sqsum_self+=val1*val1
              i+=1
           elif idx_i>idx_j:
              val2=other.vals_ptr[j]
              sqsum_other+=val2*val2
              j+=1
           else:
              val1=self.vals_ptr[i]
              val2=other.vals_ptr[j]
              sqsum_self+=val1*val1
              sqsum_other+=val2*val2
              product+=val1*val2
              i+=1
              j+=1
        if i<self.my_len:
            while i<self.my_len:
                val1=self.vals_ptr[i]
                sqsum_self+=val1*val1
                i+=1
        else:
            while j<other.my_len:
                val2=other.vals_ptr[j]
                sqsum_other+=val2*val2
                j+=1
        return product/sqrt(sqsum_self*sqsum_other)
    cpdef double jsd_unnorm(self, SparseVectorI other):
        cdef double sum=0.0
        cdef double val1, val2, avg
        cdef coordinate_t i, j, idx_i, idx_j
        i=0
        j=0
        while i<self.my_len and j<other.my_len:
           idx_i=self.idx_ptr[i]
           idx_j=other.idx_ptr[j]
           if idx_i<idx_j:
               val1=self.vals_ptr[i]
               sum+=val1*M_LN2
               i+=1
           elif idx_i>idx_j:
               val2=other.vals_ptr[j]
               sum+=val2*M_LN2
               j+=1
           else:
               val1=self.vals_ptr[i]
               val2=other.vals_ptr[j]
               i+=1
               j+=1
               avg=(val1+val2)/2.0
               sum+=val1*log(val1/avg)+val2*log(val2/avg)
        while i<self.my_len:
           val1=self.vals_ptr[i]
           i+=1
           sum+=val1*M_LN2
        while j<other.my_len:
           val2=other.vals_ptr[j]
           j+=1
           sum+=val2*M_LN2
        return sum/(2.0*M_LN2)
    cpdef double skew_unnorm(self, SparseVectorI other, double alpha):
        cdef double sum=0.0
        cdef double beta=1.0-alpha
        cdef double val1, val2, avg
        cdef coordinate_t i, j, idx_i, idx_j
        cdef double log_invbeta=log(1.0/beta)
        i=0
        j=0
        while i<self.my_len and j<other.my_len:
           idx_i=self.idx_ptr[i]
           idx_j=self.idx_ptr[j]
           if idx_i<idx_j:
               val1=self.vals_ptr[i]
               sum+=val1*log_invbeta
               i+=1
           elif idx_i>idx_j:
               j+=1
           else:
               val1=self.vals_ptr[i]
               val2=other.vals_ptr[j]
               avg=alpha*val2+beta*val1
               sum+=log(val1/avg)
               i+=1
               j+=1
        return sum
    cpdef double norm_l1(self):
        cdef double sum=0.0
        cdef unsigned int i
        for i from 0<=i<self.my_len:
            sum+=fabs(self.vals_ptr[i])
        return sum
    cpdef double norm_l2(self):
        cdef double sum=0.0
        cdef double val
        cdef unsigned int i
        for i from 0<=i<self.my_len:
            val=self.vals_ptr[i];
            sum+=val*val
        return sqrt(sum)
    cpdef double norm_lp(self, double p):
        cdef double sum=0.0
        cdef int i
        for i from 0<=i<self.my_len:
            sum+=pow(fabs(self.vals_ptr[i]),p)
        return pow(sum,1.0/p)
    def dotFull(self, numpy.ndarray[int,ndim=1] full):
        cdef int *full_ptr
        cdef unsigned int my_len=self.my_len
        cdef unsigned int full_len=len(full)
        # treat nonpresent values as 0
        while my_len>0 and self.idx_ptr[my_len-1]>full_len:
            my_len-=1
        full_ptr=<int*>full.data
        return self._dotFull_partial(full_ptr,my_len)
    def dotFull_check(self, numpy.ndarray[int,ndim=1] full):
        cdef int *full_ptr
        cdef int my_len=self.my_len
        cdef int full_len=len(full)
        # boundary check
        full[self.idx_ptr[self.my_len-1]]
        full_ptr=<int*>full.data
        return self._dotFull(full_ptr)
    def dotMatrix(self, numpy.ndarray[int,ndim=2] full):
        cdef numpy.ndarray[int,ndim=1] result
        result = numpy.zeros(full.shape[1], numpy.int32)
        for i from 0<=i<self.my_len:
            result += full[self.idx_ptr[i]] * self.vals_ptr[i]
        return result
    cdef void _axpy(self, int *x_ptr, int a):
        for i from 0<=i<self.my_len:
            x_ptr[self.idx_ptr[i]]+=a*self.vals_ptr[i]
    def axpy(self, numpy.ndarray[int,ndim=1] x, int a=1):
        # boundary check
        x[self.idx_ptr[self.my_len-1]]
        self._axpy(<int *>x.data,a)
    cpdef int sqdist(self, SparseVectorI other):
        """computes ||x-y||^2"""
        cdef int s=0
        cdef int val
        cdef int i1=0,i2=0
        while i1<self.my_len and i2<other.my_len:
            if self.idx_ptr[i1]<other.idx_ptr[i2]:
                val=self.vals_ptr[i1]
                s+=val*val
                i1+=1
            elif self.idx_ptr[i1]>other.idx_ptr[i2]:
                val=other.vals_ptr[i2]
                s+=val*val
                i2+=1
            else:
                val=self.vals_ptr[i1]-other.vals_ptr[i2]
                s+=val*val
                i1+=1
                i2+=1
        while i1<self.my_len:
            val=self.vals_ptr[i1]
            s+=val*val
            i1+=1
        while i2<other.my_len:
            val=other.vals_ptr[i2]
            s+=val*val
            i2+=1
        return s
    def write_pairs(self, f, delim=':'):
        cdef int i
        w_func=f.write
        for i from 0<=i<self.my_len:
            # svmlight does not want 0 as index
            w_func(' %d%s%s'%(self.idx_ptr[i]+1,delim,self.vals_ptr[i]))
    def __imul__(self, int a):
        cdef coordinate_t i
        for i from 0<=i<self.my_len:
            self.vals_ptr[i] *= a
        return self
    def __idiv__(self, int a):
        cdef coordinate_t i
        for i from 0<=i<self.my_len:
            self.vals_ptr[i] /= a
        return self
    cpdef SparseVectorI min_vals(self, SparseVectorI other):
        cdef SparseVectorI result
        cdef coordinate_t i1, i2, k, idx1, idx2
        result=SparseVectorI()
        if self.my_len<other.my_len:
            result.my_len=self.my_len
        else:
            result.my_len=other.my_len
        result.idx_ptr=<coordinate_t *>PyMem_Malloc(result.my_len*sizeof(coordinate_t))
        result.vals_ptr=<int *>PyMem_Malloc(result.my_len*sizeof(int))
        i1=i2=k=0
        while i1<self.my_len and i2<other.my_len:
           idx1=self.idx_ptr[i1]
           idx2=other.idx_ptr[i2]
           if idx1<idx2:
               i1+=1
           elif idx1>idx2:
               i2+=1
           else:
               result.idx_ptr[k]=idx1
               val1=self.vals_ptr[i1]
               val2=other.vals_ptr[i2]
               if val1<val2:
                   result.vals_ptr[k]=val1
               else:
                   result.vals_ptr[k]=val2
               i1+=1
               i2+=1
               k+=1
        result.my_len=k
        return result
    def scale_array(SparseVectorI self, numpy.ndarray[int, ndim=1] a):
        cdef SparseVectorI result
        cdef coordinate_t i
        result=SparseVectorI()
        result.my_len=self.my_len
        result.idx_ptr=<coordinate_t *>PyMem_Malloc(result.my_len*sizeof(coordinate_t))
        result.vals_ptr=<int *>PyMem_Malloc(result.my_len*sizeof(int))
        for i from 0<=i<self.my_len:
            result.idx_ptr[i]=self.idx_ptr[i]
        for i from 0<=i<self.my_len:
            result.vals_ptr[i]=self.vals_ptr[i]*a[self.idx_ptr[i]]
        return result
    def __div__(SparseVectorI self, int a):
        cdef SparseVectorI result
        cdef coordinate_t i
        result=SparseVectorI()
        result.my_len=self.my_len
        result.idx_ptr=<coordinate_t *>PyMem_Malloc(result.my_len*sizeof(coordinate_t))
        result.vals_ptr=<int *>PyMem_Malloc(result.my_len*sizeof(int))
        for i from 0<=i<self.my_len:
            result.idx_ptr[i]=self.idx_ptr[i]
        for i from 0<=i<self.my_len:
            result.vals_ptr[i]=self.vals_ptr[i]/a
        return result
    def __len__(self):
        return self.my_len
    def __add__(SparseVectorI self,SparseVectorI other):
        cdef coordinate_t *new_idx
        cdef int *new_vals
        cdef coordinate_t i1,i2,i_all
        cdef coordinate_t n1,n2,n_all
        cdef coordinate_t k1,k2,k
        cdef SparseVectorI result
        # first pass: determine how much space we need
        n1=self.my_len
        n2=other.my_len
        n_all=i1=i2=0
        while i1<n1 and i2<n2:
            k1=self.idx_ptr[i1]
            k2=other.idx_ptr[i2]
            if k1>=k2: i2+=1
            if k2>=k1: i1+=1
            n_all+=1
        if i1<n1: n_all+=n1-i1
        if i2<n2: n_all+=n2-i2
        new_idx=<coordinate_t *>PyMem_Malloc(n_all*sizeof(coordinate_t))
        new_vals=<int *>PyMem_Malloc(n_all*sizeof(int))
        # second pass: fill in the actual values
        i1=i2=i_all=0
        while i1<n1 and i2<n2:
            k1=self.idx_ptr[i1]
            k2=other.idx_ptr[i2]
            if k1>=k2:
                new_idx[i_all]=k2
                new_vals[i_all]=other.vals_ptr[i2]
                i2+=1
                if k1==k2:
                    new_vals[i_all]+=self.vals_ptr[i1]
                    i1+=1
            elif k2>=k1:
                new_idx[i_all]=k1
                new_vals[i_all]=self.vals_ptr[i1]
                i1+=1
            i_all+=1
        while i1<n1:
            new_idx[i_all]=self.idx_ptr[i1]
            new_vals[i_all]=self.vals_ptr[i1]
            i1+=1
            i_all+=1
        else:
            while i2<n2:
                new_idx[i_all]=other.idx_ptr[i2]
                new_vals[i_all]=other.vals_ptr[i2]
                i2+=1
                i_all+=1
        assert i_all==n_all
        result=SparseVectorI()
        result.my_len=n_all
        result.idx_ptr=new_idx
        result.vals_ptr=new_vals
        return result
    def __sub__(SparseVectorI self,SparseVectorI other):
        cdef coordinate_t *new_idx
        cdef int *new_vals
        cdef coordinate_t i1,i2,i_all
        cdef coordinate_t n1,n2,n_all
        cdef coordinate_t k1,k2,k
        cdef SparseVectorI result
        # first pass: determine how much space we need
        n1=self.my_len
        n2=other.my_len
        n_all=i1=i2=0
        while i1<n1 and i2<n2:
            k1=self.idx_ptr[i1]
            k2=other.idx_ptr[i2]
            if k1>=k2: i2+=1
            if k2>=k1: i1+=1
            n_all+=1
        if i1<n1: n_all+=n1-i1
        if i2<n2: n_all+=n2-i2
        new_idx=<coordinate_t *>PyMem_Malloc(n_all*sizeof(coordinate_t))
        new_vals=<int *>PyMem_Malloc(n_all*sizeof(int))
        # second pass: fill in the actual values
        i1=i2=i_all=0
        while i1<n1 and i2<n2:
            k1=self.idx_ptr[i1]
            k2=other.idx_ptr[i2]
            if k1>=k2:
                new_idx[i_all]=k2
                new_vals[i_all]=-other.vals_ptr[i2]
                i2+=1
                if k1==k2:
                    new_vals[i_all]+=self.vals_ptr[i1]
                    i1+=1
            elif k2>=k1:
                new_idx[i_all]=k1
                new_vals[i_all]=self.vals_ptr[i1]
                i1+=1
            i_all+=1
        while i1<n1:
            new_idx[i_all]=self.idx_ptr[i1]
            new_vals[i_all]=self.vals_ptr[i1]
            i1+=1
            i_all+=1
        else:
            while i2<n2:
                new_idx[i_all]=other.idx_ptr[i2]
                new_vals[i_all]=-other.vals_ptr[i2]
                i2+=1
                i_all+=1
        assert i_all==n_all
        result=SparseVectorI()
        result.my_len=n_all
        result.idx_ptr=new_idx
        result.vals_ptr=new_vals
        return result
    def __or__(SparseVectorI self,SparseVectorI other):
        cdef coordinate_t *new_idx
        cdef int *new_vals
        cdef coordinate_t i1,i2,i_all
        cdef coordinate_t n1,n2,n_all
        cdef coordinate_t k1,k2,k
        cdef SparseVectorI result
        cdef int val
        # first pass: determine how much space we need
        n1=self.my_len
        n2=other.my_len
        n_all=i1=i2=0
        while i1<n1 and i2<n2:
            k1=self.idx_ptr[i1]
            k2=other.idx_ptr[i2]
            if k1>=k2: i2+=1
            if k2>=k1: i1+=1
            n_all+=1
        if i1<n1: n_all+=n1-i1
        if i2<n2: n_all+=n2-i2
        new_idx=<coordinate_t *>PyMem_Malloc(n_all*sizeof(coordinate_t))
        new_vals=<int *>PyMem_Malloc(n_all*sizeof(int))
        # second pass: fill in the actual values
        i1=i2=i_all=0
        while i1<n1 and i2<n2:
            k1=self.idx_ptr[i1]
            k2=other.idx_ptr[i2]
            if k1>=k2:
                new_idx[i_all]=k2
                new_vals[i_all]=other.vals_ptr[i2]
                i2+=1
                if k1==k2:
                    val=self.vals_ptr[i1]
                    if new_vals[i_all]<val:
                        new_vals[i_all]=val
                    i1+=1
            else: # k2>k1:
                new_idx[i_all]=k1
                new_vals[i_all]=self.vals_ptr[i1]
                i1+=1
            i_all+=1
        while i1<n1:
            new_idx[i_all]=self.idx_ptr[i1]
            new_vals[i_all]=self.vals_ptr[i1]
            i1+=1
            i_all+=1
        else:
            while i2<n2:
                new_idx[i_all]=other.idx_ptr[i2]
                new_vals[i_all]=other.vals_ptr[i2]
                i2+=1
                i_all+=1
        assert i_all==n_all
        result=SparseVectorI()
        result.my_len=n_all
        result.idx_ptr=new_idx
        result.vals_ptr=new_vals
        return result
    def __lshift__(self, int k):
        cdef SparseVectorI result
        cdef coordinate_t i
        result=SparseVectorI()
        result.my_len=self.my_len
        result.idx_ptr=<coordinate_t *>PyMem_Malloc(result.my_len*sizeof(coordinate_t))
        result.vals_ptr=<int *>PyMem_Malloc(result.my_len*sizeof(int))
        for i from 0<=i<self.my_len:
            result.idx_ptr[i]=self.idx_ptr[i]+k
        for i from 0<=i<self.my_len:
            result.vals_ptr[i]=self.vals_ptr[i]
        return result
    def __ilshift__(self, int k):
        cdef SparseVectorI result
        cdef coordinate_t i
        for i from 0<=i<self.my_len:
            self.idx_ptr[i]+=k
        return self
    def __getitem__(self, int i):
        if i<0:
            i=self.my_len+i
        if i>=self.my_len or i<0:
            raise IndexError
        return (self.idx_ptr[i],self.vals_ptr[i])
    def __repr__(self):
        cdef unsigned int i
        ss=[]
        for i from 0<=i<self.my_len:
            ss.append('(%d,%s)'%(self.idx_ptr[i],self.vals_ptr[i]))
        return 'SparseVectorI([%s])'%(','.join(ss))
    def __reduce_ex__(self,protocol):
        if protocol==0:
            # choose compatibility over performance.
            return (SparseVectorI,(list(self),),())
        else:
            s_idx=PyBytes_FromStringAndSize(<char *>self.idx_ptr,self.my_len*sizeof(coordinate_t))
            s_vals=PyBytes_FromStringAndSize(<char *>self.vals_ptr,self.my_len*sizeof(int))
            return (SparseVectorI,(None,),(s_idx,s_vals))
    def __setstate__(self,state):
        cdef coordinate_t *p_idx
        cdef int *p_vals
        if len(state)==0:
            pass
        elif len(state)==2:
            assert (self.idx_ptr==NULL)
            (s_idx, s_vals)=state
            p_idx=<coordinate_t *>(<char *>s_idx)
            p_vals=<int *>(<char *>s_vals)
            self.my_len=len(s_idx)/sizeof(coordinate_t)
            self.idx_ptr=<coordinate_t *>PyMem_Malloc(self.my_len*sizeof(coordinate_t))
            self.vals_ptr=<int *>PyMem_Malloc(self.my_len*sizeof(int))
            for i from 0<=i<self.my_len:
                self.idx_ptr[i]=p_idx[i]
                self.vals_ptr[i]=p_vals[i]
    def __dealloc__(self):
        if self.buf is None:
            PyMem_Free(self.idx_ptr)
            PyMem_Free(self.vals_ptr)
        else:
            self.buf=None
        self.idx_ptr=<coordinate_t *>0
        self.vals_ptr=<int *>0
        self.my_len=0
    def to_scipy(self):
        cdef numpy.ndarray[numpy.int32_t, ndim=1] data
        cdef numpy.ndarray[numpy.int32_t, ndim=1] indices
        cdef numpy.ndarray[numpy.int32_t, ndim=1] indptr
        cdef int i,j
        indptr=numpy.zeros(2, numpy.int32)
        indices=numpy.zeros(self.my_len, numpy.int32)
        data=numpy.zeros(self.my_len, numpy.int32)
        indptr[1]=self.my_len
        for i from 0<=i<self.my_len:
            indices[i]=0
        for i from 0<=i<self.my_len:
            data[i]=self.vals_ptr[i]
        return (data,indices,indptr)

emptyvec_I=SparseVectorI([])

cdef class SparseVectorsI:
    def __init__(self):
       self.vecs = []
    cpdef SparseVectorI to_vec(self):
        cdef SparseVectorI vec
        cdef int weight
        cdef size_t i, j
        cdef VecI1 result = VecI1()
        for i from 0<=i<len(self.vecs):
            weight = self.weights.at(i)
            vec = self.vecs[i]
            for j from 0<=j<vec.my_len:
                result.add_count(vec.idx_ptr[j], weight*vec.vals_ptr[j])
        return result.to_sparse()
    cpdef add(self, SparseVectorI vec, int weight=1):
        self.vecs.append(vec)
        self.weights.push_back(weight)
    def __mul__(self, factor):
        cdef SparseVectorI vec
        cdef int weight
        cdef SparseVectorsI result = SparseVectorsI
        for i from 0<=i<len(self.vecs):
            weight = self.weights.at(i)
            vec = self.vecs[i]
            result.add(vec, weight*factor)
    def __add__(self, others):
        cdef SparseVectorI vec
        cdef int weight
        cdef SparseVectorsI oth_vecs = others
        cdef SparseVectorsI result = SparseVectorsI
        for i from 0<=i<len(self.vecs):
            weight = self.weights.at(i)
            vec = self.vecs[i]
            result.add(vec, weight)
        for i from 0<=i<len(oth_vecs.vecs):
            weight = oth_vecs.weights.at(i)
            vec = oth_vecs.vecs[i]
            result.add(vec, weight)
        return result
    def __sub__(SparseVectorsI self, SparseVectorsI others):
        cdef SparseVectorI vec
        cdef int weight
        cdef SparseVectorsI oth_vecs = others
        cdef SparseVectorsI result = SparseVectorsI()
        for i from 0<=i<len(self.vecs):
            weight = self.weights.at(i)
            vec = self.vecs[i]
            result.add(vec, weight)
        for i from 0<=i<len(oth_vecs.vecs):
            weight = oth_vecs.weights.at(i)
            vec = oth_vecs.vecs[i]
            result.add(vec, -weight)
        return result
    cdef int _dotFull(self, const_int_ptr full_ptr):
        cdef size_t i, j
        cdef int weight
        cdef SparseVectorI vec
        cdef int result=0
        for i from 0<=i<len(self.vecs):
            weight = self.weights.at(i)
            vec = self.vecs.at(i)
            for j from 0<=j<vec.my_len:
                result += full_ptr[vec.indices[j]]*weight*vec.vals_ptr[j]
        return result
    def __len__(self):
        return len(self.vecs)
    def __getitem__(self, k):
        vec = self.vecs[k]
        return (self.weights.at(k), vec)
## F -> float


cdef class IVecF1_iter

cdef class VecF1:
    """mutable sparse int matrix based on item vector"""
    def __init__(self):
        self.is_compact=True
    def item_iter(self):
        self.ensure_compact()
        return IVecF1_iter(self)
    def get_size(self,compactify=True):
        if compactify:
            self.ensure_compact()
        return self.vec.size()
    def __len__(self):
        return self.get_size(True)
    def clear(self):
        self.vec.resize(0)
        self.is_compact=True
    cdef void compactify(self):
        cdef c_SmallerAddrF1 comp
        c_IVecF1_sort(self.vec.begin(),self.vec.end(),comp)
        c_compactifyF1(&self.vec)
        self.is_compact=True
    cdef void ensure_compact(self):
        if not self.is_compact:
            self.compactify()
    cpdef float get_count(self,coordinate_t k0):
        cdef c_CItemF1 ci
        ci.addr[0]=k0
        return c_get_countF1(&self.vec,ci)
    cdef void c_add(self, coordinate_t k0, float item):
        cdef c_CItemF1 c
        c.addr[0]=k0
        c.item=item
        self.vec.push_back(c)
        self.is_compact=False
    def add_count(self, coordinate_t k0, item=1):
        self.c_add(k0, item)
    def __add__(VecF1 self, VecF1 other):
        cdef c_CItemF1 c1, c2
        cdef unsigned int i1, i2
        cdef int k, delta
        cdef VecF1 result
        self.ensure_compact()
        other.ensure_compact()
        result=VecF1()
        result.vec.reserve(max(self.vec.size(),other.vec.size()))
        i1=0
        i2=0
        while i1<self.vec.size() and i2<other.vec.size():
            c1=self.vec.at(i1)
            c2=other.vec.at(i2)
            delta=0
            for k from 0<=k<1:
                if c1.addr[k]<c2.addr[k]:
                    delta=-1
                    break
                elif c1.addr[k]>c2.addr[k]:
                    delta=1
                    break
            if delta==0:
                c1.item+=c2.item
                result.vec.push_back(c1)
                i1+=1
                i2+=1
            elif delta==-1:
                result.vec.push_back(c1)
                i1+=1
            else:
                result.vec.push_back(c2)
                i2+=1
        while i1<self.vec.size():
            c1=self.vec.at(i1)
            result.vec.push_back(c1)
            i1+=1
        while i2<other.vec.size():
            c2=other.vec.at(i2)
            result.vec.push_back(c2)
            i2+=1
        return result

    cpdef SparseVectorF to_sparse(self):
        cdef c_CItemF1 c
        cdef SparseVectorF result
        cdef coordinate_t n,i
        cdef coordinate_t *idx_ptr
        cdef float *vals_ptr
        self.ensure_compact()
        n=self.vec.size()
        idx_ptr=<coordinate_t *>PyMem_Malloc(n*sizeof(coordinate_t))
        vals_ptr=<float *>PyMem_Malloc(n*sizeof(float))
        for i from 0<=i<n:
          c=self.vec.at(i)
          idx_ptr[i]=c.addr[0]
          vals_ptr[i]=c.item
        result=SparseVectorF(None)
        result.idx_ptr=idx_ptr
        result.vals_ptr=vals_ptr
        result.my_len=n
        return result

    def tofile(self, f):
        cdef long i, n, k, k_max
        n=self.get_size()
        f.write('F1 %d\n'%(n,))
        k=0
        for i from 0<=i<n/CHUNK_SIZE:
            s=PyBytes_FromStringAndSize(<char *>c_VecF1_get_pointer(&self.vec,k), CHUNK_SIZE*sizeof(c_CItemF1))
            f.write(s)
            k+=CHUNK_SIZE
        if k<n:
            s=PyBytes_FromStringAndSize(<char *>c_VecF1_get_pointer(&self.vec,k), (n-k)*sizeof(c_CItemF1))
            f.write(s)
    def fromfile(self, f):
        cdef long i, n
        line=f.readline().strip().split()
        assert line[0]=='F1'
        n=long(line[1])
        self.clear()
        self.vec.resize(n)
        k=0
        for i from 0<=i<n/CHUNK_SIZE:
            s=f.read(CHUNK_SIZE*sizeof(c_CItemF1))
            assert len(s)==CHUNK_SIZE*sizeof(c_CItemF1)
            memcpy(<void *>c_VecF1_get_pointer(&self.vec,k), <char *>s, CHUNK_SIZE*sizeof(c_CItemF1))
            k+=CHUNK_SIZE
        if k<n:
            s=f.read((n-k)*sizeof(c_CItemF1))
            assert len(s)==(n-k)*sizeof(c_CItemF1)
            memcpy(<void *>c_VecF1_get_pointer(&self.vec,k), <char *>s, (n-k)*sizeof(c_CItemF1))
    cpdef int get_maxcol(self, int k=0):
        cdef int i, n, new_val, col
        assert k>=0 and k<1
        n=self.vec.size()
        if n==0:
            return -1
        if k==0:
            return self.vec.at(n-1).addr[0]
        col=-1
        n=self.vec.size()
        for i from 0<=i<n:
            new_val=self.vec.at(i).addr[k]
            if new_val>col:
                col=new_val
        return col
    cpdef numpy.ndarray get_type_counts(self, int k=0):
        cdef int old_val, new_val, i, n, count
        cdef numpy.ndarray[numpy.int32_t, ndim=1] result
        assert k>=0 and k<1
        if self.vec.size()==0:
            return numpy.zeros(0, numpy.int32)
        self.ensure_compact()
        n_types=self.get_maxcol(k)+1
        result=numpy.zeros(n_types,numpy.int32)
        n=self.vec.size()
        for i from 0<=i<n:
            new_val=self.vec.at(i).addr[k]
            result[new_val]+=1
        return result
    cpdef VecF1 remap(self, int k, numpy.ndarray filt):
        cdef numpy.ndarray[numpy.int8_t, ndim=1] wanted=filt.astype('b')
        cdef numpy.ndarray[numpy.int_t, ndim=1] target=filt.cumsum()-1
        self.ensure_compact()
        cdef int n=self.vec.size()
        cdef int i
        cdef VecF1 result=VecF1()
        cdef c_CItemF1 c
        for i from 0<=i<n:
            c=self.vec.at(i)
            if wanted[c.addr[k]]:
                c.addr[k]=target[c.addr[k]]
                result.vec.push_back(c)
        return result
    def append(self,tup):
        k0,val=tup
        self.c_add(k0,val)
    def extend(self,tups):
        for k0,val in tups:
            self.c_add(k0,val)
    def __iter__(self):
        return self.item_iter()
    def __reduce__(self):
        return (VecF1,(),(),self.item_iter())
    def __setstate__(self,state):
        pass
    def __repr__(self):
        return 'VecF1(%s)'%(str(list(self)),)
    def __dealloc__(self):
        # need to call destructor explicitly
        self.vec.cxx_destructor()

cdef class IVecF1_iter:
    cdef VecF1 vec
    cdef c_VecF1 *vecC
    cdef unsigned int k
    def __init__(self,vec):
        self.vec=vec
        self.vecC=&self.vec.vec
        self.k=0
    def __iter__(self):
        return self
    def __next__(self):
        if self.k>=self.vecC.size():
            raise StopIteration
        cdef c_CItemF1 res=self.vecC.at(self.k)
        self.k+=1
        return (res.addr[0],
                res.item)

cdef class LargeVecF1:
    cdef public object compact
    cdef public VecF1 loose
    def __init__(self):
        self.compact=[]
        self.loose=VecF1()
        self.loose.vec.reserve(100000)
    def ensure_compact(self):
        if self.loose.get_size()>0:
            self.compact.append(self.loose)
        self.loose=VecF1()
        while len(self.compact)>=2:
            self.compact[-2:]=[self.compact[-2]+self.compact[-1]]
    def add_count(self, coordinate_t k0, item=1):
        self.loose.c_add(k0, item)
        if self.loose.get_size(False)>=100000:
            self.loose.ensure_compact()
            self.compact.append(self.loose)
            self.loose=VecF1()
            self.loose.vec.reserve(100000)
            while (len(self.compact)>=2 and
                   self.compact[-1].get_size()>=0.6*self.compact[-2].get_size()):
                self.compact[-2:]=[self.compact[-2]+self.compact[-1]]
    def get_compact(self):
        self.ensure_compact()
        if len(self.compact)>0:
            return self.compact[0]
        else:
            return VecF1()
    def get_size(self, compactify=True):
        cdef int n
        if compactify:
            self.ensure_compact()
            if self.compact:
                return self.compact[0].get_size()
            else:
                return 0
        else:
            n=sum([c.get_size(False) for c in self.compact])
            n+=self.loose.get_size(False)
            return n
    def __iter__(self):
        self.ensure_compact()
        try:
            return iter(self.compact[0])
        except IndexError:
            return iter([])

    def get_type_counts(self, k=0):
        return self.get_compact().get_type_counts(k)
    def remap(self, k, filt):
        return self.get_compact().remap(k,filt)


cdef class IVecF2_iter

cdef class VecF2:
    """mutable sparse int matrix based on item vector"""
    def __init__(self):
        self.is_compact=True
    def item_iter(self):
        self.ensure_compact()
        return IVecF2_iter(self)
    def get_size(self,compactify=True):
        if compactify:
            self.ensure_compact()
        return self.vec.size()
    def __len__(self):
        return self.get_size(True)
    def clear(self):
        self.vec.resize(0)
        self.is_compact=True
    cdef void compactify(self):
        cdef c_SmallerAddrF2 comp
        c_IVecF2_sort(self.vec.begin(),self.vec.end(),comp)
        c_compactifyF2(&self.vec)
        self.is_compact=True
    cdef void ensure_compact(self):
        if not self.is_compact:
            self.compactify()
    cpdef float get_count(self,coordinate_t k0,coordinate_t k1):
        cdef c_CItemF2 ci
        ci.addr[0]=k0
        ci.addr[1]=k1
        return c_get_countF2(&self.vec,ci)
    cdef void c_add(self, coordinate_t k0,coordinate_t k1, float item):
        cdef c_CItemF2 c
        c.addr[0]=k0
        c.addr[1]=k1
        c.item=item
        self.vec.push_back(c)
        self.is_compact=False
    def add_count(self, coordinate_t k0,coordinate_t k1, item=1):
        self.c_add(k0,k1, item)
    def __add__(VecF2 self, VecF2 other):
        cdef c_CItemF2 c1, c2
        cdef unsigned int i1, i2
        cdef int k, delta
        cdef VecF2 result
        self.ensure_compact()
        other.ensure_compact()
        result=VecF2()
        result.vec.reserve(max(self.vec.size(),other.vec.size()))
        i1=0
        i2=0
        while i1<self.vec.size() and i2<other.vec.size():
            c1=self.vec.at(i1)
            c2=other.vec.at(i2)
            delta=0
            for k from 0<=k<2:
                if c1.addr[k]<c2.addr[k]:
                    delta=-1
                    break
                elif c1.addr[k]>c2.addr[k]:
                    delta=1
                    break
            if delta==0:
                c1.item+=c2.item
                result.vec.push_back(c1)
                i1+=1
                i2+=1
            elif delta==-1:
                result.vec.push_back(c1)
                i1+=1
            else:
                result.vec.push_back(c2)
                i2+=1
        while i1<self.vec.size():
            c1=self.vec.at(i1)
            result.vec.push_back(c1)
            i1+=1
        while i2<other.vec.size():
            c2=other.vec.at(i2)
            result.vec.push_back(c2)
            i2+=1
        return result

    cpdef CSRMatrixF to_csr(self):
        self.ensure_compact()
        cdef c_CSRMatrixF *matC=vec2csrF(&self.vec)
        cdef CSRMatrixF mat=CSRMatrixF()
        mat.set_matrix(matC)
        return mat

    def tofile(self, f):
        cdef long i, n, k, k_max
        n=self.get_size()
        f.write('F2 %d\n'%(n,))
        k=0
        for i from 0<=i<n/CHUNK_SIZE:
            s=PyBytes_FromStringAndSize(<char *>c_VecF2_get_pointer(&self.vec,k), CHUNK_SIZE*sizeof(c_CItemF2))
            f.write(s)
            k+=CHUNK_SIZE
        if k<n:
            s=PyBytes_FromStringAndSize(<char *>c_VecF2_get_pointer(&self.vec,k), (n-k)*sizeof(c_CItemF2))
            f.write(s)
    def fromfile(self, f):
        cdef long i, n
        line=f.readline().strip().split()
        assert line[0]=='F2'
        n=long(line[1])
        self.clear()
        self.vec.resize(n)
        k=0
        for i from 0<=i<n/CHUNK_SIZE:
            s=f.read(CHUNK_SIZE*sizeof(c_CItemF2))
            assert len(s)==CHUNK_SIZE*sizeof(c_CItemF2)
            memcpy(<void *>c_VecF2_get_pointer(&self.vec,k), <char *>s, CHUNK_SIZE*sizeof(c_CItemF2))
            k+=CHUNK_SIZE
        if k<n:
            s=f.read((n-k)*sizeof(c_CItemF2))
            assert len(s)==(n-k)*sizeof(c_CItemF2)
            memcpy(<void *>c_VecF2_get_pointer(&self.vec,k), <char *>s, (n-k)*sizeof(c_CItemF2))
    cpdef int get_maxcol(self, int k=0):
        cdef int i, n, new_val, col
        assert k>=0 and k<2
        n=self.vec.size()
        if n==0:
            return -1
        if k==0:
            return self.vec.at(n-1).addr[0]
        col=-1
        n=self.vec.size()
        for i from 0<=i<n:
            new_val=self.vec.at(i).addr[k]
            if new_val>col:
                col=new_val
        return col
    cpdef numpy.ndarray get_type_counts(self, int k=0):
        cdef int old_val, new_val, i, n, count
        cdef numpy.ndarray[numpy.int32_t, ndim=1] result
        assert k>=0 and k<2
        if self.vec.size()==0:
            return numpy.zeros(0, numpy.int32)
        self.ensure_compact()
        n_types=self.get_maxcol(k)+1
        result=numpy.zeros(n_types,numpy.int32)
        n=self.vec.size()
        for i from 0<=i<n:
            new_val=self.vec.at(i).addr[k]
            result[new_val]+=1
        return result
    cpdef VecF2 remap(self, int k, numpy.ndarray filt):
        cdef numpy.ndarray[numpy.int8_t, ndim=1] wanted=filt.astype('b')
        cdef numpy.ndarray[numpy.int_t, ndim=1] target=filt.cumsum()-1
        self.ensure_compact()
        cdef int n=self.vec.size()
        cdef int i
        cdef VecF2 result=VecF2()
        cdef c_CItemF2 c
        for i from 0<=i<n:
            c=self.vec.at(i)
            if wanted[c.addr[k]]:
                c.addr[k]=target[c.addr[k]]
                result.vec.push_back(c)
        return result
    def append(self,tup):
        k0,k1,val=tup
        self.c_add(k0,k1,val)
    def extend(self,tups):
        for k0,k1,val in tups:
            self.c_add(k0,k1,val)
    def __iter__(self):
        return self.item_iter()
    def __reduce__(self):
        return (VecF2,(),(),self.item_iter())
    def __setstate__(self,state):
        pass
    def __repr__(self):
        return 'VecF2(%s)'%(str(list(self)),)
    def __dealloc__(self):
        # need to call destructor explicitly
        self.vec.cxx_destructor()

cdef class IVecF2_iter:
    cdef VecF2 vec
    cdef c_VecF2 *vecC
    cdef unsigned int k
    def __init__(self,vec):
        self.vec=vec
        self.vecC=&self.vec.vec
        self.k=0
    def __iter__(self):
        return self
    def __next__(self):
        if self.k>=self.vecC.size():
            raise StopIteration
        cdef c_CItemF2 res=self.vecC.at(self.k)
        self.k+=1
        return (res.addr[0],
                res.addr[1],
                res.item)

cdef class LargeVecF2:
    cdef public object compact
    cdef public VecF2 loose
    def __init__(self):
        self.compact=[]
        self.loose=VecF2()
        self.loose.vec.reserve(100000)
    def ensure_compact(self):
        if self.loose.get_size()>0:
            self.compact.append(self.loose)
        self.loose=VecF2()
        while len(self.compact)>=2:
            self.compact[-2:]=[self.compact[-2]+self.compact[-1]]
    def add_count(self, coordinate_t k0,coordinate_t k1, item=1):
        self.loose.c_add(k0,k1, item)
        if self.loose.get_size(False)>=100000:
            self.loose.ensure_compact()
            self.compact.append(self.loose)
            self.loose=VecF2()
            self.loose.vec.reserve(100000)
            while (len(self.compact)>=2 and
                   self.compact[-1].get_size()>=0.6*self.compact[-2].get_size()):
                self.compact[-2:]=[self.compact[-2]+self.compact[-1]]
    def get_compact(self):
        self.ensure_compact()
        if len(self.compact)>0:
            return self.compact[0]
        else:
            return VecF2()
    def get_size(self, compactify=True):
        cdef int n
        if compactify:
            self.ensure_compact()
            if self.compact:
                return self.compact[0].get_size()
            else:
                return 0
        else:
            n=sum([c.get_size(False) for c in self.compact])
            n+=self.loose.get_size(False)
            return n
    def __iter__(self):
        self.ensure_compact()
        try:
            return iter(self.compact[0])
        except IndexError:
            return iter([])

    cpdef CSRMatrixF to_csr(self):
        self.ensure_compact()
        return self.compact[0].to_csr()

    def get_type_counts(self, k=0):
        return self.get_compact().get_type_counts(k)
    def remap(self, k, filt):
        return self.get_compact().remap(k,filt)


cdef class IVecF3_iter

cdef class VecF3:
    """mutable sparse int matrix based on item vector"""
    def __init__(self):
        self.is_compact=True
    def item_iter(self):
        self.ensure_compact()
        return IVecF3_iter(self)
    def get_size(self,compactify=True):
        if compactify:
            self.ensure_compact()
        return self.vec.size()
    def __len__(self):
        return self.get_size(True)
    def clear(self):
        self.vec.resize(0)
        self.is_compact=True
    cdef void compactify(self):
        cdef c_SmallerAddrF3 comp
        c_IVecF3_sort(self.vec.begin(),self.vec.end(),comp)
        c_compactifyF3(&self.vec)
        self.is_compact=True
    cdef void ensure_compact(self):
        if not self.is_compact:
            self.compactify()
    cpdef float get_count(self,coordinate_t k0,coordinate_t k1,coordinate_t k2):
        cdef c_CItemF3 ci
        ci.addr[0]=k0
        ci.addr[1]=k1
        ci.addr[2]=k2
        return c_get_countF3(&self.vec,ci)
    cdef void c_add(self, coordinate_t k0,coordinate_t k1,coordinate_t k2, float item):
        cdef c_CItemF3 c
        c.addr[0]=k0
        c.addr[1]=k1
        c.addr[2]=k2
        c.item=item
        self.vec.push_back(c)
        self.is_compact=False
    def add_count(self, coordinate_t k0,coordinate_t k1,coordinate_t k2, item=1):
        self.c_add(k0,k1,k2, item)
    def __add__(VecF3 self, VecF3 other):
        cdef c_CItemF3 c1, c2
        cdef unsigned int i1, i2
        cdef int k, delta
        cdef VecF3 result
        self.ensure_compact()
        other.ensure_compact()
        result=VecF3()
        result.vec.reserve(max(self.vec.size(),other.vec.size()))
        i1=0
        i2=0
        while i1<self.vec.size() and i2<other.vec.size():
            c1=self.vec.at(i1)
            c2=other.vec.at(i2)
            delta=0
            for k from 0<=k<3:
                if c1.addr[k]<c2.addr[k]:
                    delta=-1
                    break
                elif c1.addr[k]>c2.addr[k]:
                    delta=1
                    break
            if delta==0:
                c1.item+=c2.item
                result.vec.push_back(c1)
                i1+=1
                i2+=1
            elif delta==-1:
                result.vec.push_back(c1)
                i1+=1
            else:
                result.vec.push_back(c2)
                i2+=1
        while i1<self.vec.size():
            c1=self.vec.at(i1)
            result.vec.push_back(c1)
            i1+=1
        while i2<other.vec.size():
            c2=other.vec.at(i2)
            result.vec.push_back(c2)
            i2+=1
        return result

    def tofile(self, f):
        cdef long i, n, k, k_max
        n=self.get_size()
        f.write('F3 %d\n'%(n,))
        k=0
        for i from 0<=i<n/CHUNK_SIZE:
            s=PyBytes_FromStringAndSize(<char *>c_VecF3_get_pointer(&self.vec,k), CHUNK_SIZE*sizeof(c_CItemF3))
            f.write(s)
            k+=CHUNK_SIZE
        if k<n:
            s=PyBytes_FromStringAndSize(<char *>c_VecF3_get_pointer(&self.vec,k), (n-k)*sizeof(c_CItemF3))
            f.write(s)
    def fromfile(self, f):
        cdef long i, n
        line=f.readline().strip().split()
        assert line[0]=='F3'
        n=long(line[1])
        self.clear()
        self.vec.resize(n)
        k=0
        for i from 0<=i<n/CHUNK_SIZE:
            s=f.read(CHUNK_SIZE*sizeof(c_CItemF3))
            assert len(s)==CHUNK_SIZE*sizeof(c_CItemF3)
            memcpy(<void *>c_VecF3_get_pointer(&self.vec,k), <char *>s, CHUNK_SIZE*sizeof(c_CItemF3))
            k+=CHUNK_SIZE
        if k<n:
            s=f.read((n-k)*sizeof(c_CItemF3))
            assert len(s)==(n-k)*sizeof(c_CItemF3)
            memcpy(<void *>c_VecF3_get_pointer(&self.vec,k), <char *>s, (n-k)*sizeof(c_CItemF3))
    cpdef int get_maxcol(self, int k=0):
        cdef int i, n, new_val, col
        assert k>=0 and k<3
        n=self.vec.size()
        if n==0:
            return -1
        if k==0:
            return self.vec.at(n-1).addr[0]
        col=-1
        n=self.vec.size()
        for i from 0<=i<n:
            new_val=self.vec.at(i).addr[k]
            if new_val>col:
                col=new_val
        return col
    cpdef numpy.ndarray get_type_counts(self, int k=0):
        cdef int old_val, new_val, i, n, count
        cdef numpy.ndarray[numpy.int32_t, ndim=1] result
        assert k>=0 and k<3
        if self.vec.size()==0:
            return numpy.zeros(0, numpy.int32)
        self.ensure_compact()
        n_types=self.get_maxcol(k)+1
        result=numpy.zeros(n_types,numpy.int32)
        n=self.vec.size()
        for i from 0<=i<n:
            new_val=self.vec.at(i).addr[k]
            result[new_val]+=1
        return result
    cpdef VecF3 remap(self, int k, numpy.ndarray filt):
        cdef numpy.ndarray[numpy.int8_t, ndim=1] wanted=filt.astype('b')
        cdef numpy.ndarray[numpy.int_t, ndim=1] target=filt.cumsum()-1
        self.ensure_compact()
        cdef int n=self.vec.size()
        cdef int i
        cdef VecF3 result=VecF3()
        cdef c_CItemF3 c
        for i from 0<=i<n:
            c=self.vec.at(i)
            if wanted[c.addr[k]]:
                c.addr[k]=target[c.addr[k]]
                result.vec.push_back(c)
        return result
    def append(self,tup):
        k0,k1,k2,val=tup
        self.c_add(k0,k1,k2,val)
    def extend(self,tups):
        for k0,k1,k2,val in tups:
            self.c_add(k0,k1,k2,val)
    def __iter__(self):
        return self.item_iter()
    def __reduce__(self):
        return (VecF3,(),(),self.item_iter())
    def __setstate__(self,state):
        pass
    def __repr__(self):
        return 'VecF3(%s)'%(str(list(self)),)
    def __dealloc__(self):
        # need to call destructor explicitly
        self.vec.cxx_destructor()

cdef class IVecF3_iter:
    cdef VecF3 vec
    cdef c_VecF3 *vecC
    cdef unsigned int k
    def __init__(self,vec):
        self.vec=vec
        self.vecC=&self.vec.vec
        self.k=0
    def __iter__(self):
        return self
    def __next__(self):
        if self.k>=self.vecC.size():
            raise StopIteration
        cdef c_CItemF3 res=self.vecC.at(self.k)
        self.k+=1
        return (res.addr[0],
                res.addr[1],
                res.addr[2],
                res.item)

cdef class LargeVecF3:
    cdef public object compact
    cdef public VecF3 loose
    def __init__(self):
        self.compact=[]
        self.loose=VecF3()
        self.loose.vec.reserve(100000)
    def ensure_compact(self):
        if self.loose.get_size()>0:
            self.compact.append(self.loose)
        self.loose=VecF3()
        while len(self.compact)>=2:
            self.compact[-2:]=[self.compact[-2]+self.compact[-1]]
    def add_count(self, coordinate_t k0,coordinate_t k1,coordinate_t k2, item=1):
        self.loose.c_add(k0,k1,k2, item)
        if self.loose.get_size(False)>=100000:
            self.loose.ensure_compact()
            self.compact.append(self.loose)
            self.loose=VecF3()
            self.loose.vec.reserve(100000)
            while (len(self.compact)>=2 and
                   self.compact[-1].get_size()>=0.6*self.compact[-2].get_size()):
                self.compact[-2:]=[self.compact[-2]+self.compact[-1]]
    def get_compact(self):
        self.ensure_compact()
        if len(self.compact)>0:
            return self.compact[0]
        else:
            return VecF3()
    def get_size(self, compactify=True):
        cdef int n
        if compactify:
            self.ensure_compact()
            if self.compact:
                return self.compact[0].get_size()
            else:
                return 0
        else:
            n=sum([c.get_size(False) for c in self.compact])
            n+=self.loose.get_size(False)
            return n
    def __iter__(self):
        self.ensure_compact()
        try:
            return iter(self.compact[0])
        except IndexError:
            return iter([])

    def get_type_counts(self, k=0):
        return self.get_compact().get_type_counts(k)
    def remap(self, k, filt):
        return self.get_compact().remap(k,filt)

cdef class SparseVectorF:
    def __init__(self, pairs=None):
        cdef coordinate_t i
        if pairs is not None:
            self.my_len=len(pairs)
            self.idx_ptr=<coordinate_t *>PyMem_Malloc(self.my_len*sizeof(coordinate_t))
            self.vals_ptr=<float *>PyMem_Malloc(self.my_len*sizeof(float))
            for i from 0<=i<self.my_len:
                x,y=pairs[i]
                self.idx_ptr[i]=x
                self.vals_ptr[i]=y
        else:
            self.my_len=0
            self.idx_ptr=NULL
            self.vals_ptr=NULL
    cpdef int from_dense(self, float[:] dense):
        cdef int i, k
        assert self.my_len == 0
        k = 0
        for i from 0<=i<dense.shape[0]:
            if dense[i]!=0.0:
                k += 1
        self.my_len = k
        self.idx_ptr=<coordinate_t *>PyMem_Malloc(self.my_len*sizeof(coordinate_t))
        self.vals_ptr=<float *>PyMem_Malloc(self.my_len*sizeof(float))
        k = 0
        for i from 0<=i<dense.shape[0]:
            if dense[i] != 0.0:
                self.idx_ptr[k] = i
                self.vals_ptr[k] = dense[i]
                k += 1
        return k
    cpdef float dotSelf(self):
        cdef float s=0
        cdef coordinate_t i
        for i from 0<=i<self.my_len:
            s+=self.vals_ptr[i]*self.vals_ptr[i]
        return s
    cdef float _dotFull(self, const_float_ptr full_ptr):
        cdef float s=0
        cdef coordinate_t i
        for i from 0<=i<self.my_len:
            s+=self.vals_ptr[i]*full_ptr[self.idx_ptr[i]]
        return s
    cdef float _dotFull_partial(self, const_float_ptr full_ptr, int my_len):
        cdef float s=0
        cdef coordinate_t i
        for i from 0<=i<my_len:
            s+=self.vals_ptr[i]*full_ptr[self.idx_ptr[i]]
        return s
    cpdef double jaccard(self, SparseVectorF other):
        cdef double s_max=0
        cdef double s_min=0
        cdef float val1, val2
        cdef coordinate_t i,j, idx_i, idx_j
        i=0
        j=0
        while i<self.my_len and j<other.my_len:
            idx_i=self.idx_ptr[i]
            idx_j=other.idx_ptr[j]
            if idx_i<idx_j:
                s_max+=self.vals_ptr[i]
                i+=1
            elif idx_i>idx_j:
                s_max+=other.vals_ptr[j]
                j+=1
            else:
                val1=self.vals_ptr[i]
                val2=other.vals_ptr[j]
                if val1>val2:
                    s_max+=val1
                    s_min+=val2
                else:
                    s_max+=val2
                    s_min+=val1
                i+=1
                j+=1
        if i<self.my_len:
            while i<self.my_len:
                s_max+=self.vals_ptr[i]
                i+=1
        else:
            while j<other.my_len:
                s_max+=other.vals_ptr[j]
                j+=1
        if s_max==0:
            return 0.0
        else:
            return s_min/s_max
    cpdef float dotSparse(self, SparseVectorF other):
        cdef float product=0
        cdef float val1, val2
        cdef coordinate_t i, j, idx_i, idx_j
        i=0
        j=0
        while i<self.my_len and j<other.my_len:
           idx_i=self.idx_ptr[i]
           idx_j=other.idx_ptr[j]
           if idx_i<idx_j:
              i+=1
           elif idx_i>idx_j:
              j+=1
           else:
              val1=self.vals_ptr[i]
              val2=other.vals_ptr[j]
              product+=val1*val2
              i+=1
              j+=1
        return product
    def count_intersection(self, SparseVectorF other):
        cdef coordinate_t i, j, idx_i, idx_j
        cdef int n_ab=0
        i=0
        j=0
        while i<self.my_len and j<other.my_len:
           idx_i=self.idx_ptr[i]
           idx_j=other.idx_ptr[j]
           if idx_i<idx_j:
              i+=1
           elif idx_i>idx_j:
              j+=1
           else:
              i+=1
              j+=1
              n_ab+=1
        return n_ab
    cpdef double min_sum(self, SparseVectorF other):
        cdef double product=0.0
        cdef float val1, val2
        cdef coordinate_t i, j, idx_i, idx_j
        i=0
        j=0
        while i<self.my_len and j<other.my_len:
           idx_i=self.idx_ptr[i]
           idx_j=other.idx_ptr[j]
           if idx_i<idx_j:
              i+=1
           elif idx_i>idx_j:
              j+=1
           else:
              val1=self.vals_ptr[i]
              val2=other.vals_ptr[j]
              if val1<val2:
                  product+=val1
              else:
                  product+=val2
              i+=1
              j+=1
        return product
    cpdef double cosine(self, SparseVectorF other):
        cdef double sqsum_self=0.0
        cdef double sqsum_other=0.0
        cdef double product=0.0
        cdef float val1, val2
        cdef coordinate_t i, j, idx_i, idx_j
        i=0
        j=0
        while i<self.my_len and j<other.my_len:
           idx_i=self.idx_ptr[i]
           idx_j=other.idx_ptr[j]
           if idx_i<idx_j:
              val1=self.vals_ptr[i]
              sqsum_self+=val1*val1
              i+=1
           elif idx_i>idx_j:
              val2=other.vals_ptr[j]
              sqsum_other+=val2*val2
              j+=1
           else:
              val1=self.vals_ptr[i]
              val2=other.vals_ptr[j]
              sqsum_self+=val1*val1
              sqsum_other+=val2*val2
              product+=val1*val2
              i+=1
              j+=1
        if i<self.my_len:
            while i<self.my_len:
                val1=self.vals_ptr[i]
                sqsum_self+=val1*val1
                i+=1
        else:
            while j<other.my_len:
                val2=other.vals_ptr[j]
                sqsum_other+=val2*val2
                j+=1
        return product/sqrt(sqsum_self*sqsum_other)
    cpdef double jsd_unnorm(self, SparseVectorF other):
        cdef double sum=0.0
        cdef double val1, val2, avg
        cdef coordinate_t i, j, idx_i, idx_j
        i=0
        j=0
        while i<self.my_len and j<other.my_len:
           idx_i=self.idx_ptr[i]
           idx_j=other.idx_ptr[j]
           if idx_i<idx_j:
               val1=self.vals_ptr[i]
               sum+=val1*M_LN2
               i+=1
           elif idx_i>idx_j:
               val2=other.vals_ptr[j]
               sum+=val2*M_LN2
               j+=1
           else:
               val1=self.vals_ptr[i]
               val2=other.vals_ptr[j]
               i+=1
               j+=1
               avg=(val1+val2)/2.0
               sum+=val1*log(val1/avg)+val2*log(val2/avg)
        while i<self.my_len:
           val1=self.vals_ptr[i]
           i+=1
           sum+=val1*M_LN2
        while j<other.my_len:
           val2=other.vals_ptr[j]
           j+=1
           sum+=val2*M_LN2
        return sum/(2.0*M_LN2)
    cpdef double skew_unnorm(self, SparseVectorF other, double alpha):
        cdef double sum=0.0
        cdef double beta=1.0-alpha
        cdef double val1, val2, avg
        cdef coordinate_t i, j, idx_i, idx_j
        cdef double log_invbeta=log(1.0/beta)
        i=0
        j=0
        while i<self.my_len and j<other.my_len:
           idx_i=self.idx_ptr[i]
           idx_j=self.idx_ptr[j]
           if idx_i<idx_j:
               val1=self.vals_ptr[i]
               sum+=val1*log_invbeta
               i+=1
           elif idx_i>idx_j:
               j+=1
           else:
               val1=self.vals_ptr[i]
               val2=other.vals_ptr[j]
               avg=alpha*val2+beta*val1
               sum+=log(val1/avg)
               i+=1
               j+=1
        return sum
    cpdef double norm_l1(self):
        cdef double sum=0.0
        cdef unsigned int i
        for i from 0<=i<self.my_len:
            sum+=fabs(self.vals_ptr[i])
        return sum
    cpdef double norm_l2(self):
        cdef double sum=0.0
        cdef double val
        cdef unsigned int i
        for i from 0<=i<self.my_len:
            val=self.vals_ptr[i];
            sum+=val*val
        return sqrt(sum)
    cpdef double norm_lp(self, double p):
        cdef double sum=0.0
        cdef int i
        for i from 0<=i<self.my_len:
            sum+=pow(fabs(self.vals_ptr[i]),p)
        return pow(sum,1.0/p)
    def dotFull(self, numpy.ndarray[float,ndim=1] full):
        cdef float *full_ptr
        cdef unsigned int my_len=self.my_len
        cdef unsigned int full_len=len(full)
        # treat nonpresent values as 0
        while my_len>0 and self.idx_ptr[my_len-1]>full_len:
            my_len-=1
        full_ptr=<float*>full.data
        return self._dotFull_partial(full_ptr,my_len)
    def dotFull_check(self, numpy.ndarray[float,ndim=1] full):
        cdef float *full_ptr
        cdef int my_len=self.my_len
        cdef int full_len=len(full)
        # boundary check
        full[self.idx_ptr[self.my_len-1]]
        full_ptr=<float*>full.data
        return self._dotFull(full_ptr)
    def dotMatrix(self, numpy.ndarray[float,ndim=2] full):
        cdef numpy.ndarray[float,ndim=1] result
        result = numpy.zeros(full.shape[1], numpy.float32)
        for i from 0<=i<self.my_len:
            result += full[self.idx_ptr[i]] * self.vals_ptr[i]
        return result
    cdef void _axpy(self, float *x_ptr, float a):
        for i from 0<=i<self.my_len:
            x_ptr[self.idx_ptr[i]]+=a*self.vals_ptr[i]
    def axpy(self, numpy.ndarray[float,ndim=1] x, float a=1):
        # boundary check
        x[self.idx_ptr[self.my_len-1]]
        self._axpy(<float *>x.data,a)
    cpdef float sqdist(self, SparseVectorF other):
        """computes ||x-y||^2"""
        cdef float s=0
        cdef float val
        cdef int i1=0,i2=0
        while i1<self.my_len and i2<other.my_len:
            if self.idx_ptr[i1]<other.idx_ptr[i2]:
                val=self.vals_ptr[i1]
                s+=val*val
                i1+=1
            elif self.idx_ptr[i1]>other.idx_ptr[i2]:
                val=other.vals_ptr[i2]
                s+=val*val
                i2+=1
            else:
                val=self.vals_ptr[i1]-other.vals_ptr[i2]
                s+=val*val
                i1+=1
                i2+=1
        while i1<self.my_len:
            val=self.vals_ptr[i1]
            s+=val*val
            i1+=1
        while i2<other.my_len:
            val=other.vals_ptr[i2]
            s+=val*val
            i2+=1
        return s
    def write_pairs(self, f, delim=':'):
        cdef int i
        w_func=f.write
        for i from 0<=i<self.my_len:
            # svmlight does not want 0 as index
            w_func(' %d%s%s'%(self.idx_ptr[i]+1,delim,self.vals_ptr[i]))
    def __imul__(self, float a):
        cdef coordinate_t i
        for i from 0<=i<self.my_len:
            self.vals_ptr[i] *= a
        return self
    def __idiv__(self, float a):
        cdef coordinate_t i
        for i from 0<=i<self.my_len:
            self.vals_ptr[i] /= a
        return self
    cpdef SparseVectorF min_vals(self, SparseVectorF other):
        cdef SparseVectorF result
        cdef coordinate_t i1, i2, k, idx1, idx2
        result=SparseVectorF()
        if self.my_len<other.my_len:
            result.my_len=self.my_len
        else:
            result.my_len=other.my_len
        result.idx_ptr=<coordinate_t *>PyMem_Malloc(result.my_len*sizeof(coordinate_t))
        result.vals_ptr=<float *>PyMem_Malloc(result.my_len*sizeof(float))
        i1=i2=k=0
        while i1<self.my_len and i2<other.my_len:
           idx1=self.idx_ptr[i1]
           idx2=other.idx_ptr[i2]
           if idx1<idx2:
               i1+=1
           elif idx1>idx2:
               i2+=1
           else:
               result.idx_ptr[k]=idx1
               val1=self.vals_ptr[i1]
               val2=other.vals_ptr[i2]
               if val1<val2:
                   result.vals_ptr[k]=val1
               else:
                   result.vals_ptr[k]=val2
               i1+=1
               i2+=1
               k+=1
        result.my_len=k
        return result
    def scale_array(SparseVectorF self, numpy.ndarray[float, ndim=1] a):
        cdef SparseVectorF result
        cdef coordinate_t i
        result=SparseVectorF()
        result.my_len=self.my_len
        result.idx_ptr=<coordinate_t *>PyMem_Malloc(result.my_len*sizeof(coordinate_t))
        result.vals_ptr=<float *>PyMem_Malloc(result.my_len*sizeof(float))
        for i from 0<=i<self.my_len:
            result.idx_ptr[i]=self.idx_ptr[i]
        for i from 0<=i<self.my_len:
            result.vals_ptr[i]=self.vals_ptr[i]*a[self.idx_ptr[i]]
        return result
    def __div__(SparseVectorF self, float a):
        cdef SparseVectorF result
        cdef coordinate_t i
        result=SparseVectorF()
        result.my_len=self.my_len
        result.idx_ptr=<coordinate_t *>PyMem_Malloc(result.my_len*sizeof(coordinate_t))
        result.vals_ptr=<float *>PyMem_Malloc(result.my_len*sizeof(float))
        for i from 0<=i<self.my_len:
            result.idx_ptr[i]=self.idx_ptr[i]
        for i from 0<=i<self.my_len:
            result.vals_ptr[i]=self.vals_ptr[i]/a
        return result
    def __len__(self):
        return self.my_len
    def __add__(SparseVectorF self,SparseVectorF other):
        cdef coordinate_t *new_idx
        cdef float *new_vals
        cdef coordinate_t i1,i2,i_all
        cdef coordinate_t n1,n2,n_all
        cdef coordinate_t k1,k2,k
        cdef SparseVectorF result
        # first pass: determine how much space we need
        n1=self.my_len
        n2=other.my_len
        n_all=i1=i2=0
        while i1<n1 and i2<n2:
            k1=self.idx_ptr[i1]
            k2=other.idx_ptr[i2]
            if k1>=k2: i2+=1
            if k2>=k1: i1+=1
            n_all+=1
        if i1<n1: n_all+=n1-i1
        if i2<n2: n_all+=n2-i2
        new_idx=<coordinate_t *>PyMem_Malloc(n_all*sizeof(coordinate_t))
        new_vals=<float *>PyMem_Malloc(n_all*sizeof(float))
        # second pass: fill in the actual values
        i1=i2=i_all=0
        while i1<n1 and i2<n2:
            k1=self.idx_ptr[i1]
            k2=other.idx_ptr[i2]
            if k1>=k2:
                new_idx[i_all]=k2
                new_vals[i_all]=other.vals_ptr[i2]
                i2+=1
                if k1==k2:
                    new_vals[i_all]+=self.vals_ptr[i1]
                    i1+=1
            elif k2>=k1:
                new_idx[i_all]=k1
                new_vals[i_all]=self.vals_ptr[i1]
                i1+=1
            i_all+=1
        while i1<n1:
            new_idx[i_all]=self.idx_ptr[i1]
            new_vals[i_all]=self.vals_ptr[i1]
            i1+=1
            i_all+=1
        else:
            while i2<n2:
                new_idx[i_all]=other.idx_ptr[i2]
                new_vals[i_all]=other.vals_ptr[i2]
                i2+=1
                i_all+=1
        assert i_all==n_all
        result=SparseVectorF()
        result.my_len=n_all
        result.idx_ptr=new_idx
        result.vals_ptr=new_vals
        return result
    def __sub__(SparseVectorF self,SparseVectorF other):
        cdef coordinate_t *new_idx
        cdef float *new_vals
        cdef coordinate_t i1,i2,i_all
        cdef coordinate_t n1,n2,n_all
        cdef coordinate_t k1,k2,k
        cdef SparseVectorF result
        # first pass: determine how much space we need
        n1=self.my_len
        n2=other.my_len
        n_all=i1=i2=0
        while i1<n1 and i2<n2:
            k1=self.idx_ptr[i1]
            k2=other.idx_ptr[i2]
            if k1>=k2: i2+=1
            if k2>=k1: i1+=1
            n_all+=1
        if i1<n1: n_all+=n1-i1
        if i2<n2: n_all+=n2-i2
        new_idx=<coordinate_t *>PyMem_Malloc(n_all*sizeof(coordinate_t))
        new_vals=<float *>PyMem_Malloc(n_all*sizeof(float))
        # second pass: fill in the actual values
        i1=i2=i_all=0
        while i1<n1 and i2<n2:
            k1=self.idx_ptr[i1]
            k2=other.idx_ptr[i2]
            if k1>=k2:
                new_idx[i_all]=k2
                new_vals[i_all]=-other.vals_ptr[i2]
                i2+=1
                if k1==k2:
                    new_vals[i_all]+=self.vals_ptr[i1]
                    i1+=1
            elif k2>=k1:
                new_idx[i_all]=k1
                new_vals[i_all]=self.vals_ptr[i1]
                i1+=1
            i_all+=1
        while i1<n1:
            new_idx[i_all]=self.idx_ptr[i1]
            new_vals[i_all]=self.vals_ptr[i1]
            i1+=1
            i_all+=1
        else:
            while i2<n2:
                new_idx[i_all]=other.idx_ptr[i2]
                new_vals[i_all]=-other.vals_ptr[i2]
                i2+=1
                i_all+=1
        assert i_all==n_all
        result=SparseVectorF()
        result.my_len=n_all
        result.idx_ptr=new_idx
        result.vals_ptr=new_vals
        return result
    def __or__(SparseVectorF self,SparseVectorF other):
        cdef coordinate_t *new_idx
        cdef float *new_vals
        cdef coordinate_t i1,i2,i_all
        cdef coordinate_t n1,n2,n_all
        cdef coordinate_t k1,k2,k
        cdef SparseVectorF result
        cdef float val
        # first pass: determine how much space we need
        n1=self.my_len
        n2=other.my_len
        n_all=i1=i2=0
        while i1<n1 and i2<n2:
            k1=self.idx_ptr[i1]
            k2=other.idx_ptr[i2]
            if k1>=k2: i2+=1
            if k2>=k1: i1+=1
            n_all+=1
        if i1<n1: n_all+=n1-i1
        if i2<n2: n_all+=n2-i2
        new_idx=<coordinate_t *>PyMem_Malloc(n_all*sizeof(coordinate_t))
        new_vals=<float *>PyMem_Malloc(n_all*sizeof(float))
        # second pass: fill in the actual values
        i1=i2=i_all=0
        while i1<n1 and i2<n2:
            k1=self.idx_ptr[i1]
            k2=other.idx_ptr[i2]
            if k1>=k2:
                new_idx[i_all]=k2
                new_vals[i_all]=other.vals_ptr[i2]
                i2+=1
                if k1==k2:
                    val=self.vals_ptr[i1]
                    if new_vals[i_all]<val:
                        new_vals[i_all]=val
                    i1+=1
            else: # k2>k1:
                new_idx[i_all]=k1
                new_vals[i_all]=self.vals_ptr[i1]
                i1+=1
            i_all+=1
        while i1<n1:
            new_idx[i_all]=self.idx_ptr[i1]
            new_vals[i_all]=self.vals_ptr[i1]
            i1+=1
            i_all+=1
        else:
            while i2<n2:
                new_idx[i_all]=other.idx_ptr[i2]
                new_vals[i_all]=other.vals_ptr[i2]
                i2+=1
                i_all+=1
        assert i_all==n_all
        result=SparseVectorF()
        result.my_len=n_all
        result.idx_ptr=new_idx
        result.vals_ptr=new_vals
        return result
    def __lshift__(self, int k):
        cdef SparseVectorF result
        cdef coordinate_t i
        result=SparseVectorF()
        result.my_len=self.my_len
        result.idx_ptr=<coordinate_t *>PyMem_Malloc(result.my_len*sizeof(coordinate_t))
        result.vals_ptr=<float *>PyMem_Malloc(result.my_len*sizeof(float))
        for i from 0<=i<self.my_len:
            result.idx_ptr[i]=self.idx_ptr[i]+k
        for i from 0<=i<self.my_len:
            result.vals_ptr[i]=self.vals_ptr[i]
        return result
    def __ilshift__(self, int k):
        cdef SparseVectorF result
        cdef coordinate_t i
        for i from 0<=i<self.my_len:
            self.idx_ptr[i]+=k
        return self
    def __getitem__(self, int i):
        if i<0:
            i=self.my_len+i
        if i>=self.my_len or i<0:
            raise IndexError
        return (self.idx_ptr[i],self.vals_ptr[i])
    def __repr__(self):
        cdef unsigned int i
        ss=[]
        for i from 0<=i<self.my_len:
            ss.append('(%d,%s)'%(self.idx_ptr[i],self.vals_ptr[i]))
        return 'SparseVectorF([%s])'%(','.join(ss))
    def __reduce_ex__(self,protocol):
        if protocol==0:
            # choose compatibility over performance.
            return (SparseVectorF,(list(self),),())
        else:
            s_idx=PyBytes_FromStringAndSize(<char *>self.idx_ptr,self.my_len*sizeof(coordinate_t))
            s_vals=PyBytes_FromStringAndSize(<char *>self.vals_ptr,self.my_len*sizeof(float))
            return (SparseVectorF,(None,),(s_idx,s_vals))
    def __setstate__(self,state):
        cdef coordinate_t *p_idx
        cdef float *p_vals
        if len(state)==0:
            pass
        elif len(state)==2:
            assert (self.idx_ptr==NULL)
            (s_idx, s_vals)=state
            p_idx=<coordinate_t *>(<char *>s_idx)
            p_vals=<float *>(<char *>s_vals)
            self.my_len=len(s_idx)/sizeof(coordinate_t)
            self.idx_ptr=<coordinate_t *>PyMem_Malloc(self.my_len*sizeof(coordinate_t))
            self.vals_ptr=<float *>PyMem_Malloc(self.my_len*sizeof(float))
            for i from 0<=i<self.my_len:
                self.idx_ptr[i]=p_idx[i]
                self.vals_ptr[i]=p_vals[i]
    def __dealloc__(self):
        if self.buf is None:
            PyMem_Free(self.idx_ptr)
            PyMem_Free(self.vals_ptr)
        else:
            self.buf=None
        self.idx_ptr=<coordinate_t *>0
        self.vals_ptr=<float *>0
        self.my_len=0
    def to_scipy(self):
        cdef numpy.ndarray[numpy.float32_t, ndim=1] data
        cdef numpy.ndarray[numpy.int32_t, ndim=1] indices
        cdef numpy.ndarray[numpy.int32_t, ndim=1] indptr
        cdef int i,j
        indptr=numpy.zeros(2, numpy.int32)
        indices=numpy.zeros(self.my_len, numpy.int32)
        data=numpy.zeros(self.my_len, numpy.float32)
        indptr[1]=self.my_len
        for i from 0<=i<self.my_len:
            indices[i]=0
        for i from 0<=i<self.my_len:
            data[i]=self.vals_ptr[i]
        return (data,indices,indptr)

emptyvec_F=SparseVectorF([])

cdef class SparseVectorsF:
    def __init__(self):
       self.vecs = []
    cpdef SparseVectorF to_vec(self):
        cdef SparseVectorF vec
        cdef float weight
        cdef size_t i, j
        cdef VecF1 result = VecF1()
        for i from 0<=i<len(self.vecs):
            weight = self.weights.at(i)
            vec = self.vecs[i]
            for j from 0<=j<vec.my_len:
                result.add_count(vec.idx_ptr[j], weight*vec.vals_ptr[j])
        return result.to_sparse()
    cpdef add(self, SparseVectorF vec, float weight=1):
        self.vecs.append(vec)
        self.weights.push_back(weight)
    def __mul__(self, factor):
        cdef SparseVectorF vec
        cdef float weight
        cdef SparseVectorsF result = SparseVectorsF
        for i from 0<=i<len(self.vecs):
            weight = self.weights.at(i)
            vec = self.vecs[i]
            result.add(vec, weight*factor)
    def __add__(self, others):
        cdef SparseVectorF vec
        cdef float weight
        cdef SparseVectorsF oth_vecs = others
        cdef SparseVectorsF result = SparseVectorsF
        for i from 0<=i<len(self.vecs):
            weight = self.weights.at(i)
            vec = self.vecs[i]
            result.add(vec, weight)
        for i from 0<=i<len(oth_vecs.vecs):
            weight = oth_vecs.weights.at(i)
            vec = oth_vecs.vecs[i]
            result.add(vec, weight)
        return result
    def __sub__(SparseVectorsF self, SparseVectorsF others):
        cdef SparseVectorF vec
        cdef float weight
        cdef SparseVectorsF oth_vecs = others
        cdef SparseVectorsF result = SparseVectorsF()
        for i from 0<=i<len(self.vecs):
            weight = self.weights.at(i)
            vec = self.vecs[i]
            result.add(vec, weight)
        for i from 0<=i<len(oth_vecs.vecs):
            weight = oth_vecs.weights.at(i)
            vec = oth_vecs.vecs[i]
            result.add(vec, -weight)
        return result
    cdef float _dotFull(self, const_float_ptr full_ptr):
        cdef size_t i, j
        cdef float weight
        cdef SparseVectorF vec
        cdef float result=0
        for i from 0<=i<len(self.vecs):
            weight = self.weights.at(i)
            vec = self.vecs.at(i)
            for j from 0<=j<vec.my_len:
                result += full_ptr[vec.indices[j]]*weight*vec.vals_ptr[j]
        return result
    def __len__(self):
        return len(self.vecs)
    def __getitem__(self, k):
        vec = self.vecs[k]
        return (self.weights.at(k), vec)
## D -> double


cdef class IVecD1_iter

cdef class VecD1:
    """mutable sparse int matrix based on item vector"""
    def __init__(self):
        self.is_compact=True
    def item_iter(self):
        self.ensure_compact()
        return IVecD1_iter(self)
    def get_size(self,compactify=True):
        if compactify:
            self.ensure_compact()
        return self.vec.size()
    def __len__(self):
        return self.get_size(True)
    def clear(self):
        self.vec.resize(0)
        self.is_compact=True
    cdef void compactify(self):
        cdef c_SmallerAddrD1 comp
        c_IVecD1_sort(self.vec.begin(),self.vec.end(),comp)
        c_compactifyD1(&self.vec)
        self.is_compact=True
    cdef void ensure_compact(self):
        if not self.is_compact:
            self.compactify()
    cpdef double get_count(self,coordinate_t k0):
        cdef c_CItemD1 ci
        ci.addr[0]=k0
        return c_get_countD1(&self.vec,ci)
    cdef void c_add(self, coordinate_t k0, double item):
        cdef c_CItemD1 c
        c.addr[0]=k0
        c.item=item
        self.vec.push_back(c)
        self.is_compact=False
    def add_count(self, coordinate_t k0, item=1):
        self.c_add(k0, item)
    def __add__(VecD1 self, VecD1 other):
        cdef c_CItemD1 c1, c2
        cdef unsigned int i1, i2
        cdef int k, delta
        cdef VecD1 result
        self.ensure_compact()
        other.ensure_compact()
        result=VecD1()
        result.vec.reserve(max(self.vec.size(),other.vec.size()))
        i1=0
        i2=0
        while i1<self.vec.size() and i2<other.vec.size():
            c1=self.vec.at(i1)
            c2=other.vec.at(i2)
            delta=0
            for k from 0<=k<1:
                if c1.addr[k]<c2.addr[k]:
                    delta=-1
                    break
                elif c1.addr[k]>c2.addr[k]:
                    delta=1
                    break
            if delta==0:
                c1.item+=c2.item
                result.vec.push_back(c1)
                i1+=1
                i2+=1
            elif delta==-1:
                result.vec.push_back(c1)
                i1+=1
            else:
                result.vec.push_back(c2)
                i2+=1
        while i1<self.vec.size():
            c1=self.vec.at(i1)
            result.vec.push_back(c1)
            i1+=1
        while i2<other.vec.size():
            c2=other.vec.at(i2)
            result.vec.push_back(c2)
            i2+=1
        return result

    cpdef SparseVectorD to_sparse(self):
        cdef c_CItemD1 c
        cdef SparseVectorD result
        cdef coordinate_t n,i
        cdef coordinate_t *idx_ptr
        cdef double *vals_ptr
        self.ensure_compact()
        n=self.vec.size()
        idx_ptr=<coordinate_t *>PyMem_Malloc(n*sizeof(coordinate_t))
        vals_ptr=<double *>PyMem_Malloc(n*sizeof(double))
        for i from 0<=i<n:
          c=self.vec.at(i)
          idx_ptr[i]=c.addr[0]
          vals_ptr[i]=c.item
        result=SparseVectorD(None)
        result.idx_ptr=idx_ptr
        result.vals_ptr=vals_ptr
        result.my_len=n
        return result

    def tofile(self, f):
        cdef long i, n, k, k_max
        n=self.get_size()
        f.write('D1 %d\n'%(n,))
        k=0
        for i from 0<=i<n/CHUNK_SIZE:
            s=PyBytes_FromStringAndSize(<char *>c_VecD1_get_pointer(&self.vec,k), CHUNK_SIZE*sizeof(c_CItemD1))
            f.write(s)
            k+=CHUNK_SIZE
        if k<n:
            s=PyBytes_FromStringAndSize(<char *>c_VecD1_get_pointer(&self.vec,k), (n-k)*sizeof(c_CItemD1))
            f.write(s)
    def fromfile(self, f):
        cdef long i, n
        line=f.readline().strip().split()
        assert line[0]=='D1'
        n=long(line[1])
        self.clear()
        self.vec.resize(n)
        k=0
        for i from 0<=i<n/CHUNK_SIZE:
            s=f.read(CHUNK_SIZE*sizeof(c_CItemD1))
            assert len(s)==CHUNK_SIZE*sizeof(c_CItemD1)
            memcpy(<void *>c_VecD1_get_pointer(&self.vec,k), <char *>s, CHUNK_SIZE*sizeof(c_CItemD1))
            k+=CHUNK_SIZE
        if k<n:
            s=f.read((n-k)*sizeof(c_CItemD1))
            assert len(s)==(n-k)*sizeof(c_CItemD1)
            memcpy(<void *>c_VecD1_get_pointer(&self.vec,k), <char *>s, (n-k)*sizeof(c_CItemD1))
    cpdef int get_maxcol(self, int k=0):
        cdef int i, n, new_val, col
        assert k>=0 and k<1
        n=self.vec.size()
        if n==0:
            return -1
        if k==0:
            return self.vec.at(n-1).addr[0]
        col=-1
        n=self.vec.size()
        for i from 0<=i<n:
            new_val=self.vec.at(i).addr[k]
            if new_val>col:
                col=new_val
        return col
    cpdef numpy.ndarray get_type_counts(self, int k=0):
        cdef int old_val, new_val, i, n, count
        cdef numpy.ndarray[numpy.int32_t, ndim=1] result
        assert k>=0 and k<1
        if self.vec.size()==0:
            return numpy.zeros(0, numpy.int32)
        self.ensure_compact()
        n_types=self.get_maxcol(k)+1
        result=numpy.zeros(n_types,numpy.int32)
        n=self.vec.size()
        for i from 0<=i<n:
            new_val=self.vec.at(i).addr[k]
            result[new_val]+=1
        return result
    cpdef VecD1 remap(self, int k, numpy.ndarray filt):
        cdef numpy.ndarray[numpy.int8_t, ndim=1] wanted=filt.astype('b')
        cdef numpy.ndarray[numpy.int_t, ndim=1] target=filt.cumsum()-1
        self.ensure_compact()
        cdef int n=self.vec.size()
        cdef int i
        cdef VecD1 result=VecD1()
        cdef c_CItemD1 c
        for i from 0<=i<n:
            c=self.vec.at(i)
            if wanted[c.addr[k]]:
                c.addr[k]=target[c.addr[k]]
                result.vec.push_back(c)
        return result
    def append(self,tup):
        k0,val=tup
        self.c_add(k0,val)
    def extend(self,tups):
        for k0,val in tups:
            self.c_add(k0,val)
    def __iter__(self):
        return self.item_iter()
    def __reduce__(self):
        return (VecD1,(),(),self.item_iter())
    def __setstate__(self,state):
        pass
    def __repr__(self):
        return 'VecD1(%s)'%(str(list(self)),)
    def __dealloc__(self):
        # need to call destructor explicitly
        self.vec.cxx_destructor()

cdef class IVecD1_iter:
    cdef VecD1 vec
    cdef c_VecD1 *vecC
    cdef unsigned int k
    def __init__(self,vec):
        self.vec=vec
        self.vecC=&self.vec.vec
        self.k=0
    def __iter__(self):
        return self
    def __next__(self):
        if self.k>=self.vecC.size():
            raise StopIteration
        cdef c_CItemD1 res=self.vecC.at(self.k)
        self.k+=1
        return (res.addr[0],
                res.item)

cdef class LargeVecD1:
    cdef public object compact
    cdef public VecD1 loose
    def __init__(self):
        self.compact=[]
        self.loose=VecD1()
        self.loose.vec.reserve(100000)
    def ensure_compact(self):
        if self.loose.get_size()>0:
            self.compact.append(self.loose)
        self.loose=VecD1()
        while len(self.compact)>=2:
            self.compact[-2:]=[self.compact[-2]+self.compact[-1]]
    def add_count(self, coordinate_t k0, item=1):
        self.loose.c_add(k0, item)
        if self.loose.get_size(False)>=100000:
            self.loose.ensure_compact()
            self.compact.append(self.loose)
            self.loose=VecD1()
            self.loose.vec.reserve(100000)
            while (len(self.compact)>=2 and
                   self.compact[-1].get_size()>=0.6*self.compact[-2].get_size()):
                self.compact[-2:]=[self.compact[-2]+self.compact[-1]]
    def get_compact(self):
        self.ensure_compact()
        if len(self.compact)>0:
            return self.compact[0]
        else:
            return VecD1()
    def get_size(self, compactify=True):
        cdef int n
        if compactify:
            self.ensure_compact()
            if self.compact:
                return self.compact[0].get_size()
            else:
                return 0
        else:
            n=sum([c.get_size(False) for c in self.compact])
            n+=self.loose.get_size(False)
            return n
    def __iter__(self):
        self.ensure_compact()
        try:
            return iter(self.compact[0])
        except IndexError:
            return iter([])

    def get_type_counts(self, k=0):
        return self.get_compact().get_type_counts(k)
    def remap(self, k, filt):
        return self.get_compact().remap(k,filt)


cdef class IVecD2_iter

cdef class VecD2:
    """mutable sparse int matrix based on item vector"""
    def __init__(self):
        self.is_compact=True
    def item_iter(self):
        self.ensure_compact()
        return IVecD2_iter(self)
    def get_size(self,compactify=True):
        if compactify:
            self.ensure_compact()
        return self.vec.size()
    def __len__(self):
        return self.get_size(True)
    def clear(self):
        self.vec.resize(0)
        self.is_compact=True
    cdef void compactify(self):
        cdef c_SmallerAddrD2 comp
        c_IVecD2_sort(self.vec.begin(),self.vec.end(),comp)
        c_compactifyD2(&self.vec)
        self.is_compact=True
    cdef void ensure_compact(self):
        if not self.is_compact:
            self.compactify()
    cpdef double get_count(self,coordinate_t k0,coordinate_t k1):
        cdef c_CItemD2 ci
        ci.addr[0]=k0
        ci.addr[1]=k1
        return c_get_countD2(&self.vec,ci)
    cdef void c_add(self, coordinate_t k0,coordinate_t k1, double item):
        cdef c_CItemD2 c
        c.addr[0]=k0
        c.addr[1]=k1
        c.item=item
        self.vec.push_back(c)
        self.is_compact=False
    def add_count(self, coordinate_t k0,coordinate_t k1, item=1):
        self.c_add(k0,k1, item)
    def __add__(VecD2 self, VecD2 other):
        cdef c_CItemD2 c1, c2
        cdef unsigned int i1, i2
        cdef int k, delta
        cdef VecD2 result
        self.ensure_compact()
        other.ensure_compact()
        result=VecD2()
        result.vec.reserve(max(self.vec.size(),other.vec.size()))
        i1=0
        i2=0
        while i1<self.vec.size() and i2<other.vec.size():
            c1=self.vec.at(i1)
            c2=other.vec.at(i2)
            delta=0
            for k from 0<=k<2:
                if c1.addr[k]<c2.addr[k]:
                    delta=-1
                    break
                elif c1.addr[k]>c2.addr[k]:
                    delta=1
                    break
            if delta==0:
                c1.item+=c2.item
                result.vec.push_back(c1)
                i1+=1
                i2+=1
            elif delta==-1:
                result.vec.push_back(c1)
                i1+=1
            else:
                result.vec.push_back(c2)
                i2+=1
        while i1<self.vec.size():
            c1=self.vec.at(i1)
            result.vec.push_back(c1)
            i1+=1
        while i2<other.vec.size():
            c2=other.vec.at(i2)
            result.vec.push_back(c2)
            i2+=1
        return result

    cpdef CSRMatrixD to_csr(self):
        self.ensure_compact()
        cdef c_CSRMatrixD *matC=vec2csrD(&self.vec)
        cdef CSRMatrixD mat=CSRMatrixD()
        mat.set_matrix(matC)
        return mat

    def tofile(self, f):
        cdef long i, n, k, k_max
        n=self.get_size()
        f.write('D2 %d\n'%(n,))
        k=0
        for i from 0<=i<n/CHUNK_SIZE:
            s=PyBytes_FromStringAndSize(<char *>c_VecD2_get_pointer(&self.vec,k), CHUNK_SIZE*sizeof(c_CItemD2))
            f.write(s)
            k+=CHUNK_SIZE
        if k<n:
            s=PyBytes_FromStringAndSize(<char *>c_VecD2_get_pointer(&self.vec,k), (n-k)*sizeof(c_CItemD2))
            f.write(s)
    def fromfile(self, f):
        cdef long i, n
        line=f.readline().strip().split()
        assert line[0]=='D2'
        n=long(line[1])
        self.clear()
        self.vec.resize(n)
        k=0
        for i from 0<=i<n/CHUNK_SIZE:
            s=f.read(CHUNK_SIZE*sizeof(c_CItemD2))
            assert len(s)==CHUNK_SIZE*sizeof(c_CItemD2)
            memcpy(<void *>c_VecD2_get_pointer(&self.vec,k), <char *>s, CHUNK_SIZE*sizeof(c_CItemD2))
            k+=CHUNK_SIZE
        if k<n:
            s=f.read((n-k)*sizeof(c_CItemD2))
            assert len(s)==(n-k)*sizeof(c_CItemD2)
            memcpy(<void *>c_VecD2_get_pointer(&self.vec,k), <char *>s, (n-k)*sizeof(c_CItemD2))
    cpdef int get_maxcol(self, int k=0):
        cdef int i, n, new_val, col
        assert k>=0 and k<2
        n=self.vec.size()
        if n==0:
            return -1
        if k==0:
            return self.vec.at(n-1).addr[0]
        col=-1
        n=self.vec.size()
        for i from 0<=i<n:
            new_val=self.vec.at(i).addr[k]
            if new_val>col:
                col=new_val
        return col
    cpdef numpy.ndarray get_type_counts(self, int k=0):
        cdef int old_val, new_val, i, n, count
        cdef numpy.ndarray[numpy.int32_t, ndim=1] result
        assert k>=0 and k<2
        if self.vec.size()==0:
            return numpy.zeros(0, numpy.int32)
        self.ensure_compact()
        n_types=self.get_maxcol(k)+1
        result=numpy.zeros(n_types,numpy.int32)
        n=self.vec.size()
        for i from 0<=i<n:
            new_val=self.vec.at(i).addr[k]
            result[new_val]+=1
        return result
    cpdef VecD2 remap(self, int k, numpy.ndarray filt):
        cdef numpy.ndarray[numpy.int8_t, ndim=1] wanted=filt.astype('b')
        cdef numpy.ndarray[numpy.int_t, ndim=1] target=filt.cumsum()-1
        self.ensure_compact()
        cdef int n=self.vec.size()
        cdef int i
        cdef VecD2 result=VecD2()
        cdef c_CItemD2 c
        for i from 0<=i<n:
            c=self.vec.at(i)
            if wanted[c.addr[k]]:
                c.addr[k]=target[c.addr[k]]
                result.vec.push_back(c)
        return result
    def append(self,tup):
        k0,k1,val=tup
        self.c_add(k0,k1,val)
    def extend(self,tups):
        for k0,k1,val in tups:
            self.c_add(k0,k1,val)
    def __iter__(self):
        return self.item_iter()
    def __reduce__(self):
        return (VecD2,(),(),self.item_iter())
    def __setstate__(self,state):
        pass
    def __repr__(self):
        return 'VecD2(%s)'%(str(list(self)),)
    def __dealloc__(self):
        # need to call destructor explicitly
        self.vec.cxx_destructor()

cdef class IVecD2_iter:
    cdef VecD2 vec
    cdef c_VecD2 *vecC
    cdef unsigned int k
    def __init__(self,vec):
        self.vec=vec
        self.vecC=&self.vec.vec
        self.k=0
    def __iter__(self):
        return self
    def __next__(self):
        if self.k>=self.vecC.size():
            raise StopIteration
        cdef c_CItemD2 res=self.vecC.at(self.k)
        self.k+=1
        return (res.addr[0],
                res.addr[1],
                res.item)

cdef class LargeVecD2:
    cdef public object compact
    cdef public VecD2 loose
    def __init__(self):
        self.compact=[]
        self.loose=VecD2()
        self.loose.vec.reserve(100000)
    def ensure_compact(self):
        if self.loose.get_size()>0:
            self.compact.append(self.loose)
        self.loose=VecD2()
        while len(self.compact)>=2:
            self.compact[-2:]=[self.compact[-2]+self.compact[-1]]
    def add_count(self, coordinate_t k0,coordinate_t k1, item=1):
        self.loose.c_add(k0,k1, item)
        if self.loose.get_size(False)>=100000:
            self.loose.ensure_compact()
            self.compact.append(self.loose)
            self.loose=VecD2()
            self.loose.vec.reserve(100000)
            while (len(self.compact)>=2 and
                   self.compact[-1].get_size()>=0.6*self.compact[-2].get_size()):
                self.compact[-2:]=[self.compact[-2]+self.compact[-1]]
    def get_compact(self):
        self.ensure_compact()
        if len(self.compact)>0:
            return self.compact[0]
        else:
            return VecD2()
    def get_size(self, compactify=True):
        cdef int n
        if compactify:
            self.ensure_compact()
            if self.compact:
                return self.compact[0].get_size()
            else:
                return 0
        else:
            n=sum([c.get_size(False) for c in self.compact])
            n+=self.loose.get_size(False)
            return n
    def __iter__(self):
        self.ensure_compact()
        try:
            return iter(self.compact[0])
        except IndexError:
            return iter([])

    cpdef CSRMatrixD to_csr(self):
        self.ensure_compact()
        return self.compact[0].to_csr()

    def get_type_counts(self, k=0):
        return self.get_compact().get_type_counts(k)
    def remap(self, k, filt):
        return self.get_compact().remap(k,filt)


cdef class IVecD3_iter

cdef class VecD3:
    """mutable sparse int matrix based on item vector"""
    def __init__(self):
        self.is_compact=True
    def item_iter(self):
        self.ensure_compact()
        return IVecD3_iter(self)
    def get_size(self,compactify=True):
        if compactify:
            self.ensure_compact()
        return self.vec.size()
    def __len__(self):
        return self.get_size(True)
    def clear(self):
        self.vec.resize(0)
        self.is_compact=True
    cdef void compactify(self):
        cdef c_SmallerAddrD3 comp
        c_IVecD3_sort(self.vec.begin(),self.vec.end(),comp)
        c_compactifyD3(&self.vec)
        self.is_compact=True
    cdef void ensure_compact(self):
        if not self.is_compact:
            self.compactify()
    cpdef double get_count(self,coordinate_t k0,coordinate_t k1,coordinate_t k2):
        cdef c_CItemD3 ci
        ci.addr[0]=k0
        ci.addr[1]=k1
        ci.addr[2]=k2
        return c_get_countD3(&self.vec,ci)
    cdef void c_add(self, coordinate_t k0,coordinate_t k1,coordinate_t k2, double item):
        cdef c_CItemD3 c
        c.addr[0]=k0
        c.addr[1]=k1
        c.addr[2]=k2
        c.item=item
        self.vec.push_back(c)
        self.is_compact=False
    def add_count(self, coordinate_t k0,coordinate_t k1,coordinate_t k2, item=1):
        self.c_add(k0,k1,k2, item)
    def __add__(VecD3 self, VecD3 other):
        cdef c_CItemD3 c1, c2
        cdef unsigned int i1, i2
        cdef int k, delta
        cdef VecD3 result
        self.ensure_compact()
        other.ensure_compact()
        result=VecD3()
        result.vec.reserve(max(self.vec.size(),other.vec.size()))
        i1=0
        i2=0
        while i1<self.vec.size() and i2<other.vec.size():
            c1=self.vec.at(i1)
            c2=other.vec.at(i2)
            delta=0
            for k from 0<=k<3:
                if c1.addr[k]<c2.addr[k]:
                    delta=-1
                    break
                elif c1.addr[k]>c2.addr[k]:
                    delta=1
                    break
            if delta==0:
                c1.item+=c2.item
                result.vec.push_back(c1)
                i1+=1
                i2+=1
            elif delta==-1:
                result.vec.push_back(c1)
                i1+=1
            else:
                result.vec.push_back(c2)
                i2+=1
        while i1<self.vec.size():
            c1=self.vec.at(i1)
            result.vec.push_back(c1)
            i1+=1
        while i2<other.vec.size():
            c2=other.vec.at(i2)
            result.vec.push_back(c2)
            i2+=1
        return result

    def tofile(self, f):
        cdef long i, n, k, k_max
        n=self.get_size()
        f.write('D3 %d\n'%(n,))
        k=0
        for i from 0<=i<n/CHUNK_SIZE:
            s=PyBytes_FromStringAndSize(<char *>c_VecD3_get_pointer(&self.vec,k), CHUNK_SIZE*sizeof(c_CItemD3))
            f.write(s)
            k+=CHUNK_SIZE
        if k<n:
            s=PyBytes_FromStringAndSize(<char *>c_VecD3_get_pointer(&self.vec,k), (n-k)*sizeof(c_CItemD3))
            f.write(s)
    def fromfile(self, f):
        cdef long i, n
        line=f.readline().strip().split()
        assert line[0]=='D3'
        n=long(line[1])
        self.clear()
        self.vec.resize(n)
        k=0
        for i from 0<=i<n/CHUNK_SIZE:
            s=f.read(CHUNK_SIZE*sizeof(c_CItemD3))
            assert len(s)==CHUNK_SIZE*sizeof(c_CItemD3)
            memcpy(<void *>c_VecD3_get_pointer(&self.vec,k), <char *>s, CHUNK_SIZE*sizeof(c_CItemD3))
            k+=CHUNK_SIZE
        if k<n:
            s=f.read((n-k)*sizeof(c_CItemD3))
            assert len(s)==(n-k)*sizeof(c_CItemD3)
            memcpy(<void *>c_VecD3_get_pointer(&self.vec,k), <char *>s, (n-k)*sizeof(c_CItemD3))
    cpdef int get_maxcol(self, int k=0):
        cdef int i, n, new_val, col
        assert k>=0 and k<3
        n=self.vec.size()
        if n==0:
            return -1
        if k==0:
            return self.vec.at(n-1).addr[0]
        col=-1
        n=self.vec.size()
        for i from 0<=i<n:
            new_val=self.vec.at(i).addr[k]
            if new_val>col:
                col=new_val
        return col
    cpdef numpy.ndarray get_type_counts(self, int k=0):
        cdef int old_val, new_val, i, n, count
        cdef numpy.ndarray[numpy.int32_t, ndim=1] result
        assert k>=0 and k<3
        if self.vec.size()==0:
            return numpy.zeros(0, numpy.int32)
        self.ensure_compact()
        n_types=self.get_maxcol(k)+1
        result=numpy.zeros(n_types,numpy.int32)
        n=self.vec.size()
        for i from 0<=i<n:
            new_val=self.vec.at(i).addr[k]
            result[new_val]+=1
        return result
    cpdef VecD3 remap(self, int k, numpy.ndarray filt):
        cdef numpy.ndarray[numpy.int8_t, ndim=1] wanted=filt.astype('b')
        cdef numpy.ndarray[numpy.int_t, ndim=1] target=filt.cumsum()-1
        self.ensure_compact()
        cdef int n=self.vec.size()
        cdef int i
        cdef VecD3 result=VecD3()
        cdef c_CItemD3 c
        for i from 0<=i<n:
            c=self.vec.at(i)
            if wanted[c.addr[k]]:
                c.addr[k]=target[c.addr[k]]
                result.vec.push_back(c)
        return result
    def append(self,tup):
        k0,k1,k2,val=tup
        self.c_add(k0,k1,k2,val)
    def extend(self,tups):
        for k0,k1,k2,val in tups:
            self.c_add(k0,k1,k2,val)
    def __iter__(self):
        return self.item_iter()
    def __reduce__(self):
        return (VecD3,(),(),self.item_iter())
    def __setstate__(self,state):
        pass
    def __repr__(self):
        return 'VecD3(%s)'%(str(list(self)),)
    def __dealloc__(self):
        # need to call destructor explicitly
        self.vec.cxx_destructor()

cdef class IVecD3_iter:
    cdef VecD3 vec
    cdef c_VecD3 *vecC
    cdef unsigned int k
    def __init__(self,vec):
        self.vec=vec
        self.vecC=&self.vec.vec
        self.k=0
    def __iter__(self):
        return self
    def __next__(self):
        if self.k>=self.vecC.size():
            raise StopIteration
        cdef c_CItemD3 res=self.vecC.at(self.k)
        self.k+=1
        return (res.addr[0],
                res.addr[1],
                res.addr[2],
                res.item)

cdef class LargeVecD3:
    cdef public object compact
    cdef public VecD3 loose
    def __init__(self):
        self.compact=[]
        self.loose=VecD3()
        self.loose.vec.reserve(100000)
    def ensure_compact(self):
        if self.loose.get_size()>0:
            self.compact.append(self.loose)
        self.loose=VecD3()
        while len(self.compact)>=2:
            self.compact[-2:]=[self.compact[-2]+self.compact[-1]]
    def add_count(self, coordinate_t k0,coordinate_t k1,coordinate_t k2, item=1):
        self.loose.c_add(k0,k1,k2, item)
        if self.loose.get_size(False)>=100000:
            self.loose.ensure_compact()
            self.compact.append(self.loose)
            self.loose=VecD3()
            self.loose.vec.reserve(100000)
            while (len(self.compact)>=2 and
                   self.compact[-1].get_size()>=0.6*self.compact[-2].get_size()):
                self.compact[-2:]=[self.compact[-2]+self.compact[-1]]
    def get_compact(self):
        self.ensure_compact()
        if len(self.compact)>0:
            return self.compact[0]
        else:
            return VecD3()
    def get_size(self, compactify=True):
        cdef int n
        if compactify:
            self.ensure_compact()
            if self.compact:
                return self.compact[0].get_size()
            else:
                return 0
        else:
            n=sum([c.get_size(False) for c in self.compact])
            n+=self.loose.get_size(False)
            return n
    def __iter__(self):
        self.ensure_compact()
        try:
            return iter(self.compact[0])
        except IndexError:
            return iter([])

    def get_type_counts(self, k=0):
        return self.get_compact().get_type_counts(k)
    def remap(self, k, filt):
        return self.get_compact().remap(k,filt)

cdef class SparseVectorD:
    def __init__(self, pairs=None):
        cdef coordinate_t i
        if pairs is not None:
            self.my_len=len(pairs)
            self.idx_ptr=<coordinate_t *>PyMem_Malloc(self.my_len*sizeof(coordinate_t))
            self.vals_ptr=<double *>PyMem_Malloc(self.my_len*sizeof(double))
            for i from 0<=i<self.my_len:
                x,y=pairs[i]
                self.idx_ptr[i]=x
                self.vals_ptr[i]=y
        else:
            self.my_len=0
            self.idx_ptr=NULL
            self.vals_ptr=NULL
    cpdef int from_dense(self, double[:] dense):
        cdef int i, k
        assert self.my_len == 0
        k = 0
        for i from 0<=i<dense.shape[0]:
            if dense[i]!=0.0:
                k += 1
        self.my_len = k
        self.idx_ptr=<coordinate_t *>PyMem_Malloc(self.my_len*sizeof(coordinate_t))
        self.vals_ptr=<double *>PyMem_Malloc(self.my_len*sizeof(double))
        k = 0
        for i from 0<=i<dense.shape[0]:
            if dense[i] != 0.0:
                self.idx_ptr[k] = i
                self.vals_ptr[k] = dense[i]
                k += 1
        return k
    cpdef double dotSelf(self):
        cdef double s=0
        cdef coordinate_t i
        for i from 0<=i<self.my_len:
            s+=self.vals_ptr[i]*self.vals_ptr[i]
        return s
    cdef double _dotFull(self, const_double_ptr full_ptr):
        cdef double s=0
        cdef coordinate_t i
        for i from 0<=i<self.my_len:
            s+=self.vals_ptr[i]*full_ptr[self.idx_ptr[i]]
        return s
    cdef double _dotFull_partial(self, const_double_ptr full_ptr, int my_len):
        cdef double s=0
        cdef coordinate_t i
        for i from 0<=i<my_len:
            s+=self.vals_ptr[i]*full_ptr[self.idx_ptr[i]]
        return s
    cpdef double jaccard(self, SparseVectorD other):
        cdef double s_max=0
        cdef double s_min=0
        cdef double val1, val2
        cdef coordinate_t i,j, idx_i, idx_j
        i=0
        j=0
        while i<self.my_len and j<other.my_len:
            idx_i=self.idx_ptr[i]
            idx_j=other.idx_ptr[j]
            if idx_i<idx_j:
                s_max+=self.vals_ptr[i]
                i+=1
            elif idx_i>idx_j:
                s_max+=other.vals_ptr[j]
                j+=1
            else:
                val1=self.vals_ptr[i]
                val2=other.vals_ptr[j]
                if val1>val2:
                    s_max+=val1
                    s_min+=val2
                else:
                    s_max+=val2
                    s_min+=val1
                i+=1
                j+=1
        if i<self.my_len:
            while i<self.my_len:
                s_max+=self.vals_ptr[i]
                i+=1
        else:
            while j<other.my_len:
                s_max+=other.vals_ptr[j]
                j+=1
        if s_max==0:
            return 0.0
        else:
            return s_min/s_max
    cpdef double dotSparse(self, SparseVectorD other):
        cdef double product=0
        cdef double val1, val2
        cdef coordinate_t i, j, idx_i, idx_j
        i=0
        j=0
        while i<self.my_len and j<other.my_len:
           idx_i=self.idx_ptr[i]
           idx_j=other.idx_ptr[j]
           if idx_i<idx_j:
              i+=1
           elif idx_i>idx_j:
              j+=1
           else:
              val1=self.vals_ptr[i]
              val2=other.vals_ptr[j]
              product+=val1*val2
              i+=1
              j+=1
        return product
    def count_intersection(self, SparseVectorD other):
        cdef coordinate_t i, j, idx_i, idx_j
        cdef int n_ab=0
        i=0
        j=0
        while i<self.my_len and j<other.my_len:
           idx_i=self.idx_ptr[i]
           idx_j=other.idx_ptr[j]
           if idx_i<idx_j:
              i+=1
           elif idx_i>idx_j:
              j+=1
           else:
              i+=1
              j+=1
              n_ab+=1
        return n_ab
    cpdef double min_sum(self, SparseVectorD other):
        cdef double product=0.0
        cdef double val1, val2
        cdef coordinate_t i, j, idx_i, idx_j
        i=0
        j=0
        while i<self.my_len and j<other.my_len:
           idx_i=self.idx_ptr[i]
           idx_j=other.idx_ptr[j]
           if idx_i<idx_j:
              i+=1
           elif idx_i>idx_j:
              j+=1
           else:
              val1=self.vals_ptr[i]
              val2=other.vals_ptr[j]
              if val1<val2:
                  product+=val1
              else:
                  product+=val2
              i+=1
              j+=1
        return product
    cpdef double cosine(self, SparseVectorD other):
        cdef double sqsum_self=0.0
        cdef double sqsum_other=0.0
        cdef double product=0.0
        cdef double val1, val2
        cdef coordinate_t i, j, idx_i, idx_j
        i=0
        j=0
        while i<self.my_len and j<other.my_len:
           idx_i=self.idx_ptr[i]
           idx_j=other.idx_ptr[j]
           if idx_i<idx_j:
              val1=self.vals_ptr[i]
              sqsum_self+=val1*val1
              i+=1
           elif idx_i>idx_j:
              val2=other.vals_ptr[j]
              sqsum_other+=val2*val2
              j+=1
           else:
              val1=self.vals_ptr[i]
              val2=other.vals_ptr[j]
              sqsum_self+=val1*val1
              sqsum_other+=val2*val2
              product+=val1*val2
              i+=1
              j+=1
        if i<self.my_len:
            while i<self.my_len:
                val1=self.vals_ptr[i]
                sqsum_self+=val1*val1
                i+=1
        else:
            while j<other.my_len:
                val2=other.vals_ptr[j]
                sqsum_other+=val2*val2
                j+=1
        return product/sqrt(sqsum_self*sqsum_other)
    cpdef double jsd_unnorm(self, SparseVectorD other):
        cdef double sum=0.0
        cdef double val1, val2, avg
        cdef coordinate_t i, j, idx_i, idx_j
        i=0
        j=0
        while i<self.my_len and j<other.my_len:
           idx_i=self.idx_ptr[i]
           idx_j=other.idx_ptr[j]
           if idx_i<idx_j:
               val1=self.vals_ptr[i]
               sum+=val1*M_LN2
               i+=1
           elif idx_i>idx_j:
               val2=other.vals_ptr[j]
               sum+=val2*M_LN2
               j+=1
           else:
               val1=self.vals_ptr[i]
               val2=other.vals_ptr[j]
               i+=1
               j+=1
               avg=(val1+val2)/2.0
               sum+=val1*log(val1/avg)+val2*log(val2/avg)
        while i<self.my_len:
           val1=self.vals_ptr[i]
           i+=1
           sum+=val1*M_LN2
        while j<other.my_len:
           val2=other.vals_ptr[j]
           j+=1
           sum+=val2*M_LN2
        return sum/(2.0*M_LN2)
    cpdef double skew_unnorm(self, SparseVectorD other, double alpha):
        cdef double sum=0.0
        cdef double beta=1.0-alpha
        cdef double val1, val2, avg
        cdef coordinate_t i, j, idx_i, idx_j
        cdef double log_invbeta=log(1.0/beta)
        i=0
        j=0
        while i<self.my_len and j<other.my_len:
           idx_i=self.idx_ptr[i]
           idx_j=self.idx_ptr[j]
           if idx_i<idx_j:
               val1=self.vals_ptr[i]
               sum+=val1*log_invbeta
               i+=1
           elif idx_i>idx_j:
               j+=1
           else:
               val1=self.vals_ptr[i]
               val2=other.vals_ptr[j]
               avg=alpha*val2+beta*val1
               sum+=log(val1/avg)
               i+=1
               j+=1
        return sum
    cpdef double norm_l1(self):
        cdef double sum=0.0
        cdef unsigned int i
        for i from 0<=i<self.my_len:
            sum+=fabs(self.vals_ptr[i])
        return sum
    cpdef double norm_l2(self):
        cdef double sum=0.0
        cdef double val
        cdef unsigned int i
        for i from 0<=i<self.my_len:
            val=self.vals_ptr[i];
            sum+=val*val
        return sqrt(sum)
    cpdef double norm_lp(self, double p):
        cdef double sum=0.0
        cdef int i
        for i from 0<=i<self.my_len:
            sum+=pow(fabs(self.vals_ptr[i]),p)
        return pow(sum,1.0/p)
    def dotFull(self, numpy.ndarray[double,ndim=1] full):
        cdef double *full_ptr
        cdef unsigned int my_len=self.my_len
        cdef unsigned int full_len=len(full)
        # treat nonpresent values as 0
        while my_len>0 and self.idx_ptr[my_len-1]>full_len:
            my_len-=1
        full_ptr=<double*>full.data
        return self._dotFull_partial(full_ptr,my_len)
    def dotFull_check(self, numpy.ndarray[double,ndim=1] full):
        cdef double *full_ptr
        cdef int my_len=self.my_len
        cdef int full_len=len(full)
        # boundary check
        full[self.idx_ptr[self.my_len-1]]
        full_ptr=<double*>full.data
        return self._dotFull(full_ptr)
    def dotMatrix(self, numpy.ndarray[double,ndim=2] full):
        cdef numpy.ndarray[double,ndim=1] result
        result = numpy.zeros(full.shape[1], numpy.float64)
        for i from 0<=i<self.my_len:
            result += full[self.idx_ptr[i]] * self.vals_ptr[i]
        return result
    cdef void _axpy(self, double *x_ptr, double a):
        for i from 0<=i<self.my_len:
            x_ptr[self.idx_ptr[i]]+=a*self.vals_ptr[i]
    def axpy(self, numpy.ndarray[double,ndim=1] x, double a=1):
        # boundary check
        x[self.idx_ptr[self.my_len-1]]
        self._axpy(<double *>x.data,a)
    cpdef double sqdist(self, SparseVectorD other):
        """computes ||x-y||^2"""
        cdef double s=0
        cdef double val
        cdef int i1=0,i2=0
        while i1<self.my_len and i2<other.my_len:
            if self.idx_ptr[i1]<other.idx_ptr[i2]:
                val=self.vals_ptr[i1]
                s+=val*val
                i1+=1
            elif self.idx_ptr[i1]>other.idx_ptr[i2]:
                val=other.vals_ptr[i2]
                s+=val*val
                i2+=1
            else:
                val=self.vals_ptr[i1]-other.vals_ptr[i2]
                s+=val*val
                i1+=1
                i2+=1
        while i1<self.my_len:
            val=self.vals_ptr[i1]
            s+=val*val
            i1+=1
        while i2<other.my_len:
            val=other.vals_ptr[i2]
            s+=val*val
            i2+=1
        return s
    def write_pairs(self, f, delim=':'):
        cdef int i
        w_func=f.write
        for i from 0<=i<self.my_len:
            # svmlight does not want 0 as index
            w_func(' %d%s%s'%(self.idx_ptr[i]+1,delim,self.vals_ptr[i]))
    def __imul__(self, double a):
        cdef coordinate_t i
        for i from 0<=i<self.my_len:
            self.vals_ptr[i] *= a
        return self
    def __idiv__(self, double a):
        cdef coordinate_t i
        for i from 0<=i<self.my_len:
            self.vals_ptr[i] /= a
        return self
    cpdef SparseVectorD min_vals(self, SparseVectorD other):
        cdef SparseVectorD result
        cdef coordinate_t i1, i2, k, idx1, idx2
        result=SparseVectorD()
        if self.my_len<other.my_len:
            result.my_len=self.my_len
        else:
            result.my_len=other.my_len
        result.idx_ptr=<coordinate_t *>PyMem_Malloc(result.my_len*sizeof(coordinate_t))
        result.vals_ptr=<double *>PyMem_Malloc(result.my_len*sizeof(double))
        i1=i2=k=0
        while i1<self.my_len and i2<other.my_len:
           idx1=self.idx_ptr[i1]
           idx2=other.idx_ptr[i2]
           if idx1<idx2:
               i1+=1
           elif idx1>idx2:
               i2+=1
           else:
               result.idx_ptr[k]=idx1
               val1=self.vals_ptr[i1]
               val2=other.vals_ptr[i2]
               if val1<val2:
                   result.vals_ptr[k]=val1
               else:
                   result.vals_ptr[k]=val2
               i1+=1
               i2+=1
               k+=1
        result.my_len=k
        return result
    def scale_array(SparseVectorD self, numpy.ndarray[double, ndim=1] a):
        cdef SparseVectorD result
        cdef coordinate_t i
        result=SparseVectorD()
        result.my_len=self.my_len
        result.idx_ptr=<coordinate_t *>PyMem_Malloc(result.my_len*sizeof(coordinate_t))
        result.vals_ptr=<double *>PyMem_Malloc(result.my_len*sizeof(double))
        for i from 0<=i<self.my_len:
            result.idx_ptr[i]=self.idx_ptr[i]
        for i from 0<=i<self.my_len:
            result.vals_ptr[i]=self.vals_ptr[i]*a[self.idx_ptr[i]]
        return result
    def __div__(SparseVectorD self, double a):
        cdef SparseVectorD result
        cdef coordinate_t i
        result=SparseVectorD()
        result.my_len=self.my_len
        result.idx_ptr=<coordinate_t *>PyMem_Malloc(result.my_len*sizeof(coordinate_t))
        result.vals_ptr=<double *>PyMem_Malloc(result.my_len*sizeof(double))
        for i from 0<=i<self.my_len:
            result.idx_ptr[i]=self.idx_ptr[i]
        for i from 0<=i<self.my_len:
            result.vals_ptr[i]=self.vals_ptr[i]/a
        return result
    def __len__(self):
        return self.my_len
    def __add__(SparseVectorD self,SparseVectorD other):
        cdef coordinate_t *new_idx
        cdef double *new_vals
        cdef coordinate_t i1,i2,i_all
        cdef coordinate_t n1,n2,n_all
        cdef coordinate_t k1,k2,k
        cdef SparseVectorD result
        # first pass: determine how much space we need
        n1=self.my_len
        n2=other.my_len
        n_all=i1=i2=0
        while i1<n1 and i2<n2:
            k1=self.idx_ptr[i1]
            k2=other.idx_ptr[i2]
            if k1>=k2: i2+=1
            if k2>=k1: i1+=1
            n_all+=1
        if i1<n1: n_all+=n1-i1
        if i2<n2: n_all+=n2-i2
        new_idx=<coordinate_t *>PyMem_Malloc(n_all*sizeof(coordinate_t))
        new_vals=<double *>PyMem_Malloc(n_all*sizeof(double))
        # second pass: fill in the actual values
        i1=i2=i_all=0
        while i1<n1 and i2<n2:
            k1=self.idx_ptr[i1]
            k2=other.idx_ptr[i2]
            if k1>=k2:
                new_idx[i_all]=k2
                new_vals[i_all]=other.vals_ptr[i2]
                i2+=1
                if k1==k2:
                    new_vals[i_all]+=self.vals_ptr[i1]
                    i1+=1
            elif k2>=k1:
                new_idx[i_all]=k1
                new_vals[i_all]=self.vals_ptr[i1]
                i1+=1
            i_all+=1
        while i1<n1:
            new_idx[i_all]=self.idx_ptr[i1]
            new_vals[i_all]=self.vals_ptr[i1]
            i1+=1
            i_all+=1
        else:
            while i2<n2:
                new_idx[i_all]=other.idx_ptr[i2]
                new_vals[i_all]=other.vals_ptr[i2]
                i2+=1
                i_all+=1
        assert i_all==n_all
        result=SparseVectorD()
        result.my_len=n_all
        result.idx_ptr=new_idx
        result.vals_ptr=new_vals
        return result
    def __sub__(SparseVectorD self,SparseVectorD other):
        cdef coordinate_t *new_idx
        cdef double *new_vals
        cdef coordinate_t i1,i2,i_all
        cdef coordinate_t n1,n2,n_all
        cdef coordinate_t k1,k2,k
        cdef SparseVectorD result
        # first pass: determine how much space we need
        n1=self.my_len
        n2=other.my_len
        n_all=i1=i2=0
        while i1<n1 and i2<n2:
            k1=self.idx_ptr[i1]
            k2=other.idx_ptr[i2]
            if k1>=k2: i2+=1
            if k2>=k1: i1+=1
            n_all+=1
        if i1<n1: n_all+=n1-i1
        if i2<n2: n_all+=n2-i2
        new_idx=<coordinate_t *>PyMem_Malloc(n_all*sizeof(coordinate_t))
        new_vals=<double *>PyMem_Malloc(n_all*sizeof(double))
        # second pass: fill in the actual values
        i1=i2=i_all=0
        while i1<n1 and i2<n2:
            k1=self.idx_ptr[i1]
            k2=other.idx_ptr[i2]
            if k1>=k2:
                new_idx[i_all]=k2
                new_vals[i_all]=-other.vals_ptr[i2]
                i2+=1
                if k1==k2:
                    new_vals[i_all]+=self.vals_ptr[i1]
                    i1+=1
            elif k2>=k1:
                new_idx[i_all]=k1
                new_vals[i_all]=self.vals_ptr[i1]
                i1+=1
            i_all+=1
        while i1<n1:
            new_idx[i_all]=self.idx_ptr[i1]
            new_vals[i_all]=self.vals_ptr[i1]
            i1+=1
            i_all+=1
        else:
            while i2<n2:
                new_idx[i_all]=other.idx_ptr[i2]
                new_vals[i_all]=-other.vals_ptr[i2]
                i2+=1
                i_all+=1
        assert i_all==n_all
        result=SparseVectorD()
        result.my_len=n_all
        result.idx_ptr=new_idx
        result.vals_ptr=new_vals
        return result
    def __or__(SparseVectorD self,SparseVectorD other):
        cdef coordinate_t *new_idx
        cdef double *new_vals
        cdef coordinate_t i1,i2,i_all
        cdef coordinate_t n1,n2,n_all
        cdef coordinate_t k1,k2,k
        cdef SparseVectorD result
        cdef double val
        # first pass: determine how much space we need
        n1=self.my_len
        n2=other.my_len
        n_all=i1=i2=0
        while i1<n1 and i2<n2:
            k1=self.idx_ptr[i1]
            k2=other.idx_ptr[i2]
            if k1>=k2: i2+=1
            if k2>=k1: i1+=1
            n_all+=1
        if i1<n1: n_all+=n1-i1
        if i2<n2: n_all+=n2-i2
        new_idx=<coordinate_t *>PyMem_Malloc(n_all*sizeof(coordinate_t))
        new_vals=<double *>PyMem_Malloc(n_all*sizeof(double))
        # second pass: fill in the actual values
        i1=i2=i_all=0
        while i1<n1 and i2<n2:
            k1=self.idx_ptr[i1]
            k2=other.idx_ptr[i2]
            if k1>=k2:
                new_idx[i_all]=k2
                new_vals[i_all]=other.vals_ptr[i2]
                i2+=1
                if k1==k2:
                    val=self.vals_ptr[i1]
                    if new_vals[i_all]<val:
                        new_vals[i_all]=val
                    i1+=1
            else: # k2>k1:
                new_idx[i_all]=k1
                new_vals[i_all]=self.vals_ptr[i1]
                i1+=1
            i_all+=1
        while i1<n1:
            new_idx[i_all]=self.idx_ptr[i1]
            new_vals[i_all]=self.vals_ptr[i1]
            i1+=1
            i_all+=1
        else:
            while i2<n2:
                new_idx[i_all]=other.idx_ptr[i2]
                new_vals[i_all]=other.vals_ptr[i2]
                i2+=1
                i_all+=1
        assert i_all==n_all
        result=SparseVectorD()
        result.my_len=n_all
        result.idx_ptr=new_idx
        result.vals_ptr=new_vals
        return result
    def __lshift__(self, int k):
        cdef SparseVectorD result
        cdef coordinate_t i
        result=SparseVectorD()
        result.my_len=self.my_len
        result.idx_ptr=<coordinate_t *>PyMem_Malloc(result.my_len*sizeof(coordinate_t))
        result.vals_ptr=<double *>PyMem_Malloc(result.my_len*sizeof(double))
        for i from 0<=i<self.my_len:
            result.idx_ptr[i]=self.idx_ptr[i]+k
        for i from 0<=i<self.my_len:
            result.vals_ptr[i]=self.vals_ptr[i]
        return result
    def __ilshift__(self, int k):
        cdef SparseVectorD result
        cdef coordinate_t i
        for i from 0<=i<self.my_len:
            self.idx_ptr[i]+=k
        return self
    def __getitem__(self, int i):
        if i<0:
            i=self.my_len+i
        if i>=self.my_len or i<0:
            raise IndexError
        return (self.idx_ptr[i],self.vals_ptr[i])
    def __repr__(self):
        cdef unsigned int i
        ss=[]
        for i from 0<=i<self.my_len:
            ss.append('(%d,%s)'%(self.idx_ptr[i],self.vals_ptr[i]))
        return 'SparseVectorD([%s])'%(','.join(ss))
    def __reduce_ex__(self,protocol):
        if protocol==0:
            # choose compatibility over performance.
            return (SparseVectorD,(list(self),),())
        else:
            s_idx=PyBytes_FromStringAndSize(<char *>self.idx_ptr,self.my_len*sizeof(coordinate_t))
            s_vals=PyBytes_FromStringAndSize(<char *>self.vals_ptr,self.my_len*sizeof(double))
            return (SparseVectorD,(None,),(s_idx,s_vals))
    def __setstate__(self,state):
        cdef coordinate_t *p_idx
        cdef double *p_vals
        if len(state)==0:
            pass
        elif len(state)==2:
            assert (self.idx_ptr==NULL)
            (s_idx, s_vals)=state
            p_idx=<coordinate_t *>(<char *>s_idx)
            p_vals=<double *>(<char *>s_vals)
            self.my_len=len(s_idx)/sizeof(coordinate_t)
            self.idx_ptr=<coordinate_t *>PyMem_Malloc(self.my_len*sizeof(coordinate_t))
            self.vals_ptr=<double *>PyMem_Malloc(self.my_len*sizeof(double))
            for i from 0<=i<self.my_len:
                self.idx_ptr[i]=p_idx[i]
                self.vals_ptr[i]=p_vals[i]
    def __dealloc__(self):
        if self.buf is None:
            PyMem_Free(self.idx_ptr)
            PyMem_Free(self.vals_ptr)
        else:
            self.buf=None
        self.idx_ptr=<coordinate_t *>0
        self.vals_ptr=<double *>0
        self.my_len=0
    def to_scipy(self):
        cdef numpy.ndarray[numpy.float64_t, ndim=1] data
        cdef numpy.ndarray[numpy.int32_t, ndim=1] indices
        cdef numpy.ndarray[numpy.int32_t, ndim=1] indptr
        cdef int i,j
        indptr=numpy.zeros(2, numpy.int32)
        indices=numpy.zeros(self.my_len, numpy.int32)
        data=numpy.zeros(self.my_len, numpy.float64)
        indptr[1]=self.my_len
        for i from 0<=i<self.my_len:
            indices[i]=0
        for i from 0<=i<self.my_len:
            data[i]=self.vals_ptr[i]
        return (data,indices,indptr)

emptyvec_D=SparseVectorD([])

cdef class SparseVectorsD:
    def __init__(self):
       self.vecs = []
    cpdef SparseVectorD to_vec(self):
        cdef SparseVectorD vec
        cdef double weight
        cdef size_t i, j
        cdef VecD1 result = VecD1()
        for i from 0<=i<len(self.vecs):
            weight = self.weights.at(i)
            vec = self.vecs[i]
            for j from 0<=j<vec.my_len:
                result.add_count(vec.idx_ptr[j], weight*vec.vals_ptr[j])
        return result.to_sparse()
    cpdef add(self, SparseVectorD vec, double weight=1):
        self.vecs.append(vec)
        self.weights.push_back(weight)
    def __mul__(self, factor):
        cdef SparseVectorD vec
        cdef double weight
        cdef SparseVectorsD result = SparseVectorsD
        for i from 0<=i<len(self.vecs):
            weight = self.weights.at(i)
            vec = self.vecs[i]
            result.add(vec, weight*factor)
    def __add__(self, others):
        cdef SparseVectorD vec
        cdef double weight
        cdef SparseVectorsD oth_vecs = others
        cdef SparseVectorsD result = SparseVectorsD
        for i from 0<=i<len(self.vecs):
            weight = self.weights.at(i)
            vec = self.vecs[i]
            result.add(vec, weight)
        for i from 0<=i<len(oth_vecs.vecs):
            weight = oth_vecs.weights.at(i)
            vec = oth_vecs.vecs[i]
            result.add(vec, weight)
        return result
    def __sub__(SparseVectorsD self, SparseVectorsD others):
        cdef SparseVectorD vec
        cdef double weight
        cdef SparseVectorsD oth_vecs = others
        cdef SparseVectorsD result = SparseVectorsD()
        for i from 0<=i<len(self.vecs):
            weight = self.weights.at(i)
            vec = self.vecs[i]
            result.add(vec, weight)
        for i from 0<=i<len(oth_vecs.vecs):
            weight = oth_vecs.weights.at(i)
            vec = oth_vecs.vecs[i]
            result.add(vec, -weight)
        return result
    cdef double _dotFull(self, const_double_ptr full_ptr):
        cdef size_t i, j
        cdef double weight
        cdef SparseVectorD vec
        cdef double result=0
        for i from 0<=i<len(self.vecs):
            weight = self.weights.at(i)
            vec = self.vecs.at(i)
            for j from 0<=j<vec.my_len:
                result += full_ptr[vec.indices[j]]*weight*vec.vals_ptr[j]
        return result
    def __len__(self):
        return len(self.vecs)
    def __getitem__(self, k):
        vec = self.vecs[k]
        return (self.weights.at(k), vec)
## V -> void


cdef class IVecV1_iter

cdef class VecV1:
    """mutable sparse int matrix based on item vector"""
    def __init__(self):
        self.is_compact=True
    def item_iter(self):
        self.ensure_compact()
        return IVecV1_iter(self)
    def get_size(self,compactify=True):
        if compactify:
            self.ensure_compact()
        return self.vec.size()
    def __len__(self):
        return self.get_size(True)
    def clear(self):
        self.vec.resize(0)
        self.is_compact=True
    cdef void compactify(self):
        cdef c_SmallerAddrV1 comp
        c_IVecV1_sort(self.vec.begin(),self.vec.end(),comp)
        c_compactifyV1(&self.vec)
        self.is_compact=True
    cdef void ensure_compact(self):
        if not self.is_compact:
            self.compactify()
    cpdef bint get_count(self,coordinate_t k0):
        cdef c_CItemV1 ci
        ci.addr[0]=k0
        return c_get_countV1(&self.vec,ci)
    cdef void c_add(self, coordinate_t k0):
        cdef c_CItemV1 c
        c.addr[0]=k0
        self.vec.push_back(c)
        self.is_compact=False
    def add_count(self, coordinate_t k0):
        self.c_add(k0)
    def __add__(VecV1 self, VecV1 other):
        cdef c_CItemV1 c1, c2
        cdef unsigned int i1, i2
        cdef int k, delta
        cdef VecV1 result
        self.ensure_compact()
        other.ensure_compact()
        result=VecV1()
        result.vec.reserve(max(self.vec.size(),other.vec.size()))
        i1=0
        i2=0
        while i1<self.vec.size() and i2<other.vec.size():
            c1=self.vec.at(i1)
            c2=other.vec.at(i2)
            delta=0
            for k from 0<=k<1:
                if c1.addr[k]<c2.addr[k]:
                    delta=-1
                    break
                elif c1.addr[k]>c2.addr[k]:
                    delta=1
                    break
            if delta==0:
                result.vec.push_back(c1)
                i1+=1
                i2+=1
            elif delta==-1:
                result.vec.push_back(c1)
                i1+=1
            else:
                result.vec.push_back(c2)
                i2+=1
        while i1<self.vec.size():
            c1=self.vec.at(i1)
            result.vec.push_back(c1)
            i1+=1
        while i2<other.vec.size():
            c2=other.vec.at(i2)
            result.vec.push_back(c2)
            i2+=1
        return result

    def to_array(self):
        cdef coordinate_t n, i
        cdef numpy.ndarray[numpy.int32_t, ndim=1] result
        self.ensure_compact()
        n=self.vec.size()
        result=numpy.zeros(n, numpy.int32)
        for i from 0<=i<n:
            result[i]=self.vec.at(i).addr[0]
        return result

    def tofile(self, f):
        cdef long i, n, k, k_max
        n=self.get_size()
        f.write('V1 %d\n'%(n,))
        k=0
        for i from 0<=i<n/CHUNK_SIZE:
            s=PyBytes_FromStringAndSize(<char *>c_VecV1_get_pointer(&self.vec,k), CHUNK_SIZE*sizeof(c_CItemV1))
            f.write(s)
            k+=CHUNK_SIZE
        if k<n:
            s=PyBytes_FromStringAndSize(<char *>c_VecV1_get_pointer(&self.vec,k), (n-k)*sizeof(c_CItemV1))
            f.write(s)
    def fromfile(self, f):
        cdef long i, n
        line=f.readline().strip().split()
        assert line[0]=='V1'
        n=long(line[1])
        self.clear()
        self.vec.resize(n)
        k=0
        for i from 0<=i<n/CHUNK_SIZE:
            s=f.read(CHUNK_SIZE*sizeof(c_CItemV1))
            assert len(s)==CHUNK_SIZE*sizeof(c_CItemV1)
            memcpy(<void *>c_VecV1_get_pointer(&self.vec,k), <char *>s, CHUNK_SIZE*sizeof(c_CItemV1))
            k+=CHUNK_SIZE
        if k<n:
            s=f.read((n-k)*sizeof(c_CItemV1))
            assert len(s)==(n-k)*sizeof(c_CItemV1)
            memcpy(<void *>c_VecV1_get_pointer(&self.vec,k), <char *>s, (n-k)*sizeof(c_CItemV1))
    cpdef int get_maxcol(self, int k=0):
        cdef int i, n, new_val, col
        assert k>=0 and k<1
        n=self.vec.size()
        if n==0:
            return -1
        if k==0:
            return self.vec.at(n-1).addr[0]
        col=-1
        n=self.vec.size()
        for i from 0<=i<n:
            new_val=self.vec.at(i).addr[k]
            if new_val>col:
                col=new_val
        return col
    cpdef numpy.ndarray get_type_counts(self, int k=0):
        cdef int old_val, new_val, i, n, count
        cdef numpy.ndarray[numpy.int32_t, ndim=1] result
        assert k>=0 and k<1
        if self.vec.size()==0:
            return numpy.zeros(0, numpy.int32)
        self.ensure_compact()
        n_types=self.get_maxcol(k)+1
        result=numpy.zeros(n_types,numpy.int32)
        n=self.vec.size()
        for i from 0<=i<n:
            new_val=self.vec.at(i).addr[k]
            result[new_val]+=1
        return result
    cpdef VecV1 remap(self, int k, numpy.ndarray filt):
        cdef numpy.ndarray[numpy.int8_t, ndim=1] wanted=filt.astype('b')
        cdef numpy.ndarray[numpy.int_t, ndim=1] target=filt.cumsum()-1
        self.ensure_compact()
        cdef int n=self.vec.size()
        cdef int i
        cdef VecV1 result=VecV1()
        cdef c_CItemV1 c
        for i from 0<=i<n:
            c=self.vec.at(i)
            if wanted[c.addr[k]]:
                c.addr[k]=target[c.addr[k]]
                result.vec.push_back(c)
        return result
    def append(self,k):
        self.c_add(k)
    def extend(self,tups):
        for k in tups:
            self.c_add(k)
    def __iter__(self):
        return self.item_iter()
    def __reduce__(self):
        return (VecV1,(),(),self.item_iter())
    def __setstate__(self,state):
        pass
    def __repr__(self):
        return 'VecV1(%s)'%(str(list(self)),)
    def __dealloc__(self):
        # need to call destructor explicitly
        self.vec.cxx_destructor()

cdef class IVecV1_iter:
    cdef VecV1 vec
    cdef c_VecV1 *vecC
    cdef unsigned int k
    def __init__(self,vec):
        self.vec=vec
        self.vecC=&self.vec.vec
        self.k=0
    def __iter__(self):
        return self
    def __next__(self):
        if self.k>=self.vecC.size():
            raise StopIteration
        cdef c_CItemV1 res=self.vecC.at(self.k)
        self.k+=1
        return res.addr[0]

cdef class LargeVecV1:
    cdef public object compact
    cdef public VecV1 loose
    def __init__(self):
        self.compact=[]
        self.loose=VecV1()
        self.loose.vec.reserve(100000)
    def ensure_compact(self):
        if self.loose.get_size()>0:
            self.compact.append(self.loose)
        self.loose=VecV1()
        while len(self.compact)>=2:
            self.compact[-2:]=[self.compact[-2]+self.compact[-1]]
    def add_count(self, coordinate_t k0):
        self.loose.c_add(k0)
        if self.loose.get_size(False)>=100000:
            self.loose.ensure_compact()
            self.compact.append(self.loose)
            self.loose=VecV1()
            self.loose.vec.reserve(100000)
            while (len(self.compact)>=2 and
                   self.compact[-1].get_size()>=0.6*self.compact[-2].get_size()):
                self.compact[-2:]=[self.compact[-2]+self.compact[-1]]
    def get_compact(self):
        self.ensure_compact()
        if len(self.compact)>0:
            return self.compact[0]
        else:
            return VecV1()
    def get_size(self, compactify=True):
        cdef int n
        if compactify:
            self.ensure_compact()
            if self.compact:
                return self.compact[0].get_size()
            else:
                return 0
        else:
            n=sum([c.get_size(False) for c in self.compact])
            n+=self.loose.get_size(False)
            return n
    def __iter__(self):
        self.ensure_compact()
        try:
            return iter(self.compact[0])
        except IndexError:
            return iter([])

    def get_type_counts(self, k=0):
        return self.get_compact().get_type_counts(k)
    def remap(self, k, filt):
        return self.get_compact().remap(k,filt)


cdef class IVecV2_iter

cdef class VecV2:
    """mutable sparse int matrix based on item vector"""
    def __init__(self):
        self.is_compact=True
    def item_iter(self):
        self.ensure_compact()
        return IVecV2_iter(self)
    def get_size(self,compactify=True):
        if compactify:
            self.ensure_compact()
        return self.vec.size()
    def __len__(self):
        return self.get_size(True)
    def clear(self):
        self.vec.resize(0)
        self.is_compact=True
    cdef void compactify(self):
        cdef c_SmallerAddrV2 comp
        c_IVecV2_sort(self.vec.begin(),self.vec.end(),comp)
        c_compactifyV2(&self.vec)
        self.is_compact=True
    cdef void ensure_compact(self):
        if not self.is_compact:
            self.compactify()
    cpdef bint get_count(self,coordinate_t k0,coordinate_t k1):
        cdef c_CItemV2 ci
        ci.addr[0]=k0
        ci.addr[1]=k1
        return c_get_countV2(&self.vec,ci)
    cdef void c_add(self, coordinate_t k0,coordinate_t k1):
        cdef c_CItemV2 c
        c.addr[0]=k0
        c.addr[1]=k1
        self.vec.push_back(c)
        self.is_compact=False
    def add_count(self, coordinate_t k0,coordinate_t k1):
        self.c_add(k0,k1)
    def __add__(VecV2 self, VecV2 other):
        cdef c_CItemV2 c1, c2
        cdef unsigned int i1, i2
        cdef int k, delta
        cdef VecV2 result
        self.ensure_compact()
        other.ensure_compact()
        result=VecV2()
        result.vec.reserve(max(self.vec.size(),other.vec.size()))
        i1=0
        i2=0
        while i1<self.vec.size() and i2<other.vec.size():
            c1=self.vec.at(i1)
            c2=other.vec.at(i2)
            delta=0
            for k from 0<=k<2:
                if c1.addr[k]<c2.addr[k]:
                    delta=-1
                    break
                elif c1.addr[k]>c2.addr[k]:
                    delta=1
                    break
            if delta==0:
                result.vec.push_back(c1)
                i1+=1
                i2+=1
            elif delta==-1:
                result.vec.push_back(c1)
                i1+=1
            else:
                result.vec.push_back(c2)
                i2+=1
        while i1<self.vec.size():
            c1=self.vec.at(i1)
            result.vec.push_back(c1)
            i1+=1
        while i2<other.vec.size():
            c2=other.vec.at(i2)
            result.vec.push_back(c2)
            i2+=1
        return result

    def tofile(self, f):
        cdef long i, n, k, k_max
        n=self.get_size()
        f.write('V2 %d\n'%(n,))
        k=0
        for i from 0<=i<n/CHUNK_SIZE:
            s=PyBytes_FromStringAndSize(<char *>c_VecV2_get_pointer(&self.vec,k), CHUNK_SIZE*sizeof(c_CItemV2))
            f.write(s)
            k+=CHUNK_SIZE
        if k<n:
            s=PyBytes_FromStringAndSize(<char *>c_VecV2_get_pointer(&self.vec,k), (n-k)*sizeof(c_CItemV2))
            f.write(s)
    def fromfile(self, f):
        cdef long i, n
        line=f.readline().strip().split()
        assert line[0]=='V2'
        n=long(line[1])
        self.clear()
        self.vec.resize(n)
        k=0
        for i from 0<=i<n/CHUNK_SIZE:
            s=f.read(CHUNK_SIZE*sizeof(c_CItemV2))
            assert len(s)==CHUNK_SIZE*sizeof(c_CItemV2)
            memcpy(<void *>c_VecV2_get_pointer(&self.vec,k), <char *>s, CHUNK_SIZE*sizeof(c_CItemV2))
            k+=CHUNK_SIZE
        if k<n:
            s=f.read((n-k)*sizeof(c_CItemV2))
            assert len(s)==(n-k)*sizeof(c_CItemV2)
            memcpy(<void *>c_VecV2_get_pointer(&self.vec,k), <char *>s, (n-k)*sizeof(c_CItemV2))
    cpdef int get_maxcol(self, int k=0):
        cdef int i, n, new_val, col
        assert k>=0 and k<2
        n=self.vec.size()
        if n==0:
            return -1
        if k==0:
            return self.vec.at(n-1).addr[0]
        col=-1
        n=self.vec.size()
        for i from 0<=i<n:
            new_val=self.vec.at(i).addr[k]
            if new_val>col:
                col=new_val
        return col
    cpdef numpy.ndarray get_type_counts(self, int k=0):
        cdef int old_val, new_val, i, n, count
        cdef numpy.ndarray[numpy.int32_t, ndim=1] result
        assert k>=0 and k<2
        if self.vec.size()==0:
            return numpy.zeros(0, numpy.int32)
        self.ensure_compact()
        n_types=self.get_maxcol(k)+1
        result=numpy.zeros(n_types,numpy.int32)
        n=self.vec.size()
        for i from 0<=i<n:
            new_val=self.vec.at(i).addr[k]
            result[new_val]+=1
        return result
    cpdef VecV2 remap(self, int k, numpy.ndarray filt):
        cdef numpy.ndarray[numpy.int8_t, ndim=1] wanted=filt.astype('b')
        cdef numpy.ndarray[numpy.int_t, ndim=1] target=filt.cumsum()-1
        self.ensure_compact()
        cdef int n=self.vec.size()
        cdef int i
        cdef VecV2 result=VecV2()
        cdef c_CItemV2 c
        for i from 0<=i<n:
            c=self.vec.at(i)
            if wanted[c.addr[k]]:
                c.addr[k]=target[c.addr[k]]
                result.vec.push_back(c)
        return result
    def append(self,tup):
        k0,k1=tup
        self.c_add(k0,k1)
    def extend(self,tups):
        for k0,k1 in tups:
            self.c_add(k0,k1)
    def __iter__(self):
        return self.item_iter()
    def __reduce__(self):
        return (VecV2,(),(),self.item_iter())
    def __setstate__(self,state):
        pass
    def __repr__(self):
        return 'VecV2(%s)'%(str(list(self)),)
    def __dealloc__(self):
        # need to call destructor explicitly
        self.vec.cxx_destructor()

cdef class IVecV2_iter:
    cdef VecV2 vec
    cdef c_VecV2 *vecC
    cdef unsigned int k
    def __init__(self,vec):
        self.vec=vec
        self.vecC=&self.vec.vec
        self.k=0
    def __iter__(self):
        return self
    def __next__(self):
        if self.k>=self.vecC.size():
            raise StopIteration
        cdef c_CItemV2 res=self.vecC.at(self.k)
        self.k+=1
        return (res.addr[0],
                res.addr[1])

cdef class LargeVecV2:
    cdef public object compact
    cdef public VecV2 loose
    def __init__(self):
        self.compact=[]
        self.loose=VecV2()
        self.loose.vec.reserve(100000)
    def ensure_compact(self):
        if self.loose.get_size()>0:
            self.compact.append(self.loose)
        self.loose=VecV2()
        while len(self.compact)>=2:
            self.compact[-2:]=[self.compact[-2]+self.compact[-1]]
    def add_count(self, coordinate_t k0,coordinate_t k1):
        self.loose.c_add(k0,k1)
        if self.loose.get_size(False)>=100000:
            self.loose.ensure_compact()
            self.compact.append(self.loose)
            self.loose=VecV2()
            self.loose.vec.reserve(100000)
            while (len(self.compact)>=2 and
                   self.compact[-1].get_size()>=0.6*self.compact[-2].get_size()):
                self.compact[-2:]=[self.compact[-2]+self.compact[-1]]
    def get_compact(self):
        self.ensure_compact()
        if len(self.compact)>0:
            return self.compact[0]
        else:
            return VecV2()
    def get_size(self, compactify=True):
        cdef int n
        if compactify:
            self.ensure_compact()
            if self.compact:
                return self.compact[0].get_size()
            else:
                return 0
        else:
            n=sum([c.get_size(False) for c in self.compact])
            n+=self.loose.get_size(False)
            return n
    def __iter__(self):
        self.ensure_compact()
        try:
            return iter(self.compact[0])
        except IndexError:
            return iter([])

    def get_type_counts(self, k=0):
        return self.get_compact().get_type_counts(k)
    def remap(self, k, filt):
        return self.get_compact().remap(k,filt)


cdef class IVecV3_iter

cdef class VecV3:
    """mutable sparse int matrix based on item vector"""
    def __init__(self):
        self.is_compact=True
    def item_iter(self):
        self.ensure_compact()
        return IVecV3_iter(self)
    def get_size(self,compactify=True):
        if compactify:
            self.ensure_compact()
        return self.vec.size()
    def __len__(self):
        return self.get_size(True)
    def clear(self):
        self.vec.resize(0)
        self.is_compact=True
    cdef void compactify(self):
        cdef c_SmallerAddrV3 comp
        c_IVecV3_sort(self.vec.begin(),self.vec.end(),comp)
        c_compactifyV3(&self.vec)
        self.is_compact=True
    cdef void ensure_compact(self):
        if not self.is_compact:
            self.compactify()
    cpdef bint get_count(self,coordinate_t k0,coordinate_t k1,coordinate_t k2):
        cdef c_CItemV3 ci
        ci.addr[0]=k0
        ci.addr[1]=k1
        ci.addr[2]=k2
        return c_get_countV3(&self.vec,ci)
    cdef void c_add(self, coordinate_t k0,coordinate_t k1,coordinate_t k2):
        cdef c_CItemV3 c
        c.addr[0]=k0
        c.addr[1]=k1
        c.addr[2]=k2
        self.vec.push_back(c)
        self.is_compact=False
    def add_count(self, coordinate_t k0,coordinate_t k1,coordinate_t k2):
        self.c_add(k0,k1,k2)
    def __add__(VecV3 self, VecV3 other):
        cdef c_CItemV3 c1, c2
        cdef unsigned int i1, i2
        cdef int k, delta
        cdef VecV3 result
        self.ensure_compact()
        other.ensure_compact()
        result=VecV3()
        result.vec.reserve(max(self.vec.size(),other.vec.size()))
        i1=0
        i2=0
        while i1<self.vec.size() and i2<other.vec.size():
            c1=self.vec.at(i1)
            c2=other.vec.at(i2)
            delta=0
            for k from 0<=k<3:
                if c1.addr[k]<c2.addr[k]:
                    delta=-1
                    break
                elif c1.addr[k]>c2.addr[k]:
                    delta=1
                    break
            if delta==0:
                result.vec.push_back(c1)
                i1+=1
                i2+=1
            elif delta==-1:
                result.vec.push_back(c1)
                i1+=1
            else:
                result.vec.push_back(c2)
                i2+=1
        while i1<self.vec.size():
            c1=self.vec.at(i1)
            result.vec.push_back(c1)
            i1+=1
        while i2<other.vec.size():
            c2=other.vec.at(i2)
            result.vec.push_back(c2)
            i2+=1
        return result

    def tofile(self, f):
        cdef long i, n, k, k_max
        n=self.get_size()
        f.write('V3 %d\n'%(n,))
        k=0
        for i from 0<=i<n/CHUNK_SIZE:
            s=PyBytes_FromStringAndSize(<char *>c_VecV3_get_pointer(&self.vec,k), CHUNK_SIZE*sizeof(c_CItemV3))
            f.write(s)
            k+=CHUNK_SIZE
        if k<n:
            s=PyBytes_FromStringAndSize(<char *>c_VecV3_get_pointer(&self.vec,k), (n-k)*sizeof(c_CItemV3))
            f.write(s)
    def fromfile(self, f):
        cdef long i, n
        line=f.readline().strip().split()
        assert line[0]=='V3'
        n=long(line[1])
        self.clear()
        self.vec.resize(n)
        k=0
        for i from 0<=i<n/CHUNK_SIZE:
            s=f.read(CHUNK_SIZE*sizeof(c_CItemV3))
            assert len(s)==CHUNK_SIZE*sizeof(c_CItemV3)
            memcpy(<void *>c_VecV3_get_pointer(&self.vec,k), <char *>s, CHUNK_SIZE*sizeof(c_CItemV3))
            k+=CHUNK_SIZE
        if k<n:
            s=f.read((n-k)*sizeof(c_CItemV3))
            assert len(s)==(n-k)*sizeof(c_CItemV3)
            memcpy(<void *>c_VecV3_get_pointer(&self.vec,k), <char *>s, (n-k)*sizeof(c_CItemV3))
    cpdef int get_maxcol(self, int k=0):
        cdef int i, n, new_val, col
        assert k>=0 and k<3
        n=self.vec.size()
        if n==0:
            return -1
        if k==0:
            return self.vec.at(n-1).addr[0]
        col=-1
        n=self.vec.size()
        for i from 0<=i<n:
            new_val=self.vec.at(i).addr[k]
            if new_val>col:
                col=new_val
        return col
    cpdef numpy.ndarray get_type_counts(self, int k=0):
        cdef int old_val, new_val, i, n, count
        cdef numpy.ndarray[numpy.int32_t, ndim=1] result
        assert k>=0 and k<3
        if self.vec.size()==0:
            return numpy.zeros(0, numpy.int32)
        self.ensure_compact()
        n_types=self.get_maxcol(k)+1
        result=numpy.zeros(n_types,numpy.int32)
        n=self.vec.size()
        for i from 0<=i<n:
            new_val=self.vec.at(i).addr[k]
            result[new_val]+=1
        return result
    cpdef VecV3 remap(self, int k, numpy.ndarray filt):
        cdef numpy.ndarray[numpy.int8_t, ndim=1] wanted=filt.astype('b')
        cdef numpy.ndarray[numpy.int_t, ndim=1] target=filt.cumsum()-1
        self.ensure_compact()
        cdef int n=self.vec.size()
        cdef int i
        cdef VecV3 result=VecV3()
        cdef c_CItemV3 c
        for i from 0<=i<n:
            c=self.vec.at(i)
            if wanted[c.addr[k]]:
                c.addr[k]=target[c.addr[k]]
                result.vec.push_back(c)
        return result
    def append(self,tup):
        k0,k1,k2=tup
        self.c_add(k0,k1,k2)
    def extend(self,tups):
        for k0,k1,k2 in tups:
            self.c_add(k0,k1,k2)
    def __iter__(self):
        return self.item_iter()
    def __reduce__(self):
        return (VecV3,(),(),self.item_iter())
    def __setstate__(self,state):
        pass
    def __repr__(self):
        return 'VecV3(%s)'%(str(list(self)),)
    def __dealloc__(self):
        # need to call destructor explicitly
        self.vec.cxx_destructor()

cdef class IVecV3_iter:
    cdef VecV3 vec
    cdef c_VecV3 *vecC
    cdef unsigned int k
    def __init__(self,vec):
        self.vec=vec
        self.vecC=&self.vec.vec
        self.k=0
    def __iter__(self):
        return self
    def __next__(self):
        if self.k>=self.vecC.size():
            raise StopIteration
        cdef c_CItemV3 res=self.vecC.at(self.k)
        self.k+=1
        return (res.addr[0],
                res.addr[1],
                res.addr[2])

cdef class LargeVecV3:
    cdef public object compact
    cdef public VecV3 loose
    def __init__(self):
        self.compact=[]
        self.loose=VecV3()
        self.loose.vec.reserve(100000)
    def ensure_compact(self):
        if self.loose.get_size()>0:
            self.compact.append(self.loose)
        self.loose=VecV3()
        while len(self.compact)>=2:
            self.compact[-2:]=[self.compact[-2]+self.compact[-1]]
    def add_count(self, coordinate_t k0,coordinate_t k1,coordinate_t k2):
        self.loose.c_add(k0,k1,k2)
        if self.loose.get_size(False)>=100000:
            self.loose.ensure_compact()
            self.compact.append(self.loose)
            self.loose=VecV3()
            self.loose.vec.reserve(100000)
            while (len(self.compact)>=2 and
                   self.compact[-1].get_size()>=0.6*self.compact[-2].get_size()):
                self.compact[-2:]=[self.compact[-2]+self.compact[-1]]
    def get_compact(self):
        self.ensure_compact()
        if len(self.compact)>0:
            return self.compact[0]
        else:
            return VecV3()
    def get_size(self, compactify=True):
        cdef int n
        if compactify:
            self.ensure_compact()
            if self.compact:
                return self.compact[0].get_size()
            else:
                return 0
        else:
            n=sum([c.get_size(False) for c in self.compact])
            n+=self.loose.get_size(False)
            return n
    def __iter__(self):
        self.ensure_compact()
        try:
            return iter(self.compact[0])
        except IndexError:
            return iter([])

    def get_type_counts(self, k=0):
        return self.get_compact().get_type_counts(k)
    def remap(self, k, filt):
        return self.get_compact().remap(k,filt)



def run_test():
    cdef VecI2 counts
    counts=VecI2()
    counts.c_add(0,1,12)
    counts.c_add(0,13,1312)
    counts.c_add(1,0,8)
    counts.c_add(0,0,23)
    counts.c_add(0,1,12)
    counts.c_add(1,0,8)
    for xs in counts.item_iter():
        print ' '.join([str(x) for x in xs])
    cdef CSRMatrixI mat
    mat=counts.to_csr()
    mat.print_csr()
    print "mat[0,1]==%d"%(mat.get_count(0,1))
    counts.clear()
    counts.c_add(2,1,12)
    counts.c_add(1,13,1312)
    counts.c_add(0,12,2412)
    counts.c_add(1,0,8)
    counts.c_add(0,2,23)
    counts.c_add(0,1,12)
    counts.c_add(1,0,8)
    for xs in counts.item_iter():
        print ' '.join([str(x) for x in xs])
    mat2=counts.to_csr()
    mat2.print_csr()
    mat3=mat+mat2
    mat3.print_csr()
    mat3.write_binary(open('/tmp/mat3','w'))
    mat4=mmapCSR(open('/tmp/mat3'))
    mat4.print_csr()
    mat4=None
    mat5=mat3.transpose()
    mat5.print_csr()
    mat6=mat5.transpose()
    mat6.print_csr()
    for xs in mat6.item_iter():
        print ' '.join([str(x) for x in xs])
    for sv in mat6:
        print sv