cimport lurrn.sparsmat as sparsmat
from lurrn import sparsmat
from python_list cimport PyList_Append, PyList_GET_SIZE
from python_mem cimport PyMem_Malloc, PyMem_Free, PyMem_Realloc
import simplejson as json
import alphabet
cimport alphabet
import numpy
cimport numpy

try:
    from itertools import izip
except ImportError:
    # Python 3+
    izip = zip

import sys
from collections import defaultdict

cdef extern from *:
    ctypedef char *const_char_ptr "const char *"

cdef extern from "strings.h":
    ctypedef unsigned int size_t
    void bzero(void *s, size_t n)
cdef extern from "math.h":
    double pow(double x, double y)

cdef extern from "cxx_gram.h":
    cdef char *escape_amis(const_char_ptr x)
    cdef char *unescape_amis(const_char_ptr x)

cdef class ExtensibleArray:
    cdef numpy.ndarray values
    cdef object tp
    cdef int max_pos
    def __init__(self, tp='i'):
        self.values = numpy.zeros(16,tp)
        self.tp = tp
        self.max_pos = 0
    def __getitem__(self, pos):
        if pos >= len(self.values):
            return 0
        else:
            return self.values[pos]
    def __setitem__(self, pos, val):
        if pos >= len(self.values):
            new_len = <int>pos*1.25
            vals_new = numpy.zeros(new_len, self.tp)
            vals_new[:len(self.values)] = self.values
            self.values = vals_new
        if pos >= self.max_pos:
            self.max_pos = pos+1
        self.values[pos] = val
    cpdef do_remap(self, numpy.int8_t[:] filt):
        cdef int i, k
        cdef int old_max_pos
        k = 0
        for i from 0<=i<self.max_pos:
            if filt[i]:
                self.values[k] = self.values[i]
                k += 1
        old_max_pos = k
        while k < self.max_pos:
            self.values[k] = 0
            k += 1
        self.max_pos = old_max_pos
    def __len__(self):
        return self.max_pos
    def get_array(self):
        return self.values[:self.max_pos]

def flist_sort_order(x):
    if isinstance(x,basestring):
        return x
    else:
        return x[0]

cdef class FeatureList:
    cdef object features
    cdef double *vals
    def __init__(self, flist):
        cdef int i,n
        n=len(flist)
        flist=sorted(flist, key=flist_sort_order)
        self.features=[]
        self.vals=<double *>PyMem_Malloc(n*sizeof(double))
        for i from 0<=i<n:
            feat=flist[i]
            if isinstance(feat,basestring):
                PyList_Append(self.features,feat)
                self.vals[i]=1.0
            else:
                f,c = feat
                PyList_Append(self.features,f)
                self.vals[i]=c
    def __getitem__(self,n):
        return (self.features[n],self.vals[n])
    def __len__(self):
        return PyList_GET_SIZE(self.features)
    cdef object get_feat(self,int n):
        return self.features[n]
    cdef double get_val(self,int n):
        return self.vals[n]
    cdef double get_p_norm(self,p):
        cdef int i,n
        cdef double v, cumsum
        cumsum=0.0
        n=PyList_GET_SIZE(self.features)
        for i from 0<=i<n:
            cumsum+=pow(self.vals[i],p)
        return pow(cumsum,1.0/p)
    def __add__(self, other):
        xs=list(self)+list(other)
        return FeatureList(xs)
    def diff_with(self, FeatureList other):
        cdef bint matches
        cdef int i
        cdef int idx1, idx2
        if len(self)==len(other):
            matches=True
            for i from 0<=i<len(self):
                if self.features[i]!=other.features[i]:
                    matches=False
                    break
            if matches:
                for i from 0<=i<len(self):
                    if self.vals[i]!=other.vals[i]:
                        matches=False
                        break
                if matches:
                    return None
        result=[]
        idx1=0; idx2=0
        while idx1<len(self) and idx2<len(other):
            (f1,v1)=self[idx1]
            (f2,v2)=other[idx2]
            if f1<f2:
                result.append(('-',f1,v1))
                idx1+=1
            elif f2<f1:
                result.append(('+',f2,v2))
                idx2+=1
            else:
                result.append((' ',f1,v1))
                idx1+=1
                idx2+=1
        while idx1<len(self):
            (f1,v1)=self[idx1]
            result.append(('-',f1,v1))
            idx1+=1
        while idx2<len(other):
            (f2,v2)=other[idx2]
            result.append(('+',f2,v2))
            idx2+=1
        return result
    def __str__(self):
        return 'FeatureList(%r)'%(list(self),)
    def __repr__(self):
        return 'FeatureList(%r)'%(list(self),)
    def __dealloc__(self):
        PyMem_Free(self.vals)

cdef class InfoNode:
    cdef public int nodeId
    cdef public object tp
    cdef public object features
    cdef public object edges_out
    def __init__(self, tp, feats=None):
        self.nodeId=-1
        self.tp=tp
        if feats==None:
            self.features=[]
        else:
            self.features=feats
        self.edges_out=[]
    def add_edge(self, other_node, edge_label=None):
        self.edges_out.append((other_node,edge_label))
    def as_tree(self,parts_out):
        if self.features or self.edges_out:
            parts_out.append('(%s '%(self.tp,))
        else:
            parts_out.append('(%s'%(self.tp,))
        parts_out.append(''.join(['(%s %s)'%(x,x) for x in self.features]))
        for edge,lbl in self.edges_out:
            edge.as_tree(parts_out)
        parts_out.append(')')
    def as_nodes(self, offset, node_list, edge_list):
        new_offset=offset
        if self.nodeId==-1:
            self.nodeId=new_offset
            new_offset+=1
        node_list.append((self.nodeId,self.tp))
        for edge,lbl in self.edges_out:
            new_offset=edge.as_nodes(new_offset, node_list, edge_list)
            edge_list.append((self.nodeId, edge.nodeId, lbl))
        for i,feat in enumerate(self.features):
            node_list.append((new_offset+i,feat))
            edge_list.append((self.nodeId, new_offset+i, '_f'))
        new_offset+=len(self.features)
        return new_offset
    def add_to_list(self, lst):
        if self.nodeId!=-1:
            pass
        else:
            self.nodeId=len(lst)
            lst.append(self)
            for edge,lbl in self.edges_out:
                edge.add_to_list(lst)

cdef class InfoTree:
    cdef public object nodes
    cdef public object roots
    def __init__(self,data=None):
        all_nodes=[]
        roots=[]
        if data is not None:
            nlist=data['nodes']
            for i,(lb,attrs,edges) in enumerate(nlist):
                node=InfoNode(lb,attrs)
                node.nodeId=i
                all_nodes.append(node)
            for i,(lb,attrs,edges) in enumerate(nlist):
                node=all_nodes[i]
                for x in edges:
                    if isinstance(x,int):
                        node.add_edge(all_nodes[x])
                    else:
                        node.add_edge(all_nodes[x[0]],x[1])
            for r in data['roots']:
                roots.append(all_nodes[r])
        self.nodes=all_nodes
        self.roots=roots
    def as_tree(self,parts_out):
        self.roots[0].as_tree(parts_out)
    def as_json(self):
        cdef InfoNode n
        roots=[]
        for n in self.roots:
            roots.append(n.nodeId)
        node_list=[]
        for n in self.nodes:
            node_list.append((n.tp, n.features, [(n1.nodeId, lbl) for (n1,lbl) in n.edges_out]))
        return {'nodes':node_list, 'roots':roots}
    def as_subdue(self,start_offset=1):
        node_list=[]
        for n in self.nodes:
            node_list.append(('v',n.nodeId+start_offset,n.tp))
        k=len(self.nodes)
        for n in self.nodes:
            for feat in n.features:
                node_list.append(('v',k+start_offset,feat))
                k+=1
        for n in self.nodes:
            for (n1,lbl) in n.edges_out:
                node_list.append(('e', n.nodeId+start_offset, n1.nodeId+start_offset, lbl))
        k=len(self.nodes)
        for n in self.nodes:
            for feat in n.features:
                node_list.append(('e',n.nodeId+start_offset,k+start_offset,'_f'))
                k+=1
        return (node_list,start_offset+k)
    def add_node(self, n, as_root=False):
        n.add_to_list(self.nodes)
        if as_root:
            self.roots.append(n)
    def diff_with(self, InfoTree other):
        cdef int i
        cdef InfoNode n1, n2
        # check that the backbones are isomorphic
        if len(self.nodes)!=len(other.nodes):
            return (self,other)
        for i from 0<=i<len(self.nodes):
            n1=self.nodes[i]
            n2=other.nodes[i]
            if n1.tp!=n2.tp:
                return (self,other)
            for ((n1a,lbl1),(n2a,lbl2)) in izip(n1.edges_out,n2.edges_out):
                if lbl1!=lbl2 or n1a.nodeId!=n2a.nodeId:
                    return (self,other)
        return None

cdef class Multipart:
    cdef public object parts
    cdef public object trees
    def __init__(self, parts):
        self.parts=map(FeatureList,parts)
        self.trees=[]
    def __getitem__(self,n):
        return self.parts[n]
    def __len__(self):
        return len(self.parts)
    def add_tree(self,data):
        if self.trees is None:
            self.trees=[]
        self.trees.append(InfoTree(data))
    def diff_with(self, Multipart other):
        result_p=[]
        for part1, part2 in izip(self.parts,other.parts):
            d=part1.diff_with(part2)
            if d is not None:
                result_p.append(d)
        result_m={}
        if result_p:
            result_m['parts']=result_p
        result_g=[]
        for g1,g2 in izip(self.trees,other.trees):
            d=g1.diff_with(g2)
            if d is not None:
                result_g.append(d)
        if result_g:
            result_m['trees']=result_g
        if result_m:
            return result_m
        else:
            return None
    def as_json(self):
        return {'_type':'multipart',
                'parts':[list(x) for x in self.parts],
                'trees':[x.as_json() for x in self.trees]}

cdef class FCombo:
    cdef public alphabet.StringAlphabet dict
    cdef public alphabet.StringAlphabet dict_aux
    cdef public object escaped_words
    cdef public object codec
    cdef public object encoding
    cdef public double scale
    cdef public double norm_p
    cdef int bias_item
    cdef int n
    def __init__(self,n,mydict=None,bias_item=None, want_utf8=True,
                 scale=1.0, norm_p=0.0):
        self.n=n
        if mydict is None:
            self.dict=alphabet.StringAlphabet(want_utf8)
        else:
            self.dict=mydict
        if bias_item is None:
            self.bias_item=-1
        else:
            self.bias_item=self.dict[bias_item]
        self.escaped_words=[]
        self.codec=None
        self.norm_p=norm_p
        self.scale=scale
        if want_utf8:
            self.encoding = 'UTF-8'
        else:
            self.encoding = 'ISO-8859-15'
    def __call__(self,flist,labels=None):
        cdef bint is_multipart
        cdef FeatureList lst
        cdef Multipart flist_mp
        cdef double part_norm
        a=sparsmat.VecD1()
        is_multipart = isinstance(flist,Multipart)
        if self.bias_item>=0:
            if labels is None:
                a.add_count(self.bias_item,1.0)
            else:
                suffix='^'+self.dict.words[self.bias_item]
                for lbl in labels:
                    try:
                        a.add_count(self.dict[lbl+suffix],1.0)
                    except KeyError:
                        pass
        if is_multipart:
            flist_mp=flist
            lst=flist_mp.parts[0]
        else:
            try:
                lst=flist
            except TypeError:
                lst=FeatureList(flist)
        if labels is None:
            self.mkvec(self.n,lst,0,[],1.0,a)
        else:
            for lbl in labels:
                self.mkvec(self.n,lst,0,[lbl],1.0,a)
        if is_multipart:
            for i from 1<=i<len(flist):
                lst=flist_mp.parts[i]
                if len(lst)>0:
                    if self.norm_p==0.0:
                        part_norm=self.scale
                    else:
                        part_norm=self.scale/lst.get_p_norm(self.norm_p)
                    if labels is None:
                        self.mkvec(1,lst,0,[],part_norm,a)
                    else:
                        for lbl in labels:
                            self.mkvec(1,lst,0,[lbl],part_norm,a)
        return a.to_sparse()
    def unmunge(self, vec):
        result = []
        for k,v in vec:
            result.append((self.dict.get_sym(k), v))
        return result
    def munge_uni(self, flist):
        """
        takes a feature list and creates a list of sparse vectors
        (without feature combination)
        """
        is_multipart = isinstance(flist,Multipart)
        if is_multipart:
            parts=flist.parts
        else:
            parts=[FeatureList(flist)]
        vecs=[]
        if self.dict_aux is None:
            self.dict_aux=alphabet.StringAlphabet(self.dict.use_utf8)
        for lst in parts:
            a=sparsmat.VecD1()
            self.mkvec1(lst,a)
            vecs.append(a.to_sparse())
        return vecs
    def munge_vec(self, vecs, mask, labels=None):
        """
        applies feature combination to a list of sparse vectors,
        combining with a set of (target) labels
        """
        a=sparsmat.VecD1()
        if labels is None:
            self.mkmunge(self.n,vecs,mask,0,0,[],1.0,a)
        else:
            for lbl in labels:
                self.mkmunge(self.n,vecs,mask,0,0,[lbl],1.0,a)
        return a.to_sparse()
    def munge_vec_2(self, vecs, mask, kinds):
        a=sparsmat.VecD1()
        vecs1=[]
        mask1=[]
        vecs2=[]
        mask2=[]
        for v,m,k in izip(vecs, mask, kinds):
            if k==0:
                # ignore
                pass
            elif k==1:
                vecs1.append(v)
                mask1.append(m)
            elif k==2:
                vecs2.append(v)
                mask2.append(m)
            else:
                sys.stderr.write('Unknown part type %d\n'%(k,))
        self.mkmunge(self.n,vecs2,mask2,0,0,[],1.0,a)
        if vecs1:
            self.mkmunge(1,vecs1,mask1,0,0,[],1.0,a)
        return a.to_sparse()
    def to_svmltk(self,flist,f):
        cdef Multipart mp
        cdef FeatureList lst
        cdef InfoTree tree
        cdef int i
        mp=flist
        if mp.trees is not None:
            for i from 0<=i<len(mp.trees):
                lst_parts=[]
                tree=mp.trees[i]
                tree.as_tree(lst_parts)
                f.write(' |BT| ')
                for part in lst_parts:
                    if isinstance(part,unicode):
                        f.write(part.encode('UTF-8'))
                    else:
                        f.write(part)
            f.write('|ET|')
        self(mp[0]).write_pairs(f)
        for i from 1<=i<len(mp):
            lst=mp[i]
            f.write('|BV|')
            self(lst).write_pairs(f)
            f.write(' |EV| ')
        lst_parts=[]
    def __reduce__(self):
        return (FCombo,(self.n,self.dict),None)
    def __setstate__(self):
        pass
    cpdef escaped(self, int n):
        cdef int i
        if len(self.escaped_words)<=n:
            for i from len(self.escaped_words)<=i<=n:
                w=self.dict.num2sym(i)
                self.escaped_words.append(escape_amis(w))
        return self.escaped_words[n]
    cpdef unescaped(self, w0):
        cdef const_char_ptr w1
        w1=unescape_amis(w0)
        if self.codec is None:
            w2=w1
        else:
            w2=self.codec.decode(w1)[0]
        return self.dict[w2]
    cdef mkvec(self,int n,FeatureList flist,int pos,partial,double w, sparsmat.VecD1 a):
        cdef int i
        cdef object feat
        cdef double ww, c
        for i from pos<=i<len(flist):
            partial.append(flist.get_feat(i))
            ww=w*flist.get_val(i)
            try:
                a.add_count(self.dict['^'.join(partial)],ww)
            except KeyError:
                pass
            if n>1:
                self.mkvec(n-1,flist,i+1,partial,ww,a)
            del partial[-1]
    cdef mkvec1(self,FeatureList flist, sparsmat.VecD1 a):
        cdef int i
        cdef object feat
        for i from 0<=i<len(flist):
            try:
                a.add_count(self.dict_aux[flist.get_feat(i)],flist.get_val(i))
            except KeyError:
                pass
    cdef mkmunge1(self, int n, flist, masks, int posL, int posF, partial, double w, sparsmat.VecD1 a):
        # looks at features from the current part posL,
        # starting at posF
        cdef int i
        cdef object feat
        cdef double ww, c
        cdef sparsmat.SparseVectorD vec=flist[posL]
        cdef object mask
        if posL>=len(masks):
            mask=None
        else:
            mask=masks[posL]
        for i from posF<=i<len(vec):
            if mask is None or mask[vec.idx_ptr[i]]:
                partial.append(self.dict_aux.num2sym(vec.idx_ptr[i]))
                ww=w*vec.vals_ptr[i]
                try:
                    a.add_count(self.dict['^'.join(partial)],ww)
                except KeyError:
                    pass
                if n>1:
                    self.mkmunge(n-1,flist,masks,posL,i+1,partial,ww,a)
                del partial[-1]
    cdef mkmunge(self, int n, flist, masks, int posL, int posF, partial, double w, sparsmat.VecD1 a):
        # looks at features, starting with part posL, feature posF
        cdef int i
        self.mkmunge1(n,flist,masks,posL,posF,partial,w,a)
        if posL<len(flist):
            for i from posL<i<len(flist):
                self.mkmunge1(n,flist,masks,i,0,partial,w,a)
    def set_dict(self,new_dict):
        if self.bias_item!=-1:
            self.bias_item=new_dict[self.dict.words[self.bias_item]]
        self.dict=new_dict
    def load_weights(self,f):
        cdef alphabet.StringAlphabet mydict
        mydict=alphabet.StringAlphabet()
        ws=[]
        for l in f:
            line=l.strip().split()
            mydict.sym2num(unescape_amis(line[0]))
            ws.append(float(line[1]))
        mydict.growing=False
        self.dict=mydict
        return numpy.array(ws,'d')

    def serialize(self, d):
        if self.bias_item == -1:
            bias_item = None
        else:
            bias_item = self.dict.num2sym(self.bias_item)
        d['opts'] = json.dumps([self.n, bias_item, self.encoding])
        self.dict.tofile_utf8(d.writer('alph'))
        return 'FCombo'

def deserialize_fcombo(d):
    opts = json.loads(d['opts'])
    fc = FCombo(opts[0], bias_item=opts[1],
                want_utf8 = (opts[2].upper() == 'UTF-8'))
    fc.dict.fromfile_utf8(d.reader('alph'))
    return fc


cdef class CountingBloomFilter:
    cdef unsigned char *values
    cdef unsigned int size
    cdef int n_addrs
    def __init__(self, size=805306457, n_addrs=5):
        self.n_addrs = n_addrs
        self.size = size
        self.values = <unsigned char *>PyMem_Malloc(size)
        bzero(self.values, size)
    def __del__(self):
        PyMem_Free(self.values)
    cpdef int get_count(self, s):
        cdef int a = hash(s)
        cdef int b, i, max_val, other_val
        max_val = self.values[a%self.size]
        if max_val == 0:
            return 0
        b = hash('foo|%s'%(s,))
        for i from 1<=i<self.n_bits:
            other_val = self.values[(a+i*b)%self.size]
            if other_val < max_val:
                max_val = other_val
        return max_val
    cpdef int add_count(self, s, int n):
        cdef int a = hash(s)
        cdef int b, i, max_val, other_val, addr
        max_val = 255
        b = hash('foo|%s'%(s,))
        for i from 0<=i<self.n_addrs:
            addr = (a+i*b)%self.size
            other_val = self.values[addr] + n
            self.values[addr] = other_val
            if other_val < max_val:
                max_val = other_val
        return max_val
    cpdef int remove_count(self, s, int n):
        cdef int a = hash(s)
        cdef int b, i, max_val, other_val, addr
        max_val = 255
        b = hash('foo|%s'%(s,))
        for i from 0<=i<self.n_addrs:
            addr = (a+i*b)%self.size
            other_val = self.values[addr] - n
            if other_val < 0:
                print >>sys.stderr, "Holy Blooming Filters! There's something wrong!"
                other_val = 0
            self.values[addr] = other_val
            if other_val < max_val:
                max_val = other_val
        return max_val

cdef class BloomFMap:
    '''
    maps string lists to features, but skips the first 3 instances
    of a feature; this helps to deal with features that are too
    sparse and do not occur frequently.
    '''
    cdef public CountingBloomFilter bloom
    cdef public int min_count
    cdef public alphabet.StringAlphabet dict
    def __init__(self):
        self.bloom = CountingBloomFilter()
        self.min_count = 3
        self.dict = alphabet.StringAlphabet()
    def __call__(self, lst):
        result = sparsmat.VecD1()
        for s in lst:
            if s in self.dict:
                result.add_count(self.dict[s])
            else:
                count = self.bloom.add_count(s, 1)
                if count >= self.min_count:
                    result.add_count(self.dict[s])
                    self.bloom.remove_count(s, self.min_count)
        return result.to_sparse()

cdef class FeatureHasher:
    '''
    Hash-kernel transformation of features to ints
    suitable primes for num_dimensions:
        98317 393241 1572869 6291469
    see also http://planetmath.org/goodhashtableprimes
    '''
    cdef readonly int num_dimensions
    cdef long int *hash_cache
    cdef long hash_cache_size
    cdef sparsmat.VecD1 result_cache
    def __init__(self, dimensions=514229):
        self.num_dimensions = dimensions
        self.hash_cache = NULL
        self.result_cache = sparsmat.VecD1()
    def __dealloc__(self):
        if self.hash_cache != NULL:
            PyMem_Free(self.hash_cache)
        self.hash_cache = NULL
    def __call__(self, lst, target=None):
        cdef long int tgt_hash
        cdef sparsmat.VecD1 result
        tgt_hash = hash(target) * 97
        result = sparsmat.VecD1()
        for s in lst:
            k = (hash(s)+tgt_hash)%self.num_dimensions
            result.add_count(k, 1)
        return result.to_sparse()
    def cross(self, lst, lst2):
        cdef long *tgt_hash
        cdef long hash_s
        cdef int i
        cdef int n
        cdef sparsmat.VecD1 result = self.result_cache
        result.clear()
        n = len(lst2)
        if self.hash_cache == NULL:
            self.hash_cache = <long *> PyMem_Malloc(n * sizeof(long))
            self.hash_cache_size = n
        elif self.hash_cache_size < n:
            self.hash_cache = <long *> PyMem_Realloc(self.hash_cache, n * sizeof(long))
            self.hash_cache_size = n
        tgt_hash = self.hash_cache
        for i from 0<=i<n:
            tgt_hash[i] = hash(lst2[i])*97
        result = sparsmat.VecD1()
        for s in lst:
            hash_s = hash(s)
            for i from 0<=i<n:
                k = (hash_s+tgt_hash[i])%self.num_dimensions
                result.add_count(k, 1)
        return result.to_sparse()

cdef empty_sv=sparsmat.VecD1().to_sparse()

cdef class GraphNode:
    cdef bint isDisjunctive
    cdef int nodeId
    cdef public object label
    cdef public object children
    cdef public object features
    cdef public sparsmat.SparseVectorD featuresD
    cdef public double localScore
    cdef public double insideScore
    def __cinit__(self,label,isDisj=False):
        self.label=label
        self.children=[]
        self.nodeId=-1
        self.features=[]
        self.localScore=0.0
        self.insideScore=-1.0
        self.isDisjunctive=isDisj
    def __str__(self):
        return 'GraphNode(%s,%s)#%d'%(self.label,self.isDisjunctive,self.nodeId)
    cpdef assign_ids(self, int start):
        cdef int last_id=start
        cdef GraphNode n
        if self.nodeId!=-1:
            return last_id
        for n in self.children:
            last_id=n.assign_ids(last_id)
        self.nodeId=last_id
        last_id+=1
        return last_id
    cpdef double calcPotential(self, weights, FCombo fc, done):
        cdef GraphNode n
        if self.nodeId==-1:
            raise ValueError("Need nodeIDs (%s)"%(self,))
        if done[self.nodeId]:
            return self.insideScore
        done[self.nodeId]=True
        if self.featuresD is None:
            self.featuresD=fc(self.features)
        self.localScore=self.featuresD.dotFull(weights)
        for n in self.children:
            n.calcPotential(weights,fc,done)
    cpdef double calcInsideMin(self, done) except? -1000.0:
        cdef GraphNode n
        cdef double val, val2
        if self.nodeId==-1:
            raise ValueError("Need nodeIDs (%s)"%(self,))
        if done[self.nodeId]:
            return self.insideScore
        done[self.nodeId]=True
        if self.isDisjunctive:
            # OR node - take MAX
            val=-1000.0
            for n in self.children:
                val2=n.calcInsideMin(done)
                if val2>=val:
                    val=val2
            self.insideScore=val+self.localScore
            return self.insideScore
        else:
            # AND node - take SUM
            val=self.localScore
            for n in self.children:
                val+=n.calcInsideMin(done)
            self.insideScore=val
            return self.insideScore
    cpdef GraphNode extractBest(self, FCombo fc):
        cdef GraphNode n, n_best
        cdef double val
        if self.isDisjunctive:
            val=-1000.0
            n_best=None
            for n in self.children:
                val2=n.insideScore
                ## print "\t%s=>%s"%(n.label,n.insideScore)
                if val2>=val:
                    val=val2
                    n_best=n
            assert n_best is not None, self
            ## print "(%s) best: %s val=%s"%(self,n_best,val)
            if n_best.featuresD is None:
                n_best.featuresD=fc(n_best.features)
            n_best=n_best.extractBest(fc)
            if self.featuresD is None:
                self.featuresD=fc(self.features)
            if len(self.featuresD)>0:
                #print "adding self[%s] to n_best[%s]"%(self.featuresD,
                #                                       n_best.featuresD)
                n_best.featuresD+=self.featuresD
            return n_best
        else:
            lst=[]
            for n in self.children:
                n_best=n.extractBest(fc)
                lst.append(n_best)
            n_best=GraphNode(self.label,False)
            n_best.features=self.features
            n_best.featuresD=self.featuresD
            n_best.children=lst
            return n_best
    cpdef extractLabels(self, FCombo fc, result):
        cdef GraphNode n, n_best
        cdef double val
        #print "extractLabels: %s"%(self,)
        if self.label is not None:
            result.append(self.label)
        if self.isDisjunctive:
            val=-1000.0
            n_best=None
            for n in self.children:
                val2=n.insideScore
                ## print "\t%s=>%s"%(n.label,n.insideScore)
                if val2>=val:
                    val=val2
                    n_best=n
            assert n_best is not None, self
            ## print "(%s) best: %s val=%s"%(self,n_best,val)
            n_best.extractLabels(fc, result)
        else:
            for n in self.children:
                n.extractLabels(fc,result)
    cpdef write_amis(self, features,
                     bint in_disj,
                     written,
                     out, FCombo fcomb):
        #cdef object[bint] written
        cdef GraphNode n
        if self.featuresD is None:
            self.featuresD=fcomb(self.features)
        if in_disj:
            if self.isDisjunctive:
                for n in self.children:
                    if features is None:
                        n.write_amis(self.featuresD,
                                     True, written, out, fcomb)
                    else:
                        n.write_amis(features+self.featuresD,
                                     True, written, out, fcomb)
            else:
                out.write('( _ ')
                if features:
                    for k,v in features:
                        out.write('%s:%s '%(fcomb.escaped(k),v))
                for k,v in self.featuresD:
                    out.write('%s:%s '%(fcomb.escaped(k),v))
                for n in self.children:
                    n.write_amis(empty_sv, False,
                                 written, out, fcomb)
                out.write(') ')
        else:
            if self.isDisjunctive:
                if self.nodeId==-1:
                    raise ValueError("Need nodeIDs (%s)"%(self,))
                assert features is None or len(features)==0
                if written[self.nodeId]:
                    out.write('$n%d '%(self.nodeId))
                else:
                    out.write('{ n%d '%(self.nodeId))
                    written[self.nodeId]=True
                    for n in self.children:
                        n.write_amis(self.featuresD, True,
                                     written, out, fcomb)
                    out.write('} ')
            else:
                if features:
                    for k,v in features:
                        out.write('%s:%s '%(fcomb.escaped(k),v))
                for k,v in self.featuresD:
                    out.write('%s:%s '%(fcomb.escaped(k),v))
                for n in self.children:
                    n.write_amis(None, False, written,
                                 out, fcomb)

def AndNode(label):
    return GraphNode(label)
def OrNode(label):
    return GraphNode(label,True)

class crf_dict(defaultdict):
    def __missing__(self,x):
        val=OrNode(None)
        self[x]=val
        return val

def make_simple_crf(nodes):
    cdef GraphNode n0, n1, n2
    states=[crf_dict() for nlist in nodes]
    cur_states={None:AndNode('*start*')}
    for k,ns in enumerate(nodes):
        next_states=states[k]
        for st_old,node_old in cur_states.iteritems():
            for n0 in ns:
                lbl=n0.label
                n1=AndNode('[%s]%s->%s'%(k,st_old,lbl))
                n1.children+=[n0,node_old]
                n1.features+=[('T%s-%s'%(st_old,lbl),1.0)]
                n2=next_states[lbl]
                n2.children.append(n1)
        cur_states=next_states
    n2=OrNode('*stop*')
    for k,st_old in cur_states.iteritems():
        n2.children.append(st_old)
        for n in st_old.children:
            n.features+=[('STOP',1.0)]
    return n2

def decode_simple_crf(GraphNode n,weights,fc):
    cdef GraphNode n_best, n2
    cdef int num_nodes
    num_nodes=n.assign_ids(0)
    n.calcPotential(weights,fc,numpy.zeros(num_nodes,'b'))
    n.calcInsideMin(numpy.zeros(num_nodes,'b'))
    n_best=n.extractBest(fc)
    result=[]
    n2=n_best
    while n2.children:
        result.append(n2.children[0])
        n2=n2.children[1]
    result.reverse()
    return result

def make_multilabel(labels,otherlabels,data,fc):
    cdef GraphNode top_node, and_node, inter_node, label_node
    top_node=OrNode('*start*')
    label_nodes={}
    for lbls in [labels]+otherlabels:
        and_node=AndNode(str(lbls))
        top_node.children.append(and_node)
        for lbl in lbls:
            if lbl in label_nodes:
                and_node.children.append(label_nodes[lbl])
            else:
                label_node=GraphNode(lbl)
                label_node.features=None
                label_node.featuresD=fc(data,labels=[lbl])
                inter_node=OrNode('OR-'+lbl)
                inter_node.children.append(label_node)
                label_nodes[lbl]=inter_node
                and_node.children.append(inter_node)
    top_node.children[0].localScore=1
    return top_node

def dump_example(GraphNode n, buf, FCombo fc):
    n.assign_ids(0)
    n.calcInsideMin(defaultdict(bool))
    n2=n.extractBest(fc)
    n2.write_amis(None,False,None,buf,fc)
    buf.write('\n')
    n.write_amis(None,False,defaultdict(bool),buf,fc)
    buf.write('\n\n')

def test():
    cdef GraphNode n, n2, n3
    a=FCombo(2)
    print a([('a',1),('b',1),('c',2),('d',2)])
    print a.dict.words
    # CRF example
    crf_len=3
    labels=[[AndNode(x) for x in 'abc']
            for k in xrange(crf_len)]
    for lbls in labels:
        for n in lbls:
            n.features.append(('L'+n.label,1.0))
            if n.label=='a':
                n.localScore=1.0
    n2=make_simple_crf(labels)
    n2.assign_ids(0)
    val=n2.calcInsideMin(defaultdict(bool))
    print val
    n3=n2.extractBest(a)
    print str(n2), n2.children
    print str(n3), n3.children
    n3.write_amis(None,False,None,sys.stdout,a)
    print
    n2.write_amis(None,False,defaultdict(bool),
                  sys.stdout,a)
    print
