#ifndef CXX_GRAM
#define CXX_GRAM
#include <vector>
#include <algorithm>
#include <stdint.h>

#define CSR_MAGIC 0x90ddea7

typedef uint32_t coordinate_t;

int as_typecode(const char* str, int n)
{
  return str[0]+(str[1]<<8)+
    (str[2]<<16)+(('0'+n)<<24);
}

template <typename T>
struct typecode {};

template <>
struct typecode<int>
{
  int operator()() {
    return as_typecode("INT",sizeof(int));
  }
};

template<>
struct typecode<float> {
  int operator()() {
    return as_typecode("FLT",sizeof(float));
  }
};

template<>
struct typecode<double> {
  int operator()() {
    return as_typecode("FLT",sizeof(double));
  }
};

template<int n, typename T>
  struct CountItem {
    coordinate_t addr[n];
    T item;
  };

template<int n>
  struct CountItem<n,void> {
    coordinate_t addr[n];
  };

template<int k, int n, typename T>
  struct smallerAddr_k {
    static bool isSmaller(CountItem<n,T> a, CountItem<n,T> b) {
      if (a.addr[k]<b.addr[k]) return true;
      if (a.addr[k]>b.addr[k]) return false;
      return smallerAddr_k<k+1,n,T>::isSmaller(a,b);
    }
  };

template<int n, typename T>
  struct smallerAddr_k<n,n,T> {
    static bool isSmaller(CountItem<n,T> a, CountItem<n,T> b) {
      return false;
    }
};

template<int n, typename T>
  struct smallerAddr {
    bool operator () (CountItem<n,T> a, CountItem<n,T> b) {
      for (int i=0; i<n; i++) {
	if (a.addr[i]<b.addr[i]) return true;
	if (a.addr[i]>b.addr[i]) return false;
      }
      return false;
    }
  };

template<int n, typename T>
  struct eqAddr {
    bool operator () (CountItem<n,T> a, CountItem<n,T> b) {
      for (int i=0; i<n; i++) {
	if (a.addr[i]!=b.addr[i]) return false;
      }
      return true;
    }
  };

template <typename T>
  T get_count_v2(std::vector<CountItem<2,T> >*v,
	       coordinate_t k1, coordinate_t k2)
{
  typename std::vector<CountItem<2,T> >::iterator it;
  CountItem<2,T> ci;
  ci.addr[0]=k1;
  ci.addr[1]=k2;
  it=lower_bound(v->begin(),v->end(),ci,smallerAddr<2,T>());
  if (eqAddr<2,T>()(*it,ci)) {
    return it->item;
  } else {
    return 0;
  }
}

template <typename T, int n>
  T get_count_v(std::vector<CountItem<n,T> >*v,
		CountItem<n,T> ci)
{
  typename std::vector<CountItem<n,T> >::iterator it;
  it=lower_bound(v->begin(),v->end(),ci,smallerAddr<n,T>());
  if (eqAddr<n,T>()(*it,ci)) {
    return it->item;
  } else {
    return 0;
  }
}

template <typename T, int n>
  CountItem<n,T> *get_pointer(std::vector<CountItem<n,T> >*v, size_t k)
{
  return &((*v)[k]);
}

template <int n>
int get_count_set(std::vector<CountItem<n,void> >*v,
		CountItem<n,void> ci)
{
  typename std::vector<CountItem<n,void> >::iterator it;
  it=lower_bound(v->begin(),v->end(),ci,smallerAddr<n,void>());
  if (eqAddr<n,void>()(*it,ci)) {
    return 1;
  } else {
    return 0;
  }
}


template <int n, typename T>
void compactify(std::vector<CountItem<n,T> >*v)
{
  typename std::vector<CountItem<n,T> >::iterator it, it2;
  it=it2=v->begin();
  eqAddr<n,T> eq;
  for(++it2;it2!=v->end(); it2++)
  {
    if (eq(*it,*it2)) {
      it->item+=it2->item;
    } else {
      ++it;
      *it=*it2;
    }
  }
  v->erase(++it,v->end());
}

template <int n>
void compactify_set(std::vector<CountItem<n,void> >*v)
{
  typename std::vector<CountItem<n,void> >::iterator it, it2;
  it=it2=v->begin();
  eqAddr<n,void> eq;
  for(++it2;it2!=v->end(); it2++)
  {
    if (!eq(*it,*it2)) {
      ++it;
      *it=*it2;
    }
  }
  v->erase(++it,v->end());
}

//TODO: partial_sort (sorts second half and does an in-place merge)

template<typename T>
struct CSRMatrix {
  coordinate_t num_rows;
  int *offsets;
  coordinate_t *rightColumns;
  T *values;
  void write_binary(int fileno);
  CSRMatrix<T> *transpose();
  void compute_left_marginals(T *result);
  void compute_right_marginals(T *result);
  void compute_right_squared_marginals(T *result);
};

template<typename T>
CSRMatrix<T> *new_csr(int numRows, int numNonZero)
{
  CSRMatrix<T> *result=new CSRMatrix<T>();
  if (numRows==0) {
    result->num_rows=0;
    result->offsets=new int[1];
    result->offsets[0]=0;
    result->rightColumns=NULL;
    result->values=NULL;
    return result;
  } else {
    result->num_rows=numRows;
    result->offsets=new int[numRows+1];
    result->rightColumns=new coordinate_t[numNonZero];
    result->values=new T[numNonZero];
  }
  return result;
}

template<typename T>
void CSRMatrix<T>::write_binary(int fileno)
{
    int magic=CSR_MAGIC;
    write(fileno,&magic,sizeof(int));
    magic=typecode<T>()();
    write(fileno,&magic,sizeof(int));
    write(fileno,&num_rows,sizeof(coordinate_t));
    write(fileno,offsets,(num_rows+1)*sizeof(int));
    write(fileno,rightColumns,(offsets[num_rows])*sizeof(int));
    write(fileno,values,(offsets[num_rows])*sizeof(T));
}

template<typename T>
CSRMatrix<T> *CSRMatrix<T>::transpose()
{
  CSRMatrix<T> *result=new CSRMatrix<T>();
  coordinate_t i;
  if (num_rows==0)
  {
    result->num_rows=0;
    result->offsets=new int[1];
    result->offsets[0]=0;
    result->rightColumns=NULL;
    result->values=NULL;
  }
  coordinate_t max_rows=0;
  // determine necessary number of rows
  int oldoff=0;
  for (i=0; i<num_rows;i++) {
    int newoff=offsets[i+1];
    if (newoff>oldoff) {
      coordinate_t last_row=rightColumns[newoff-1];
      if (last_row>max_rows) max_rows=last_row;
    }
  }
  max_rows+=1;
  result->num_rows=max_rows;
  result->offsets=new int[max_rows+1];
  result->rightColumns=new coordinate_t[offsets[num_rows]];
  result->values=new T[offsets[num_rows]];
  int *filled=result->offsets;
  for (i=0; i<max_rows; i++) filled[i]=0;
  int off;
  // first, we count how many values each k2 has
  for (off=0;off<offsets[num_rows];off++)
  {
    int k2=rightColumns[off];
    filled[k2]++;
  }
  off=0;
  // then, we cumulatively sum these values
  // to get the real offset table
  for (i=0; i<max_rows;i++) {
    int tmp=filled[i];
    filled[i]=off;
    off+=tmp;
  }
  // we butcher the offset table by misusing each entry as a pointer
  // where the next thing has to go
  for (off=0,i=0; i<num_rows; i++)
  {
    for (;off<offsets[i+1]; off++)
    {
      coordinate_t k2=rightColumns[off];
      int offR=filled[k2];
      result->rightColumns[offR]=i;
      result->values[offR]=values[off];
      filled[k2]=offR+1;
    }
  }
  // and, since every offset is now the beginning of the *next*
  // column, we copy everything one to the left. phew.
  for (int i=max_rows;i>0;i--)
  {
    filled[i]=filled[i-1];
  }
  filled[0]=0;
  return result;
}

template<typename T>
void CSRMatrix<T>::compute_left_marginals(T *result)
{
  T sum;
  for (coordinate_t i=0; i<num_rows; i++)
  {
    sum=0;
    for (int off=offsets[i];off<offsets[i+1];off++)
    {
      sum+=values[off];
    }
    result[i]=sum;
  }
}

template<typename T>
void CSRMatrix<T>::compute_right_marginals(T *result)
{
  for (coordinate_t i=0; i<num_rows; i++)
  {
    for (int off=offsets[i];off<offsets[i+1]; off++)
    {
      result[rightColumns[off]]+=values[off];
    }
  }
}

template<typename T>
void CSRMatrix<T>::compute_right_squared_marginals(T *result)
{
  for (coordinate_t i=0; i<num_rows; i++)
  {
    for (int off=offsets[i];off<offsets[i+1]; off++)
    {
      T val=values[off];
      result[rightColumns[off]]+=val*val;
    }
  }
}
    
template<typename T>
CSRMatrix<T> *vec2csr(std::vector<CountItem<2,T> > *v)
{
  CSRMatrix<T> *result;
  result=new CSRMatrix<T>();
  if (v->empty())
  {
    result->num_rows=0;
    result->offsets=new int[1];
    result->offsets[0]=0;
    result->rightColumns=NULL;
    result->values=NULL;
    return result;
  }
  int num_rows=v->end()[-1].addr[0]+1;
  result->num_rows=num_rows;
  result->offsets=new int[num_rows+1];
  result->rightColumns=new coordinate_t[v->size()];
  result->values=new T[v->size()];
  typename std::vector<CountItem<2,T> >::iterator it;
  coordinate_t old_row=0;
  coordinate_t *cur_rc=result->rightColumns;
  T *cur_val=result->values;
  result->offsets[0]=0;
  for (it=v->begin();it!=v->end();++it)
  {
    while (old_row<it->addr[0])
    {
      if (it->addr[0]>result->num_rows) {
	fprintf(stderr,"num_rows[=%d] < addr[%td][0]=%d\n",result->num_rows,it-v->begin(),it->addr[0]);
	abort();
      }
      ++old_row;
      result->offsets[old_row]=it - v->begin();
    }
    *(cur_rc++)=it->addr[1];
    *(cur_val++)=it->item;
  }
  result->offsets[old_row+1]=v->size();
  return result;
}

/**
 * adds two sparse matrices (by joining the rows)
 */
template<typename T>
CSRMatrix<T> *add_csr(CSRMatrix<T> *m1, CSRMatrix<T> *m2)
{
  CSRMatrix<T> *result;
  result=new CSRMatrix<T>();
  coordinate_t smaller_rows;
  int num_rows;
  if (m2->num_rows>=m1->num_rows)
  {
    num_rows=m2->num_rows;
    smaller_rows=m1->num_rows;
  }
  else
  {
    num_rows=m1->num_rows;
    smaller_rows=m2->num_rows;
  }
  result->num_rows=num_rows;
  result->offsets=new int[num_rows+1];
  result->offsets[0]=0;
  int off1=0;
  int off2=0;
  int space_needed=0;
  // first pass: determine how much space we need
  for (coordinate_t i=1;i<=smaller_rows;i++)
  {
    while (off1<m1->offsets[i] && off2<m2->offsets[i])
    {
      int k1=m1->rightColumns[off1];
      int k2=m2->rightColumns[off2];
      if (k1>=k2) off2++;
      if (k2>=k1) off1++;
      space_needed++;
    }
    if (off1<m1->offsets[i]) {
      space_needed+=m1->offsets[i]-off1;
      off1=m1->offsets[i];
    } else if (off2<m2->offsets[i]) {
      space_needed+=m2->offsets[i]-off2;
      off2=m2->offsets[i];
    }
    result->offsets[i]=space_needed;
  }
  if (m1->num_rows>smaller_rows) {
    for (int i=smaller_rows+1;i<=num_rows;i++) {
      space_needed+=m1->offsets[i]-m1->offsets[i-1];
      result->offsets[i]=space_needed;
    }
  } else if (m2->num_rows>smaller_rows) {
    for (int i=smaller_rows+1;i<=num_rows;i++) {
      space_needed+=m2->offsets[i]-m2->offsets[i-1];
      result->offsets[i]=space_needed;
    }
  }    
  result->rightColumns=new coordinate_t[space_needed];
  result->values=new T[space_needed];
  // second pass: actually fill in the data
  int offR=0;
  off1=0;
  off2=0;
  for (coordinate_t i=1;i<=smaller_rows;i++)
  {
    //printf("Row %d\n",i);
    while (off1<m1->offsets[i] && off2<m2->offsets[i])
    {
      coordinate_t k1=m1->rightColumns[off1];
      coordinate_t k2=m2->rightColumns[off2];
      //printf("k1[%d]=%d k2[%d]=%d offR=%d\n",
      //     off1,k1,off2,k2,offR);
      if (k1==k2) {
	result->rightColumns[offR]=k1;
	result->values[offR]=
	  m1->values[off1++]+
	  m2->values[off2++];
      } else if (k1<k2) {
	result->rightColumns[offR]=k1;
	result->values[offR]=
	  m1->values[off1++];
      } else {
	result->rightColumns[offR]=k2;
	result->values[offR]=
	  m2->values[off2++];
      }
      offR++;
    }
    if (off1<m1->offsets[i]) {
      int delta=m1->offsets[i]-off1;
      memcpy(result->rightColumns+offR,
	     m1->rightColumns+off1,
	     sizeof(coordinate_t) * delta);
      memcpy(result->values+offR,
	     m1->values+off1,
	     sizeof(T) * delta);
      offR+=delta;
      off1=m1->offsets[i];
    } else if (off2<m2->offsets[i]) {
      int delta=m2->offsets[i]-off2;
      memcpy(result->rightColumns+offR,
	     m2->rightColumns+off2,
	     sizeof(coordinate_t) * delta);
      memcpy(result->values+offR,
	     m2->values+off2,
	     sizeof(T) * delta);
      offR+=delta;
      off2=m2->offsets[i];
    }
  }
  if (m1->num_rows>smaller_rows) {
    int delta=m1->offsets[num_rows]-m1->offsets[smaller_rows];
    memcpy(result->rightColumns+offR,
	   m1->rightColumns+off1,
	   sizeof(coordinate_t) * delta);
    memcpy(result->values+offR,
	   m1->values+off1,
	   sizeof(T) * delta);
  } else if (m2->num_rows>smaller_rows) {
    int delta=m2->offsets[num_rows]-m2->offsets[smaller_rows];
    memcpy(result->rightColumns+offR,
	   m2->rightColumns+off2,
	   sizeof(coordinate_t) * delta);
    memcpy(result->values+offR,
	   m2->values+off2,
	   sizeof(T) * delta);
  }    
  return result;
}

/*
template <int N>
void print_itemsI(std::vector<CountItem<N,int> > *v)
{
  std::vector<CountItem<N,int> >::iterator it;
  for (it=v->begin(); it!=v->end(); it++)
  {
    for (int i=0; i<N; i++)
    {
      printf("%u ",it->addr[i]);
    }
    printf("%d\n",it->item);
  }
}

template <int N>
void print_itemsF(std::vector<CountItem<N,float> > *v)
{
  std::vector<CountItem<N,float> >::iterator it;
  for (it=v->begin(); it!=v->end(); it++)
  {
    for (int i=0; i<N; i++)
    {
      printf("%u ",it->addr[i]);
    }
    printf("%f\n",it->item);
  }
  }*/

template<typename T>
void print_csr(CSRMatrix<T> *m)
{
  printf("%d rows.\n",m->num_rows);
  for (coordinate_t i=0;i<m->num_rows;i++)
  {
    printf("%d: [%d,%d]|",i,m->offsets[i],m->offsets[i+1]-1);
    for (int j=m->offsets[i];j<m->offsets[i+1];j++)
    {
      printf(" %d:%f",m->rightColumns[j],(double)m->values[j]);
    }
    printf("\n");
  }
}

template<>
void print_csr<int>(CSRMatrix<int> *m)
{
  printf("%d rows.\n",m->num_rows);
  for (coordinate_t i=0;i<m->num_rows;i++)
  {
    printf("%d: [%d,%d]|",i,m->offsets[i],m->offsets[i+1]-1);
    for (int j=m->offsets[i];j<m->offsets[i+1];j++)
    {
      printf(" %d:%d",m->rightColumns[j],m->values[j]);
    }
    printf("\n");
  }
}

template<typename T>
CSRMatrix<T> *readCSR_binary(int fileno)
{
  int magic;
  CSRMatrix<T> *m;
  read(fileno,&magic,sizeof(int));
  if (magic!=CSR_MAGIC) {
    return NULL;
  }
  read(fileno,&magic,sizeof(int));
  if (magic!=typecode<T>()()) {
    return NULL;
  }  
  m=new CSRMatrix<T>();
  read(fileno,&m->num_rows,sizeof(int));
  read(fileno,m->offsets,(m->num_rows+1)*sizeof(int));
  read(fileno,m->rightColumns,(m->offsets[m->num_rows])*sizeof(int));
  read(fileno,m->values,(m->offsets[m->num_rows])*sizeof(T));
}

template<typename T>
int csrFromBuffer(void *buf, CSRMatrix<T> *m)
{
  int *ptr=(int *)buf;
  coordinate_t *ptr2;
  if (*ptr++!=CSR_MAGIC) {
    fprintf(stderr,"no CSR_MAGIC found!\n");
    return -1;
  }
  if (*ptr++!=typecode<T>()()) {
    fprintf(stderr,"wrong typecode!\n");
    return -1;
  }
  m->num_rows=*ptr++;
  m->offsets=ptr;
  ptr+=m->num_rows+1;
  ptr2=(coordinate_t *)ptr;
  m->rightColumns=ptr2;
  ptr2+=m->offsets[m->num_rows];
  m->values=(T *)ptr2;
  return 0;
}

static const char need_escape[]="x:(){}$#= \n\\";
const char *escape_amis(const char *x) {
  static char buf[8192];
  char *p;
  const char *q, *r;
  p=buf;
  q=x;
  while (*q) {
    if ((r=strchr(need_escape, *q))!=NULL) {
      *p++='x';
      *p++='A'+(r-need_escape);
    } else {
      *p++=*q;
    }
    q++;
  }
  *p='\0';
  return buf;
}

const char *unescape_amis(const char *x) {
  static char buf[8192];
  char *p;
  const char *q;
  p=buf;
  q=x;
  while (*q) {
    if (*q=='x') {
      *p++=need_escape[*(++q)-'A'];
    } else {
      *p++=*q;
    }
    q++;
  }
  *p='\0';
  return buf;
}
  
#endif
