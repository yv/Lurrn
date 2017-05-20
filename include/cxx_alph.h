#ifndef CXX_ALPH_H
#define CXX_ALPH_H
#include <unordered_map>
#include <vector>
#include <Python.h>

#ifdef __APPLE__
#include <malloc/malloc.h>
#else
#include <obstack.h>
#define obstack_chunk_alloc PyMem_Malloc
#define obstack_chunk_free PyMem_Free
#endif

struct eqstr {
  bool operator()(const char *s1,
		  const char *s2) const
  {
    return (strcmp(s1,s2)==0);
  }
};

struct hashstr {
    std::size_t operator()(const char *s) const {
        std::size_t hash = 0xbaffe701;
        while (*s) {
            hash = 101 * hash + *s++;
        }
        return hash;
    }
};

typedef std::unordered_map<
  const char *, int,
  hashstr, eqstr> t_dict;

struct CPPAlphabet{
  t_dict dictionary;
  std::vector<const char *> words;
#ifdef __APPLE__
  malloc_zone_t *space;
#else
  struct obstack space;
#endif
  int growing;
  CPPAlphabet()
  {
#ifdef __APPLE__
    space = malloc_create_zone(2 << 20, 0);
#else
    obstack_init(&space);
#endif
    growing=1;
  }
  ~CPPAlphabet() {
#ifdef __APPLE__
    malloc_destroy_zone(space);
#else
    obstack_free(&space,NULL);
#endif
  }
  int size() { return words.size(); }
  int sym2num(const char *sym, bool add=true) {
    t_dict::iterator it=dictionary.find(sym);
    if (it==dictionary.end()) {
      if (growing && add) {
        int n=words.size();
#ifdef __APPLE__
	char *w=(char *)malloc_zone_malloc(space, strlen(sym)+1);
#else
        char *w=(char *)obstack_alloc(&space,strlen(sym)+1);
#endif
        strcpy(w,sym);
        words.push_back(w);
        dictionary[w]=n;
        return n;
      } else {
        return -1;
      }
    } else {
      return it->second;
    }
  }
  const char *num2sym(int num) {
    if (num<0 || num>=(int)words.size()) {
      return NULL;
    }
    return words[num];
  }
};

/* hideous macro to use placement-new in Cython,
   since Cython doesn't initialize C++ objects in
   extension types :-/
*/
#define placement_new_CPPAlphabet(where) (new(where) CPPAlphabet())
#endif /* CXX_ALPH_H */
