from libcpp.unordered_map cimport unordered_map
from libcpp.utility cimport pair
from libcpp.vector cimport vector

cdef extern from "util.hh" nogil:
    cdef cppclass replacement_map:
        cppclass iterator:
            pair[pair[int, int], int] &operator*()
            iterator& operator++()
            bint operator==(const iterator&)
            bint operator!=(const iterator&)

        iterator begin()
        int& operator[](const pair[int, int] &)
        iterator end()