#ifndef __POINT_H__
#define __POINT_H__

#include <iostream>
#include <iomanip>

// FIXME multiple definition of `printErrDesc(int)', first defined here #include "errors.hpp"

using std::ostream;
using std::setw;
using std::setprecision;

template <typename T> 
class Point {

  private:
    T *x;
    unsigned int cluster, d;
  
  public:
    Point() {
      this->x = NULL;
      this->d = 0;
      this->cluster = 0;
    }

    Point(T *_x, unsigned int _d): cluster(0), d(_d) {
      this->x = new T[_d];
      for (unsigned int i = 0; i < _d; i++) {
        x[i] = _x[i];
      }
    }

    Point(Point<T>* const p): cluster(p->cluster), d(p->d) {
      x = new T[d];
      for (unsigned int i = 0; i < d; i++) {
        x[i] = p->x[i];
      }
    }

    int getCluster () { return cluster; }
    void setCluster (unsigned int c) { cluster = c; }

    T get(unsigned int i) {
      if (i >= d) {
        // FIXME printErrDesc(EXIT_POINT_IOB); exit(EXIT_POINT_IOB);
        exit(3);
      }
      return x[i];
    }

    ~Point() {
      delete[] x;
    }

    Point& operator= (Point const &p) {
      if (x) { delete[] x; }
      x = new T[p.d];
      
      for (unsigned int i = 0; i < p.d; i++) {
        x[i] = p.x[i];
      }
      d = p.d;

      return *this;
    }

    bool operator== (const Point<T> &p) const {
      if (d != p.d) return false;
      for (unsigned int i = 0; i < d; ++i)
        if (x[i] != p.x[i])
          return false;
      return true;
    }

    friend ostream& operator<< (ostream &os, Point const &p) {
      os << setw(9) << p.cluster;
      for (unsigned int i = 0; i < p.d; i++) {
        os << setw(9) << setprecision(5) << p.x[i];
      }
      return os;
    }
};

#endif
