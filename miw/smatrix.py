# A really simple matrix package in pure Python
# Co-exists with numpy, sympy, theano and other symbolic packages.
import __builtin__
class matrix(list):
    def transpose(m):
        return matrix(map(list,zip(*m)))

    # ~a
    def __invert__(m): return m.transpose()

    def minor(m,i,j):
        return matrix([row[:j] + row[j+1:] for row in (m[:i]+m[i+1:])])

    def determinant2(m):
        if len(m) == 0:
            det = 1
        elif len(m) == 1:
            det = m[0][0]
        elif len(m) == 2:
            det= m[0][0]*m[1][1]-m[0][1]*m[1][0]
        else:
            det = matrix(
                [[m[i][j]*m[0][0]-m[i][0]*m[0][j]
                    for j in range(1,len(m[i]))]
                        for i in range(1,len(m))]).determinant2()
            for i in range(1,len(m)-1): det = det/m[0][0]
        return det

    def determinant(m):
        if len(m) == 0:
            return 1
        else:
            if not isinstance(m[0],list) or len(m[0]) != len(m):
                raise ValueError("not square")
            return __builtin__.sum([
                    ((-1)**c)*m[0][c]*m.minor(0,c).determinant()
                        for c in range(len(m))])

    #def __abs__(m): return m.determinant()

    def cofactors(m):
        cofactors = matrix([])
        for r in range(len(m)):
            cofactorRow = []
            for c in range(len(m)):
                minor = m.minor(r,c)
                cofactorRow.append(((-1)**(r+c)) * minor.determinant())
            cofactors.append(cofactorRow)
        return cofactors

    def inverse(m):
        determinant = m.determinant()
        return matrix([[c/determinant for c in r]
                for r in m.cofactors().transpose()])

    def __mul__(p,q):
        if isinstance(q,matrix):
            qt = q.transpose()
            if len(p)!=len(qt):
                raise ValueError("incompatible")
            return matrix([[__builtin__.sum([i*j for i,j in zip(r,c)])
                     for c in qt] for r in p])
        else: # assume scalar
            return matrix([[i*q for i in r] for r in p])

    def __rmul__(p,q):
        return matrix([[q*i for i in r] for r in p])

    def __neg__(p):
        return p*-1

    def __truediv__(p,q):
        return p*q.inverse()

    def __rtruediv__(p,q):
        return q*p.inverse()

    def __pow__(p,n):
        if n>0:
            return p*(p**(n-1))
        elif n<0:
            return p.inverse()*(p**(n+1))
        else:
            return matrix([[1 if i==j else 0 for i in range(len(p[j]))] for j in range(len(p))])

    def __add__(p,q):
        if len(p)!=len(q):
            raise ValueError("incompatible")
        return matrix([[i+j for i,j in zip(rp,rq)] for rp,rq in zip(p,q)])

    def __sub__(p,q):
        if len(p)!=len(q):
            raise ValueError("incompatible")
        return matrix([[i-j for i,j in zip(rp,rq)] for rp,rq in zip(p,q)])

    def __eq__(p,q):
        if len(p)!=len(q):
            raise ValueError("incompatible")
        return [[i==j for i,j in zip(rp,rq)] for rp,rq in zip(p,q)]

    def __getitem__(self, key):
        if isinstance(key,tuple) and len(key)==2:
            return self[key[0]][key[1]]
        else:
            return list(self)[key]

    def __str__(self):
        maxlen = max(1,1,*[len(str(e).strip()) for row in self for e in row])
        return '\n\n'.join('  '.join(str(e).strip().rjust(maxlen)
                         for e in row)
               for row in self
        )

    def __repr__(self):
        return '<%s %sx%s 0x%x>' % (
            self.__class__.__name__, len(list(self)), 0 if len(list(self))==0 else len(list(self)[0]), id(self)
        )
