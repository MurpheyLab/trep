import trep
from _trep import _Spline
import numpy as np
import numpy.linalg

class Spline(_Spline):
    def __init__(self, data):
        self._data = data

        self._x_points = None
        self._coeffs = None
        
        x = [float(d[0]) for d in data]
        y = [float(d[1]) for d in data]

        dy = [None]*len(x)
        ddy = [None]*len(x)
        constraints = 0
        for i in range(len(x)):
            if len(data[i]) > 2 and data[i][2] is not None:
                dy[i] = float(data[i][2])
                constraints += 1
            if len(data[i]) > 3 and data[i][3] is not None:
                ddy[i] = float(data[i][3])
                constraints += 1
        
        if ddy[0] is None and constraints < 2:
            print "Spline: Not enough constraints - Assuming initial second derivative is zero"
            ddy[0] = 0.0
            constraints += 1
        if ddy[-1] is None and constraints < 2:
            print "Spline: Not enough constraints - Assuming final second derivative is zero"
            ddy[-1] = 0.0
            constraints += 1


        N = len(x)-2
        A = np.zeros( (5*(N+1), 5*(N+1)) )
        b = np.zeros( (5*(N+1), ) )

        #print A.shape
        #print b.shape

        def ai(i): return 5*i + 0
        def bi(i): return 5*i + 1
        def ci(i): return 5*i + 2
        def di(i): return 5*i + 3
        def ei(i): return 5*i + 4

        row = 0
        _borrow = [0]*(N+1)

        def borrow(index):
            if index > 0 and _borrow[index-1] == 0:
                index = index-1
            elif _borrow[index] == 0:
                pass
            elif index > 0 and _borrow[index-1] == 1:
                index = index-1
            elif _borrow[index] == 1:
                pass
            else:
                # If we get here, it means that
                #print _borrow
                while index > 0:
                    if _borrow[index] < 2:
                        break
                    index -= 1
                #print "borrowing from index %d" % index
                
            _borrow[index] += 1
            if _borrow[index] == 1:
                #print "Killing a_%d" % index
                return ai(index)
            else:
                #print "Killing b_%d" % index
                return bi(index)
                    
        def borrow2(index):
            if _borrow[index] == 0:
                pass
            elif index > 0 and _borrow[index-1] == 0:
                index = index-1
            elif _borrow[index] == 1:
                pass
            elif index > 0 and _borrow[index-1] == 1:
                index = index-1
            else:
                # If we get here, it means that
                #print "FUCK: "
                #print _borrow
                while index > 0:
                    if _borrow[index] < 2:
                        break
                    index -= 1
                #print "borrowing from index %d" % index
                
            _borrow[index] += 1
            if _borrow[index] == 1:
                #print "Killing a_%d" % index
                return ai(index)
            else:
                #print "Killing b_%d" % index
                return bi(index)

        for i in range(0, N+1):
            dx = x[i+1] - x[i]

            A[row, ai(i)] = dx**5
            A[row, bi(i)] = dx**4
            A[row, ci(i)] = dx**3
            A[row, di(i)] = dx**2
            A[row, ei(i)] = dx
            b[row] = y[i+1] - y[i]
            row += 1

            if i < N:
                A[row, ai(i)] = 5.0*dx**4
                A[row, bi(i)] = 4.0*dx**3
                A[row, ci(i)] = 3.0*dx**2
                A[row, di(i)] = 2.0*dx
                A[row, ei(i)] = 1.0
                A[row, ei(i+1)] = -1.0
                b[row] = 0.0
                row +=1

                A[row, ai(i)] = 20.0*dx**3
                A[row, bi(i)] = 12.0*dx**2
                A[row, ci(i)] =  6.0*dx
                A[row, di(i)] = 2.0
                A[row, di(i+1)] = -2.0
                b[row] = 0.0
                row += 1

            if dy[i] is None:
                index = borrow(i)
                A[row, index] = 1.0
                b[row] = 0.0
                #pass
            else:
                A[row, ei(i)] = 1.0
                b[row] = dy[i]
            row += 1

            if ddy[i] is None:
                index = borrow(i)
                A[row, index] = 1.0
                b[row] = 0.0
                #pass
            else:
                A[row, di(i)] = 2.0
                b[row] = ddy[i]
            row += 1

        dx = x[N+1] - x[N]
        if dy[N+1] is None:
            index = borrow2(N)
            A[row, index] = 1.0
            b[row] = 0.0
        else:
            #print "row: %d, N: %d" % (row, N)
            #print "len(dy): %d" % len(dy)
            A[row, ai(N)] = 5.0*dx**4
            A[row, bi(N)] = 4.0*dx**3
            A[row, ci(N)] = 3.0*dx**2
            A[row, di(N)] = 2.0*dx
            A[row, ei(N)] = 1.0
            b[row] = dy[N+1]
        row += 1
        
        #print "crr"
        if ddy[N+1] is None:
            index = borrow2(N)
            A[row, index] = 1.0
            b[row] = 0.0
        else:
            A[row, ai(N)] = 20.0*dx**3
            A[row, bi(N)] = 12.0*dx**2
            A[row, ci(N)] =  6.0*dx
            A[row, di(N)] =  2.0
            b[row] = ddy[N+1]
        row += 1

        sol = np.linalg.solve(A,b)

        coeffs = []

        for i in range(0, N+1):
            coeffs.append( (sol[ai(i)],
                            sol[bi(i)],
                            sol[ci(i)],
                            sol[di(i)],
                            sol[ei(i)],
                            y[i]) )
                            
        # Add polynomial to beginning
        y0 = y[0]
        dy0 = coeffs[0][4]
        ddy0 = 2*coeffs[0][3]

        xi = x[0] - (x[-1] - x[-2]) # Arbitrary, just something less than x[0]
        dx = x[0] - xi
        d = 0.5 * ddy0
        e = dy0 - ddy0*dx
        f = y0 - dy0*dx + 0.5*ddy0*dx**2
        coeffs.insert(0, (0, 0, 0, d, e, f))
        x = [xi] + x
        y = [f] + y

        # Add polynomial to end
        dx = x[-1] - x[-2]
        yN = y[-1]
        dyN = (5*coeffs[-1][0]*dx**4 + 4*coeffs[-1][1]*dx**3 + 3*coeffs[-1][2]*dx**2
               + 2*coeffs[-1][3]*dx + coeffs[-1][4])
        ddyN = 20*coeffs[-1][0]*dx**3 + 12*coeffs[-1][1]*dx**2 + 6*coeffs[-1][2]*dx + 2*coeffs[-1][3]
        
        d = 0.5 * ddyN
        e = dyN
        f = yN
        coeffs.append( (0, 0, 0, d, e, f) )
        x = x + [x[-1] + dx]
        y = y + [d*dx**2 + e*dx + f]

        self._x_points = np.array(x, np.double, order='C')
        self._y_points = np.array(y, np.double, order='C')
        self._coefficients = np.array(coeffs, np.double, order='C')

    @property
    def x_points(self):
        """A list of the x points that define this spline."""
        return self._x_points.copy()

    @property
    def y_points(self):
        """A list of the y points that define this spline."""
        return self._y_points.copy()

    @property
    def coefficients(self):
        """Coefficients of the interpolating polynomials."""
        return self._coefficients.copy()

    def y(self, x):
        """Evaluate the value of the spline at x."""
        return self._y(x)

    def dy(self, x):
        """Evaluate the first derivative of the spline at x."""
        return self._dy(x)

    def ddy(self, x):
        """Evaluate the second derivative of the spline at x."""
        return self._ddy(x)

    def copy(self):
        """Create a new copy of this spline."""
        return Spline(self._data[:])


