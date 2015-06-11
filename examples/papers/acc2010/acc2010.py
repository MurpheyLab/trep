import numpy as np
import trep

# set mass, length, and gravity:
m = 1.0; l = 1.0; g = 9.8;

# create system
system = trep.System()

# define frames
frames = [
    trep.rz("theta_1", name="Link1"), [
        trep.ty(-l, name="Mass1", mass=m), [
            trep.rz("theta_2", name="Link2"), [
                trep.ty(-l, name="Mass2", mass=m)]]],
    trep.tx(2*l, name="Link3Anchor")]

# add frames to system
system.import_frames(frames)

# add link 3 as a distance constraint
trep.constraints.Distance(system, "Mass2", "Link3Anchor", l)

# set gravity
trep.potentials.Gravity(system, (0, -g, 0))

# add and set torque input on theta_1
trep.forces.ConfigForce(system, "theta_1", "torque1")
system.get_input('torque1').u = 2.0

# solve for equilibrium configuration
system.q = system.minimize_potential_energy()

# compute null space and set velocities
h1 = system.constraints[0].h_dq(system.get_config('theta_1'))
h2 = system.constraints[0].h_dq(system.get_config('theta_2'))
system.dq = [1.000, -h1/h2]

# print configuration and linearizations
print "===================="
print "TREP RESULTS:"
print "===================="
print "q = ", system.q
print "v = ", system.dq, "\r\n"
print "State Linearization:"
print np.vstack([np.hstack([np.zeros([2,2]),np.eye(2)]),
                 np.hstack([system.f_dq(),system.f_ddq()])]), "\r\n"
print "Input Linearization:"
print system.f_du()

#################################################################
# numerical tests for validation:
def test_ddq_dq(system, q=None, eps=0.001):
    if q == None:
        q = system.q
    ddq_dq = np.zeros((system.nQ,system.nQ))
    system.q = q
    f = system.f()
    for j in range(system.nQ):
        system.configs[j].q += eps
        fp = system.f()
        system.configs[j].q -= 2*eps
        fm = system.f()
        df_approx = ((fp-f)/eps + (f-fm)/eps)/2.0
        ddq_dq[:,j] = df_approx
    return ddq_dq

def test_ddq_ddq(system, dq=None, eps=0.001):
    if dq == None:
        dq = system.dq
    ddq_ddq = np.zeros((system.nQ,system.nQ))
    system.dq = dq
    f = system.f()
    for j in range(system.nQ):
        system.configs[j].dq += eps
        fp = system.f()
        system.configs[j].dq -= 2*eps
        fm = system.f()
        df_approx = ((fp-f)/eps + (f-fm)/eps)/2.0
        ddq_ddq[:,j] = df_approx
    return ddq_ddq


def test_ddq_du(system, u=None, eps=0.001):
    if u == None:
        u = system.u
    ddq_du = np.zeros((system.nQ,system.nu))
    system.u = u
    f = system.f()
    for j in range(system.nu):
        system.inputs[j].u += eps
        fp = system.f()
        system.inputs[j].u -= 2*eps
        fm = system.f()
        df_approx = ((fp-f)/eps + (f-fm)/eps)/2.0
        ddq_du[:,j] = df_approx
    return ddq_du

print ""
print "===================="
print "NUMERICAL TESTS:"
print "===================="
print "q = ", system.q
print "v = ", system.dq, "\r\n"
print "State Linearization:"
print np.vstack([np.hstack([np.zeros([2,2]),np.eye(2)]),
                 np.hstack([test_ddq_dq(system, eps=1e-6),
                 test_ddq_ddq(system, eps=1e-6)])]), "\r\n"
print "Input Linearization:"
print test_ddq_du(system, eps=1e-6)
