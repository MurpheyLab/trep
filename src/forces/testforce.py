import trep

__all__ = ["TestForce"]

class TestForce(trep.Force):
    def __init__(self, system, name=None):
        trep.Force.__init__(self, system, name)

    def f(self, q):
        result = 1.0
        for config in self.system.configs:
            result *= config.q
            result *= config.dq
        for u in self.system.inputs:
            result *= u.u
        return result
    
    def f_dq(self, q, q1):
        result = 1.0
        for config in self.system.configs:
            if config != q1:
                result *= config.q
            result *= config.dq
        for u in self.system.inputs:
            result *= u.u
        return result
                    
    def f_dqdq(self, q, q1, q2):
        if q1 == q2:
            return 0.0;
        result = 1.0
        for config in self.system.configs:
            if config != q1 and config != q2:
                result *= config.q
            result *= config.dq
        for u in self.system.inputs:
            result *= u.u
        return result

    def f_ddq(self, q, dq1):
        result = 1.0
        for config in self.system.configs:
            result *= config.q
            if config != dq1:
                result *= config.dq
        for u in self.system.inputs:
            result *= u.u
        return result
                    
    def f_ddqdq(self, q, dq1, q2):
        result = 1.0
        for config in self.system.configs:
            if config != q2:
                result *= config.q
            if config != dq1:
                result *= config.dq
        for u in self.system.inputs:
            result *= u.u
        return result

    def f_ddqddq(self, q, dq1, dq2):
        if dq1 == dq2:
            return 0.0
        result = 1.0
        for config in self.system.configs:
            result *= config.q
            if config != dq1 and config != dq2:
                result *= config.dq
        for u in self.system.inputs:
            result *= u.u
        return result

    def f_du(self, q, u1):
        result = 1.0
        for config in self.system.configs:
            result *= config.q
            result *= config.dq
        for u in self.system.inputs:
            if u != u1:
                result *= u.u
        return result
            
    def f_dudq(self, q, u1, q2):
        result = 1.0
        for config in self.system.configs:
            if config != q2:
                result *= config.q
            result *= config.dq
        for u in self.system.inputs:
            if u != u1:
                result *= u.u
        return result

    def f_duddq(self, q, u1, dq2):
        result = 1.0
        for config in self.system.configs:
            result *= config.q
            if config != dq2:
                result *= config.dq
        for u in self.system.inputs:
            if u != u1:
                result *= u.u
        return result    

    def f_dudu(self, q, u1, u2):
        if u1 == u2:
            return 0.0
        result = 1.0
        for config in self.system.configs:
            result *= config.q
            result *= config.dq
        for u in self.system.inputs:
            if u != u1 and u != u2:
                result *= u.u
        return result    

