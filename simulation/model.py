class Airplane:
    def __init__(self, x, y, h, phi, v, h_min=0, h_max=38000, v_min=100, v_max=300):
        self.x = x
        self.y = y
        self.h = h
        self.v = v
        self.phi = phi
        if (v < v_min) or (v > v_max):
            raise ValueError("invalid velocity")
        if (h < h_min) or (h > h_max):
            raise ValueError("invalid altitude")
        self.h_dot = [-1000,0,1000]
        self.v_dot = [-5,0,5]
        self.phi_dot = [-3,0,3]
        self.h_set = None
        self.v_set = None
        self.phi_set = None
        self.type = "no airplane defined"    
        
    def overMVA(self, MVA):
        if self.h >= MVA:
            return True
        else: return False

    def command(self, h_set=None, v_set=None, phi_set=None):
        self.h_set = h_set
        self.v_set = v_set
        self.phi_set = phi_set
        
class Airspace:
    def __init__(self, *area):#every area needs a MVA
        self.areas = []
        for i in area:
            self.areas.append(i)
